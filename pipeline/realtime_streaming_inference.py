#!/usr/bin/env python3
"""
Real-Time Streaming Inference Pipeline

A completely new pipeline designed for real-time user interaction:
- Starts generation when first prompt is sent
- Switches prompts immediately when user clicks "send" 
- Generates continuously for 1 minute (16fps = 960 frames)
- Uses dynamic switch points based on user actions, not pre-defined indices
"""

import torch
import torch.nn as nn
import time
import threading
import queue
from typing import List, Optional, Dict, Callable
from dataclasses import dataclass
from omegaconf import OmegaConf
import numpy as np

from pipeline.interactive_causal_inference import InteractiveCausalInferencePipeline
from utils.misc import set_seed
from utils.memory import get_cuda_free_memory_gb
from typing import Any


@dataclass
class PromptSwitchEvent:
    """Event triggered when user sends a new prompt"""
    prompt: str
    timestamp: float
    target_frame: int  # Frame where this prompt should take effect


@dataclass 
class GenerationStatus:
    """Current generation status"""
    is_running: bool
    current_frame: int
    total_frames: int
    current_prompt: str
    fps: float
    elapsed_time: float
    start_time: Optional[float]


class RealTimeStreamingPipeline:
    """
    Real-time streaming inference pipeline for dynamic prompt switching
    
    Key differences from batch inference:
    1. Generates frames continuously, one block at a time
    2. Switches prompts immediately when user sends new ones
    3. Uses frame-by-frame recaching for smooth transitions
    4. Stops after exactly 1 minute of generation
    """
    
    def __init__(self, config_path: str, device: str = "cuda"):
        self.config = OmegaConf.load(config_path)
        self.device = torch.device(device)
        
        # Video parameters
        # Read fps from config if provided; fall back to 16
        self.fps = int(getattr(self.config, "fps", 16))
        # Use config-provided total frames (e.g., 240). Derive duration from fps.
        self.total_frames = int(getattr(self.config, "num_output_frames", 960))
        self.duration_seconds = self.total_frames / self.fps
        self.block_size = self.config.num_frame_per_block  # Typically 3
        
        # Initialize base pipeline
        set_seed(self.config.seed)
        torch.set_grad_enabled(False)
        self.base_pipeline = InteractiveCausalInferencePipeline(self.config, device=self.device)
        self._setup_pipeline()
        
        # Real-time state
        self.is_running = False
        self.generation_thread = None
        self.current_frame = 0
        self.current_prompt = ""
        self.start_time = None
        # Keep recent latents to support recache on prompt switch (store all up to total_frames)
        from collections import deque
        self._latents_history = deque(maxlen=int(self.total_frames))
        
        # Prompt switching
        self.prompt_switch_queue = queue.Queue()
        self.pending_prompt_switch = None
        self.next_switch_frame = None
        
        # Frame output
        self.frame_callbacks: List[Callable[[np.ndarray, int], None]] = []
        self.latest_frame = None
        
        # Pre-allocated noise for entire video
        self.noise = torch.randn([
            1, self.total_frames, 16, 60, 104
        ], device=self.device, dtype=torch.bfloat16)
        
        print(f"[RealTime] Pipeline initialized for {self.duration_seconds}s video ({self.total_frames} frames)")
        
    def _setup_pipeline(self):
        """Setup the base pipeline for real-time use"""
        # Load weights
        if self.config.generator_ckpt:
            self._load_generator_weights()
        
        # Configure pipeline
        self.base_pipeline = self.base_pipeline.to(dtype=torch.bfloat16)
        self.base_pipeline.generator.to(device=self.device)
        self.base_pipeline.vae.to(device=self.device)
        
        # Optional: setup LoRA if configured
        try:
            adapter_cfg: Any = getattr(self.config, "adapter", None)
            if adapter_cfg:
                from utils.lora_utils import configure_lora_for_model
                import peft
                print(f"[RealTime] Enabling LoRA with config: {adapter_cfg}")
                self.base_pipeline.generator.model = configure_lora_for_model(
                    self.base_pipeline.generator.model,
                    model_name="generator",
                    lora_config=adapter_cfg,
                    is_main_process=True,
                )
                lora_ckpt_path = getattr(self.config, "lora_ckpt", None)
                if lora_ckpt_path:
                    print(f"[RealTime] Loading LoRA checkpoint from {lora_ckpt_path}")
                    lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
                    if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
                        peft.set_peft_model_state_dict(self.base_pipeline.generator.model, lora_checkpoint["generator_lora"])
                    else:
                        peft.set_peft_model_state_dict(self.base_pipeline.generator.model, lora_checkpoint)
                    print("[RealTime] LoRA weights loaded")
        except Exception as e:
            print(f"[RealTime] LoRA setup skipped due to error: {e}")
        
        # Pre-allocate timestep tensors for efficiency
        self.base_pipeline._pre_allocate_timestep_tensors(1, self.block_size, self.device)
        
        # Initialize base pipeline caches and attention configuration
        local_attn_cfg = getattr(self.config.model_kwargs, "local_attn_size", -1)
        if local_attn_cfg != -1:
            kv_cache_size = local_attn_cfg * self.base_pipeline.frame_seq_length
        else:
            kv_cache_size = self.total_frames * self.base_pipeline.frame_seq_length
        # Fresh caches
        self.base_pipeline._initialize_kv_cache(
            batch_size=1,
            dtype=torch.bfloat16,
            device=self.device,
            kv_cache_size_override=kv_cache_size
        )
        self.base_pipeline._initialize_crossattn_cache(
            batch_size=1,
            dtype=torch.bfloat16,
            device=self.device
        )
        # Apply attention size to all modules
        self.base_pipeline.generator.model.local_attn_size = self.base_pipeline.local_attn_size
        self.base_pipeline._set_all_modules_max_attention_size(self.base_pipeline.local_attn_size)
        
        # Per-frame fallback control
        self._force_per_frame = False
        
    def _load_generator_weights(self):
        """Load generator weights"""
        state_dict = torch.load(self.config.generator_ckpt, map_location="cpu")
        raw_gen_state_dict = state_dict["generator_ema" if self.config.use_ema else "generator"]

        if self.config.use_ema:
            def _clean_key(name: str) -> str:
                return name.replace("_fsdp_wrapped_module.", "")
            cleaned_state_dict = {_clean_key(k): v for k, v in raw_gen_state_dict.items()}
            self.base_pipeline.generator.load_state_dict(cleaned_state_dict, strict=False)
        else:
            self.base_pipeline.generator.load_state_dict(raw_gen_state_dict)
    
    def add_frame_callback(self, callback: Callable[[np.ndarray, int], None]):
        """Add callback to receive generated frames"""
        self.frame_callbacks.append(callback)
        
    def remove_frame_callback(self, callback: Callable[[np.ndarray, int], None]):
        """Remove frame callback"""
        if callback in self.frame_callbacks:
            self.frame_callbacks.remove(callback)
    
    def start_generation(self, initial_prompt: str):
        """Start real-time generation with the first prompt"""
        if self.is_running:
            print("[RealTime] Generation already running!")
            return
            
        print(f"[RealTime] Starting generation with prompt: '{initial_prompt}'")
        self.current_prompt = initial_prompt
        self.current_frame = 0
        self.start_time = time.time()
        self.is_running = True
        
        # Start generation thread
        self.generation_thread = threading.Thread(
            target=self._generation_loop, 
            args=(initial_prompt,),
            daemon=True
        )
        self.generation_thread.start()
        
    def stop_generation(self):
        """Stop generation immediately"""
        print("[RealTime] Stopping generation...")
        self.is_running = False
        if self.generation_thread:
            self.generation_thread.join(timeout=5.0)
            
    def reset_for_new_video(self):
        """Reset pipeline state for a new video generation"""
        print("[RealTime] Resetting pipeline for new video...")
        
        # Stop current generation if running
        if self.is_running:
            self.stop_generation()
            
        # Reset state variables
        self.current_frame = 0
        self.current_prompt = ""
        self.start_time = None
        
        # Clear prompt queue
        while not self.prompt_switch_queue.empty():
            try:
                self.prompt_switch_queue.get_nowait()
            except:
                break
                
        # Reset frame data
        self.latest_frame = None
        
        # Generate new noise for the next video
        self.noise = torch.randn([
            1, self.total_frames, 16, 60, 104
        ], device=self.device, dtype=torch.bfloat16)
        
        print("[RealTime] Pipeline reset complete - ready for new video!")
        
    def is_finished(self) -> bool:
        """Check if current video generation is finished"""
        if not self.start_time:
            return False
        elapsed = time.time() - self.start_time
        return (elapsed >= self.duration_seconds or 
                self.current_frame >= self.total_frames or 
                not self.is_running)
        
    def send_prompt(self, prompt: str):
        """Send a new prompt to switch at the next available frame"""
        if not self.is_running:
            print("[RealTime] Not running - use start_generation() first")
            return False
            
        # Check if prompt is actually different from current one
        if prompt.strip() == self.current_prompt.strip():
            print(f"[RealTime] Prompt unchanged - skipping switch: '{prompt[:50]}...'")
            return False
            
        # Calculate the target frame for this switch (current + small buffer)
        buffer_frames = max(1, self.block_size)  # Minimum buffer for smooth switching
        target_frame = self.current_frame + buffer_frames
        
        switch_event = PromptSwitchEvent(
            prompt=prompt,
            timestamp=time.time(),
            target_frame=target_frame
        )
        
        self.prompt_switch_queue.put(switch_event)
        print(f"[RealTime] Prompt queued for frame {target_frame}: '{prompt[:50]}...'")
        return True
        
    def get_status(self) -> GenerationStatus:
        """Get current generation status"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        fps_actual = self.current_frame / elapsed if elapsed > 0 else 0
        
        return GenerationStatus(
            is_running=self.is_running,
            current_frame=self.current_frame,
            total_frames=self.total_frames,
            current_prompt=self.current_prompt,
            fps=fps_actual,
            elapsed_time=elapsed,
            start_time=self.start_time
        )
    
    def _generation_loop(self, initial_prompt: str):
        """Main generation loop - runs in background thread"""
        try:
            # Encode initial prompt
            with torch.no_grad():
                current_encoded_prompt = self.base_pipeline.text_encoder([initial_prompt])["prompt_embeds"]
            
            # Initialize caches for a fresh run
            self._initialize_cache_state()
            
            # Generate blocks continuously
            while self.is_running and self.current_frame < self.total_frames:
                # Check for prompt switches
                if self._check_and_apply_prompt_switch():
                    # Re-encode new prompt
                    with torch.no_grad():
                        current_encoded_prompt = self.base_pipeline.text_encoder([self.current_prompt])["prompt_embeds"]
                    # Rebuild caches using recent frames to match Interactive behavior
                    try:
                        self._recache_after_switch(
                            prompt_embeds=current_encoded_prompt,
                            current_start_frame=self.current_frame
                        )
                    except Exception as e:
                        print(f"[RealTime] Recache step failed (continuing): {e}")
                
                # Generate next block
                block_start = self.current_frame
                block_end = min(self.current_frame + self.block_size, self.total_frames)
                actual_block_size = block_end - block_start
                
                if actual_block_size > 0:
                    self._generate_and_stream_block(
                        block_start, actual_block_size, current_encoded_prompt
                    )
                    
                    # current_frame is advanced inside _generate_and_stream_block
                
                # Do not stop based on wall-clock time; stop by total_frames only
                    
        except Exception as e:
            print(f"[RealTime] Generation error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
            print("[RealTime] Generation stopped")
    
    def _initialize_cache_state(self):
        """Initialize KV and cross-attention caches owned by base pipeline."""
        local_attn_cfg = getattr(self.config.model_kwargs, "local_attn_size", -1)
        if local_attn_cfg != -1:
            kv_cache_size = local_attn_cfg * self.base_pipeline.frame_seq_length
        else:
            kv_cache_size = self.total_frames * self.base_pipeline.frame_seq_length
        self.base_pipeline._initialize_kv_cache(
            batch_size=1,
            dtype=torch.bfloat16,
            device=self.device,
            kv_cache_size_override=kv_cache_size
        )
        self.base_pipeline._initialize_crossattn_cache(
            batch_size=1,
            dtype=torch.bfloat16,
            device=self.device
        )
        self.base_pipeline.generator.model.local_attn_size = self.base_pipeline.local_attn_size
        self.base_pipeline._set_all_modules_max_attention_size(self.base_pipeline.local_attn_size)
    
    def _check_and_apply_prompt_switch(self) -> bool:
        """Check if we should switch prompts at current frame"""
        try:
            # Check if there's a pending switch
            while not self.prompt_switch_queue.empty():
                switch_event = self.prompt_switch_queue.get_nowait()
                
                if self.current_frame >= switch_event.target_frame:
                    # Apply switch immediately
                    old_prompt = self.current_prompt
                    self.current_prompt = switch_event.prompt
                    print(f"[RealTime] Prompt switched at frame {self.current_frame}")
                    print(f"  From: '{old_prompt[:30]}...'")
                    print(f"  To:   '{self.current_prompt[:30]}...'")
                    
                    # Clear caches for smooth transition (match Interactive pipeline)
                    self._clear_cache_for_switch()
                    # Force per-frame generation on the next block to avoid size mismatches right after a switch
                    self._force_per_frame = True
                    return True
                else:
                    # Not time yet, put it back
                    self.prompt_switch_queue.put(switch_event)
                    break
                    
        except queue.Empty:
            pass
        
        return False
    
    def _clear_cache_for_switch(self):
        """Clear relevant cache entries for smoother prompt switching (match Interactive)."""
        # Match the InteractiveCausalInferencePipeline's cache clearing logic
        
        # Check if we should clear KV cache (assuming no global_sink mode for real-time)
        global_sink = getattr(self.config, "global_sink", False)
        
        if not global_sink:
            # Reset KV cache (using fill_ to preserve dimensions)
            for cache_entry in self.base_pipeline.kv_cache1:
                cache_entry["k"].fill_(0.0)
                cache_entry["v"].fill_(0.0)
                # Reset index pointers if they exist
                if "global_end_index" in cache_entry:
                    cache_entry["global_end_index"].fill_(0)
                if "local_end_index" in cache_entry:
                    cache_entry["local_end_index"].fill_(0)
            print("[RealTime] KV cache cleared for prompt switch")
        
        # Reset cross-attention cache (using fill_ to preserve dimensions)
        for cache_entry in self.base_pipeline.crossattn_cache:
            cache_entry["k"].fill_(0.0)
            cache_entry["v"].fill_(0.0)
            cache_entry["is_init"] = False
        
        print("[RealTime] Cross-attention cache cleared for prompt switch")
        
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    
    def _reset_cache_for_first_block(self):
        """Reset cache state for first block generation"""
        # Use the same cache clearing approach but for first block
        self._clear_cache_for_switch()
        print("[RealTime] Cache reset for first block")
                
    def _generate_and_stream_block(self, start_frame: int, block_size: int, encoded_prompt: torch.Tensor):
        """Generate a block using base pipeline cached denoising and stream frames"""
        try:
            # Extract noise for this block
            noise_block = self.noise[:, start_frame:start_frame + block_size]
            
            # Prefer fast block generation; fallback to per-frame if a switch just happened or if a mismatch occurs
            def generate_blockwise() -> torch.Tensor:
                if start_frame == 0:
                    return self.base_pipeline._generate_block(
                        noise=noise_block,
                        prompt_embeds=encoded_prompt,
                        start_frame=start_frame
                    )
                else:
                    return self.base_pipeline._generate_block_with_cache(
                        noise=noise_block,
                        prompt_embeds=encoded_prompt,
                        start_frame=start_frame
                    )
            
            def generate_per_frame() -> torch.Tensor:
                per_frame_latents = []
                for offset in range(block_size):
                    cur_start = start_frame + offset
                    noise_single = noise_block[:, offset:offset + 1]
                    if cur_start == 0:
                        lat_single = self.base_pipeline._generate_block(
                            noise=noise_single,
                            prompt_embeds=encoded_prompt,
                            start_frame=cur_start
                        )
                    else:
                        lat_single = self.base_pipeline._generate_block_with_cache(
                            noise=noise_single,
                            prompt_embeds=encoded_prompt,
                            start_frame=cur_start
                        )
                    per_frame_latents.append(lat_single)
                return torch.cat(per_frame_latents, dim=1)
            
            with torch.no_grad():
                if self._force_per_frame:
                    latents = generate_per_frame()
                    # Clear flag after one block
                    self._force_per_frame = False
                else:
                    try:
                        latents = generate_blockwise()
                    except RuntimeError as e:
                        # Fallback if shapes mismatch or any cache-related error occurs
                        print(f"[RealTime] Blockwise generation failed, falling back to per-frame: {e}")
                        latents = generate_per_frame()
            
            # Update KV cache with context timestep for next block (optional; skip if low memory)
            self._maybe_update_kv_context(latents, encoded_prompt, start_frame, block_size)
            
            # Append latents to history for potential recache
            try:
                # Split along time dimension and store per-frame
                for t_idx in range(latents.shape[1]):
                    self._latents_history.append(latents[:, t_idx:t_idx+1].detach())
            except Exception:
                pass
            
            # Decode latents to pixel frames
            with torch.no_grad():
                video_frames = self.base_pipeline.vae.decode_to_pixel(latents, use_cache=False)
                
            # Convert to numpy and stream frames
            video_np = video_frames.cpu().numpy()  # [B, T, C, H, W]
            video_np = (video_np * 0.5 + 0.5).clip(0, 1)  # Normalize to [0,1]
            
            print(f"[RealTime] Video frames shape after processing: {video_np.shape}")
            print(f"[RealTime] Streaming {video_np.shape[1]} frames...")
            
            # Stream all decoded frames, capped by remaining total frames
            start_video_frame = self.current_frame
            remaining_frames = max(0, self.total_frames - start_video_frame)
            num_frames_to_stream = min(video_np.shape[1], remaining_frames)
            
            for t in range(num_frames_to_stream):
                frame = video_np[0, t].transpose(1, 2, 0)  # [H, W, C]
                frame_uint8 = (frame * 255).astype(np.uint8)
                frame_idx = start_video_frame + t
                
                # Store latest frame
                self.latest_frame = frame_uint8
                print(f"[RealTime] Streaming frame {frame_idx}, shape: {frame_uint8.shape}")
                
                # Call all registered callbacks
                for callback in self.frame_callbacks:
                    try:
                        callback(frame_uint8, frame_idx)
                        print(f"[RealTime] Frame {frame_idx} sent to callback")
                    except Exception as e:
                        print(f"[RealTime] Callback error: {e}")
            
            # Advance global video frame counter
            self.current_frame += num_frames_to_stream
                        
        except Exception as e:
            print(f"[RealTime] Block generation error: {e}")
            raise
    
    def _recache_after_switch(self, prompt_embeds: torch.Tensor, current_start_frame: int):
        """Recache previous frames with new prompt conditioning to match Interactive pipeline."""
        try:
            # Determine number of frames to recache
            if self.base_pipeline.local_attn_size == -1:
                num_recache_frames = current_start_frame
            else:
                num_recache_frames = min(self.base_pipeline.local_attn_size, current_start_frame)
            if num_recache_frames <= 0:
                return
            recache_start_frame = current_start_frame - num_recache_frames
            
            # Collect frames from history
            frames_list = list(self._latents_history)[-num_recache_frames:]
            if len(frames_list) != num_recache_frames:
                # Not enough history to recache
                return
            frames_to_recache = torch.cat(frames_list, dim=1)  # [1, F, 16, 60, 104]
            device = self.device
            if frames_to_recache.device.type != 'cuda':
                frames_to_recache = frames_to_recache.to(device)
            
            # Prepare block mask for recache
            try:
                block_mask = self.base_pipeline._get_block_mask(
                    device=device,
                    num_frames=num_recache_frames,
                    local_attn_size=self.base_pipeline.local_attn_size
                )
                self.base_pipeline.generator.model.block_mask = block_mask
            except Exception:
                pass
            
            # Prepare context timestep (on device)
            batch_size = frames_to_recache.shape[0]
            cache_key = (batch_size, num_recache_frames)
            context_timestep = self.base_pipeline._recache_timestep_cache.get(cache_key)
            if context_timestep is None:
                context_timestep = torch.full(
                    [batch_size, num_recache_frames],
                    self.config.context_noise if hasattr(self.config, "context_noise") else 0,
                    device=device,
                    dtype=torch.int64
                )
                self.base_pipeline._recache_timestep_cache[cache_key] = context_timestep
            
            conditional_dict = {"prompt_embeds": prompt_embeds}
            
            # Run recache forward to rebuild KV with new prompt
            with torch.no_grad():
                self.base_pipeline.generator._compiled_forward(
                    noisy_image_or_video=frames_to_recache,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.base_pipeline.kv_cache1,
                    crossattn_cache=self.base_pipeline.crossattn_cache,
                    current_start=recache_start_frame * self.base_pipeline.frame_seq_length,
                )
            # Clear cross-attention cache again after recache
            for cache_entry in self.base_pipeline.crossattn_cache:
                cache_entry["k"].zero_()
                cache_entry["v"].zero_()
                cache_entry["is_init"] = False
        except Exception as e:
            print(f"[RealTime] Recache after switch failed (continuing): {e}")

    def _maybe_update_kv_context(self, latents: torch.Tensor, encoded_prompt: torch.Tensor, start_frame: int, block_size: int):
        """Optionally perform KV context update if enough free memory is available."""
        try:
            free_gb = get_cuda_free_memory_gb(self.device)
            # Require at least ~1.0 GB free to attempt context update to reduce OOM risk
            if free_gb < 1.0:
                print(f"[RealTime] Skipping KV context update due to low free GPU memory ({free_gb:.2f} GB)")
                return
            if block_size == self.base_pipeline.num_frame_per_block:
                context_timestep = self.base_pipeline._context_timestep_base
            else:
                context_timestep = self.base_pipeline._context_timestep_base[:, :block_size]
            conditional_dict = {"prompt_embeds": encoded_prompt}
            with torch.no_grad():
                self.base_pipeline.generator._compiled_forward(
                    noisy_image_or_video=latents,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.base_pipeline.kv_cache1,
                    crossattn_cache=self.base_pipeline.crossattn_cache,
                    current_start=start_frame * self.base_pipeline.frame_seq_length,
                )
        except Exception as e:
            print(f"[RealTime] KV context update failed (continuing): {e}")
    
    def _generate_first_block(self, noise: torch.Tensor, encoded_prompt: torch.Tensor) -> torch.Tensor:
        """Generate first block latents via base pipeline"""
        with torch.no_grad():
            return self.base_pipeline._generate_block(
                noise=noise,
                prompt_embeds=encoded_prompt,
                start_frame=0
            )
    
    def _generate_cached_block(self, noise: torch.Tensor, encoded_prompt: torch.Tensor, start_frame: int) -> torch.Tensor:
        """Generate cached block latents via base pipeline"""
        with torch.no_grad():
            return self.base_pipeline._generate_block_with_cache(
                noise=noise,
                prompt_embeds=encoded_prompt,
                start_frame=start_frame
            )
    def _generate_block_simple(self, noise: torch.Tensor, prompt: str) -> torch.Tensor:
        """Simple block generation - encode prompt and generate"""
        batch_size, num_frames = noise.shape[:2]
        
        # Encode prompt
        with torch.no_grad():
            encoded = self.base_pipeline.text_encoder([prompt])
            conditional_dict = {"prompt_embeds": encoded["prompt_embeds"]}
        
        # Use a very simple approach: just call the base pipeline with minimal args
        try:
            # Create minimal text prompts list
            text_prompts_list = [[prompt]]
            switch_frame_indices = []
            
            # Call the base pipeline inference method
            output = self.base_pipeline.inference(
                noise=noise,
                text_prompts_list=text_prompts_list,
                switch_frame_indices=switch_frame_indices,
                return_latents=True
            )
            # Handle return value - might be tuple or tensor
            if isinstance(output, tuple):
                return output[0]  # Take first element (usually latents)
            return output
        except Exception as e:
            print(f"[RealTime] Simple generation failed: {e}")
            # Return noise as last resort (will look bad but won't crash)
            print("[RealTime] Returning noise as fallback - video will look noisy")
            return noise


# Example usage for testing
if __name__ == "__main__":
    pipeline = RealTimeStreamingPipeline("configs/longlive_interactive_inference.yaml")
    
    def frame_callback(frame: np.ndarray, frame_idx: int):
        print(f"Received frame {frame_idx}: {frame.shape}")
    
    pipeline.add_frame_callback(frame_callback)
    
    # Start generation
    pipeline.start_generation("A beautiful sunset")
    
    # Simulate user interactions
    import time
    time.sleep(5)
    pipeline.send_prompt("A stormy ocean")
    time.sleep(5)
    pipeline.send_prompt("A peaceful forest")
    
    # Let it run
    time.sleep(20)
    pipeline.stop_generation()