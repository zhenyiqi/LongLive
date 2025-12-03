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
                
                # Generate next block
                block_start = self.current_frame
                block_end = min(self.current_frame + self.block_size, self.total_frames)
                actual_block_size = block_end - block_start
                
                if actual_block_size > 0:
                    self._generate_and_stream_block(
                        block_start, actual_block_size, current_encoded_prompt
                    )
                    
                    self.current_frame = block_end
                
                # Check if we've reached 1 minute
                elapsed = time.time() - self.start_time
                if elapsed >= self.duration_seconds:
                    print(f"[RealTime] Reached {self.duration_seconds}s duration limit")
                    break
                    
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
                    
                    # Clear cache for smooth transition
                    self._clear_cache_for_switch()
                    return True
                else:
                    # Not time yet, put it back
                    self.prompt_switch_queue.put(switch_event)
                    break
                    
        except queue.Empty:
            pass
        
        return False
    
    def _clear_cache_for_switch(self):
        """Clear relevant cache entries for smoother prompt switching"""
        # Clear cross-attention cache (prompt-dependent) - clear token dimension (dim=1)
        for cache_entry in self.base_pipeline.crossattn_cache:
            cache_entry["k"] = cache_entry["k"][:, 0:0, :, :]
            cache_entry["v"] = cache_entry["v"][:, 0:0, :, :]
            cache_entry["is_init"] = False
        
        # Keep some KV cache tokens for temporal consistency, but reduce it
        # Convert frames to tokens using frame_seq_length
        keep_frames = min(1, self.block_size)  # Keep at most 1 recent frame
        keep_tokens = keep_frames * self.base_pipeline.frame_seq_length
        for cache_entry in self.base_pipeline.kv_cache1:
            token_len = cache_entry["k"].shape[1]
            if token_len > keep_tokens:
                cache_entry["k"] = cache_entry["k"][:, -keep_tokens:, :, :]
                cache_entry["v"] = cache_entry["v"][:, -keep_tokens:, :, :]
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    
    def _reset_cache_for_first_block(self):
        """Reset cache state for first block generation"""
        # Reset all cache entries to initial state
        for cache_entry in self.kv_cache:
            cache_entry["k"] = torch.zeros_like(cache_entry["k"])
            cache_entry["v"] = torch.zeros_like(cache_entry["v"])
            cache_entry["is_init"] = False
            
        for cache_entry in self.crossattn_cache:
            cache_entry["k"] = torch.zeros_like(cache_entry["k"])
            cache_entry["v"] = torch.zeros_like(cache_entry["v"])
            cache_entry["is_init"] = False
                
    def _generate_and_stream_block(self, start_frame: int, block_size: int, encoded_prompt: torch.Tensor):
        """Generate a block using base pipeline cached denoising and stream frames"""
        try:
            # Extract noise for this block
            noise_block = self.noise[:, start_frame:start_frame + block_size]
            
            # Generate latents one frame at a time to match causal model expectations (1560 tokens per call)
            per_frame_latents = []
            with torch.no_grad():
                for offset in range(block_size):
                    cur_start = start_frame + offset
                    noise_single = noise_block[:, offset:offset + 1]  # shape [B, 1, C, H, W]
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
            
            # Concatenate per-frame latents along time dimension
            latents = torch.cat(per_frame_latents, dim=1)  # [B, block_size, C, H, W]
            
            # Update KV cache with context timestep for next block (optional; skip if low memory)
            self._maybe_update_kv_context(latents, encoded_prompt, start_frame, block_size)
            
            # Decode latents to pixel frames
            with torch.no_grad():
                video_frames = self.base_pipeline.vae.decode_to_pixel(latents, use_cache=False)
                
            # Convert to numpy and stream frames
            video_np = video_frames.cpu().numpy()  # [B, T, C, H, W]
            video_np = (video_np * 0.5 + 0.5).clip(0, 1)  # Normalize to [0,1]
            
            print(f"[RealTime] Video frames shape after processing: {video_np.shape}")
            print(f"[RealTime] Streaming {video_np.shape[1]} frames...")
            
            # Only take the first block_size frames to avoid the extra frames
            num_frames_to_stream = min(block_size, video_np.shape[1])
            
            for t in range(num_frames_to_stream):
                frame = video_np[0, t].transpose(1, 2, 0)  # [H, W, C]
                frame_uint8 = (frame * 255).astype(np.uint8)
                frame_idx = start_frame + t
                
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
                        
        except Exception as e:
            print(f"[RealTime] Block generation error: {e}")
            raise

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
    
    def _generate_with_kv_cache(self, noise: torch.Tensor, conditional_dict: dict, 
                               start_frame: int, num_frames: int, frame_seq_length: int) -> torch.Tensor:
        """Generate using KV cache for temporal consistency like interactive pipeline"""
        # Get denoising steps
        denoising_step_list = self.base_pipeline.noise_scheduler.timesteps
        
        # Initialize input noise
        noisy_input = noise
        
        # Denoising loop with KV cache
        for index, current_timestep in enumerate(denoising_step_list):
            timestep_val = float(current_timestep)
            
            # Create timestep tensor
            timestep = torch.full(
                (1, num_frames), timestep_val, device=self.device, dtype=torch.float32
            )
            
            if index < len(denoising_step_list) - 1:
                # Intermediate denoising step - use KV cache
                torch.compiler.cudagraph_mark_step_begin()
                _, denoised_pred = self.base_pipeline.generator._compiled_forward(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start=start_frame * frame_seq_length,
                )
                
                # Add noise for next step (matching interactive pipeline approach)
                next_timestep = denoising_step_list[index + 1]
                next_timestep_val = float(next_timestep)
                
                # Create noise timestep tensor
                noise_timestep = torch.full(
                    (num_frames,), next_timestep_val, device=self.device, dtype=torch.float32
                )
                
                # Add noise using the scheduler
                noisy_input = self.base_pipeline.noise_scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    noise_timestep
                ).unflatten(0, (1, num_frames))
                
            else:
                # Final denoising step - generates clean output
                torch.compiler.cudagraph_mark_step_begin() 
                clean_output, _ = self.base_pipeline.generator._compiled_forward(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start=start_frame * frame_seq_length,
                )
                
                return clean_output
    
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