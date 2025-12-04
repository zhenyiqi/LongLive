#!/usr/bin/env python3
"""
Real-Time Streaming Inference Pipeline V2

Based closely on InteractiveCausalInferencePipeline with key differences:
1. Real-time prompt injection (no pre-defined switching frames/prompts)
2. Asynchronous VAE decoding - decode frames as latents are generated
3. Continuous generation with dynamic prompt switching
"""

import torch
import torch.nn as nn
import time
import threading
import queue
from typing import List, Optional, Dict, Callable, Tuple
from dataclasses import dataclass
from omegaconf import OmegaConf
import numpy as np
from collections import deque

from pipeline.interactive_causal_inference import InteractiveCausalInferencePipeline
from utils.misc import set_seed
from utils.memory import get_cuda_free_memory_gb


@dataclass
class PromptSwitchEvent:
    """Event for real-time prompt switching"""
    prompt: str
    timestamp: float
    target_frame: int


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


class RealTimeStreamingPipelineV2:
    """
    Real-time streaming pipeline based on InteractiveCausalInferencePipeline
    
    Key features:
    1. Real-time prompt injection without pre-defined switch points
    2. Asynchronous VAE decoding as latents are generated
    3. Frame-by-frame streaming to callbacks
    4. Maintains temporal continuity through KV cache management
    """
    
    def __init__(self, config_path: str, device: str = "cuda"):
        self.config = OmegaConf.load(config_path)
        self.device = torch.device(device)
        
        # Video parameters
        self.fps = int(getattr(self.config, "fps", 16))
        self.total_frames = int(getattr(self.config, "num_output_frames", 240))
        self.duration_seconds = self.total_frames / self.fps
        self.block_size = self.config.num_frame_per_block  # Typically 3
        
        # Initialize base pipeline (copy constructor pattern)
        set_seed(self.config.seed)
        torch.set_grad_enabled(False)
        self.base_pipeline = InteractiveCausalInferencePipeline(self.config, device=self.device)
        self._setup_pipeline()
        
        # Real-time state
        self.is_running = False
        self.generation_thread = None
        self.vae_thread = None
        self.current_frame = 0
        self.current_prompt = ""
        self.start_time = None
        
        # Prompt switching (real-time injection)
        self.prompt_switch_queue = queue.Queue()
        self.pending_prompt_switch = None
        
        # Asynchronous VAE decoding
        self.latents_queue = queue.Queue()  # (latents_chunk, start_frame)
        self.frame_callbacks: List[Callable[[np.ndarray, int], None]] = []
        self.latest_frame = None
        
        # Pre-allocated noise and output tensors
        self.noise = torch.randn([
            1, self.total_frames, 16, 60, 104
        ], device=self.device, dtype=torch.bfloat16)
        
        # Output latents accumulator (like interactive pipeline)
        output_device = torch.device('cpu') if getattr(self.config, 'low_memory', False) else self.device
        self.output = torch.zeros(
            [1, self.total_frames, 16, 60, 104],
            device=output_device,
            dtype=torch.bfloat16
        )
        
        print(f"[RealTimeV2] Pipeline initialized for {self.duration_seconds}s video ({self.total_frames} frames)")
        
    def _setup_pipeline(self):
        """Setup pipeline - copy initialization from InteractiveCausalInferencePipeline"""
        # Load weights if specified
        if getattr(self.config, 'generator_ckpt', None):
            self._load_generator_weights()
        
        # Configure dtype and device
        self.base_pipeline = self.base_pipeline.to(dtype=torch.bfloat16)
        self.base_pipeline.generator.to(device=self.device)
        self.base_pipeline.vae.to(device=self.device)
        
        # Setup LoRA if configured
        self._setup_lora()
        
        # Initialize caches (copy from interactive pipeline logic)
        self._initialize_caches()
        
        print("[RealTimeV2] Pipeline setup complete")
        
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
            
    def _setup_lora(self):
        """Setup LoRA if configured"""
        try:
            adapter_cfg = getattr(self.config, "adapter", None)
            if adapter_cfg:
                from utils.lora_utils import configure_lora_for_model
                import peft
                print(f"[RealTimeV2] Enabling LoRA with config: {adapter_cfg}")
                self.base_pipeline.generator.model = configure_lora_for_model(
                    self.base_pipeline.generator.model,
                    model_name="generator",
                    lora_config=adapter_cfg,
                    is_main_process=True,
                )
                lora_ckpt_path = getattr(self.config, "lora_ckpt", None)
                if lora_ckpt_path:
                    print(f"[RealTimeV2] Loading LoRA checkpoint from {lora_ckpt_path}")
                    lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
                    if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
                        peft.set_peft_model_state_dict(self.base_pipeline.generator.model, lora_checkpoint["generator_lora"])
                    else:
                        peft.set_peft_model_state_dict(self.base_pipeline.generator.model, lora_checkpoint)
                    print("[RealTimeV2] LoRA weights loaded")
        except Exception as e:
            print(f"[RealTimeV2] LoRA setup skipped due to error: {e}")
            
    def _initialize_caches(self):
        """Initialize caches following InteractiveCausalInferencePipeline pattern"""
        # Cache configuration (copy from interactive pipeline)
        local_attn_cfg = getattr(self.config.model_kwargs, "local_attn_size", -1)
        if local_attn_cfg != -1:
            # Local attention
            kv_cache_size = local_attn_cfg * self.base_pipeline.frame_seq_length
        else:
            # Global attention
            kv_cache_size = self.total_frames * self.base_pipeline.frame_seq_length

        print(f"[RealTimeV2] Initializing caches with size: {kv_cache_size}")
        
        # Initialize KV and cross-attention caches
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
        
        # Pre-allocate timestep tensors
        self.base_pipeline._pre_allocate_timestep_tensors(
            batch_size=1,
            max_frames=self.block_size,
            device=self.device
        )
        
        # Set attention configuration
        self.base_pipeline.generator.model.local_attn_size = self.base_pipeline.local_attn_size
        self.base_pipeline._set_all_modules_max_attention_size(self.base_pipeline.local_attn_size)
        
    def add_frame_callback(self, callback: Callable[[np.ndarray, int], None]):
        """Add callback to receive decoded frames"""
        self.frame_callbacks.append(callback)
        
    def start_generation(self, initial_prompt: str):
        """Start real-time generation"""
        if self.is_running:
            print("[RealTimeV2] Generation already running!")
            return
            
        print(f"[RealTimeV2] Starting generation with prompt: '{initial_prompt}'")
        self.current_prompt = initial_prompt
        self.current_frame = 0
        self.start_time = time.time()
        self.is_running = True
        
        # Clear queues
        while not self.latents_queue.empty():
            try:
                self.latents_queue.get_nowait()
            except:
                break
                
        # Start generation thread
        self.generation_thread = threading.Thread(
            target=self._generation_loop, 
            args=(initial_prompt,),
            daemon=True
        )
        self.generation_thread.start()
        
        # Start VAE decoding thread
        self.vae_thread = threading.Thread(
            target=self._vae_decode_loop,
            daemon=True
        )
        self.vae_thread.start()
        
    def stop_generation(self):
        """Stop generation"""
        print("[RealTimeV2] Stopping generation...")
        self.is_running = False
        
        if self.generation_thread:
            self.generation_thread.join(timeout=5.0)
        if self.vae_thread:
            self.vae_thread.join(timeout=5.0)
            
    def send_prompt(self, prompt: str) -> bool:
        """Inject new prompt for real-time switching"""
        if not self.is_running:
            print("[RealTimeV2] Not running - use start_generation() first")
            return False
            
        if prompt.strip() == self.current_prompt.strip():
            print(f"[RealTimeV2] Prompt unchanged - skipping: '{prompt[:50]}...'")
            return False
            
        # Queue prompt switch for next available frame
        switch_event = PromptSwitchEvent(
            prompt=prompt,
            timestamp=time.time(),
            target_frame=self.current_frame + 1  # Apply at next frame
        )
        
        self.prompt_switch_queue.put(switch_event)
        print(f"[RealTimeV2] Prompt queued for frame {switch_event.target_frame}: '{prompt[:50]}...'")
        return True
        
    def get_status(self) -> GenerationStatus:
        """Get current status"""
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
        """Main generation loop - closely follows InteractiveCausalInferencePipeline.inference"""
        try:
            # Encode initial prompt (like interactive pipeline)
            print("[RealTimeV2] Encoding initial prompt...")
            with torch.no_grad():
                initial_encoded = self.base_pipeline.text_encoder([initial_prompt])
                current_conditional_dict = {"prompt_embeds": initial_encoded["prompt_embeds"]}
            
            # Initialize frame-by-frame generation (like interactive pipeline temporal loop)
            current_start_frame = 0
            self.base_pipeline.generator.model.local_attn_size = self.base_pipeline.local_attn_size
            
            # Temporal denoising by blocks (copy from interactive pipeline)
            all_num_frames = [self.block_size] * (self.total_frames // self.block_size)
            if self.total_frames % self.block_size != 0:
                all_num_frames.append(self.total_frames % self.block_size)
            
            for block_idx, current_num_frames in enumerate(all_num_frames):
                if not self.is_running:
                    break
                    
                print(f"[RealTimeV2] Generating block {block_idx}: frames {current_start_frame}-{current_start_frame+current_num_frames-1}")
                
                # Check for real-time prompt switching
                if self._check_and_apply_prompt_switch():
                    # Re-encode new prompt
                    with torch.no_grad():
                        new_encoded = self.base_pipeline.text_encoder([self.current_prompt])
                        current_conditional_dict = {"prompt_embeds": new_encoded["prompt_embeds"]}
                    # Perform recaching (like interactive pipeline)
                    self._recache_after_switch(self.output, current_start_frame, current_conditional_dict)
                
                # Generate current block (copy from interactive pipeline block generation)
                block_output = self._generate_block(
                    current_start_frame, current_num_frames, current_conditional_dict
                )
                
                # Store output (like interactive pipeline)
                self.output[:, current_start_frame:current_start_frame + current_num_frames] = block_output.to(self.output.device)
                
                # Queue latents for asynchronous VAE decoding
                latents_chunk = block_output.clone()
                self.latents_queue.put((latents_chunk, current_start_frame))
                
                # Update frame counter
                self.current_frame = current_start_frame + current_num_frames
                current_start_frame += current_num_frames
                
        except Exception as e:
            print(f"[RealTimeV2] Generation error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
            print("[RealTimeV2] Generation loop finished")
            
    def _generate_block(self, start_frame: int, num_frames: int, conditional_dict: dict) -> torch.Tensor:
        """Generate a single block - copy from InteractiveCausalInferencePipeline temporal denoising"""
        # Extract noise for this block
        noisy_input = self.noise[:, start_frame:start_frame + num_frames]
        
        # Spatial denoising loop (copy from interactive pipeline)
        for index, current_timestep in enumerate(self.base_pipeline.denoising_step_list):
            timestep_val = float(current_timestep)
            
            # Use pre-allocated timestep tensor (like interactive pipeline)
            timestep_key = int(current_timestep.item()) if hasattr(current_timestep, 'item') else int(current_timestep)
            if num_frames == self.block_size:
                timestep = self.base_pipeline._timestep_tensors[timestep_key]
            else:
                timestep = self.base_pipeline._timestep_tensors[timestep_key][:, :num_frames]

            if index < len(self.base_pipeline.denoising_step_list) - 1:
                # Intermediate denoising step (copy from interactive pipeline)
                torch.compiler.cudagraph_mark_step_begin()
                _, denoised_pred = self.base_pipeline.generator._compiled_forward(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.base_pipeline.kv_cache1,
                    crossattn_cache=self.base_pipeline.crossattn_cache,
                    current_start=start_frame * self.base_pipeline.frame_seq_length,
                )
                
                # Noise scheduling (copy from interactive pipeline)
                next_timestep = self.base_pipeline.denoising_step_list[index + 1]
                next_timestep_key = int(next_timestep.item()) if hasattr(next_timestep, 'item') else int(next_timestep)
                if num_frames == self.block_size:
                    noise_timestep = self.base_pipeline._noise_timestep_tensors[next_timestep_key]
                else:
                    noise_timestep = self.base_pipeline._noise_timestep_tensors[next_timestep_key][:num_frames]
                
                noisy_input = self.base_pipeline.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    noise_timestep,
                ).unflatten(0, denoised_pred.shape[:2])
            else:
                # Final denoising step (copy from interactive pipeline)
                torch.compiler.cudagraph_mark_step_begin()
                _, denoised_pred = self.base_pipeline.generator._compiled_forward(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.base_pipeline.kv_cache1,
                    crossattn_cache=self.base_pipeline.crossattn_cache,
                    current_start=start_frame * self.base_pipeline.frame_seq_length,
                )

        # KV cache update with clean context (copy from interactive pipeline)
        if num_frames == self.base_pipeline.num_frame_per_block:
            context_timestep = self.base_pipeline._context_timestep_base
        else:
            context_timestep = self.base_pipeline._context_timestep_base[:, :num_frames]
        
        torch.compiler.cudagraph_mark_step_begin()
        self.base_pipeline.generator._compiled_forward(
            noisy_image_or_video=denoised_pred,
            conditional_dict=conditional_dict,
            timestep=context_timestep,
            kv_cache=self.base_pipeline.kv_cache1,
            crossattn_cache=self.base_pipeline.crossattn_cache,
            current_start=start_frame * self.base_pipeline.frame_seq_length,
        )
        
        return denoised_pred
        
    def _check_and_apply_prompt_switch(self) -> bool:
        """Check for real-time prompt switches"""
        try:
            while not self.prompt_switch_queue.empty():
                switch_event = self.prompt_switch_queue.get_nowait()
                
                if self.current_frame >= switch_event.target_frame:
                    # Apply switch
                    old_prompt = self.current_prompt
                    self.current_prompt = switch_event.prompt
                    print(f"[RealTimeV2] Prompt switched at frame {self.current_frame}")
                    print(f"  From: '{old_prompt[:30]}...'")
                    print(f"  To:   '{self.current_prompt[:30]}...'")
                    return True
                else:
                    # Not time yet, put it back
                    self.prompt_switch_queue.put(switch_event)
                    break
        except queue.Empty:
            pass
        return False
        
    def _recache_after_switch(self, output: torch.Tensor, current_start_frame: int, new_conditional_dict: dict):
        """Recache after prompt switch - copy from InteractiveCausalInferencePipeline._recache_after_switch"""
        if current_start_frame == 0:
            return
            
        # Determine recache frames (copy from interactive pipeline)
        num_recache_frames = current_start_frame if self.base_pipeline.local_attn_size == -1 else min(self.base_pipeline.local_attn_size, current_start_frame)
        recache_start_frame = current_start_frame - num_recache_frames
        
        frames_to_recache = output[:, recache_start_frame:current_start_frame]
        
        # Move to GPU if needed
        if frames_to_recache.device.type == 'cpu':
            frames_to_recache = frames_to_recache.to(self.device)
        
        print(f"[RealTimeV2] Recaching {num_recache_frames} frames for prompt switch")
        
        # Prepare block mask (copy from interactive pipeline)
        device = frames_to_recache.device
        block_mask = self.base_pipeline._get_block_mask(
            device=device,
            num_frames=num_recache_frames,
            local_attn_size=self.base_pipeline.local_attn_size
        )
        self.base_pipeline.generator.model.block_mask = block_mask
        
        # Context timestep (copy from interactive pipeline)
        batch_size = frames_to_recache.shape[0]
        cache_key = (batch_size, num_recache_frames)
        if cache_key not in self.base_pipeline._recache_timestep_cache:
            self.base_pipeline._recache_timestep_cache[cache_key] = torch.full(
                [batch_size, num_recache_frames], 
                self.config.context_noise,
                device=device, 
                dtype=torch.int64
            )
        context_timestep = self.base_pipeline._recache_timestep_cache[cache_key]
        
        # Perform recaching (copy from interactive pipeline)
        with torch.no_grad():
            self.base_pipeline.generator(
                noisy_image_or_video=frames_to_recache,
                conditional_dict=new_conditional_dict,
                timestep=context_timestep,
                kv_cache=self.base_pipeline.kv_cache1,
                crossattn_cache=self.base_pipeline.crossattn_cache,
                current_start=recache_start_frame * self.base_pipeline.frame_seq_length,
            )
        
        # Reset cross-attention cache after recaching (copy from interactive pipeline)
        for blk in self.base_pipeline.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False
            
    def _vae_decode_loop(self):
        """Asynchronous VAE decoding loop"""
        print("[RealTimeV2] Starting VAE decode loop...")
        
        try:
            while self.is_running or not self.latents_queue.empty():
                try:
                    # Get latents chunk from queue
                    latents_chunk, start_frame = self.latents_queue.get(timeout=1.0)
                    
                    print(f"[RealTimeV2] Decoding latents for frames {start_frame}-{start_frame + latents_chunk.shape[1] - 1}")
                    
                    # Decode to video frames
                    with torch.no_grad():
                        video_frames = self.base_pipeline.vae.decode_to_pixel(latents_chunk, use_cache=False)
                    
                    # Convert to numpy and stream
                    video_np = video_frames.cpu().numpy()  # [B, T, C, H, W]
                    video_np = (video_np * 0.5 + 0.5).clip(0, 1)  # Normalize to [0,1]
                    
                    # Stream individual frames
                    for t in range(video_np.shape[1]):
                        frame = video_np[0, t].transpose(1, 2, 0)  # [H, W, C]
                        frame_uint8 = (frame * 255).astype(np.uint8)
                        frame_idx = start_frame + t
                        
                        # Store latest frame
                        self.latest_frame = frame_uint8
                        
                        # Call all registered callbacks
                        for callback in self.frame_callbacks:
                            try:
                                callback(frame_uint8, frame_idx)
                            except Exception as e:
                                print(f"[RealTimeV2] Callback error: {e}")
                                
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[RealTimeV2] VAE decode error: {e}")
                    
        except Exception as e:
            print(f"[RealTimeV2] VAE decode loop error: {e}")
        finally:
            print("[RealTimeV2] VAE decode loop finished")
            
    def reset_for_new_video(self):
        """Reset for new video generation"""
        print("[RealTimeV2] Resetting for new video...")
        
        if self.is_running:
            self.stop_generation()
            
        # Reset state
        self.current_frame = 0
        self.current_prompt = ""
        self.start_time = None
        
        # Clear queues
        while not self.prompt_switch_queue.empty():
            try:
                self.prompt_switch_queue.get_nowait()
            except:
                break
        while not self.latents_queue.empty():
            try:
                self.latents_queue.get_nowait()
            except:
                break
                
        # Reset frame data
        self.latest_frame = None
        
        # Generate new noise
        self.noise = torch.randn([
            1, self.total_frames, 16, 60, 104
        ], device=self.device, dtype=torch.bfloat16)
        
        # Reset output tensor
        self.output.zero_()
        
        print("[RealTimeV2] Reset complete")
        
    def is_finished(self) -> bool:
        """Check if generation is finished"""
        return not self.is_running and self.current_frame >= self.total_frames


# Example usage
if __name__ == "__main__":
    pipeline = RealTimeStreamingPipelineV2("configs/longlive_interactive_inference.yaml")
    
    def frame_callback(frame: np.ndarray, frame_idx: int):
        print(f"Received frame {frame_idx}: {frame.shape}")
    
    pipeline.add_frame_callback(frame_callback)
    
    # Start generation
    pipeline.start_generation("A beautiful sunset")
    
    # Simulate real-time prompt injection
    import time
    time.sleep(3)
    pipeline.send_prompt("A stormy ocean with waves")
    time.sleep(3)
    pipeline.send_prompt("A peaceful forest with animals")
    
    # Let it run
    time.sleep(15)
    pipeline.stop_generation()