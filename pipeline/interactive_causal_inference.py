# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional
import torch
import time
import os

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
from pipeline.causal_inference import CausalInferencePipeline
import torch.distributed as dist
from utils.debug_option import DEBUG

# Import comprehensive timing components
try:
    import sys
    script_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts_zhenyi')
    if script_dir not in sys.path:
        sys.path.append(script_dir)
    from comprehensive_latency_tracker import (
        ComprehensiveLatencyTracker, 
        time_kv_operations, 
        time_attention_kernel,
        time_device_sync,
        time_quantization,
        instrument_wan_components
    )
    COMPREHENSIVE_TIMING_AVAILABLE = True
except ImportError as e:
    print(f"Comprehensive timing not available: {e}")
    COMPREHENSIVE_TIMING_AVAILABLE = False


class InteractiveCausalInferencePipeline(CausalInferencePipeline):
    def __init__(
        self,
        args,
        device,
        *,
        generator: WanDiffusionWrapper | None = None,
        text_encoder: WanTextEncoder | None = None,
        vae: WanVAEWrapper | None = None,
    ):
        super().__init__(args, device, generator=generator, text_encoder=text_encoder, vae=vae)
        self.global_sink = getattr(args, "global_sink", False)
        
        # Initialize latency tracking
        self.latency_tracker = None
        self.enable_timing_logs = True

    def set_latency_tracker(self, tracker):
        """Set the comprehensive latency tracker"""
        self.latency_tracker = tracker
        if COMPREHENSIVE_TIMING_AVAILABLE and tracker:
            # Instrument WAN components for timing
            instrument_wan_components()
            self.text_encoder.latency_tracker = tracker
            self.generator.latency_tracker = tracker  
            self.vae.latency_tracker = tracker
            
    def _log_timing(self, message: str, elapsed_ms: float = None):
        """Log timing information"""
        if self.enable_timing_logs:
            if elapsed_ms is not None:
                print(f"[Interactive Timing] {message}: {elapsed_ms:.2f} ms")
            else:
                print(f"[Interactive] {message}")

    # Internal helpers
    def _recache_after_switch(self, output, current_start_frame, new_conditional_dict, segment_idx: int):
        """Recache previous frames with new prompt conditioning after a prompt switch"""
        
        switch_start_time = time.perf_counter()
        self._log_timing(f"Starting prompt switch to segment {segment_idx} at frame {current_start_frame}")
        
        cache_reset_start = time.perf_counter()
        
        if not self.global_sink:
            # Reset KV cache with timing
            if self.latency_tracker and COMPREHENSIVE_TIMING_AVAILABLE:
                with self.latency_tracker.time_component('prompt_switch_kv_reset', 
                                                       segment_idx=segment_idx,
                                                       frame_idx=current_start_frame):
                    for block_idx in range(self.num_transformer_blocks):
                        cache = self.kv_cache1[block_idx]
                        cache["k"].zero_()
                        cache["v"].zero_()
            else:
                for block_idx in range(self.num_transformer_blocks):
                    cache = self.kv_cache1[block_idx]
                    cache["k"].zero_()
                    cache["v"].zero_()
            
        # Reset cross-attention cache
        with time_kv_operations(self.latency_tracker, "prompt_switch_crossattn_reset", 
                              segment_idx=segment_idx, frame_idx=current_start_frame):
            for blk in self.crossattn_cache:
                blk["k"].zero_()
                blk["v"].zero_()
                blk["is_init"] = False
        
        cache_reset_time = (time.perf_counter() - cache_reset_start) * 1000
        self._log_timing(f"Cache reset completed", cache_reset_time)

        # Early exit if no frames to recache
        if current_start_frame == 0:
            switch_total_time = (time.perf_counter() - switch_start_time) * 1000
            self._log_timing(f"Prompt switch completed (no recaching needed)", switch_total_time)
            return
        
        recache_setup_start = time.perf_counter()
        
        num_recache_frames = current_start_frame if self.local_attn_size == -1 else min(self.local_attn_size, current_start_frame)
        recache_start_frame = current_start_frame - num_recache_frames
        
        frames_to_recache = output[:, recache_start_frame:current_start_frame]
        
        # Move to GPU if needed
        if frames_to_recache.device.type == 'cpu':
            target_device = next(self.generator.parameters()).device
            frames_to_recache = frames_to_recache.to(target_device)
        
        batch_size = frames_to_recache.shape[0]
        self._log_timing(f"Recaching {num_recache_frames} frames (from {recache_start_frame} to {current_start_frame})")
        
        # Prepare blockwise causal mask
        device = frames_to_recache.device
        
        with time_attention_kernel(self.latency_tracker, operation="prompt_switch_mask_prep",
                                 segment_idx=segment_idx, num_frames=num_recache_frames):
            block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
                device=device,
                num_frames=num_recache_frames,
                frame_seqlen=self.frame_seq_length,
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=self.local_attn_size
            )
        
        context_timestep = torch.ones([batch_size, num_recache_frames], 
                                    device=device, dtype=torch.int64) * self.args.context_noise
        
        self.generator.model.block_mask = block_mask
        
        recache_setup_time = (time.perf_counter() - recache_setup_start) * 1000
        self._log_timing(f"Recache setup completed", recache_setup_time)
        
        # Perform recaching with timing
        recache_forward_start = time.perf_counter()
        
        with torch.no_grad():
            if self.latency_tracker and COMPREHENSIVE_TIMING_AVAILABLE:
                with self.latency_tracker.time_component('prompt_switch_recache', 
                                                       segment_idx=segment_idx,
                                                       frame_idx=current_start_frame,
                                                       num_frames=num_recache_frames):
                    self.generator(
                        noisy_image_or_video=frames_to_recache,
                        conditional_dict=new_conditional_dict,
                        timestep=context_timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=recache_start_frame * self.frame_seq_length,
                    )
            else:
                self.generator(
                    noisy_image_or_video=frames_to_recache,
                    conditional_dict=new_conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=recache_start_frame * self.frame_seq_length,
                )
        
        recache_forward_time = (time.perf_counter() - recache_forward_start) * 1000
        self._log_timing(f"Recache forward pass completed", recache_forward_time)
        
        # Reset cross-attention cache again after recaching
        with time_kv_operations(self.latency_tracker, "prompt_switch_final_reset", 
                              segment_idx=segment_idx):
            for blk in self.crossattn_cache:
                blk["k"].zero_()
                blk["v"].zero_()
                blk["is_init"] = False
        
        switch_total_time = (time.perf_counter() - switch_start_time) * 1000
        self._log_timing(f"Prompt switch to segment {segment_idx} completed", switch_total_time)
        
        # Record switch timing for analysis
        if self.latency_tracker and COMPREHENSIVE_TIMING_AVAILABLE:
            # Record as a completed timing event
            self.latency_tracker.component_timings['prompt_switch_total'].append({
                'segment_idx': segment_idx,
                'frame_idx': current_start_frame,
                'num_recache_frames': num_recache_frames,
                'cache_reset_ms': cache_reset_time,
                'recache_setup_ms': recache_setup_time,
                'recache_forward_ms': recache_forward_time,
                'total_ms': switch_total_time,
                'gpu_time_ms': switch_total_time,
                'cpu_time_ms': switch_total_time
            })

    def inference(
        self,
        noise: torch.Tensor,
        *,
        text_prompts_list: List[List[str]],
        switch_frame_indices: List[int],
        return_latents: bool = False,
        low_memory: bool = False,
    ):
        """Generate a video and switch prompts at specified frame indices."""
        return self.inference_with_timing(
            noise=noise,
            text_prompts_list=text_prompts_list,
            switch_frame_indices=switch_frame_indices,
            return_latents=return_latents,
            low_memory=low_memory,
            latency_tracker=self.latency_tracker
        )
        
    def inference_with_timing(
        self,
        noise: torch.Tensor,
        *,
        text_prompts_list: List[List[str]],
        switch_frame_indices: List[int],
        return_latents: bool = False,
        low_memory: bool = False,
        latency_tracker=None
    ):
        """Generate a video and switch prompts at specified frame indices with comprehensive timing.

        Args:
            noise: Noise tensor, shape = (B, T_out, C, H, W).
            text_prompts_list: List[List[str]], length = N_seg. Prompt list used for segment i (aligned with batch).
            switch_frame_indices: List[int], length = N_seg - 1. The i-th value indicates that when generation reaches this frame (inclusive)
                we start using the prompts for segment i+1.
            return_latents: Whether to also return the latent tensor.
            low_memory: Enable low-memory mode.
            latency_tracker: ComprehensiveLatencyTracker for detailed timing analysis.
        """
        # Set up timing tracking
        if latency_tracker:
            self.set_latency_tracker(latency_tracker)
            latency_tracker.start_generation()
        
        # Store switch indices for frame saving
        self.set_current_switch_indices(switch_frame_indices)
        
        generation_start_time = time.perf_counter()
        self._log_timing(f"Starting interactive inference with {len(text_prompts_list)} segments")
        self._log_timing(f"Switch points: {switch_frame_indices}")
        
        # Validate inputs
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert len(text_prompts_list) >= 1, "text_prompts_list must not be empty"
        assert len(switch_frame_indices) == len(text_prompts_list) - 1, (
            "length of switch_frame_indices should be one less than text_prompts_list"
        )
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        
        # Encode all prompts with timing
        prompt_encoding_start = time.perf_counter()
        self._log_timing(f"Encoding {len(text_prompts_list)} prompt segments...")
        print(text_prompts_list)
        
        if latency_tracker and COMPREHENSIVE_TIMING_AVAILABLE:
            cond_list = []
            for i, prompts in enumerate(text_prompts_list):
                with latency_tracker.time_component('interactive_text_encoding', 
                                                   segment_idx=i, 
                                                   num_prompts=len(prompts)):
                    cond_list.append(self.text_encoder(text_prompts=prompts))
        else:
            cond_list = [self.text_encoder(text_prompts=p) for p in text_prompts_list]
            
        prompt_encoding_time = (time.perf_counter() - prompt_encoding_start) * 1000
        self._log_timing(f"All prompts encoded", prompt_encoding_time)

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder,
                target_device=gpu,
                preserved_memory_gb=gpu_memory_preservation,
            )

        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype
        )

        # initialize caches
        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        kv_policy = ""
        if local_attn_cfg != -1:
            # local attention
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        else:
            # global attention
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        print(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")

        self._initialize_kv_cache(
            batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )

        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)

        # temporal denoising by blocks
        all_num_frames = [self.num_frame_per_block] * num_blocks
        segment_idx = 0  # current segment index
        next_switch_pos = (
            switch_frame_indices[segment_idx]
            if segment_idx < len(switch_frame_indices)
            else None
        )

        if DEBUG:
            print("[MultipleSwitch] all_num_frames", all_num_frames)
            print("[MultipleSwitch] switch_frame_indices", switch_frame_indices)

        # Temporal denoising loop with comprehensive timing
        block_idx = 0
        frame_idx = 0
        
        for current_num_frames in all_num_frames:
            block_start_time = time.perf_counter()
            
            # Check for prompt switching
            if next_switch_pos is not None and current_start_frame >= next_switch_pos:
                segment_idx += 1
                self._recache_after_switch(output, current_start_frame, cond_list[segment_idx], segment_idx)
                if DEBUG:
                    print(
                        f"[MultipleSwitch] Switch to segment {segment_idx} at frame {current_start_frame}"
                    )
                next_switch_pos = (
                    switch_frame_indices[segment_idx]
                    if segment_idx < len(switch_frame_indices)
                    else None
                )
                self._log_timing(f"Now using segment {segment_idx}: {text_prompts_list[segment_idx]}")
                
            cond_in_use = cond_list[segment_idx]

            # Start block timing
            if latency_tracker:
                latency_tracker.start_block(block_idx)

            with time_device_sync(latency_tracker, "interactive_block_input_prep"):
                noisy_input = noise[:, current_start_frame : current_start_frame + current_num_frames]

            # Spatial denoising loop with timing
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep_val = float(current_timestep)
                
                with time_device_sync(latency_tracker, "interactive_timestep_prep"):
                    timestep = (
                        torch.ones([batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64)
                        * current_timestep
                    )

                if index < len(self.denoising_step_list) - 1:
                    # Intermediate denoising step
                    if latency_tracker and COMPREHENSIVE_TIMING_AVAILABLE:
                        with latency_tracker.time_component('interactive_denoising_step', 
                                                          timestep=timestep_val,
                                                          step_index=index, 
                                                          block_idx=block_idx,
                                                          segment_idx=segment_idx,
                                                          is_final_step=False):
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=cond_in_use,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length,
                            )
                    else:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=cond_in_use,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                        )
                        
                    with time_device_sync(latency_tracker, "interactive_noise_scheduling"):
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep
                            * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long
                            ),
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # Final denoising step
                    if latency_tracker and COMPREHENSIVE_TIMING_AVAILABLE:
                        with latency_tracker.time_component('interactive_denoising_step_final', 
                                                          timestep=timestep_val,
                                                          step_index=index,
                                                          block_idx=block_idx,
                                                          segment_idx=segment_idx,
                                                          is_final_step=True):
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=cond_in_use,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length,
                            )
                    else:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=cond_in_use,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                        )

            # Record output
            with time_device_sync(latency_tracker, "interactive_output_recording"):
                output[:, current_start_frame : current_start_frame + current_num_frames] = denoised_pred.to(output.device)

            # Record frame completion timing
            if latency_tracker:
                for frame_offset in range(current_num_frames):
                    latency_tracker.record_frame_completion(frame_idx + frame_offset)

            # KV cache update with clean context
            with time_device_sync(latency_tracker, "interactive_context_prep"):
                context_timestep = torch.ones_like(timestep) * self.args.context_noise
                
            with time_kv_operations(latency_tracker, "interactive_kv_cache_update", 
                                  block_idx=block_idx, 
                                  segment_idx=segment_idx,
                                  context_noise=self.args.context_noise):
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=cond_in_use,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )

            # End block timing
            if latency_tracker:
                latency_tracker.end_block()
                
            block_time = (time.perf_counter() - block_start_time) * 1000
            self._log_timing(f"Block {block_idx} (frames {current_start_frame}-{current_start_frame+current_num_frames-1}, segment {segment_idx}) completed", block_time)

            # Update pointers
            current_start_frame += current_num_frames
            frame_idx += current_num_frames
            block_idx += 1

        # VAE decoding with timing
        vae_decode_start = time.perf_counter()
        self._log_timing("Starting VAE decode...")
        
        with time_device_sync(latency_tracker, "interactive_vae_input_prep"):
            output_for_vae = output.to(noise.device)
            
        # Chunked VAE decode for better memory efficiency and potential speed improvement
        chunk_size = 60  # Process 60 frames at a time (quarter of total)
        video_chunks = []
        
        for chunk_start in range(0, num_output_frames, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_output_frames)
            chunk_latents = output_for_vae[:, chunk_start:chunk_end]
            
            if latency_tracker and COMPREHENSIVE_TIMING_AVAILABLE:
                with latency_tracker.time_component('interactive_vae_decode_chunk', 
                                                  chunk_idx=chunk_start//chunk_size,
                                                  chunk_frames=chunk_end-chunk_start,
                                                  total_chunks=(num_output_frames + chunk_size - 1) // chunk_size):
                    chunk_video = self.vae.decode_to_pixel(chunk_latents, use_cache=False)
            else:
                chunk_video = self.vae.decode_to_pixel(chunk_latents, use_cache=False)
                
            video_chunks.append(chunk_video)
            
        # Concatenate all chunks
        video = torch.cat(video_chunks, dim=1)
            
        with time_device_sync(latency_tracker, "interactive_vae_postprocess"):
            video = (video * 0.5 + 0.5).clamp(0, 1)
        
        vae_decode_time = (time.perf_counter() - vae_decode_start) * 1000
        self._log_timing(f"VAE decode completed", vae_decode_time)
        
        # Complete timing and log summary
        generation_total_time = (time.perf_counter() - generation_start_time) * 1000
        self._log_timing(f"Interactive inference completed", generation_total_time)
        
        if latency_tracker:
            # Generation completed - timing data is already collected
            
            # Log timing summary
            print("\n" + "="*60)
            print("INTERACTIVE INFERENCE TIMING SUMMARY")
            print("="*60)
            print(f"Total generation time: {generation_total_time:.2f} ms")
            print(f"Prompt encoding time: {prompt_encoding_time:.2f} ms")
            print(f"VAE decode time: {vae_decode_time:.2f} ms")
            print(f"Number of segments: {len(text_prompts_list)}")
            print(f"Number of blocks: {num_blocks}")
            print(f"Frames per block: {self.num_frame_per_block}")
            print(f"Switch frame indices: {switch_frame_indices}")
            print("="*60)

        # Save sample frames for inspection
        self._save_sample_frames(video, generation_start_time)
        
        if return_latents:
            return video, output
        return video
    
    def _save_sample_frames(self, video: torch.Tensor, generation_start_time: float):
        """Save a few sample frames with frame numbers for inspection"""
        try:
            import os
            from torchvision.utils import save_image
            
            # Create frames directory
            frames_dir = "/tmp/interactive_sample_frames"
            os.makedirs(frames_dir, exist_ok=True)
            
            batch_size, num_frames = video.shape[0], video.shape[1]
            timestamp = int(generation_start_time)
            
            # Save frames at different points: start, middle, switches, end
            frames_to_save = [
                0,  # First frame
                num_frames // 4,  # 25% through
                num_frames // 2,  # Middle frame  
                3 * num_frames // 4,  # 75% through
                num_frames - 1  # Last frame
            ]
            
            # Add switch frames if we have switch indices
            if hasattr(self, '_current_switch_indices'):
                for switch_idx in self._current_switch_indices:
                    if 0 <= switch_idx < num_frames:
                        # Save the switch frame and 4 frames after
                        frames_to_save.append(switch_idx)  # Switch frame
                        for offset in range(1, 5):  # 4 frames after switch
                            next_frame = switch_idx + offset
                            if next_frame < num_frames:
                                frames_to_save.append(next_frame)
            
            # Remove duplicates and sort
            frames_to_save = sorted(list(set(frames_to_save)))
            
            for batch_idx in range(min(batch_size, 1)):  # Save only first batch item
                for frame_idx in frames_to_save:
                    if frame_idx < num_frames:
                        # Extract frame: [C, H, W]
                        frame = video[batch_idx, frame_idx]  # Shape: [3, H, W]
                        
                        # Save frame
                        filename = f"frame_batch{batch_idx}_frame{frame_idx:03d}_t{timestamp}.png"
                        filepath = os.path.join(frames_dir, filename)
                        save_image(frame, filepath, normalize=True, value_range=(0, 1))
            
            self._log_timing(f"Sample frames saved to {frames_dir} (frames: {frames_to_save})")
            
        except Exception as e:
            # Don't fail the entire inference if frame saving fails
            print(f"Warning: Could not save sample frames: {e}")
            
    def set_current_switch_indices(self, switch_indices: List[int]):
        """Store switch indices for frame saving reference"""
        self._current_switch_indices = switch_indices 