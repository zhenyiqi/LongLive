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
import os

# Set PyTorch CUDA memory allocator to use expandable segments to reduce fragmentation
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from utils.misc import set_seed
from utils.memory import get_cuda_free_memory_gb

# Direct imports for pipeline components
from pipeline.causal_inference import CausalInferencePipeline


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
        
        # Initialize components directly (no base pipeline)
        set_seed(self.config.seed)
        torch.set_grad_enabled(False)
        self._setup_pipeline_direct()
        
        # Real-time state
        self.is_running = False
        self.generation_thread = None
        self.current_frame = 0
        self.current_prompt = ""
        self.start_time = None
        
        # Prompt switching (real-time injection)
        self.prompt_switch_queue = queue.Queue()
        self.pending_prompt_switch = None
        
        # Frame callbacks for streaming decoded frames
        self.frame_callbacks: List[Callable[[np.ndarray, int], None]] = []
        self.latest_frame = None
        
        # Pre-allocate tensors (like InteractiveCausalInferencePipeline does)
        # Use low_memory mode to put output on CPU if configured
        print("[RealTimeV2] Pre-allocating noise and output tensors...")
        
        self.noise = torch.randn([
            1, self.total_frames, 16, 60, 104
        ], device=self.device, dtype=torch.bfloat16)
        
        # Output tensor - use CPU if low_memory is enabled (like InteractiveCausalInferencePipeline)
        output_device = torch.device('cpu') if getattr(self.config, 'low_memory', False) else self.device
        self.output = torch.zeros([
            1, self.total_frames, 16, 60, 104
        ], device=output_device, dtype=torch.bfloat16)
        
        print(f"[RealTimeV2] Allocated noise on {self.noise.device}, output on {self.output.device}")
        
        print(f"[RealTimeV2] Pipeline initialized for {self.duration_seconds}s video ({self.total_frames} frames)")
        
    def _setup_pipeline_direct(self):
        """Setup pipeline components directly without creating another pipeline"""
        # Create the base causal inference pipeline  
        self.pipeline = CausalInferencePipeline(self.config, device=self.device)
        
        # Load model weights
        if getattr(self.config, 'generator_ckpt', None):
            self._load_generator_weights()
        
        # Match working pipelines: set dtype/device BEFORE applying LoRA
        try:
            self.pipeline = self.pipeline.to(dtype=torch.bfloat16)
            self.pipeline.generator.to(device=self.device)
            self.pipeline.vae.to(device=self.device)
            # Sweep for any lingering FP32 params/buffers (e.g., Conv3d.bias) and convert to BF16
            for module in self.pipeline.generator.modules():
                for p in module.parameters(recurse=False):
                    if p is not None and p.dtype == torch.float32:
                        p.data = p.data.to(torch.bfloat16)
                for name, b in module.named_buffers(recurse=False):
                    if b is not None and b.dtype == torch.float32:
                        setattr(module, name, b.to(torch.bfloat16))
        except Exception as e:
            print(f"[RealTimeV2] Warning: dtype/device preset failed: {e}")
        
        # Setup LoRA if needed
        if getattr(self.config, 'adapter', None):
            self._setup_lora()
            
        # Initialize caches and other components (like InteractiveCausalInferencePipeline)
        self._initialize_additional_components()
            
    def _load_generator_weights(self):
        """Load generator weights"""
        import torch
        print(f"[RealTimeV2] Loading generator weights from {self.config.generator_ckpt}")
        
        state_dict = torch.load(self.config.generator_ckpt, map_location="cpu")
        raw_gen_state_dict = state_dict["generator_ema" if self.config.use_ema else "generator"]
        
        if self.config.use_ema:
            def _clean_key(name: str) -> str:
                return name.replace("_fsdp_wrapped_module.", "")
            
            cleaned_state_dict = {_clean_key(k): v for k, v in raw_gen_state_dict.items()}
            missing, unexpected = self.pipeline.generator.load_state_dict(cleaned_state_dict, strict=False)
            
            if missing:
                print(f"[RealTimeV2] Warning: {len(missing)} parameters missing")
            if unexpected:
                print(f"[RealTimeV2] Warning: {len(unexpected)} unexpected params")
        else:
            self.pipeline.generator.load_state_dict(raw_gen_state_dict)
            
    def _setup_lora(self):
        """Setup LoRA weights"""
        from utils.lora_utils import configure_lora_for_model
        import peft
        
        print(f"[RealTimeV2] Applying LoRA with config: {self.config.adapter}")
        
        self.pipeline.generator.model = configure_lora_for_model(
            self.pipeline.generator.model,
            model_name="generator", 
            lora_config=self.config.adapter,
            is_main_process=True,
        )
        
        # Load LoRA weights if specified
        lora_ckpt_path = getattr(self.config, "lora_ckpt", None)
        if lora_ckpt_path:
            print(f"[RealTimeV2] Loading LoRA checkpoint from {lora_ckpt_path}")
            lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
            
            if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
                peft.set_peft_model_state_dict(self.pipeline.generator.model, lora_checkpoint["generator_lora"])
            else:
                peft.set_peft_model_state_dict(self.pipeline.generator.model, lora_checkpoint)
                
            print("[RealTimeV2] LoRA weights loaded")
    
    def _initialize_additional_components(self):
        """Initialize components missing from CausalInferencePipeline"""
        # Set up block mask cache and prefetch (from InteractiveCausalInferencePipeline)
        self.global_sink = getattr(self.config, "global_sink", False)
        self._block_mask_cache = {}
        self._prefetch_stream = torch.cuda.Stream(device=self.device) if torch.cuda.is_available() else None
        self._prefetch_cache = None
        
        # (dtype/device already aligned in _setup_pipeline_direct before LoRA)
        
        # Set up compiled forward method
        self._generator_compiled = False
        try:
            self.pipeline.generator._compiled_forward = self.pipeline.generator.__call__
        except Exception:
            pass
            
        # Compile safe models (text encoder only; skip generator/vae graphs)
        self._compile_safe_models_for_h100()
        
        # Set up pre-allocated timestep tensors
        self._setup_timestep_tensors()
        
    def _compile_safe_models_for_h100(self):
        """Align with Interactive: compile text encoder call only; keep generator/vae eager."""
        print("[RealTimeV2] Preparing text encoder compiled forward (reduce-overhead)...")
        try:
            self.pipeline.text_encoder._compiled_forward = torch.compile(
                self.pipeline.text_encoder.__call__,
                mode="reduce-overhead",
                dynamic=False,
                fullgraph=False
            )
        except Exception as e:
            print(f"[RealTimeV2] Text encoder compiled forward setup failed, using eager: {e}")
            self.pipeline.text_encoder._compiled_forward = self.pipeline.text_encoder.__call__
        # Ensure generator compiled-forward exists (kept eager)
        try:
            self.pipeline.generator._compiled_forward = self.pipeline.generator.__call__
        except Exception:
            pass
        # Skip VAE compilation; provide unified callable
        print("[RealTimeV2] Skipping VAE compilation; using eager decode_to_pixel")
        self.pipeline.vae._compiled_decode = self.pipeline.vae.decode_to_pixel
            
    def _setup_timestep_tensors(self):
        """Pre-allocate timestep tensors for performance (from InteractiveCausalInferencePipeline)"""
        self._timestep_tensors = {}
        self._noise_timestep_tensors = {}
        self._context_timestep_base = torch.zeros([1, self.block_size], device=self.device, dtype=torch.long)
        
        # Pre-allocate for common timestep values
        for timestep_val in self.pipeline.denoising_step_list:
            timestep_key = int(timestep_val.item()) if hasattr(timestep_val, 'item') else int(timestep_val)
            self._timestep_tensors[timestep_key] = torch.full(
                [1, self.block_size], timestep_val, device=self.device, dtype=torch.long
            )
            
        # Pre-allocate noise timestep tensors for scheduling
        for timestep_val in self.pipeline.denoising_step_list:
            timestep_key = int(timestep_val.item()) if hasattr(timestep_val, 'item') else int(timestep_val)
            self._noise_timestep_tensors[timestep_key] = torch.full(
                [self.block_size], timestep_val, device=self.device, dtype=self.pipeline.denoising_step_list.dtype
            )
    
    def _initialize_generation_caches(self):
        """Initialize KV and cross-attention caches for generation (from InteractiveCausalInferencePipeline)"""
        batch_size = 1  # Always 1 for real-time generation
        dtype = torch.bfloat16
        device = self.device
        
        # Determine cache size based on attention configuration
        local_attn_cfg = getattr(self.pipeline, 'local_attn_size', -1)
        if local_attn_cfg != -1:
            kv_cache_size = local_attn_cfg * self.pipeline.frame_seq_length
            kv_policy = f"local, size={local_attn_cfg}"
        else:
            kv_cache_size = self.total_frames * self.pipeline.frame_seq_length 
            kv_policy = "global (-1)"
        
        print(f"[RealTimeV2] Initializing caches: kv_size={kv_cache_size}, policy={kv_policy}")
        
        # Initialize KV cache
        self.pipeline._initialize_kv_cache(
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            kv_cache_size_override=kv_cache_size
        )
        
        # Initialize cross-attention cache  
        self.pipeline._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=dtype,
            device=device
        )
        
        # Set attention size on all modules
        self.pipeline._set_all_modules_max_attention_size(local_attn_cfg)
        
    def _old_setup_pipeline(self):
        """Setup pipeline - copy initialization from InteractiveCausalInferencePipeline"""
        # Load weights if specified
        if getattr(self.config, 'generator_ckpt', None):
            self._load_generator_weights()
        
        # Configure dtype and device
        self.pipeline = self.pipeline.to(dtype=torch.bfloat16)
        self.pipeline.generator.to(device=self.device)
        self.pipeline.vae.to(device=self.device)
        
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
            self.pipeline.generator.load_state_dict(cleaned_state_dict, strict=False)
        else:
            self.pipeline.generator.load_state_dict(raw_gen_state_dict)
            
    def _setup_lora(self):
        """Setup LoRA if configured"""
        try:
            adapter_cfg = getattr(self.config, "adapter", None)
            if adapter_cfg:
                from utils.lora_utils import configure_lora_for_model
                import peft
                print(f"[RealTimeV2] Enabling LoRA with config: {adapter_cfg}")
                self.pipeline.generator.model = configure_lora_for_model(
                    self.pipeline.generator.model,
                    model_name="generator",
                    lora_config=adapter_cfg,
                    is_main_process=True,
                )
                lora_ckpt_path = getattr(self.config, "lora_ckpt", None)
                if lora_ckpt_path:
                    print(f"[RealTimeV2] Loading LoRA checkpoint from {lora_ckpt_path}")
                    lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
                    if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
                        peft.set_peft_model_state_dict(self.pipeline.generator.model, lora_checkpoint["generator_lora"])
                    else:
                        peft.set_peft_model_state_dict(self.pipeline.generator.model, lora_checkpoint)
                    print("[RealTimeV2] LoRA weights loaded")
        except Exception as e:
            print(f"[RealTimeV2] LoRA setup skipped due to error: {e}")
            
    def _initialize_caches(self):
        """Initialize caches following InteractiveCausalInferencePipeline pattern"""
        # Cache configuration (copy from interactive pipeline)
        local_attn_cfg = getattr(self.config.model_kwargs, "local_attn_size", -1)
        if local_attn_cfg != -1:
            # Local attention
            kv_cache_size = local_attn_cfg * self.pipeline.frame_seq_length
        else:
            # Global attention
            kv_cache_size = self.total_frames * self.pipeline.frame_seq_length

        print(f"[RealTimeV2] Initializing caches with size: {kv_cache_size}")
        
        # Initialize KV and cross-attention caches
        self.pipeline._initialize_kv_cache(
            batch_size=1,
            dtype=torch.bfloat16,
            device=self.device,
            kv_cache_size_override=kv_cache_size
        )
        self.pipeline._initialize_crossattn_cache(
            batch_size=1,
            dtype=torch.bfloat16,
            device=self.device
        )
        
        # Pre-allocate timestep tensors
        self.pipeline._pre_allocate_timestep_tensors(
            batch_size=1,
            max_frames=self.block_size,
            device=self.device
        )
        
        # Set attention configuration
        self.pipeline.generator.model.local_attn_size = self.pipeline.local_attn_size
        self.pipeline._set_all_modules_max_attention_size(self.pipeline.local_attn_size)
        
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
        
        # Initialize caches for this generation session (if not already done)
        if not hasattr(self.pipeline, 'kv_cache1') or self.pipeline.kv_cache1 is None:
            self._initialize_generation_caches()
        
        # Start generation thread
        self.generation_thread = threading.Thread(
            target=self._generation_loop, 
            args=(initial_prompt,),
            daemon=True
        )
        self.generation_thread.start()
        
    def stop_generation(self):
        """Stop generation"""
        print("[RealTimeV2] Stopping generation...")
        self.is_running = False
        
        if self.generation_thread:
            self.generation_thread.join(timeout=5.0)
            
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
            # Safety: enforce BF16 on generator before first use (covers any late dtype changes)
            try:
                self.pipeline.generator.to(dtype=torch.bfloat16, device=self.device)
                for module in self.pipeline.generator.modules():
                    for p in module.parameters(recurse=False):
                        if p is not None and p.dtype == torch.float32:
                            p.data = p.data.to(torch.bfloat16)
                    for name, b in module.named_buffers(recurse=False):
                        if b is not None and b.dtype == torch.float32:
                            setattr(module, name, b.to(torch.bfloat16))
            except Exception:
                pass
            # Encode initial prompt (like interactive pipeline)
            print("[RealTimeV2] Encoding initial prompt...")
            with torch.no_grad():
                out = self.pipeline.text_encoder._compiled_forward(text_prompts=[initial_prompt])
                try:
                    if isinstance(out, dict) and "prompt_embeds" in out:
                        out["prompt_embeds"] = out["prompt_embeds"].clone()
                except Exception:
                    pass
                current_conditional_dict = {"prompt_embeds": out["prompt_embeds"]}
            
            # Initialize frame-by-frame generation (like interactive pipeline temporal loop)
            current_start_frame = 0
            self.pipeline.generator.model.local_attn_size = self.pipeline.local_attn_size
            
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
                        out = self.pipeline.text_encoder._compiled_forward(text_prompts=[self.current_prompt])
                        try:
                            if isinstance(out, dict) and "prompt_embeds" in out:
                                out["prompt_embeds"] = out["prompt_embeds"].clone()
                        except Exception:
                            pass
                        current_conditional_dict = {"prompt_embeds": out["prompt_embeds"]}
                    # Perform recaching (like interactive pipeline)
                    self._recache_after_switch(self.output, current_start_frame, current_conditional_dict)
                
                # Generate current block (copy from interactive pipeline block generation)
                block_output = self._generate_block(
                    current_start_frame, current_num_frames, current_conditional_dict
                )
                
                # Store output (like interactive pipeline)
                self.output[:, current_start_frame:current_start_frame + current_num_frames] = block_output.to(self.output.device)
                
                # Decode latents synchronously and stream frames
                try:
                    with torch.no_grad():
                        video_frames = self.pipeline.vae._compiled_decode(block_output, use_cache=False)
                    video_np = video_frames.cpu().numpy()
                    video_np = (video_np * 0.5 + 0.5).clip(0, 1)
                    for t in range(video_np.shape[1]):
                        frame = video_np[0, t].transpose(1, 2, 0)
                        frame_uint8 = (frame * 255).astype(np.uint8)
                        frame_idx = current_start_frame + t
                        self.latest_frame = frame_uint8
                        for callback in self.frame_callbacks:
                            try:
                                callback(frame_uint8, frame_idx)
                            except Exception as e:
                                print(f"[RealTimeV2] Callback error: {e}")
                finally:
                    try:
                        del video_frames, video_np
                    except Exception:
                        pass
                
                # Update frame counter
                self.current_frame = current_start_frame + current_num_frames
                current_start_frame += current_num_frames
                
                # Memory cleanup after each block
                del block_output
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                
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
        for index, current_timestep in enumerate(self.pipeline.denoising_step_list):
            timestep_val = float(current_timestep)
            
            # Use pre-allocated timestep tensor (like interactive pipeline)
            timestep_key = int(current_timestep.item()) if hasattr(current_timestep, 'item') else int(current_timestep)
            if num_frames == self.block_size:
                timestep = self._timestep_tensors[timestep_key]
            else:
                timestep = self._timestep_tensors[timestep_key][:, :num_frames]

            if index < len(self.pipeline.denoising_step_list) - 1:
                # Intermediate denoising step (copy from interactive pipeline)
                torch.compiler.cudagraph_mark_step_begin()
                _, denoised_pred = self.pipeline.generator._compiled_forward(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.pipeline.kv_cache1,
                    crossattn_cache=self.pipeline.crossattn_cache,
                    current_start=start_frame * self.pipeline.frame_seq_length,
                )
                
                # Noise scheduling (copy from interactive pipeline)
                next_timestep = self.pipeline.denoising_step_list[index + 1]
                next_timestep_key = int(next_timestep.item()) if hasattr(next_timestep, 'item') else int(next_timestep)
                if num_frames == self.block_size:
                    noise_timestep = self._noise_timestep_tensors[next_timestep_key]
                else:
                    noise_timestep = self._noise_timestep_tensors[next_timestep_key][:num_frames]
                
                noisy_input = self.pipeline.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    noise_timestep,
                ).unflatten(0, denoised_pred.shape[:2])
            else:
                # Final denoising step (copy from interactive pipeline)
                torch.compiler.cudagraph_mark_step_begin()
                _, denoised_pred = self.pipeline.generator._compiled_forward(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.pipeline.kv_cache1,
                    crossattn_cache=self.pipeline.crossattn_cache,
                    current_start=start_frame * self.pipeline.frame_seq_length,
                )

        # KV cache update with clean context (copy from interactive pipeline)
        if num_frames == self.pipeline.num_frame_per_block:
            context_timestep = self._context_timestep_base
        else:
            context_timestep = self._context_timestep_base[:, :num_frames]
        
        torch.compiler.cudagraph_mark_step_begin()
        self.pipeline.generator._compiled_forward(
            noisy_image_or_video=denoised_pred,
            conditional_dict=conditional_dict,
            timestep=context_timestep,
            kv_cache=self.pipeline.kv_cache1,
            crossattn_cache=self.pipeline.crossattn_cache,
            current_start=start_frame * self.pipeline.frame_seq_length,
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
        num_recache_frames = current_start_frame if self.pipeline.local_attn_size == -1 else min(self.pipeline.local_attn_size, current_start_frame)
        recache_start_frame = current_start_frame - num_recache_frames
        
        frames_to_recache = output[:, recache_start_frame:current_start_frame]
        
        # Move to GPU if needed
        if frames_to_recache.device.type == 'cpu':
            frames_to_recache = frames_to_recache.to(self.device)
        
        print(f"[RealTimeV2] Recaching {num_recache_frames} frames for prompt switch")
        
        # Prepare block mask (copy from interactive pipeline)
        device = frames_to_recache.device
        block_mask = self.pipeline._get_block_mask(
            device=device,
            num_frames=num_recache_frames,
            local_attn_size=self.pipeline.local_attn_size
        )
        self.pipeline.generator.model.block_mask = block_mask
        
        # Context timestep (copy from interactive pipeline)
        batch_size = frames_to_recache.shape[0]
        cache_key = (batch_size, num_recache_frames)
        if cache_key not in self.pipeline._recache_timestep_cache:
            self.pipeline._recache_timestep_cache[cache_key] = torch.full(
                [batch_size, num_recache_frames], 
                self.config.context_noise,
                device=device, 
                dtype=torch.int64
            )
        context_timestep = self.pipeline._recache_timestep_cache[cache_key]
        
        # Perform recaching (copy from interactive pipeline)
        with torch.no_grad():
            self.pipeline.generator(
                noisy_image_or_video=frames_to_recache,
                conditional_dict=new_conditional_dict,
                timestep=context_timestep,
                kv_cache=self.pipeline.kv_cache1,
                crossattn_cache=self.pipeline.crossattn_cache,
                current_start=recache_start_frame * self.pipeline.frame_seq_length,
            )
        
        # Reset cross-attention cache after recaching (copy from interactive pipeline)
        for blk in self.pipeline.crossattn_cache:
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
                        video_frames = self.pipeline.vae.decode_to_pixel(latents_chunk, use_cache=False)
                    
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
                    
                    # Memory cleanup after processing each batch
                    del latents_chunk, video_frames, video_np
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                                
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
                
        # Reset frame data
        self.latest_frame = None
        
        # Clear existing output and noise to free memory
        if hasattr(self, 'output') and self.output is not None:
            del self.output
        if hasattr(self, 'noise') and self.noise is not None:
            del self.noise
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
            
        # Generate new noise
        self.noise = torch.randn([
            1, self.total_frames, 16, 60, 104
        ], device=self.device, dtype=torch.bfloat16)
        
        # Create new output tensor
        self.output = torch.zeros_like(self.noise)
        
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