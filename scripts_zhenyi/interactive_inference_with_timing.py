# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interactive Causal Inference with Comprehensive Latency Analysis

This script extends the standard interactive inference to include:
- Comprehensive timing breakdown of all major components
- Prompt switching analysis
- Frame generation latency tracking
- Component-level performance metrics
- Cross-block vs intra-block analysis
"""

import argparse
import os
import sys
import json
from typing import List

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.io import write_video
from torchvision import transforms
from einops import rearrange

# Add the parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.misc import set_seed
from utils.distributed import barrier  
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller
from pipeline.interactive_causal_inference import InteractiveCausalInferencePipeline
from utils.dataset import MultiTextDataset

# Import timing components
try:
    from comprehensive_latency_tracker import ComprehensiveLatencyTracker
    COMPREHENSIVE_TIMING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Comprehensive timing not available: {e}")
    COMPREHENSIVE_TIMING_AVAILABLE = False

# ----------------------------- Argument parsing -----------------------------
parser = argparse.ArgumentParser("Interactive causal inference with timing analysis")
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--enable_timing", action="store_true", default=False, 
                   help="Enable comprehensive timing analysis")
parser.add_argument("--timing_output_dir", type=str, default="/tmp/interactive_timing_results",
                   help="Directory to save timing analysis results")
args = parser.parse_args()

config = OmegaConf.load(args.config_path)

# ----------------------------- Distributed setup -----------------------------
if "LOCAL_RANK" in os.environ:
    os.environ["NCCL_CROSS_NIC"] = "1"
    os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
    os.environ["NCCL_TIMEOUT"] = os.environ.get("NCCL_TIMEOUT", "1800")
    
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", str(local_rank)))
    
    # Set device first
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Initialize process group with backend and timeout
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.constants.default_pg_timeout
        )
    
    set_seed(config.seed + local_rank)
    print(f"[Rank {rank}] Initialized distributed processing on device {device}")
else:
    local_rank = 0
    rank = 0
    device = torch.device("cuda")
    set_seed(config.seed)
    print(f"Single GPU mode on device {device}")

low_memory = get_cuda_free_memory_gb(device) < 40
torch.set_grad_enabled(False)

# ----------------------------- Initialize Pipeline -----------------------------
pipeline = InteractiveCausalInferencePipeline(config, device=device)

# Load generator checkpoint
if config.generator_ckpt:
    state_dict = torch.load(config.generator_ckpt, map_location="cpu")
    raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]

    if config.use_ema:
        def _clean_key(name: str) -> str:
            return name.replace("_fsdp_wrapped_module.", "")

        cleaned_state_dict = {_clean_key(k): v for k, v in raw_gen_state_dict.items()}
        missing, unexpected = pipeline.generator.load_state_dict(
            cleaned_state_dict, strict=False
        )
        if local_rank == 0:
            if missing:
                print(f"[Warning] {len(missing)} parameters missing: {missing[:8]} ...")
            if unexpected:
                print(f"[Warning] {len(unexpected)} unexpected params: {unexpected[:8]} ...")
    else:
        pipeline.generator.load_state_dict(raw_gen_state_dict)

# ----------------------------- LoRA support (optional) ---------------------------
from utils.lora_utils import configure_lora_for_model
import peft

pipeline.is_lora_enabled = False
if getattr(config, "adapter", None) and configure_lora_for_model is not None:
    if local_rank == 0:
        print(f"LoRA enabled with config: {config.adapter}")
        print("Applying LoRA to generator (inference)...")
    # After loading base weights, apply LoRA wrapper to the generator's transformer model
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=(local_rank == 0),
    )

    # Load LoRA weights (if lora_ckpt is provided)
    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if lora_ckpt_path:
        if local_rank == 0:
            print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
        # Support both formats: containing the `generator_lora` key or a raw LoRA state dict
        if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])
        else:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)
        if local_rank == 0:
            print("LoRA weights loaded for generator")
    else:
        if local_rank == 0:
            print("No LoRA checkpoint specified; using base weights with LoRA adapters initialized")

    pipeline.is_lora_enabled = True

# Move pipeline to appropriate dtype and device
print("dtype", pipeline.generator.model.dtype)
pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)

# ----------------------------- Setup Timing Analysis -----------------------------
latency_tracker = None
if args.enable_timing and COMPREHENSIVE_TIMING_AVAILABLE:
    if local_rank == 0:
        os.makedirs(args.timing_output_dir, exist_ok=True)
    
    latency_tracker = ComprehensiveLatencyTracker(
        num_frame_per_block=config.num_frame_per_block,
        device=device
    )
    
    if local_rank == 0:
        print(f"Comprehensive timing enabled. Results will be saved to: {args.timing_output_dir}")
elif args.enable_timing:
    print("Warning: Timing analysis requested but comprehensive_latency_tracker not available")

# ----------------------------- Build dataset -----------------------------
# Parse switch_frame_indices
switch_frame_indices: List[int] = [int(x) for x in config.switch_frame_indices.split(",") if x.strip()]

# Create dataset
dataset = MultiTextDataset(config.data_path)

# Validate number of segments & switch_frame_indices length
num_segments = len(dataset[0]["prompts_list"])
assert len(switch_frame_indices) == num_segments - 1, (
    "The number of switch_frame_indices should be the number of prompt segments minus 1"
)

print("Number of segments:", num_segments)
print("Switch frame indices:", switch_frame_indices)

num_prompts_total = len(dataset)
print(f"Number of prompt lines: {num_prompts_total}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)

dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory
if local_rank == 0:
    os.makedirs(config.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

# ----------------------------- Inference loop with timing -----------------------------
print("\n" + "="*80)
print("STARTING INTERACTIVE INFERENCE WITH COMPREHENSIVE TIMING")
print("="*80)

for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data["idx"].item()
    prompts_list: List[List[str]] = batch_data["prompts_list"]
    
    if local_rank == 0:
        print(f"\n[Prompt {idx}] Processing prompts: {prompts_list}")
        print(f"Switch points: {switch_frame_indices}")

    sampled_noise = torch.randn(
        [
            config.num_samples,
            config.num_output_frames,
            16,
            60,
            104,
        ],
        device=device,
        dtype=torch.bfloat16,
    )

    # Run inference with timing
    if latency_tracker:
        # Store current prompt info for analysis
        latency_tracker.current_prompt_idx = idx
        latency_tracker.current_prompts = prompts_list  
        latency_tracker.current_switch_indices = switch_frame_indices
        latency_tracker.current_rank = rank

    video = pipeline.inference_with_timing(
        noise=sampled_noise,
        text_prompts_list=prompts_list,
        switch_frame_indices=switch_frame_indices,
        return_latents=False,
        latency_tracker=latency_tracker
    )

    current_video = rearrange(video, "b t c h w -> b t h w c").cpu() * 255.0

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # Determine model type for filename
    if hasattr(pipeline, 'is_lora_enabled') and pipeline.is_lora_enabled:
        model_type = "lora"
    elif getattr(config, 'use_ema', False):
        model_type = "ema"
    else:
        model_type = "regular"

    # Save videos
    for seed_idx in range(config.num_samples):
        if config.save_with_index:
            output_path = os.path.join(config.output_folder, f"rank{rank}-{idx}-{seed_idx}_{model_type}.mp4")
        else:
            # Use the first prompt segment as the filename prefix to avoid overly long names
            short_name = prompts_list[0][0][:100].replace("/", "_")
            output_path = os.path.join(config.output_folder, f"rank{rank}-{short_name}-{seed_idx}_{model_type}.mp4")
        write_video(output_path, current_video[seed_idx].to(torch.uint8), fps=16)

    # Save timing analysis if enabled
    if latency_tracker and local_rank == 0:
        timing_filename = f"interactive_timing_analysis_rank{rank}_prompt{idx}.json"
        timing_filepath = os.path.join(args.timing_output_dir, timing_filename)
        
        analysis_data = latency_tracker.get_comprehensive_statistics()
        with open(timing_filepath, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"\nTiming analysis saved to: {timing_filepath}")
        
        # Print brief summary
        if "performance_metrics" in analysis_data:
            metrics = analysis_data["performance_metrics"]
            print("\nTIMING SUMMARY:")
            print(f"  Total generation time: {metrics.get('total_generation_time_ms', 0):.2f} ms")
            print(f"  Average time per frame: {metrics.get('avg_time_per_frame_ms', 0):.2f} ms")
            print(f"  Total frames: {analysis_data.get('configuration', {}).get('total_frames', 0)}")
            
            if "component_breakdown" in analysis_data:
                print("\nTOP COMPONENTS BY TIME:")
                comp_data = analysis_data["component_breakdown"]
                # Sort by mean GPU time and show top 5
                sorted_components = sorted(
                    [(name, data.get('gpu_times', {}).get('mean', 0)) for name, data in comp_data.items()],
                    key=lambda x: x[1], reverse=True
                )[:5]
                for comp_name, mean_time in sorted_components:
                    print(f"  {comp_name}: {mean_time:.2f} ms (avg)")

    if config.inference_iter != -1 and i >= config.inference_iter:
        break

# Final cleanup
if dist.is_initialized():
    dist.destroy_process_group()

if latency_tracker and local_rank == 0:
    print(f"\nAll timing analysis results saved to: {args.timing_output_dir}")
    print("Use scripts_zhenyi/analyze_latency_breakdown.py to analyze the results in detail.")


"""
  python scripts_zhenyi/interactive_inference_with_timing.py \
      --config_path configs/your_config.yaml \
      --enable_timing \
      --timing_output_dir /tmp/interactive_timing_results
"""