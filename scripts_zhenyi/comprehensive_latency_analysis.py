#!/usr/bin/env python3
# Comprehensive latency analysis with detailed component-level breakdown
# Measures latency across:
# - Model steps / denoising iterations  
# - Attention kernels
# - KV operations (frame-sink concat / recache)
# - Quant/dequant operations
# - VAE encode/decode
# - Device ↔ Host synchronization

import argparse
import torch
import os
import time
import json
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import (
    CausalInferencePipeline,
)
from utils.dataset import TextDataset
from utils.misc import set_seed
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, log_gpu_memory

# Import comprehensive timing system
from comprehensive_latency_tracker import ComprehensiveLatencyTracker

def print_component_breakdown(stats, component_name, title=None):
    """Print detailed breakdown for a component"""
    if component_name not in stats['component_breakdown']:
        return
        
    component = stats['component_breakdown'][component_name]
    title = title or component_name.replace('_', ' ').title()
    
    print(f"\n{title}:")
    print("-" * len(title))
    
    if 'gpu_timing' in component:
        gpu = component['gpu_timing']
        print(f"  Count: {component['count']}")
        print(f"  Mean: {gpu['mean_ms']:.2f} ms")
        print(f"  Median: {gpu['median_ms']:.2f} ms")
        print(f"  P95: {gpu['p95_ms']:.2f} ms")  
        print(f"  Max: {gpu['max_ms']:.2f} ms")
        print(f"  Total: {gpu['total_ms']:.2f} ms")
        
        if 'per_block_variance' in component:
            print("  Per-block breakdown:")
            for block_key, block_stats in component['per_block_variance'].items():
                print(f"    {block_key}: {block_stats['mean_ms']:.2f} ms (n={block_stats['count']})")

def print_comprehensive_summary(stats):
    """Print comprehensive timing summary"""
    print("\n" + "="*80)
    print("COMPREHENSIVE LATENCY ANALYSIS SUMMARY")
    print("="*80)
    
    config = stats['configuration']
    print(f"Configuration:")
    print(f"  - Frames per block: {config['num_frame_per_block']}")
    print(f"  - Total frames: {config['total_frames']}")
    print(f"  - Total blocks: {config['total_blocks']}")
    print(f"  - Warmup frames: {config['warmup_frames']}")
    
    # Performance metrics
    if 'performance_metrics' in stats:
        perf = stats['performance_metrics']
        print(f"\nOverall Performance:")
        print(f"  - Total generation time: {perf['total_generation_time_s']:.2f} s")
        print(f"  - Average FPS: {perf['average_fps']:.2f}")
        print(f"  - Average ms per frame: {perf['ms_per_frame']:.2f}")
    
    # Cross-block vs intra-block analysis
    if 'cross_vs_intra_block' in stats:
        cross_block = stats['cross_vs_intra_block']
        print(f"\nWorst Case Analysis (Cross-block vs Intra-block):")
        
        if 'cross_block_worst_case' in cross_block:
            worst = cross_block['cross_block_worst_case']
            print(f"  Cross-block transitions ({worst['count']} instances):")
            print(f"    - Mean latency: {worst['mean_ms']:.2f} ms")
            print(f"    - P95 latency: {worst['p95_ms']:.2f} ms")
            print(f"    - Max latency: {worst['max_ms']:.2f} ms")
            
        if 'intra_block_best_case' in cross_block:
            best = cross_block['intra_block_best_case']
            print(f"  Intra-block transitions ({best['count']} instances):")
            print(f"    - Mean latency: {best['mean_ms']:.2f} ms")
            print(f"    - P95 latency: {best['p95_ms']:.2f} ms")
            print(f"    - Max latency: {best['max_ms']:.2f} ms")
            
        if 'comparison' in cross_block:
            comp = cross_block['comparison']
            print(f"  Impact Analysis:")
            print(f"    - Cross-block is {comp['latency_multiplier']:.1f}x slower")
            print(f"    - Additional latency: +{comp['additional_latency_ms']:.2f} ms")
            print(f"    - Cross-block percentage: {comp['cross_block_percentage']:.1f}%")
    
    # Component-level breakdown
    print(f"\n" + "="*60)
    print("COMPONENT-LEVEL BREAKDOWN")
    print("="*60)
    
    # Core components
    print_component_breakdown(stats, 'text_encoding', 'Text Encoder')
    print_component_breakdown(stats, 'denoising_step', 'Denoising Steps')
    print_component_breakdown(stats, 'denoising_step_final', 'Final Denoising Steps')
    print_component_breakdown(stats, 'vae_decode_final', 'VAE Decoding')
    
    # KV operations
    print_component_breakdown(stats, 'kv_operations', 'KV Cache Operations')
    print_component_breakdown(stats, 'attention_kernel', 'Attention Kernels')
    
    # Device synchronization
    print_component_breakdown(stats, 'device_sync', 'Device Synchronization')
    
    # Other components
    for component_name in stats['component_breakdown']:
        if component_name not in ['text_encoding', 'denoising_step', 'denoising_step_final', 
                                'vae_decode_final', 'kv_operations', 'attention_kernel', 'device_sync']:
            print_component_breakdown(stats, component_name)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--output_dir", type=str, default="./comprehensive_latency_analysis", help="Directory to save analysis results")
args = parser.parse_args()

config = OmegaConf.load(args.config_path)

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    os.environ["NCCL_CROSS_NIC"] = "1"
    os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
    os.environ["NCCL_TIMEOUT"] = os.environ.get("NCCL_TIMEOUT", "1800")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", str(local_rank)))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.constants.default_pg_timeout,
        )
    set_seed(config.seed + local_rank)
    config.distributed = True
    if rank == 0:
        print(f"[Rank {rank}] Initialized distributed processing on device {device}")
else:
    local_rank = 0
    rank = 0
    device = torch.device("cuda")
    set_seed(config.seed)
    config.distributed = False
    print(f"Single GPU mode on device {device}")

print(f'Free VRAM {get_cuda_free_memory_gb(device)} GB')
low_memory = get_cuda_free_memory_gb(device) < 40
low_memory = True

torch.set_grad_enabled(False)

# Initialize pipeline
pipeline = CausalInferencePipeline(config, device=device)

# Load generator checkpoint
if config.generator_ckpt:
    state_dict = torch.load(config.generator_ckpt, map_location="cpu")
    if "generator" in state_dict or "generator_ema" in state_dict:
        raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
    elif "model" in state_dict:
        raw_gen_state_dict = state_dict["model"]
    else:
        raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")
    if config.use_ema:
        def _clean_key(name: str) -> str:
            """Remove FSDP / checkpoint wrapper prefixes from parameter names."""
            name = name.replace("_fsdp_wrapped_module.", "")
            return name

        cleaned_state_dict = { _clean_key(k): v for k, v in raw_gen_state_dict.items() }
        missing, unexpected = pipeline.generator.load_state_dict(cleaned_state_dict, strict=False)
        if local_rank == 0:
            if len(missing) > 0:
                print(f"[Warning] {len(missing)} parameters are missing when loading checkpoint: {missing[:8]} ...")
            if len(unexpected) > 0:
                print(f"[Warning] {len(unexpected)} unexpected parameters encountered when loading checkpoint: {unexpected[:8]} ...")
    else:
        pipeline.generator.load_state_dict(raw_gen_state_dict)

# LoRA support (optional)
from utils.lora_utils import configure_lora_for_model
import peft

pipeline.is_lora_enabled = False
if getattr(config, "adapter", None) and configure_lora_for_model is not None:
    if local_rank == 0:
        print(f"LoRA enabled with config: {config.adapter}")
        print("Applying LoRA to generator (inference)...")
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=(local_rank == 0),
    )

    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if lora_ckpt_path:
        if local_rank == 0:
            print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
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
pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)

extended_prompt_path = config.data_path
dataset = TextDataset(prompt_path=config.data_path, extended_prompt_path=extended_prompt_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directories
if local_rank == 0:
    os.makedirs(config.output_folder, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]

    all_video = []
    num_generated_frames = 0

    prompt = batch['prompts'][0]
    extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
    if extended_prompt is not None:
        prompts = [extended_prompt] * config.num_samples
    else:
        prompts = [prompt] * config.num_samples

    sampled_noise = torch.randn(
        [config.num_samples, config.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
    )

    print("Starting comprehensive latency analysis...")
    print(f"Measuring detailed component timing for {config.num_output_frames} frames")
    print(f"Components: text encoding, denoising steps, attention, KV ops, VAE, device sync")
    
    # Create comprehensive latency tracker
    comprehensive_tracker = ComprehensiveLatencyTracker(
        num_frame_per_block=config.num_frame_per_block,
        device=device
    )
    
    # Use comprehensive timing-instrumented inference
    video_result = pipeline.inference_with_timing(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        low_memory=low_memory,
        profile=False,
        latency_tracker=comprehensive_tracker
    )
    
    video, latents = video_result
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Generate and display comprehensive statistics
    stats = comprehensive_tracker.get_comprehensive_statistics()
    if local_rank == 0:
        print_comprehensive_summary(stats)

    # Save comprehensive analysis
    if local_rank == 0:
        log_path = os.path.join(args.output_dir, f'comprehensive_analysis_rank{rank}_prompt{idx}.json')
        comprehensive_tracker.save_comprehensive_log(log_path)
        print(f"\nComprehensive analysis saved to: {log_path}")

    # Final output video
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # Save the video
    if idx < num_prompts:
        if hasattr(pipeline, 'is_lora_enabled') and pipeline.is_lora_enabled:
            model_type = "lora"
        elif getattr(config, 'use_ema', False):
            model_type = "ema"
        else:
            model_type = "regular"
            
        for seed_idx in range(config.num_samples):
            if config.save_with_index:
                output_path = os.path.join(config.output_folder, f'rank{rank}-{idx}-{seed_idx}_{model_type}_comprehensive.mp4')
            else:
                output_path = os.path.join(config.output_folder, f'rank{rank}-{prompt[:100]}-{seed_idx}_comprehensive.mp4')
            write_video(output_path, video[seed_idx], fps=16)

    if config.inference_iter != -1 and i >= config.inference_iter:
        break

if dist.is_initialized():
    dist.destroy_process_group()