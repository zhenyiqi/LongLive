#!/usr/bin/env python3
# Enhanced latency analysis focusing on worst-case scenarios:
# Inter-block frame transitions vs intra-block frame transitions

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

class EnhancedLatencyTracker:
    def __init__(self, num_frame_per_block):
        self.frame_times = []
        self.inter_frame_latencies = []
        self.block_boundaries = []
        self.generation_start_time = None
        self.last_frame_time = None
        self.warmup_frames = 5
        self.num_frame_per_block = num_frame_per_block
        
    def start_generation(self):
        """Mark the start of video generation"""
        self.generation_start_time = time.perf_counter()
        self.last_frame_time = self.generation_start_time
        
    def record_frame_completion(self, frame_idx):
        """Record when a frame is completed"""
        current_time = time.perf_counter()
        self.frame_times.append({
            'frame_idx': frame_idx,
            'timestamp': current_time,
            'elapsed_since_start': current_time - self.generation_start_time
        })
        
        # Calculate inter-frame latency (skip warmup frames for steady-state)
        if frame_idx > self.warmup_frames:
            inter_frame_latency = current_time - self.last_frame_time
            
            # Determine if this is a cross-block transition (worst case)
            is_cross_block = self._is_cross_block_transition(frame_idx)
            
            self.inter_frame_latencies.append({
                'frame_idx': frame_idx,
                'latency_ms': inter_frame_latency * 1000,
                'is_cross_block': is_cross_block,
                'transition_type': 'cross-block' if is_cross_block else 'intra-block'
            })
            
        self.last_frame_time = current_time
    
    def _is_cross_block_transition(self, frame_idx):
        """
        Determine if this frame represents a cross-block transition.
        Cross-block transitions occur when moving from the last frame of one block
        to the first frame of the next block.
        """
        if frame_idx == 0:
            return False  # First frame has no predecessor
        
        # Check if the previous frame was the last frame of a block
        prev_frame_idx = frame_idx - 1
        return (prev_frame_idx + 1) % self.num_frame_per_block == 0
    
    def get_enhanced_statistics(self):
        """Calculate enhanced latency statistics including worst-case analysis"""
        if not self.inter_frame_latencies:
            return {}
        
        all_latencies = [x['latency_ms'] for x in self.inter_frame_latencies]
        cross_block_latencies = [x['latency_ms'] for x in self.inter_frame_latencies if x['is_cross_block']]
        intra_block_latencies = [x['latency_ms'] for x in self.inter_frame_latencies if not x['is_cross_block']]
        
        stats = {
            'total_frames': len(self.inter_frame_latencies),
            'cross_block_transitions': len(cross_block_latencies),
            'intra_block_transitions': len(intra_block_latencies),
            'num_frame_per_block': self.num_frame_per_block
        }
        
        # Overall statistics
        if all_latencies:
            stats['overall'] = {
                'mean_latency_ms': np.mean(all_latencies),
                'median_latency_ms': np.median(all_latencies),
                'std_latency_ms': np.std(all_latencies),
                'min_latency_ms': np.min(all_latencies),
                'max_latency_ms': np.max(all_latencies),
                'p95_latency_ms': np.percentile(all_latencies, 95),
                'p99_latency_ms': np.percentile(all_latencies, 99)
            }
        
        # Cross-block (worst case) statistics  
        if cross_block_latencies:
            stats['cross_block_worst_case'] = {
                'mean_latency_ms': np.mean(cross_block_latencies),
                'median_latency_ms': np.median(cross_block_latencies),
                'std_latency_ms': np.std(cross_block_latencies),
                'min_latency_ms': np.min(cross_block_latencies),
                'max_latency_ms': np.max(cross_block_latencies),
                'p95_latency_ms': np.percentile(cross_block_latencies, 95),
                'p99_latency_ms': np.percentile(cross_block_latencies, 99)
            }
        
        # Intra-block (best case) statistics
        if intra_block_latencies:
            stats['intra_block_best_case'] = {
                'mean_latency_ms': np.mean(intra_block_latencies),
                'median_latency_ms': np.median(intra_block_latencies),
                'std_latency_ms': np.std(intra_block_latencies),
                'min_latency_ms': np.min(intra_block_latencies),
                'max_latency_ms': np.max(intra_block_latencies),
                'p95_latency_ms': np.percentile(intra_block_latencies, 95),
                'p99_latency_ms': np.percentile(intra_block_latencies, 99)
            }
        
        # Worst case analysis
        if cross_block_latencies and intra_block_latencies:
            cross_block_mean = np.mean(cross_block_latencies)
            intra_block_mean = np.mean(intra_block_latencies)
            stats['worst_case_analysis'] = {
                'latency_multiplier': cross_block_mean / intra_block_mean if intra_block_mean > 0 else float('inf'),
                'additional_latency_ms': cross_block_mean - intra_block_mean,
                'cross_block_percentage': (len(cross_block_latencies) / len(self.inter_frame_latencies)) * 100
            }
        
        if self.frame_times:
            stats['timing'] = {
                'total_generation_time_s': self.frame_times[-1]['elapsed_since_start'],
                'average_fps': len(self.frame_times) / self.frame_times[-1]['elapsed_since_start']
            }
        
        return stats
        
    def save_enhanced_log(self, output_path):
        """Save detailed frame timing information with enhanced analysis"""
        data = {
            'frame_times': self.frame_times,
            'inter_frame_latencies': self.inter_frame_latencies,
            'enhanced_statistics': self.get_enhanced_statistics(),
            'analysis_config': {
                'num_frame_per_block': self.num_frame_per_block,
                'warmup_frames': self.warmup_frames
            }
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--output_dir", type=str, default="./enhanced_latency_analysis", help="Directory to save latency analysis results")
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

    print("Starting enhanced latency analysis...")
    print(f"Generating {config.num_output_frames} frames in blocks of {config.num_frame_per_block}")
    print(f"Expected cross-block transitions: {(config.num_output_frames // config.num_frame_per_block) - 1}")
    
    # Create enhanced latency tracker for this run
    latency_tracker = EnhancedLatencyTracker(config.num_frame_per_block)
    
    # Use the timing-instrumented inference
    video_result = pipeline.inference_with_timing(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        low_memory=low_memory,
        profile=False,
        latency_tracker=latency_tracker
    )
    
    video, latents = video_result
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Calculate and display enhanced statistics
    stats = latency_tracker.get_enhanced_statistics()
    if local_rank == 0:
        print("\n" + "="*60)
        print("ENHANCED INTER-FRAME LATENCY ANALYSIS")
        print("="*60)
        print(f"Configuration:")
        print(f"  - Frames per block: {stats.get('num_frame_per_block', 'N/A')}")
        print(f"  - Total frames: {stats.get('total_frames', 0)}")
        print(f"  - Cross-block transitions: {stats.get('cross_block_transitions', 0)}")
        print(f"  - Intra-block transitions: {stats.get('intra_block_transitions', 0)}")
        
        if 'overall' in stats:
            print(f"\nOVERALL STATISTICS:")
            overall = stats['overall']
            print(f"  - Mean latency: {overall['mean_latency_ms']:.2f} ms")
            print(f"  - 95th percentile: {overall['p95_latency_ms']:.2f} ms")
            print(f"  - Max latency: {overall['max_latency_ms']:.2f} ms")
        
        if 'cross_block_worst_case' in stats:
            print(f"\nWORST CASE (Cross-Block Transitions):")
            worst = stats['cross_block_worst_case']
            print(f"  - Mean latency: {worst['mean_latency_ms']:.2f} ms")
            print(f"  - 95th percentile: {worst['p95_latency_ms']:.2f} ms")
            print(f"  - Max latency: {worst['max_latency_ms']:.2f} ms")
        
        if 'intra_block_best_case' in stats:
            print(f"\nBEST CASE (Intra-Block Transitions):")
            best = stats['intra_block_best_case']
            print(f"  - Mean latency: {best['mean_latency_ms']:.2f} ms")
            print(f"  - 95th percentile: {best['p95_latency_ms']:.2f} ms")
            print(f"  - Max latency: {best['max_latency_ms']:.2f} ms")
        
        if 'worst_case_analysis' in stats:
            print(f"\nWORST CASE IMPACT:")
            impact = stats['worst_case_analysis']
            print(f"  - Cross-block latency is {impact['latency_multiplier']:.1f}x higher")
            print(f"  - Additional latency: +{impact['additional_latency_ms']:.2f} ms")
            print(f"  - Cross-block transitions: {impact['cross_block_percentage']:.1f}% of all transitions")
        
        print("="*60)

    # Save enhanced latency log
    if local_rank == 0:
        log_path = os.path.join(args.output_dir, f'enhanced_latency_analysis_rank{rank}_prompt{idx}.json')
        latency_tracker.save_enhanced_log(log_path)
        print(f"Enhanced latency log saved to: {log_path}")

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
                output_path = os.path.join(config.output_folder, f'rank{rank}-{idx}-{seed_idx}_{model_type}_enhanced.mp4')
            else:
                output_path = os.path.join(config.output_folder, f'rank{rank}-{prompt[:100]}-{seed_idx}_enhanced.mp4')
            write_video(output_path, video[seed_idx], fps=16)

    if config.inference_iter != -1 and i >= config.inference_iter:
        break

if dist.is_initialized():
    dist.destroy_process_group()