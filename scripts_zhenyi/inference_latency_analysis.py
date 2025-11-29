#!/usr/bin/env python3
# Modified inference.py for analyzing steady-state inter-frame latency
# Measures gap between consecutive frames during continuous generation

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

class LatencyTracker:
    def __init__(self):
        self.frame_times = []
        self.inter_frame_latencies = []
        self.generation_start_time = None
        self.last_frame_time = None
        self.warmup_frames = 5  # Skip first few frames for steady-state analysis
        
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
            self.inter_frame_latencies.append({
                'frame_idx': frame_idx,
                'latency_ms': inter_frame_latency * 1000
            })
            
        self.last_frame_time = current_time
        
    def get_statistics(self):
        """Calculate latency statistics"""
        if not self.inter_frame_latencies:
            return {}
            
        latencies = [x['latency_ms'] for x in self.inter_frame_latencies]
        
        return {
            'num_frames': len(self.inter_frame_latencies),
            'mean_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'total_generation_time_s': self.frame_times[-1]['elapsed_since_start'] if self.frame_times else 0,
            'average_fps': len(self.frame_times) / self.frame_times[-1]['elapsed_since_start'] if self.frame_times else 0
        }
        
    def save_detailed_log(self, output_path):
        """Save detailed frame timing information"""
        data = {
            'frame_times': self.frame_times,
            'inter_frame_latencies': self.inter_frame_latencies,
            'statistics': self.get_statistics()
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--output_dir", type=str, default="./latency_analysis", help="Directory to save latency analysis results")
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

# The pipeline now has a built-in inference_with_timing method

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

    print("Starting latency-tracked inference...")
    print(f"Generating {config.num_output_frames} frames")
    
    # Create latency tracker for this run
    latency_tracker = LatencyTracker()
    
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

    # Calculate and display statistics
    stats = latency_tracker.get_statistics()
    if local_rank == 0:
        print("\n" + "="*50)
        print("INTER-FRAME LATENCY ANALYSIS")
        print("="*50)
        print(f"Total frames analyzed: {stats.get('num_frames', 0)}")
        print(f"Mean inter-frame latency: {stats.get('mean_latency_ms', 0):.2f} ms")
        print(f"Median inter-frame latency: {stats.get('median_latency_ms', 0):.2f} ms")
        print(f"Std deviation: {stats.get('std_latency_ms', 0):.2f} ms")
        print(f"Min latency: {stats.get('min_latency_ms', 0):.2f} ms")
        print(f"Max latency: {stats.get('max_latency_ms', 0):.2f} ms")
        print(f"95th percentile: {stats.get('p95_latency_ms', 0):.2f} ms")
        print(f"99th percentile: {stats.get('p99_latency_ms', 0):.2f} ms")
        print(f"Average FPS: {stats.get('average_fps', 0):.2f}")
        print(f"Total generation time: {stats.get('total_generation_time_s', 0):.2f} s")
        print("="*50)

    # Save detailed latency log
    if local_rank == 0:
        log_path = os.path.join(args.output_dir, f'latency_analysis_rank{rank}_prompt{idx}.json')
        latency_tracker.save_detailed_log(log_path)
        print(f"Detailed latency log saved to: {log_path}")

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
                output_path = os.path.join(config.output_folder, f'rank{rank}-{idx}-{seed_idx}_{model_type}_latency.mp4')
            else:
                output_path = os.path.join(config.output_folder, f'rank{rank}-{prompt[:100]}-{seed_idx}_latency.mp4')
            write_video(output_path, video[seed_idx], fps=16)

    if config.inference_iter != -1 and i >= config.inference_iter:
        break

if dist.is_initialized():
    dist.destroy_process_group()
