#!/usr/bin/env python3
# Comprehensive latency tracker for detailed component-level analysis
# Breaks down latency across:
# - Model steps / denoising iterations  
# - Attention kernels
# - KV operations (frame-sink concat / recache)
# - Quant/dequant operations
# - VAE encode/decode
# - Device ↔ Host synchronization

import time
import torch
import numpy as np
import json
from typing import Dict, List, Optional, Any
from collections import defaultdict, OrderedDict
import functools
import contextlib

class ComprehensiveLatencyTracker:
    def __init__(self, num_frame_per_block: int, device: torch.device):
        self.num_frame_per_block = num_frame_per_block
        self.device = device
        self.warmup_frames = 5
        
        # Frame-level tracking
        self.frame_times = []
        self.frame_idx = 0
        self.generation_start_time = None
        self.last_frame_time = None
        
        # Component-level tracking  
        self.component_timings = defaultdict(list)
        self.current_block_timings = defaultdict(list)
        self.block_summaries = []
        
        # Context stack for nested timing
        self.timing_stack = []
        self.current_context = {}
        
        # CUDA events for precise GPU timing
        self.cuda_events = {}
        
    def start_generation(self):
        """Initialize generation timing"""
        torch.cuda.synchronize()
        self.generation_start_time = time.perf_counter()
        self.last_frame_time = self.generation_start_time
        self.frame_idx = 0
        
    def start_block(self, block_idx: int):
        """Start timing a new frame block"""
        torch.cuda.synchronize()
        self.current_block_timings = defaultdict(list)
        self.current_context = {
            'block_idx': block_idx,
            'block_start_time': time.perf_counter(),
            'frame_start_idx': self.frame_idx
        }
        
    def end_block(self):
        """End timing for current frame block"""
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        block_summary = {
            'block_idx': self.current_context['block_idx'],
            'frame_range': [self.current_context['frame_start_idx'], self.frame_idx - 1],
            'total_time_ms': (end_time - self.current_context['block_start_time']) * 1000,
            'component_timings': dict(self.current_block_timings),
            'avg_time_per_frame_ms': (end_time - self.current_context['block_start_time']) * 1000 / self.num_frame_per_block
        }
        
        self.block_summaries.append(block_summary)
        
    @contextlib.contextmanager
    def time_component(self, component_name: str, **metadata):
        """Context manager for timing individual components"""
        # Create CUDA events for precise GPU timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # CPU timing as backup
        cpu_start = time.perf_counter()
        
        # Record GPU start
        start_event.record()
        
        try:
            yield
        finally:
            # Record GPU end and synchronize
            end_event.record()
            torch.cuda.synchronize()
            
            cpu_end = time.perf_counter()
            gpu_time_ms = start_event.elapsed_time(end_event)
            cpu_time_ms = (cpu_end - cpu_start) * 1000
            
            # Store timing data
            timing_data = {
                'component': component_name,
                'gpu_time_ms': gpu_time_ms,
                'cpu_time_ms': cpu_time_ms,
                'frame_idx': self.frame_idx,
                'block_idx': self.current_context.get('block_idx', -1),
                'timestamp': time.perf_counter(),
                **metadata
            }
            
            self.component_timings[component_name].append(timing_data)
            self.current_block_timings[component_name].append(timing_data)
    
    def record_frame_completion(self, frame_idx: int):
        """Record when a frame is completed"""
        torch.cuda.synchronize()
        current_time = time.perf_counter()
        
        self.frame_times.append({
            'frame_idx': frame_idx,
            'timestamp': current_time,
            'elapsed_since_start': current_time - self.generation_start_time
        })
        
        # Update frame counter
        self.frame_idx = frame_idx + 1
        
        # Calculate inter-frame latency after warmup
        if frame_idx > self.warmup_frames:
            inter_frame_latency = current_time - self.last_frame_time
            is_cross_block = self._is_cross_block_transition(frame_idx)
            
            # Store inter-frame timing
            self.component_timings['inter_frame_latency'].append({
                'frame_idx': frame_idx,
                'latency_ms': inter_frame_latency * 1000,
                'is_cross_block': is_cross_block,
                'transition_type': 'cross-block' if is_cross_block else 'intra-block'
            })
            
        self.last_frame_time = current_time
        
    def _is_cross_block_transition(self, frame_idx: int) -> bool:
        """Check if this is a cross-block transition"""
        if frame_idx == 0:
            return False
        prev_frame_idx = frame_idx - 1
        return (prev_frame_idx + 1) % self.num_frame_per_block == 0
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive timing statistics"""
        stats = {
            'configuration': {
                'num_frame_per_block': self.num_frame_per_block,
                'warmup_frames': self.warmup_frames,
                'total_frames': len(self.frame_times),
                'total_blocks': len(self.block_summaries)
            },
            'component_breakdown': {},
            'block_analysis': self.block_summaries,
            'cross_vs_intra_block': self._analyze_cross_vs_intra_block(),
            'performance_metrics': self._calculate_performance_metrics()
        }
        
        # Analyze each component
        for component_name, timings in self.component_timings.items():
            if timings:
                stats['component_breakdown'][component_name] = self._analyze_component(timings)
        
        return stats
    
    def _analyze_component(self, timings: List[Dict]) -> Dict[str, Any]:
        """Analyze timing data for a specific component"""
        if not timings:
            return {}
            
        # Extract GPU times (primary metric)
        gpu_times = [t.get('gpu_time_ms', t.get('latency_ms', 0)) for t in timings]
        cpu_times = [t.get('cpu_time_ms', 0) for t in timings]
        
        # Basic statistics
        analysis = {
            'count': len(gpu_times),
            'gpu_timing': {
                'mean_ms': np.mean(gpu_times),
                'median_ms': np.median(gpu_times),
                'std_ms': np.std(gpu_times),
                'min_ms': np.min(gpu_times),
                'max_ms': np.max(gpu_times),
                'p95_ms': np.percentile(gpu_times, 95),
                'p99_ms': np.percentile(gpu_times, 99),
                'total_ms': np.sum(gpu_times)
            }
        }
        
        if cpu_times and any(t > 0 for t in cpu_times):
            analysis['cpu_timing'] = {
                'mean_ms': np.mean(cpu_times),
                'total_ms': np.sum(cpu_times)
            }
            
        # Per-block analysis if applicable
        block_breakdown = defaultdict(list)
        for timing in timings:
            block_idx = timing.get('block_idx', -1)
            if block_idx >= 0:
                block_breakdown[block_idx].append(timing.get('gpu_time_ms', timing.get('latency_ms', 0)))
        
        if block_breakdown:
            analysis['per_block_variance'] = {
                f'block_{idx}': {
                    'mean_ms': np.mean(times),
                    'count': len(times)
                }
                for idx, times in block_breakdown.items()
            }
            
        return analysis
    
    def _analyze_cross_vs_intra_block(self) -> Dict[str, Any]:
        """Analyze cross-block vs intra-block latency"""
        inter_frame_data = self.component_timings.get('inter_frame_latency', [])
        if not inter_frame_data:
            return {}
            
        cross_block = [t['latency_ms'] for t in inter_frame_data if t.get('is_cross_block', False)]
        intra_block = [t['latency_ms'] for t in inter_frame_data if not t.get('is_cross_block', False)]
        
        analysis = {}
        
        if cross_block:
            analysis['cross_block_worst_case'] = {
                'count': len(cross_block),
                'mean_ms': np.mean(cross_block),
                'median_ms': np.median(cross_block),
                'max_ms': np.max(cross_block),
                'p95_ms': np.percentile(cross_block, 95)
            }
            
        if intra_block:
            analysis['intra_block_best_case'] = {
                'count': len(intra_block),
                'mean_ms': np.mean(intra_block),
                'median_ms': np.median(intra_block),
                'max_ms': np.max(intra_block),
                'p95_ms': np.percentile(intra_block, 95)
            }
            
        if cross_block and intra_block:
            analysis['comparison'] = {
                'latency_multiplier': np.mean(cross_block) / np.mean(intra_block),
                'additional_latency_ms': np.mean(cross_block) - np.mean(intra_block),
                'cross_block_percentage': len(cross_block) / (len(cross_block) + len(intra_block)) * 100
            }
            
        return analysis
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate overall performance metrics"""
        if not self.frame_times:
            return {}
            
        total_time_s = self.frame_times[-1]['elapsed_since_start']
        total_frames = len(self.frame_times)
        
        return {
            'total_generation_time_s': total_time_s,
            'average_fps': total_frames / total_time_s,
            'frames_per_second': total_frames / total_time_s,
            'ms_per_frame': (total_time_s * 1000) / total_frames
        }
    
    def save_comprehensive_log(self, output_path: str):
        """Save detailed comprehensive analysis"""
        data = {
            'frame_times': self.frame_times,
            'component_timings': dict(self.component_timings),
            'comprehensive_statistics': self.get_comprehensive_statistics(),
            'metadata': {
                'device': str(self.device),
                'timestamp': time.time(),
                'warmup_frames': self.warmup_frames,
                'num_frame_per_block': self.num_frame_per_block
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)


class TimingInstrumentationMixin:
    """Mixin to add timing instrumentation to pipeline components"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latency_tracker = None
        
    def set_latency_tracker(self, tracker: ComprehensiveLatencyTracker):
        """Set the latency tracker for this component"""
        self.latency_tracker = tracker
        
    def _time_method(self, method_name: str, component_name: str):
        """Decorator to time a method call"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if self.latency_tracker:
                    with self.latency_tracker.time_component(
                        component_name, 
                        method=method_name,
                        args_info=str(type(args[1:]))[:100] if len(args) > 1 else ""
                    ):
                        return func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            return wrapper
        return decorator


# Monkey patch key methods to add timing
def instrument_wan_components():
    """Add timing instrumentation to WAN components"""
    
    # Import classes here to avoid circular imports
    from utils.wan_wrapper import WanTextEncoder, WanVAEWrapper, WanDiffusionWrapper
    
    # Instrument text encoder
    original_text_forward = WanTextEncoder.forward
    def timed_text_forward(self, *args, **kwargs):
        if hasattr(self, 'latency_tracker') and self.latency_tracker:
            with self.latency_tracker.time_component('text_encoder', operation='forward'):
                return original_text_forward(self, *args, **kwargs)
        return original_text_forward(self, *args, **kwargs)
    WanTextEncoder.forward = timed_text_forward
    
    # Instrument VAE encode/decode
    original_vae_encode = WanVAEWrapper.encode_to_latent
    def timed_vae_encode(self, *args, **kwargs):
        if hasattr(self, 'latency_tracker') and self.latency_tracker:
            with self.latency_tracker.time_component('vae_encode', operation='encode_to_latent'):
                return original_vae_encode(self, *args, **kwargs)
        return original_vae_encode(self, *args, **kwargs)
    WanVAEWrapper.encode_to_latent = timed_vae_encode
    
    original_vae_decode = WanVAEWrapper.decode_to_pixel
    def timed_vae_decode(self, *args, **kwargs):
        if hasattr(self, 'latency_tracker') and self.latency_tracker:
            with self.latency_tracker.time_component('vae_decode', operation='decode_to_pixel', use_cache=kwargs.get('use_cache', False)):
                return original_vae_decode(self, *args, **kwargs)
        return original_vae_decode(self, *args, **kwargs)
    WanVAEWrapper.decode_to_pixel = timed_vae_decode
    
    # Instrument diffusion model forward
    original_diffusion_forward = WanDiffusionWrapper.forward
    def timed_diffusion_forward(self, *args, **kwargs):
        if hasattr(self, 'latency_tracker') and self.latency_tracker:
            timestep_info = kwargs.get('timestep', 'unknown')
            timestep_val = timestep_info[0].item() if torch.is_tensor(timestep_info) and timestep_info.numel() > 0 else str(timestep_info)
            with self.latency_tracker.time_component('diffusion_forward', 
                                                   operation='forward',
                                                   timestep=timestep_val):
                return original_diffusion_forward(self, *args, **kwargs)
        return original_diffusion_forward(self, *args, **kwargs)
    WanDiffusionWrapper.forward = timed_diffusion_forward


# Context managers for specific operations
@contextlib.contextmanager
def time_kv_operations(tracker: ComprehensiveLatencyTracker, operation: str, **metadata):
    """Time KV cache operations"""
    if tracker:
        with tracker.time_component('kv_operations', operation=operation, **metadata):
            yield
    else:
        yield

@contextlib.contextmanager  
def time_attention_kernel(tracker: ComprehensiveLatencyTracker, **metadata):
    """Time attention kernel operations"""
    if tracker:
        with tracker.time_component('attention_kernel', **metadata):
            yield
    else:
        yield

@contextlib.contextmanager
def time_device_sync(tracker: ComprehensiveLatencyTracker, sync_type: str = "cuda_sync"):
    """Time device synchronization operations"""
    if tracker:
        with tracker.time_component('device_sync', sync_type=sync_type):
            torch.cuda.synchronize()
            yield
    else:
        torch.cuda.synchronize() 
        yield

@contextlib.contextmanager
def time_quantization(tracker: ComprehensiveLatencyTracker, operation: str = "quant"):
    """Time quantization/dequantization operations"""
    if tracker:
        with tracker.time_component('quantization', operation=operation):
            yield
    else:
        yield