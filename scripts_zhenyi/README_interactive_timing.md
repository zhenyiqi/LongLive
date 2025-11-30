# Interactive Causal Inference Latency Analysis

This directory contains tools for comprehensive latency analysis of the interactive causal inference pipeline, which supports multiple prompt segments with dynamic switching during video generation.

## Overview

The interactive causal inference pipeline differs from the standard pipeline in that it supports:
- **Multiple prompt segments**: Different text prompts for different parts of the video
- **Dynamic prompt switching**: Switching prompts at specified frame indices during generation
- **Cache management**: Sophisticated KV cache handling during prompt transitions

## Key Components

### 1. Enhanced Pipeline (`pipeline/interactive_causal_inference.py`)

The enhanced `InteractiveCausalInferencePipeline` includes:
- Comprehensive timing instrumentation throughout the inference process
- Detailed logging of prompt switching operations
- Component-level latency breakdown for all major operations
- Cache management timing analysis

**Key timing features:**
- **Prompt switching analysis**: Measures cache reset, recaching setup, and forward pass times
- **Segment-aware timing**: Tracks which prompt segment is being used for each operation
- **Frame-level granularity**: Records completion time for individual frames
- **Cross-attention cache management**: Times cache reset and update operations

### 2. Timing Analysis Script (`interactive_inference_with_timing.py`)

Enhanced version of the standard interactive inference script that adds:
- Comprehensive latency tracking integration
- Detailed timing output and logging
- JSON export of timing data for analysis
- Real-time performance feedback

**Usage:**
```bash
python scripts_zhenyi/interactive_inference_with_timing.py \
    --config_path configs/your_config.yaml \
    --enable_timing \
    --timing_output_dir /tmp/interactive_timing_results
```

### 3. Runner Script (`run_interactive_timing_analysis.sh`)

Convenient wrapper script that:
- Sets up the environment
- Runs the timing analysis
- Provides guidance on analyzing results

**Usage:**
```bash
./scripts_zhenyi/run_interactive_timing_analysis.sh [config_path] [output_dir]
```

## Interactive vs Standard Pipeline Differences

### Prompt Switching Operations

The interactive pipeline introduces unique operations not present in standard inference:

1. **Cache Reset on Switch**
   - KV cache clearing for attention mechanisms
   - Cross-attention cache reset
   - Memory cleanup operations

2. **Recaching Operations**
   - Reprocessing previous frames with new prompt conditioning
   - Blockwise causal mask preparation
   - Forward pass through recached frames

3. **Multi-segment Text Encoding**
   - Encoding multiple prompt segments upfront
   - Managing different conditional dictionaries
   - Switching between prompts during generation

### Timing Categories

The interactive analysis includes these specialized timing categories:

#### Prompt-Specific Operations
- `interactive_text_encoding`: Encoding each prompt segment
- `prompt_switch_*`: Various prompt switching operations
- `prompt_switch_total`: Complete prompt switch timing

#### Frame Generation
- `interactive_denoising_step`: Individual denoising iterations
- `interactive_denoising_step_final`: Final denoising step
- `interactive_vae_decode`: VAE decoding with frame count info

#### Cache Operations
- `prompt_switch_kv_reset`: KV cache reset during switches
- `prompt_switch_crossattn_reset`: Cross-attention cache reset
- `prompt_switch_recache`: Recaching previous frames
- `interactive_kv_cache_update`: Regular KV cache updates

#### Device Operations
- `interactive_vae_input_prep`: VAE input preparation
- `interactive_vae_postprocess`: VAE output processing
- `interactive_block_input_prep`: Block input preparation

## Analysis and Visualization

### 1. Latency Breakdown Analysis

```bash
python scripts_zhenyi/analyze_latency_breakdown.py \
    /tmp/interactive_timing_results/interactive_timing_analysis_rank0_prompt0.json \
    --detailed
```

This will show:
- Time fraction breakdown by component
- Prompt switching overhead analysis
- Frame generation efficiency metrics
- Component hierarchy analysis (avoiding double-counting)

### 2. Visualization

```bash
python scripts_zhenyi/visualize_comprehensive_latency.py \
    --input_dir /tmp/interactive_timing_results \
    --output_dir /tmp/interactive_timing_results/visualizations
```

Generates:
- Timeline plots showing prompt switches
- Component breakdown charts
- Frame completion timing
- Comparative analysis across segments

## Key Metrics to Monitor

### 1. Prompt Switching Overhead
- **Switch timing**: How long each prompt switch takes
- **Recache performance**: Time to reprocess previous frames
- **Cache reset efficiency**: Speed of memory cleanup operations

### 2. Cross-Segment Comparison
- **Per-segment generation time**: Performance differences between prompt segments
- **Transition costs**: Additional overhead when switching prompts
- **Frame generation consistency**: Whether switching affects frame generation speed

### 3. Memory and Cache Performance
- **Cache hit rates**: How effectively the KV cache is utilized
- **Memory transfer times**: GPU ↔ CPU data movement during switches
- **Cache reset efficiency**: Speed of clearing and reinitializing caches

## Example Output

When running with timing enabled, you'll see logs like:

```
[Interactive Timing] Starting interactive inference with 3 segments: 152.30 ms
[Interactive Timing] Switch points: [30, 60]
[Interactive Timing] All prompts encoded: 1205.45 ms
[Interactive] Starting prompt switch to segment 1 at frame 30
[Interactive Timing] Cache reset completed: 12.34 ms
[Interactive Timing] Recaching 12 frames (from 18 to 30)
[Interactive Timing] Recache setup completed: 8.92 ms
[Interactive Timing] Recache forward pass completed: 156.78 ms
[Interactive Timing] Prompt switch to segment 1 completed: 189.23 ms
...
```

## Performance Optimization Tips

Based on timing analysis, you can:

1. **Optimize prompt switching**: Reduce recaching overhead by adjusting local attention size
2. **Balance segments**: Ensure prompt segments are roughly equal in complexity
3. **Cache tuning**: Adjust KV cache settings based on memory vs. performance trade-offs
4. **Switch point selection**: Choose switch points that minimize recaching overhead

## Integration with Existing Tools

The interactive timing analysis is fully compatible with:
- Existing latency analysis tools (`comprehensive_latency_tracker.py`)
- Visualization scripts (`visualize_comprehensive_latency.py`)
- Analysis scripts (`analyze_latency_breakdown.py`)

All timing data is saved in the same JSON format for consistent analysis across pipeline types.