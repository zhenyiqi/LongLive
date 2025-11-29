# Inter-Frame Latency Analysis Scripts

This directory contains scripts for analyzing "steady-state inter-frame latency" - the gap between consecutive frames during continuous video generation.

## Files Overview

### Core Scripts
- `inference_latency_analysis.py` - Modified inference script with frame timing instrumentation
- `inference_latency_analysis.sh` - Bash script to run latency analysis
- `visualize_latency.py` - Visualization and analysis tool for latency data
- `count_video_frames.py` - Utility to count frames in generated videos

### Analysis Features

The latency analysis measures:
- **Inter-frame latency**: Time gap between consecutive frame completions
- **Steady-state analysis**: Excludes warmup frames (first 5 frames) 
- **Statistical metrics**: Mean, median, std dev, percentiles
- **Target FPS analysis**: Percentage of frames meeting 16/24/30 FPS targets

## Usage

### 1. Run Latency Analysis
```bash
# Basic usage with default config
./scripts_zhenyi/inference_latency_analysis.sh

# With custom config and output directory
./scripts_zhenyi/inference_latency_analysis.sh configs/your_config.yaml ./custom_output_dir
```

### 2. Visualize Results
```bash
# Generate plots and summary report
python scripts_zhenyi/visualize_latency.py --input_dir ./latency_analysis_results
```

### 3. Count Frames in Generated Videos
```bash
python scripts_zhenyi/count_video_frames.py /path/to/video.mp4
```

## Output Files

### Latency Data
- `latency_analysis_rank{X}_prompt{Y}.json` - Detailed timing data per run
  - Frame-by-frame timestamps
  - Inter-frame latencies
  - Statistical summary

### Visualizations
- `inter_frame_latency_timeline.png` - Latency over time
- `latency_distribution.png` - Histogram of latency values  
- `latency_boxplot.png` - Box plot comparison across runs
- `running_statistics.png` - Running mean/std deviation
- `latency_analysis_report.txt` - Text summary with key metrics

### Generated Videos
- Video files with `_latency` suffix for identification

## Key Metrics Explained

### Steady-State Inter-Frame Latency
- **Definition**: Time between consecutive frame completions after initial warmup
- **Measurement**: Excludes first 5 frames to avoid initialization overhead
- **Target**: Lower and more consistent latencies indicate better streaming performance

### Statistical Measures
- **Mean/Median**: Central tendency of frame timing
- **95th/99th percentile**: Worst-case latency behavior
- **Standard deviation**: Consistency of frame timing

### Target FPS Analysis
- **16 FPS target**: ≤62.5ms per frame
- **24 FPS target**: ≤41.67ms per frame  
- **30 FPS target**: ≤33.33ms per frame

## Important Notes

### Limitations of Current Implementation
1. **Simulated Timing**: The current script uses simulated frame timing for demonstration
2. **Real Implementation**: For actual measurements, you need to instrument the pipeline's internal frame generation loop
3. **Streaming vs Batch**: This analysis is designed for streaming generation scenarios

### For Real Measurements
To get actual frame-by-frame timing, you would need to:
1. Modify the `CausalInferencePipeline` class to emit timing events
2. Hook into the actual diffusion step completions
3. Measure latency in the autoregressive generation loop

### Memory Considerations
- The analysis stores timing data for all frames
- For very long videos, consider streaming the timing data to disk
- Monitor memory usage during analysis

## Example Output

```
INTER-FRAME LATENCY ANALYSIS
==================================================
Total frames analyzed: 115
Mean inter-frame latency: 15.23 ms
Median inter-frame latency: 14.87 ms
Std deviation: 3.45 ms
Min latency: 10.12 ms
Max latency: 28.95 ms
95th percentile: 21.34 ms
99th percentile: 25.67 ms
Average FPS: 65.71
Total generation time: 1.75 s
==================================================

TARGET FPS ANALYSIS
Frames meeting 16 FPS target: 115/115 (100.0%)
Frames meeting 24 FPS target: 115/115 (100.0%)  
Frames meeting 30 FPS target: 115/115 (100.0%)
```

This analysis helps identify performance bottlenecks and optimize streaming video generation pipelines.
