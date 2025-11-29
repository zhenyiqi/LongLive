#!/bin/bash

# Script to run inference with inter-frame latency analysis
# Usage: ./inference_latency_analysis.sh [config_path] [output_dir]

CONFIG_PATH=${1:-"configs/longlive_inference.yaml"}
OUTPUT_DIR=${2:-"./latency_analysis_results"}

echo "Running inference with latency analysis..."
echo "Config: $CONFIG_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "================================"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the latency analysis
torchrun \
  --nproc_per_node=1 \
  --master_port=29500 \
  scripts_zhenyi/inference_latency_analysis.py \
  --config_path "$CONFIG_PATH" \
  --output_dir "$OUTPUT_DIR"

echo "================================"
echo "Latency analysis complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Files generated:"
echo "- JSON files with detailed timing data"
echo "- Video files with '_latency' suffix"
echo ""
echo "To visualize results, run:"
echo "python scripts_zhenyi/visualize_latency.py --input_dir $OUTPUT_DIR"