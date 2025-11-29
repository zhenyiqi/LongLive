#!/bin/bash

# Script to run enhanced inference with worst-case latency analysis
# Focuses on cross-block vs intra-block frame transitions
# Usage: ./enhanced_latency_analysis.sh [config_path] [output_dir]

CONFIG_PATH=${1:-"configs/longlive_inference.yaml"}
OUTPUT_DIR=${2:-"./enhanced_latency_analysis_results"}

echo "Running enhanced latency analysis..."
echo "Focus: Worst-case (cross-block) vs best-case (intra-block) transitions"
echo "Config: $CONFIG_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "================================"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the enhanced latency analysis
torchrun \
  --nproc_per_node=1 \
  --master_port=29501 \
  scripts_zhenyi/enhanced_latency_analysis.py \
  --config_path "$CONFIG_PATH" \
  --output_dir "$OUTPUT_DIR"

echo "================================"
echo "Enhanced latency analysis complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Analysis includes:"
echo "- Cross-block transitions (worst case)"
echo "- Intra-block transitions (best case)"  
echo "- Latency multiplier between worst/best case"
echo "- Detailed JSON logs with frame-by-frame timing"
echo ""
echo "To visualize results, run:"
echo "python scripts_zhenyi/visualize_enhanced_latency.py --input_dir $OUTPUT_DIR"