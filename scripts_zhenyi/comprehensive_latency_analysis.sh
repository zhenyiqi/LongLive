#!/bin/bash

# Script to run comprehensive latency analysis with detailed component breakdown
# Measures: model steps, attention kernels, KV operations, device sync, VAE, etc.
# Usage: ./comprehensive_latency_analysis.sh [config_path] [output_dir]

CONFIG_PATH=${1:-"configs/longlive_inference.yaml"}
OUTPUT_DIR=${2:-"./comprehensive_latency_results"}

echo "Running comprehensive component-level latency analysis..."
echo "Components measured:"
echo "  - Model steps / denoising iterations"
echo "  - Attention kernels"  
echo "  - KV operations (frame-sink concat / recache)"
echo "  - Quant/dequant operations"
echo "  - VAE encode/decode"
echo "  - Device ↔ Host synchronization"
echo ""
echo "Config: $CONFIG_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "================================"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the comprehensive latency analysis
torchrun \
  --nproc_per_node=1 \
  --master_port=29502 \
  scripts_zhenyi/comprehensive_latency_analysis.py \
  --config_path "$CONFIG_PATH" \
  --output_dir "$OUTPUT_DIR"

echo "================================"
echo "Comprehensive latency analysis complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Analysis includes:"
echo "- Component-level timing breakdown"
echo "- Cross-block vs intra-block detailed analysis"  
echo "- Per-component statistics (mean, P95, max)"
echo "- Block-level variance analysis"
echo "- JSON logs with frame-by-frame and component timing"
echo ""
echo "To visualize results, run:"
echo "python scripts_zhenyi/visualize_comprehensive_latency.py --input_dir $OUTPUT_DIR"