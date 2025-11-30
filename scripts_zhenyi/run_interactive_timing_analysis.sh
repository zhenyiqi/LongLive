#!/bin/bash

# Interactive Inference Timing Analysis Runner
# This script runs the interactive inference with comprehensive latency analysis
#   ./scripts_zhenyi/run_interactive_timing_analysis.sh configs/your_config.yaml /tmp/results

set -e

# Default config (can be overridden)
CONFIG_PATH=${1:-"configs/longlive_interactive_inference.yaml"}
OUTPUT_DIR=${2:-"/tmp/interactive_timing_results"}

echo "Interactive Inference Timing Analysis"
echo "====================================="
echo "Config: $CONFIG_PATH"
echo "Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Run the timing analysis
echo "Running interactive inference with comprehensive timing..."
cd "$PROJECT_ROOT"

python scripts_zhenyi/interactive_inference_with_timing.py \
    --config_path "$CONFIG_PATH" \
    --enable_timing \
    --timing_output_dir "$OUTPUT_DIR"

echo ""
echo "Analysis complete! Results saved to: $OUTPUT_DIR"
echo ""
echo "To analyze the timing results, run:"
echo "  python scripts_zhenyi/analyze_latency_breakdown.py $OUTPUT_DIR/*.json --detailed"
echo ""
echo "To visualize the results, run:"
echo "  python scripts_zhenyi/visualize_comprehensive_latency.py --input_dir $OUTPUT_DIR --output_dir $OUTPUT_DIR/visualizations"
