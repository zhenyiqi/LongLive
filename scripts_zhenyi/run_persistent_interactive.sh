#!/bin/bash

# Persistent Interactive Pipeline Runner
# Keeps compiled models loaded to avoid recompilation

set -e

CONFIG_PATH=${1:-"configs/longlive_interactive_inference.yaml"}
MODE=${2:-"interactive"}
DATA_PATH=${3:-""}
OUTPUT_DIR=${4:-"/tmp/persistent_outputs"}

echo "Persistent Interactive Pipeline"
echo "==============================="
echo "Config: $CONFIG_PATH"
echo "Mode: $MODE"
echo ""

if [ "$MODE" = "help" ]; then
    echo "Usage: $0 [config_path] [mode] [data_path] [output_dir]"
    echo ""
    echo "Modes:"
    echo "  interactive - Enter interactive mode for multiple runs"
    echo "  data        - Run once on data file (requires data_path)"
    echo "  help        - Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                                          # Interactive mode with default config"
    echo "  $0 configs/my_config.yaml interactive      # Interactive mode with custom config"
    echo "  $0 configs/my_config.yaml data /path/to/data.txt  # Run once on data file"
    echo ""
    echo "Benefits of persistent mode:"
    echo "  - First inference: ~90s (includes 30s compilation)" 
    echo "  - Subsequent runs: ~60s (no compilation overhead)"
    echo "  - 33% time savings for multiple runs!"
    exit 0
fi

echo "This persistent pipeline will:"
echo "1. Load and compile text encoder + VAE once (~10 seconds setup)"
echo "2. First inference includes compilation (~90 seconds total)"
echo "3. Subsequent inferences skip compilation (~60 seconds each)"
echo "4. Keep the Python process running to preserve compiled graphs"
echo ""

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Run the persistent server
cd "$PROJECT_ROOT"

if [ "$MODE" = "data" ]; then
    if [ -z "$DATA_PATH" ]; then
        echo "Error: data mode requires data_path"
        echo "Usage: $0 $CONFIG_PATH data /path/to/data.txt [output_dir]"
        exit 1
    fi
    echo "Running inference on data file: $DATA_PATH"
    echo "Output directory: $OUTPUT_DIR"
    python scripts_zhenyi/persistent_interactive_server.py \
        --config_path "$CONFIG_PATH" \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR"
else
    echo "Starting interactive mode..."
    python scripts_zhenyi/persistent_interactive_server.py \
        --config_path "$CONFIG_PATH" \
        --interactive
fi

echo ""
echo "Persistent pipeline session ended."
