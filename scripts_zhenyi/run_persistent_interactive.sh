#!/bin/bash

# Persistent Interactive Pipeline Runner
# Keeps compiled models loaded to avoid recompilation

set -e

# Parse arguments
CONFIG_PATH="configs/longlive_interactive_inference.yaml"
MODE="interactive"
DATA_PATH=""
OUTPUT_DIR="/tmp/persistent_outputs"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --fp8)
            EXTRA_ARGS="$EXTRA_ARGS --fp8"
            shift
            ;;
        --no-fp8)
            EXTRA_ARGS="$EXTRA_ARGS --no-fp8"
            shift
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        help)
            MODE="help"
            shift
            ;;
        *)
            # Positional arguments (old style)
            if [[ -z "$CONFIG_SET" ]]; then
                CONFIG_PATH="$1"
                CONFIG_SET=1
            elif [[ -z "$MODE_SET" ]]; then
                MODE="$1"
                MODE_SET=1
            elif [[ -z "$DATA_SET" ]]; then
                DATA_PATH="$1"
                DATA_SET=1
            elif [[ -z "$OUTPUT_SET" ]]; then
                OUTPUT_DIR="$1"
                OUTPUT_SET=1
            fi
            shift
            ;;
    esac
done

echo "Persistent Interactive Pipeline"
echo "==============================="
echo "Config: $CONFIG_PATH"
echo "Mode: $MODE"
echo ""

if [ "$MODE" = "help" ]; then
    echo "Usage: $0 [options] [config_path] [mode] [data_path] [output_dir]"
    echo ""
    echo "Options:"
    echo "  --fp8                Enable FP8 quantization (overrides config)"
    echo "  --no-fp8             Disable FP8 quantization (overrides config)"
    echo "  --config PATH        Configuration file path"
    echo "  --mode MODE          Mode: interactive, data, help"
    echo "  --data-path PATH     Data file path (for data mode)"
    echo "  --output-dir DIR     Output directory"
    echo ""
    echo "Modes:"
    echo "  interactive - Enter interactive mode for multiple runs"
    echo "  data        - Run once on data file (requires data_path)"
    echo "  help        - Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                                          # Interactive mode with default config"
    echo "  $0 --fp8                                    # Interactive mode with FP8 quantization"
    echo "  $0 --config configs/my_config.yaml         # Custom config"
    echo "  $0 --mode data --data-path /path/to/data.txt # Run once on data file"
    echo ""
    echo "Benefits of persistent mode:"
    echo "  - First inference: ~90s (includes 30s compilation)" 
    echo "  - Subsequent runs: ~60s (no compilation overhead)"
    echo "  - FP8 quantization: Additional ~20% speedup (24.8 FPS)"
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
        echo "Usage: $0 --mode data --data-path /path/to/data.txt"
        exit 1
    fi
    echo "Running inference on data file: $DATA_PATH"
    echo "Output directory: $OUTPUT_DIR"
    python scripts_zhenyi/persistent_interactive_server.py \
        --config_path "$CONFIG_PATH" \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        $EXTRA_ARGS
else
    echo "Starting interactive mode..."
    python scripts_zhenyi/persistent_interactive_server.py \
        --config_path "$CONFIG_PATH" \
        --interactive \
        $EXTRA_ARGS
fi

echo ""
echo "Persistent pipeline session ended."
