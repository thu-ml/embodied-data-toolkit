#!/bin/bash

# Configuration
SRC_ROOT="/path/to/src/dir"
DEST_ROOT="/path/to/dest/dir"
CONFIG_FILE="/path/to/config.json"
WORKERS=16

# Ensure script is run from its directory or handle paths relative to it
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CD_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Running Data Format Conversion"
echo "=========================================="
echo "Source: $SRC_ROOT"
echo "Dest:   $DEST_ROOT"
echo "Config: $CONFIG_FILE"
echo "Workers: $WORKERS"
echo "------------------------------------------"

# Run conversion
cd "$CD_DIR"
python unified_data_converter/run_conversion.py \
    --config "$CONFIG_FILE" \
    --src_root "$SRC_ROOT" \
    --dest_root "$DEST_ROOT" \
    --workers "$WORKERS" \
    --no-resume

echo "=========================================="
echo "Conversion Completed"
echo "=========================================="

