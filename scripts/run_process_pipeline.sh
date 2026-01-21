#!/bin/bash

WORK_DIR=$(dirname $(realpath $0))
PROJECT_ROOT=$(dirname $WORK_DIR)

cd $PROJECT_ROOT


# Conda 
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate embodied-data-toolkit

export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Starting Data Process Pipeline (Config Driven)..."

python process_pipeline/process_pipeline.py \
    --config process_pipeline/configs/config.yaml
