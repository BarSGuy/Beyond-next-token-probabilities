#!/bin/bash

# Default base directory for raw data
DEFAULT_BASE_RAW_DATA_DIR="/home/guy_b/big-storage/raw_data"

# Use provided argument as base directory, otherwise use default
BASE_RAW_DATA_DIR=${1:-$DEFAULT_BASE_RAW_DATA_DIR}

# Define datasets
DATASETS=("WikiMIA_32" "WikiMIA_64" "BookMIA_128")

# Define models for each dataset
declare -A MODELS
MODELS[WikiMIA_32]="EleutherAI/pythia-6.9b EleutherAI/pythia-12b huggyllama/llama-13b huggyllama/llama-30b state-spaces/mamba-1.4b-hf"
MODELS[WikiMIA_64]="EleutherAI/pythia-6.9b EleutherAI/pythia-12b huggyllama/llama-13b huggyllama/llama-30b state-spaces/mamba-1.4b-hf"
MODELS[BookMIA_128]="EleutherAI/pythia-6.9b EleutherAI/pythia-12b huggyllama/llama-13b huggyllama/llama-30b"



# Loop through datasets and models
for DATASET in "${DATASETS[@]}"; do
  for MODEL in ${MODELS[$DATASET]}; do
    echo "Creating dataset for $DATASET with model $MODEL..."
    python create_DC_datasets.py \
      --dataset "$DATASET" \
      --LLM "$MODEL" \
      --base_raw_data_dir "$BASE_RAW_DATA_DIR"
  done
done

echo "All datasets have been created successfully."
