#!/bin/bash

# Default base directory for raw data
DEFAULT_BASE_RAW_DATA_DIR="/home/guy_b/big-storage/raw_data"
# Default base directory for pre-processed data
DEFAULT_BASE_PRE_PROCESSED_DATA_DIR="/home/guy_b/LOS-Net/pre_processed_data"

# Use provided arguments as base directories, otherwise use defaults
BASE_RAW_DATA_DIR=${1:-$DEFAULT_BASE_RAW_DATA_DIR}
BASE_PRE_PROCESSED_DATA_DIR=${2:-$DEFAULT_BASE_PRE_PROCESSED_DATA_DIR}

# Allow specifying the number of parallel chunks (default to 8)
MAX_PARALLEL_JOBS=${3:-8}

# Define datasets
DATASETS=("WikiMIA_32" "WikiMIA_64" "BookMIA_128" "imdb" "imdb_test" "movies" "movies_test" "hotpotqa" "hotpotqa_test")

# Define models for each dataset
declare -A MODELS
MODELS[WikiMIA_32]="EleutherAI/pythia-6.9b EleutherAI/pythia-12b huggyllama/llama-13b huggyllama/llama-30b state-spaces/mamba-1.4b-hf"
MODELS[WikiMIA_64]="EleutherAI/pythia-6.9b EleutherAI/pythia-12b huggyllama/llama-13b huggyllama/llama-30b state-spaces/mamba-1.4b-hf"
MODELS[BookMIA_128]="EleutherAI/pythia-6.9b EleutherAI/pythia-12b huggyllama/llama-13b huggyllama/llama-30b"

MODELS[imdb]="mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3-8B-Instruct"
MODELS[imdb_test]="mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3-8B-Instruct"
MODELS[movies]="mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3-8B-Instruct"
MODELS[movies_test]="mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3-8B-Instruct"
MODELS[hotpotqa]="mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3-8B-Instruct"
MODELS[hotpotqa_test]="mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3-8B-Instruct"

# Define input_output_types
INPUT_OUTPUT_TYPES=("input" "output")

# Track running jobs
RUNNING_JOBS=0

# Log file
LOG_FILE="Datasets_preprocess_ACT.log"

## TODO: Delete
# declare -A MODELS
MODELS[BookMIA_128]="EleutherAI/pythia-6.9b EleutherAI/pythia-12b huggyllama/llama-13b huggyllama/llama-30b"

DATASETS=("BookMIA_128")
INPUT_OUTPUT_TYPES=("input")
## TODO: ^

echo "Starting dataset preprocessing process..." | tee "$LOG_FILE"
echo "---------------------------------------------" | tee -a "$LOG_FILE"
echo "Datasets: ${DATASETS[*]}" | tee -a "$LOG_FILE"
echo "Models: ${MODELS[*]}" | tee -a "$LOG_FILE"
echo "Types: ${INPUT_OUTPUT_TYPES[*]}" | tee -a "$LOG_FILE"
echo "---------------------------------------------" | tee -a "$LOG_FILE"


# Loop through datasets and models
for DATASET in "${DATASETS[@]}"; do
  for MODEL in ${MODELS[$DATASET]}; do
    for TYPE in "${INPUT_OUTPUT_TYPES[@]}"; do
    printf "Running preprocessing for dataset %s with model and type %s...\n" "$DATASET" "$MODEL" "$TYPE" | tee -a "$LOG_FILE"
      python preprocess_datasets.py \
        --LLM "$MODEL" \
        --dataset "$DATASET" \
        --base_raw_data_dir "$BASE_RAW_DATA_DIR" \
        --base_pre_processed_data_dir "$BASE_PRE_PROCESSED_DATA_DIR" \
        --input_output_type "$TYPE" \
        --N_max 100 \
        --input_type "activations" \
        --L_max 50 2>&1 | tee -a "$LOG_FILE" &

      ((RUNNING_JOBS++))

      # If the number of jobs reaches the limit, wait for the first one to finish
      if ((RUNNING_JOBS >= MAX_PARALLEL_JOBS)); then
        wait -n  # Waits for ANY one job to finish
        ((RUNNING_JOBS--))  # Reduce the running jobs counter
        printf "Finished preprocessing for dataset %s with model and type %s...\n" "$DATASET" "$MODEL" "$TYPE" | tee -a "$LOG_FILE"
      fi
    done
  done
done

# Ensure any remaining processes finish
wait

echo "All preprocessing tasks have been completed successfully."
