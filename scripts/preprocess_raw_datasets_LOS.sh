#!/bin/bash

# ./scripts/preprocess_raw_datasets_LOS.sh /mnt/storage/guy_b/LOS_NET /home/guy_b/LOS-Net-cleaned/LLM-Output-Signatures-Network/pre_processed_data 4

# Default base directory for raw data
DEFAULT_BASE_RAW_DATA_DIR="/mnt/storage/guy_b/LOS_NET_with_raw_logits"
# Default base directory for pre-processed data
DEFAULT_BASE_PRE_PROCESSED_DATA_DIR="/home/guy_b/LOS-Net-cleaned/LLM-Output-Signatures-Network/pre_processed_data"

# Use provided arguments as base directories, otherwise use defaults
BASE_RAW_DATA_DIR=${1:-$DEFAULT_BASE_RAW_DATA_DIR}
BASE_PRE_PROCESSED_DATA_DIR=${2:-$DEFAULT_BASE_PRE_PROCESSED_DATA_DIR}

# Allow specifying the number of parallel chunks (default to 8)
MAX_PARALLEL_JOBS=${3:-8}

# Define datasets
DATASETS=("WikiMIA_32" "WikiMIA_64" "BookMIA_128" "imdb" "imdb_test" "movies" "movies_test" "hotpotqa" "hotpotqa_test")

# Define models for each dataset
declare -A MODELS
MODELS[WikiMIA_32]="EleutherAI/pythia-6.9b huggyllama/llama-13b huggyllama/llama-30b state-spaces/mamba-1.4b-hf"
MODELS[WikiMIA_64]="EleutherAI/pythia-6.9b huggyllama/llama-13b huggyllama/llama-30b state-spaces/mamba-1.4b-hf"
MODELS[BookMIA_128]="EleutherAI/pythia-6.9b EleutherAI/pythia-12b huggyllama/llama-13b huggyllama/llama-30b"

MODELS[imdb]="mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3-8B-Instruct Qwen/Qwen2.5-7B-Instruct"
MODELS[imdb_test]="mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3-8B-Instruct Qwen/Qwen2.5-7B-Instruct"
MODELS[movies]="mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3-8B-Instruct Qwen/Qwen2.5-7B-Instruct"
MODELS[movies_test]="mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3-8B-Instruct Qwen/Qwen2.5-7B-Instruct"
MODELS[hotpotqa]="mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3-8B-Instruct Qwen/Qwen2.5-7B-Instruct"
MODELS[hotpotqa_test]="mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3-8B-Instruct Qwen/Qwen2.5-7B-Instruct"

declare -A INPUT_OUTPUT_TYPES
INPUT_OUTPUT_TYPES[WikiMIA_32]="input"
INPUT_OUTPUT_TYPES[WikiMIA_64]="input"
INPUT_OUTPUT_TYPES[BookMIA_128]="input"

INPUT_OUTPUT_TYPES[imdb]="output"
INPUT_OUTPUT_TYPES[imdb_test]="output"
INPUT_OUTPUT_TYPES[movies]="output"
INPUT_OUTPUT_TYPES[movies_test]="output"
INPUT_OUTPUT_TYPES[hotpotqa]="output"
INPUT_OUTPUT_TYPES[hotpotqa_test]="output"

declare -A TOPK_PREPROCESS  # Explicitly declare an associative array

TOPK_PREPROCESS[WikiMIA_32]=1_000_000
TOPK_PREPROCESS[WikiMIA_64]=1_000_000
TOPK_PREPROCESS[BookMIA_128]=1_000

TOPK_PREPROCESS[imdb]=1_000
TOPK_PREPROCESS[imdb_test]=1_000
TOPK_PREPROCESS[movies]=1_000
TOPK_PREPROCESS[movies_test]=1_000
TOPK_PREPROCESS[hotpotqa]=1_000
TOPK_PREPROCESS[hotpotqa_test]=1_000


# Track running jobs
RUNNING_JOBS=0

# Log file
LOG_FILE="Datasets_preprocess_LOS.log"

echo "Starting dataset preprocessing process..." | tee "$LOG_FILE"
echo "---------------------------------------------" | tee -a "$LOG_FILE"
echo "Datasets: ${DATASETS[*]}" | tee -a "$LOG_FILE"
echo "Models: ${MODELS[*]}" | tee -a "$LOG_FILE"
echo "---------------------------------------------" | tee -a "$LOG_FILE"


# Loop through datasets and models
for DATASET in "${DATASETS[@]}"; do
  for MODEL in ${MODELS[$DATASET]}; do

    printf "Running preprocessing for dataset %s with model and type %s...\n" "$DATASET" "$MODEL" "$TYPE" | tee -a "$LOG_FILE"
    

    python preprocess_datasets.py \
      --LLM "$MODEL" \
      --dataset "$DATASET" \
      --base_raw_data_dir "$BASE_RAW_DATA_DIR" \
      --base_pre_processed_data_dir "$BASE_PRE_PROCESSED_DATA_DIR" \
      --topk_preprocess "${TOPK_PREPROCESS[$DATASET]}" \
      --input_output_type "${INPUT_OUTPUT_TYPES[$DATASET]}" \
      --N_max 100 \
      --input_type "LOS" 2>&1 | tee -a "$LOG_FILE" &
    
    ((RUNNING_JOBS++))

    # If the number of jobs reaches the limit, wait for the first one to finish
    if ((RUNNING_JOBS >= MAX_PARALLEL_JOBS)); then
      wait -n  # Waits for ANY one job to finish
      ((RUNNING_JOBS--))  # Reduce the running jobs counter
      printf "Finished preprocessing for dataset %s with model %s...\n" "$DATASET" "$MODEL" | tee -a "$LOG_FILE"
    fi

  done
done

# Ensure any remaining processes finish
wait

echo "All preprocessing tasks have been completed successfully." | tee -a "$LOG_FILE"
