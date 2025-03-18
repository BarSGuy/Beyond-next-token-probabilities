#!/bin/bash

# Default base directory for raw data
DEFAULT_BASE_RAW_DATA_DIR="/home/guy_b/big-storage/raw_data"

# Use provided argument as base directory, otherwise use default
BASE_RAW_DATA_DIR=${1:-$DEFAULT_BASE_RAW_DATA_DIR}

# Allow specifying the number of parallel chunks (default to 8)
MAX_PARALLEL_JOBS=${2:-8}

# Define datasets
DATASETS=("imdb" "imdb_test" "movies" "movies_test" "hotpotqa" "hotpotqa_test")

# Define models for each dataset
MODELS=("mistralai/Mistral-7B-Instruct-v0.2" "meta-llama/Meta-Llama-3-8B-Instruct")

# Define chunks
CHUNKS=(1 2 3 4 5 6 7 8 9 10)

# Track running jobs
RUNNING_JOBS=0


# Log file
LOG_FILE="Raw_HD_datasets_creation.log"

echo "Starting dataset creation process..." | tee "$LOG_FILE"
echo "---------------------------------------------" | tee -a "$LOG_FILE"
echo "Datasets: ${DATASETS[*]}" | tee -a "$LOG_FILE"
echo "Models: ${MODELS[*]}" | tee -a "$LOG_FILE"
echo "Chunks: ${CHUNKS[*]}" | tee -a "$LOG_FILE"
echo "---------------------------------------------" | tee -a "$LOG_FILE"



# Loop through datasets and models
for DATASET in "${DATASETS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for CHUNK in "${CHUNKS[@]}"; do
      printf "Running chunk %d of dataset %s with model %s...\n" "$CHUNK" "$DATASET" "$MODEL" | tee -a "$LOG_FILE"

      # Run the process in the background and log both stdout and stderr
      python create_HD_datasets.py \
        --dataset "$DATASET" \
        --LLM "$MODEL" \
        --chunk "$CHUNK" \
        --base_raw_data_dir "$BASE_RAW_DATA_DIR" \
        --n_samples 10000 2>&1 | tee -a "$LOG_FILE" &

      ((RUNNING_JOBS++))

      # If the number of jobs reaches the limit, wait for the first one to finish
      if ((RUNNING_JOBS >= MAX_PARALLEL_JOBS)); then
        wait -n  # Waits for ANY one job to finish
        ((RUNNING_JOBS--))  # Reduce the running jobs counter
        printf "Chunk %d of dataset %s with model %s has finished.\n" "$CHUNK" "$DATASET" "$MODEL" | tee -a "$LOG_FILE"
      fi
    done
  done
done

# Ensure any remaining processes finish
wait

echo "All datasets have been created successfully." | tee -a "$LOG_FILE"
