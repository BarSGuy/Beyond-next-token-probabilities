# 📦 LOS-Net Official Repository

This repository contains the official code of the paper
[**Beyond Next Token Probabilities: Learnable, Fast Detection of Hallucinations and Data Contamination on LLM Output Distributions**](https://arxiv.org/pdf/2503.14043) (*AAAI 2026*)


# Overview


We explore predicting problematic behaviors of LLMs, such as Hallucination Detection (HD) and Data Contamination (DC) detection, by learning from LLM Output Signatures (LOS), which are defined as the pair of TDS and ATP (see illustration below). We develop LOS-Net, a learnable gray-box model trained on LOS to predict these problematic behaviors.


<p align="center">
  <img src="./Figures/LOS.png" width="100%" height="50%">
</p>




⭐ If you find our implementation and paper helpful, please consider citing our work ⭐ :

```bibtex
@article{bar2025learning,
  title={Learning on LLM Output Signatures for Gray-Box Behavior Analysis},
  author={Bar-Shalom, Guy and Frasca, Fabrizio and Lim, Derek and Gelberg, Yoav and Ziser, Yftah and El-Yaniv, Ran and Chechik, Gal and Maron, Haggai},
  journal={arXiv preprint arXiv:2503.14043},
  year={2025}
}
```

Below we present the instructions to reproduce all the experiments we conducted in the paper.

## Table of Contents

- [Installation](#installation)
- [Handeling Datasets](#handeling-datasets)
    - [Generating Raw Datasets (DC)](#generating-raw-datasets-dc)
    - [Generating Raw Datasets (HD)](#generating-raw-datasets-hd)
  - [Preprocess Raw Datasets](#preprocess-raw-datasets)
- [Reproducibility](#reproducibility)
  - [Standard Experiments](#standard-experiemnts)
  - [Transferability Experiments](#transferability-experiments)
    - [Zero-Shot Generalization on BookMIA](#zero-shot-generalization-on-bookmia)
    - [Generalization Across Datasets on Hallucinations Datasets](#generalization-across-datasets-on-hallucinations-datasets)
    - [Plot heatmaps](#plot-heatmaps)


# Installation

First create a conda environment
```
conda env create -f los_net_env.yml
```
and activate it
```
conda activate los_net_env
```

# Handeling Datasets
### Generating Raw Datasets (DC)

#### Available Datasets & Supported Models

| Dataset      | Supported Models |
|-------------|-----------------|
| **WikiMIA_32 / WikiMIA_64** | `EleutherAI/pythia-6.9b`<br>`huggyllama/llama-13b`<br>`huggyllama/llama-30b`<br>`state-spaces/mamba-1.4b-hf` |
| **BookMIA**   | `EleutherAI/pythia-6.9b`<br>`EleutherAI/pythia-12b`<br>`huggyllama/llama-13b`<br>`huggyllama/llama-30b` |

---

#### Dataset Construction Instructions:
To construct a dataset for a specific model, execute the following command, replacing placeholders with the desired dataset and model:

```bash
python create_DC_datasets.py \
  --dataset <DATASET_NAME> \
  --LLM <MODEL_NAME> \
  --base_raw_data_dir <BASE_RAW_DATA_DIRECTORY>
```

- **Example:** 
  ```bash
  python create_DC_datasets.py \
    --dataset BookMIA_128 \
    --LLM huggyllama/llama-13b \
    --base_raw_data_dir /home/guy_b/big-storage/raw_data
  ```

To automatically generate all the raw datasets for all the models and datasets, run the following command:

1. **Make the script executable:**
   ```bash
   chmod +x ./scripts/generate_DC_raw_datasets.sh
   ```

2. **Run the script:**
   ```bash
   ./scripts/generate_DC_raw_datasets.sh [BASE_RAW_DATA_DIR]
   ```


Example run:
  ```bash
  ./scripts/generate_DC_raw_datasets.sh /home/guy_b/big-storage/raw_data
  ```

You can monitor the progress of this execution through the log file: Raw_DC_datasets_creation.log.

## HD
### Generating Raw Datasets (HD)
#### Available Datasets & Supported Models
Supported datasets are:
- `imdb`,`imdb_test` 
- `movies`, `movies_test`
- `hotpotqa`, `hotpotqa_test`

Supported models for all datasets:
- `mistralai/Mistral-7B-Instruct-v0.2`
- `meta-llama/Meta-Llama-3-8B-Instruct`

#### Dataset Construction Instructions:
To construct a dataset for a specific model, execute the following command, replacing placeholders with the desired dataset and model:

```bash
python create_HD_datasets.py \
  --dataset <DATASET_NAME> \
  --LLM <MODEL_NAME> \
  --base_raw_data_dir <BASE_RAW_DATA_DIRECTORY> \
  --n_samples <NUMBER_OF_SAMPLES_TO_USE_FROM_DATASET (default is 10_000 which effectivly is everything)> \
  --chunk <CHUNK_ID> 
```
**Note:** We split the generation to 10 chunks (for efficiency reasons), indexed from 1 to 10, where the i-th chunk stand for samples i000 -> (i+1)000.

- **Example:** 
  ```bash
  python create_HD_datasets.py \
    --dataset imdb \
    --LLM mistralai/Mistral-7B-Instruct-v0.2 \
    --base_raw_data_dir /home/guy_b/big-storage/raw_data \
    --n_samples 10_000 \
    --chunk 1
  ```

To automatically generate all the raw (full) datasets for all the models, datasets and chunks, run the following command:

1. **Make the script executable:**
   ```bash
   chmod +x ./scripts/generate_HD_raw_datasets.sh
   ```

2. **Run the script:**
   ```bash
   ./scripts/generate_HD_raw_datasets.sh [BASE_RAW_DATA_DIR] [NUM_PARALLEL_JOBS]
   ```


Example run:
  ```bash
  ./scripts/generate_HD_raw_datasets.sh /home/guy_b/big-storage/raw_data 5
  ```

You can monitor the progress of this execution through the log file: Raw_HD_datasets_creation.log.

---

## Preprocess Raw Datasets

To preprocess a raw dataset for a specific model (to create the proper datatype for LOS-Net), use the following commands:

```bash
python preprocess_datasets.py \
  --LLM <MODEL_NAME> \
  --dataset <DATASET_NAME> \
  --base_raw_data_dir <BASE_RAW_DATA_DIRECTORY> \
  --topk_preprocess <TOP_K> \
  --base_pre_processed_data_dir <BASE_PRE_PROCESSED_DATA_DIRECTORY> \
  --input_output_type <input/output> \
  --N_max <MAX_SEQUENCE_LENGTH> \
  --input_type LOS
```
- **Example:** 
  ```bash
  python preprocess_datasets.py \
    --LLM EleutherAI/pythia-6.9b \
    --dataset WikiMIA_32 \
    --topk_preprocess 1_000_000 \
    --base_raw_data_dir /home/guy_b/big-storage/raw_data \
    --base_pre_processed_data_dir /home/guy_b/LOS-Net/pre_processed_data \
    --input_output_type input \
    --N_max 100 \
    --input_type LOS
  ```

### Automate Dataset Preprocessing
To preprocess all raw datasets for all models:
  ```bash
  chmod +x ./scripts/preprocess_raw_datasets_LOS.sh

  bash ./scripts/preprocess_raw_datasets_LOS.sh [BASE_RAW_DATA_DIR] [BASE_PRE_PROCESSED_DATA_DIR] [NUM_PARALLEL_JOBS]
  ```
  - example:
  ```bash
  bash ./scripts/preprocess_raw_datasets_LOS.sh /home/guy_b/big-storage/raw_data /home/guy_b/LOS-Net/pre_processed_data 2
  ```


# Reproducibility
## Standard Experiemnts 
To reproduce the experiments from the paper, run the commands below:

- DC:

  | Dataset  | Command |
  |------------|---------|
  | **llama-13b -- BookMIA** | ```wandb sweep ./sweeps/LOS/DC/llama_13b_BookMIA.yaml``` | 
  | **llama-13b -- WikiMIA-32** | ```wandb sweep ./sweeps/LOS/DC/llama_13b_WikiMIA_32.yaml``` | 
  | **llama-13b -- WikiMIA-64** | ```wandb sweep ./sweeps/LOS/DC/llama_13b_WikiMIA_64.yaml``` | 
  |------------|---------|
  | **llama-30b -- BookMIA** | ```wandb sweep ./sweeps/LOS/DC/llama_30b_BookMIA.yaml``` | 
  | **llama-30b -- WikiMIA-32** | ```wandb sweep ./sweeps/LOS/DC/llama_30b_WikiMIA_32.yaml``` | 
  | **llama-30b -- WikiMIA-64** | ```wandb sweep ./sweeps/LOS/DC/llama_30b_WikiMIA_64.yaml``` | 
  |------------|---------|
  | **mamba-1-4 -- WikiMIA-32** | ```wandb sweep ./sweeps/LOS/DC/mamba_1_4_WikiMIA_32.yaml``` | 
  | **mamba-1-4 -- WikiMIA-64** | ```wandb sweep ./sweeps/LOS/DC/mamba_1_4_WikiMIA_64.yaml``` | 
  |------------|---------|
  | **pythia-6-9 -- BookMIA** | ```wandb sweep ./sweeps/LOS/DC/pythia_6_9_BookMIA.yaml``` | 
  | **pythia-6-9 -- WikiMIA-32** | ```wandb sweep ./sweeps/LOS/DC/pythia_6_9_WikiMIA_32.yaml``` | 
  | **pythia-6-9 -- WikiMIA-64** | ```wandb sweep ./sweeps/LOS/DC/pythia_6_9_WikiMIA_64.yaml``` | 
  |------------|---------|
  | **pythia-12 -- BookMIA** | ```wandb sweep ./sweeps/LOS/DC/pythia_12_BookMIA.yaml``` | 

- HD:
  | Dataset  | Command |
  |------------|---------|
    | **mistral -- HotpotQA** | ```wandb sweep ./sweeps/LOS/DC/mistral_hotpotqa_output.yaml``` | 
  | **mistral -- IMDB** | ```wandb sweep ./sweeps/LOS/DC/mistral_imdb_output.yaml``` | 
  | **mistral -- Movies** | ```wandb sweep ./sweeps/LOS/DC/mistral_movies_output.yaml``` | 
  |------------|---------|
  | **meta -- HotpotQA** | ```wandb sweep ./sweeps/LOS/DC/meta_hotpotqa_output.yaml``` | 
  | **meta -- IMDB** | ```wandb sweep ./sweeps/LOS/DC/meta_imdb_output.yaml``` | 
  | **meta -- Movies** | ```wandb sweep ./sweeps/LOS/DC/meta_movies_output.yaml``` | 


## Transferability Experiments

To ensure the functions below operate correctly, please make sure that:
- The project name for all sweeps is **"LOS-Net"**.
- The `probe_model` argument in each sweep is also set to **"LOS-Net"**.

This consistency is essential for proper functionality.

### Cross-LLMs Experiments

This part provides scripts to evaluate zero-shot generalization and transferability of fine-tuned models across different LLMs and datasets. All of the functions are implemented in `fine_tune_main.py`.

#### Zero-Shot Generalization on BookMIA

To evaluate zero-shot transferability on the **BookMIA** dataset, follow these steps:

1. Modify the function `run_tranferability_bookmia` to include a dictionary with the sweep IDs of the runs you conducted:

    ```python    
    sweep_map_bookmia = {
        # 'MODEL_NAME': 'SWEEP_ID'
        'EleutherAI/pythia-6.9b': 'bzpdd8iv',
        'EleutherAI/pythia-12b': 'fpp7ig8b',
        'huggyllama/llama-13b': 'pip01m6c',
        'huggyllama/llama-30b': 'hqfx6e5n'
    }
    ```  

2. Run the following command:

    ```bash
    python fine_tune_main.py
    ```

This script retrieves the best model configuration from the sweeps, tests it across other LLMs, and logs results in:

```
./Results/transferability/DC_zero_shot_cross_model_results_for_dataset_BookMIA_128.csv
```

---

#### Generalization Across Models on Hallucinations Datasets

To assess generalization across different models on **Hallucinations Datasets**, follow these steps:

1. Modify the function `run_transferability_cross_LLMs_HD` to include the corresponding sweep IDs:

    ```python
    imdb_sweep_map_HD_cross_models = {
        # 'MODEL_NAME': 'SWEEP_ID'
        ## Example:
        'meta-llama/Meta-Llama-3-8B-Instruct': 'lgn2rztq',
        'mistralai/Mistral-7B-Instruct-v0.2': 'x8vs8t2x'
    }
    
    movies_sweep_map_HD_cross_models = {
        # 'MODEL_NAME': 'SWEEP_ID'
    }
    
    hotpotqa_sweep_map_HD_cross_models = {
        # 'MODEL_NAME': 'SWEEP_ID'
    }
    ```

2. Run the following command:

    ```bash
    python fine_tune_main.py
    ```

The script retrieves the best model configuration from the sweeps, tests it across different LLMs, and logs results in:

```
./Results/transferability/HD_Finetuning_cross_models_results_for_dataset_<DATASET>.csv
./Results/transferability/HD_Training_from_scratch_cross_models_results_for_dataset_<DATASET>.csv
```

where `<DATASET>` is the name of the dataset being tested, and the second file corresponds to training from scratch (as a natural baseline).

---

### Generalization Across Datasets on Hallucinations Datasets

To evaluate generalization across datasets on **Hallucinations Datasets**, follow these steps:

1. Modify the function `run_transferability_cross_datasets_HD` to include the corresponding sweep IDs:

    ```python
    mistral_sweep_map_HD_cross_datasets = {
        # 'DATASET_NAME': 'SWEEP_ID'
        ## Example:
        'imdb': 'x8vs8t2x',
        'movies': 'w364p4yg',
    }
    
    meta_sweep_map_HD_cross_datasets = {
        # 'DATASET_NAME': 'SWEEP_ID'
        ## Example:
        'imdb': 'x8vs8t2x',
        'movies': 'w364p4yg',
    }
    ```

2. Run the following command:

    ```bash
    python fine_tune_main.py
    ```

The script retrieves the best model configuration from the sweeps, tests it across different datasets, and logs results in:

```
./Results/transferability/HD_Finetuning_cross_datasets_results_for_model_<MODEL>.csv
./Results/transferability/HD_Training_from_scratch_cross_datasets_results_for_model_<MODEL>.csv
```

where `<MODEL>` represents the model being evaluated, and the second file corresponds to training from scratch (as a natural baseline).

### Plot heatmaps
1. **Modify `utils/results_extraction/plots.py`**: 
   For each function in `utils/results_extraction/plots.py`, replace the existing values with the results from your sweeps for the cross LLMs/Datasets experiments.

2. **Example: Updating `plot_heatmaps_for_meta_fine_tune`**:
   Locate the function:

    ```python
    def plot_heatmaps_for_meta_fine_tune():
    ```

    Replace the following arrays:
    ```python
    meta_fine_tune_data = np.array([
        # imdb # movies # hotpotqa
        [89.44, 62.96, 56.52],
        [86.09, 77.04, 68.40],
        [85.54, 75.21, 72.97]
    ])

    meta_fine_tune_errors = np.array([
        [0.32, 0.32, 0.88],
        [0.50, 0.77, 0.47],
        [1.32, 0.18, 0.41]
    ])

    meta_better_than_baselines_indicators = np.array([
        ['*', '', ''],
        ['*', '*', '*'],
        ['*', '*', '*']
    ])
    meta_better_than_from_scratch_indicators = np.array([
        ['*', '*', ''],
        ['*', '*', '*'],
        ['*', '*', '*']
    ])
    ```
  With your updated results from the sweeps for `meta-llama/Meta-Llama-3-8B-Instruct` on the HD datasets (`Imdb`, `Movies`, and `HotpotQA`).
  
  3. run `main.py` to generate the heatmaps.
  