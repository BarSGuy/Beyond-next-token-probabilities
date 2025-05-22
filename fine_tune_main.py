from tqdm import tqdm
import pandas as pd
import torch
from sklearn.metrics import auc, roc_curve
import numpy as np
from utils.constants import LIST_OF_DATASETS_DC
import os
from torch.utils.data import Subset, DataLoader
from utils.Architectures import *
from transformers import set_seed
import wandb
from pathlib import Path
from utils.logger import get_logger
from utils.dataset_preprocess import *
from main import get_train_test_datasets, get_train_test_val_subsets
from typing import Dict, List

# NOTE: Change this to the project name of your sweeps
PROJ = 'LOS-Net'

# -- Wandb helpers -- #
def get_num_seeds_for_sweep(project_name: str, sweep_name: str):
    api = wandb.Api()
    sweep = api.sweep(f"{project_name}/sweeps/{sweep_name}")
    
    num_seeds = len(list(set([run.config['seed'] for run in sweep.runs])))
    return num_seeds

def get_runs_for_sweep(project_name: str, sweep_name: str, metric: str):
    api = wandb.Api()
    sweep = api.sweep(f"{project_name}/sweeps/{sweep_name}")
    # NOTE: it expects the probe model to be called "LOS-Net", and the metric to be "best_val_AUC"
    runs = [run for run in sweep.runs if ((run.state == "finished") and run.config['probe_model'] == "LOS-Net")]
    return runs
    
def get_nth_best_run(project_name: str, sweep_name: str, metric: str, n: int, higher_is_better: bool = True, logger=None):
    api = wandb.Api()
    sweep = api.sweep(f"{project_name}/sweeps/{sweep_name}")
    # NOTE: it expects the probe model to be called "LOS-Net", and the metric to be "best_val_AUC"
    sorted_runs = sorted(
        [run for run in sweep.runs if ((run.state == "finished") and run.config['probe_model'] == "LOS-Net")],  # Filtering out failed runs
        key=lambda run: run.summary.get(metric, float("inf")),
        reverse=higher_is_better
    )
    if 0 < n <= len(sorted_runs):
        return sorted_runs[n - 1]
    else:
        logger.info(f"Invalid index: {n}. Only {len(sorted_runs)} runs available.")
        return None


def extract_searched_hyperparameters_excluding_seeds(project_name: str, sweep_name: str):
    """
    Extracts only the hyperparameters that were searched in the grid, excluding the seed.
    """
    api = wandb.Api()
    sweep = api.sweep(f"{project_name}/sweeps/{sweep_name}")
    all_params = sweep.config["parameters"]

    # Keep only parameters where the "values" list has more than one item
    grid_params = {k: v for k, v in all_params.items() if "values" in v and isinstance(v["values"], list) and len(v["values"]) > 1}
    del grid_params["seed"]
    del grid_params["probe_model"]
    grid_params_keys = grid_params.keys()
    return grid_params_keys

def find_runs_with_hyperparams(best_args_filtered_wrt_grid, all_runs, grid_params_keys):
    """
    Finds all runs that share the same hyperparameter configuration.
    """
    matching_runs = []
    for run in all_runs:
        run_params_filtered_wrt_grid =  {k: v for k, v in run.config.items() if k in grid_params_keys}
        if run_params_filtered_wrt_grid == best_args_filtered_wrt_grid:
            matching_runs.append(run)
    return matching_runs



# -- General function for finetune and evaluation -- #
def fine_tune_and_evaluate(
                       args,
                       num_epochs=10,
                       lr= 0.01,
                       target_LLM="huggyllama/llama-13B",
                       target_dataset='BookMIASplit',
                       from_scratch=False,
                       verbose=False):
    logger = get_logger()
    
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__
    args = dotdict(args)

    set_seed(args.seed)
    
    logger.info("Loading the dataset.")
    
    def get_datasets(args, logger, target_LLM, target_dataset):
        """
        Temporarily modifies args to set target LLM and dataset, retrieves train and test datasets,
        and then restores the original args values.
        
        Args:
            args: The arguments object containing LLM and dataset settings.
            logger: Logger instance for logging (if needed in get_train_test_datasets).
            target_LLM: The target LLM to be set temporarily.
            target_dataset: The target training dataset.
        
        Returns:
            tuple: (dataset_train, dataset_test) - The retrieved train and test datasets.
        """
        # Store original values
        original_LLM = args.LLM
        original_train_dataset = args.train_dataset
        original_test_dataset = args.test_dataset
        
        # try:
        # Modify args temporarily
        args.LLM = target_LLM
        args.train_dataset = target_dataset
        args.test_dataset = target_dataset + "_test"
        
        dataset_train, dataset_test = get_train_test_datasets(args, logger)
    
        # finally:
            # Restore original args values
        args.LLM = original_LLM
        args.train_dataset = original_train_dataset
        args.test_dataset = original_test_dataset
        
        return dataset_train, dataset_test

    dataset_train, dataset_test = get_datasets(args, logger, target_LLM, target_dataset)


    logger.info("Splitting dataset into train, validation, and test indices.")
    assert args.num_folds == 5, "num_folds should be 5."
    splits = stratified_split(dataset_train, percentage=1/args.num_folds, random_state=42)
    train_indices, val_indices, test_indices = get_train_val_test_indices(splits=splits)


        
        
    train_indices =  [train_indices[0], val_indices[0]]
    train_indices = [idx for sublist in train_indices for idx in sublist]
    val_indices = [test_indices[0]]
    val_indices = [idx for sublist in val_indices for idx in sublist]
    train_data = Subset(dataset_train, train_indices)
    val_data = Subset(dataset_train, val_indices)

    
    assert args.fold_to_run < args.num_folds, "fold_to_run should be less than num_folds."
        
 


    best_model_path = Path(args.best_model_path) / args.LLM / args.train_dataset
    checkpoint_path = os.path.join(best_model_path, f"{args.random_number}_best_model.pth")
    artifact = torch.load(checkpoint_path)
    

    
    # Instantiate model (and potentially load weights)
    logger.info(f"Instantiating the model.")
    device = f"cuda" if torch.cuda.is_available() else "cpu"
    # NOTE: max sequence length is 200
    max_sequence_length = 200
    model = get_model(args=args,
                    max_sequence_length=max_sequence_length,
                    actual_sequence_length=train_data[0][0].shape[-2],
                    input_dim=train_data[0][0].shape[-1],
                    input_shape=train_data[0][0].shape).to(device=device)
        
    
    if not from_scratch:
        logger.info(f"Loading the model from the checkpoint.")
        model.load_state_dict(artifact['model_state_dict'])
    else:
        logger.info(f"Training from scratch.")


    dataloader_train = DataLoader(
            train_data,        # Your dataset instance
            batch_size=64,     # Number of samples per batch
            shuffle=True,      # Shuffle data for training
            prefetch_factor=2,
            num_workers=4,
            pin_memory=True)
    dataloader_val = DataLoader(
            val_data,          # Your dataset instance
            batch_size=64,     # Number of samples per batch
            shuffle=False,     # Shuffle data for training
            prefetch_factor=2,
            num_workers=4,     # Number of worker threads for data loading
            pin_memory=True)
        
    dataloader_test = DataLoader(
            dataset_test,         # Your dataset instance
            batch_size=64,     # Number of samples per batch
            shuffle=False,     # Shuffle data for training
            prefetch_factor=2,
            num_workers=4,     # Number of worker threads for data loading
            pin_memory=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCELoss()
    
    best_val_auc = 0
    best_val_tpr_5_fpr = 0
    best_test_auc = 0
    best_test_tpr_5_fpr = 0
    
    if num_epochs == 0:
        inference_only = True
        num_epochs = 1
    else:
        inference_only = False
    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        
        if not inference_only:
            ## train loop ##
            train_loss = 0
            acc_train = 0
            model.train()
            for batch in dataloader_train:
                
                # FWD
                batch = [item.to(device) for item in batch]
                normalized_vocab_prob, normalized_mark, one_hot_rank, labels = batch
                predictions = model(normalized_vocab_prob, normalized_mark, one_hot_rank).reshape(-1)
                
                # BWD
                loss_train = criterion(predictions, labels.float())
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()
                
                # Saving
                acc_train += sum(predictions.round() == labels) / len(labels)
                train_loss += loss_train.item()
        
        ## val loop ##
        model.eval()
        with torch.no_grad():
            val_loss = 0
            acc_val = 0
            all_labels_val = []
            all_predictions_val = []
            for batch in dataloader_val:
                batch = [item.to(device) for item in batch]
                normalized_vocab_prob, normalized_mark, one_hot_rank, labels = batch
                predictions = model(normalized_vocab_prob, normalized_mark, one_hot_rank).reshape(-1)
                loss_val = criterion(predictions, labels.float())
                acc_val += sum(predictions.round() == labels) / len(labels)
                val_loss += loss_val.item()
                all_labels_val.append(labels.cpu().tolist())
                all_predictions_val.append(predictions.detach().cpu().tolist())
                
        all_labels_val = [item for sublist in all_labels_val for item in sublist]
        all_predictions_val = [item for sublist in all_predictions_val for item in sublist]
        fpr_val, tpr_val, _ = roc_curve(np.array(all_labels_val, dtype=bool), np.array(all_predictions_val))
        AUC_val = auc(fpr_val, tpr_val)
        tpr_5_fpr_val = tpr_val[np.where(fpr_val < 0.05)[0][-1]]
                            
        ## test loop ##
        model.eval()
        with torch.no_grad():
            test_loss = 0
            acc_test = 0
            all_labels_test = []
            all_predictions_test = []
            for batch in dataloader_test:
                batch = [item.to(device) for item in batch]
                normalized_vocab_prob, normalized_mark, one_hot_rank, labels = batch
                predictions = model(normalized_vocab_prob, normalized_mark, one_hot_rank).reshape(-1)
                loss_test = criterion(predictions, labels.float())
                acc_test += sum(predictions.round() == labels) / len(labels)
                test_loss += loss_test.item()
                all_labels_test.append(labels.cpu().tolist())
                all_predictions_test.append(predictions.detach().cpu().tolist())
        all_labels_test = [item for sublist in all_labels_test for item in sublist]
        all_predictions_test = [item for sublist in all_predictions_test for item in sublist]
        fpr_test, tpr_test, _ = roc_curve(np.array(all_labels_test, dtype=bool), np.array(all_predictions_test))
        AUC_test = auc(fpr_test, tpr_test)
        tpr_5_fpr_test = tpr_test[np.where(fpr_test< 0.05)[0][-1]]
        
        if AUC_val > best_val_auc:
            best_val_auc = AUC_val
            best_test_auc = AUC_test
            
        if tpr_5_fpr_val > best_val_tpr_5_fpr:
            best_val_tpr_5_fpr = tpr_5_fpr_val
            best_test_tpr_5_fpr = tpr_5_fpr_test
            
        if verbose:
            logger.info(f"[i] Epoch {epoch}")
            logger.info(f'                   Val. AUC: {AUC_val} (best: {best_val_auc})')
            logger.info(f'                   Test AUC: {AUC_test} (best: {best_test_auc})')


    if verbose:
        logger.info(f"""
        Test Results:
        --------------
        From Scratch:                {from_scratch}
        Model Number:                {args['random_number']}
        Pretrain LLM                 {args.LLM}
        Finetune/Test LLM            {target_LLM}
        Pretrain dataset             {args.train_dataset}
        Finetune/Test dataset        {target_dataset}
        Learning Rate (LR):          {lr}
        Weight Decay:                {args.weight_decay}
        Dropout:                     {args.dropout}
        Epochs:                      {num_epochs}
        
        Validation Metrics:
        - AUC:                       {best_val_auc}
        - TPR at 5% FPR:             {best_val_tpr_5_fpr}
        
        Test Metrics:
        - AUC:                       {best_test_auc}
        - TPR at 5% FPR:             {best_test_tpr_5_fpr}
        """)
    return best_test_auc, best_test_tpr_5_fpr

# -- BookMIA zero-shot generalization helpers -- #
def get_results_for_bookmia_cross_LLM(sweep_map):
    logger = get_logger()
    results = []  # List to store results
    source_llms = list(sweep_map.keys())
    target_llms = list(sweep_map.keys())

    
    for source_llm in tqdm(sweep_map):
        row_results = []
        sweep_name = sweep_map[source_llm]
        
        grid_params_keys = extract_searched_hyperparameters_excluding_seeds(PROJ, sweep_name)
        logger.info(f'[i] The grid params keys for {source_llm} are: {grid_params_keys}')
        
        best = get_nth_best_run(PROJ, sweep_name, "best_val_AUC", 2, higher_is_better=True, logger=logger)
        logger.info(f'[i] The best run has id: {best.id}')
        
        best_args = best.config
        best_args_filtered_wrt_grid =  {k: v for k, v in best_args.items() if k in grid_params_keys}
        logger.info(f'[i] Extracting args of best run filtered wrt grid: {best_args_filtered_wrt_grid}')
        
        all_runs = get_runs_for_sweep(PROJ, sweep_name, "best_val_AUC")        
        logger.info("Extracted all runs for the sweep.")
        
        matching_runs = find_runs_with_hyperparams(best_args_filtered_wrt_grid=best_args_filtered_wrt_grid, all_runs=all_runs, grid_params_keys=grid_params_keys)
        
        logger.info(f'[i] keeping only runs that match the best run hyperparameters, found {len(matching_runs)} matching runs.')
        expected_seeds = get_num_seeds_for_sweep(PROJ, sweep_name)
        actual_runs = len(matching_runs)
        
        assert actual_runs == expected_seeds, (
            f"Mismatch in the number of runs: Expected {expected_seeds}, but found {actual_runs}."
        )
    
        
        for target_llm in sweep_map:
            if source_llm == target_llm:
                row_results.append((0, 0))
                continue

            target_LLM = target_llm
        
            target_dataset = "BookMIA_128"


            auc_scores = []
            for i, run in enumerate(matching_runs):
                logger.info(f"Running source LLM {source_llm} VS target LLM {target_llm}, over model {i+1}/{len(matching_runs)}...")
                args_of_run = run.config
                best_test_auc, best_test_tpr_5_fpr = fine_tune_and_evaluate(
                                args_of_run,
                                num_epochs=0,
                                lr=0,
                                target_LLM=target_LLM,
                                target_dataset=target_dataset,
                                from_scratch=False)
                auc_scores.append(best_test_auc)
            mean_auc = np.mean(auc_scores)
            std_auc = np.std(auc_scores)
            row_results.append((mean_auc, std_auc))
        results.append(row_results)


    logger.info("\n" + "#" * 50)
    logger.info("Final Results for zero shot generalization:")
    
    df_results = pd.DataFrame(
            [[f"{mean:.4f} ± {std:.4f}" if isinstance(mean, float) else mean for mean, std in row] for row in results],
            index=source_llms, columns=target_llms
        )

    
    logger.info("RESULTS".center(50))
    logger.info("#" * 50 + "\n")
    logger.info(df_results)
    # Define the folder path
    folder_path = f"./Results/transferability_BookMIA"

    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    df_results.to_csv(f"./{folder_path}/DC_zero_shot_cross_model_results_for_dataset_{target_dataset}.csv")
    logger.info("\n" + "#" * 50)

# -- HD transferability helpers -- #
def get_results_for_HD_cross_Datasets_seeds(sweep_map, num_epochs=0, lr=0.001, from_scratch=False, num_seeds=10):
    logger = get_logger()
    results = []  # List to store results
    source_datasets = list(sweep_map.keys())
    target_datasets = list(sweep_map.keys())

    
    for source_dataset in tqdm(sweep_map):
        row_results = []
        for target_dataset in sweep_map:
            sweep_name = sweep_map[source_dataset]
            # if source_dataset == target_dataset:
            #     row_results.append((0, 0))  # Diagonal values are irrelevant
            #     continue
            
            # NOTE: it expects to project name to be called "LOS-Net", and the metric to be "best_val_AUC"
            # NOTE: taken 2nd best run (not 1st), which makes sense if the sweep is over 3 seeds
            best = get_nth_best_run(PROJ, sweep_name, "best_val_AUC", 2, higher_is_better=True, logger=logger)
            
            logger.info(f'[i] The best run has id: {best.id}')
            best_args = best.config
            auc_scores = []
            for seed in range(num_seeds):
                best_args['seed'] = seed  # Change seed for each run
                logger.info(f"Running source dataset {source_dataset} VS target dataset {target_dataset}...")
                best_test_auc, best_test_tpr_5_fpr = fine_tune_and_evaluate(
                                best_args,
                                num_epochs=num_epochs,
                                lr=lr,
                                target_LLM=best_args['LLM'],
                                target_dataset=target_dataset,
                                from_scratch=from_scratch)
                auc_scores.append(best_test_auc)
            mean_auc = np.mean(auc_scores)
            std_auc = np.std(auc_scores)
            row_results.append((mean_auc, std_auc))
        results.append(row_results)
    
    logger.info(f"NOTE! the diagonal values are irrelevant! They show what happens when you take the checkpoint, and test it on the same dataset, but after fine tunning/training from scratch.")
    logger.info("\n" + "#" * 50)
    if from_scratch:
        logger.info("Final Results for training from scratch:")
    else:
        logger.info(f"Final Results for fine-tuning for {num_epochs} epochs, with constant lr of {lr}:")



    df_results = pd.DataFrame(
        [[f"{mean:.4f} ± {std:.4f}" if isinstance(mean, float) else mean for mean, std in row] for row in results],
        index=source_datasets, columns=target_datasets
    )

    logger.info("RESULTS".center(50))
    logger.info("#" * 50 + "\n")
    logger.info(df_results)
    # Define the folder path
    folder_path = f"./Results/transferability_HD||num_epochs_{num_epochs}_lr_{lr}_num_seeds_{num_seeds}"

    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    if from_scratch:
        df_results.to_csv(f"./{folder_path}/HD_Training_from_scratch_cross_datasets_results_for_model_{best_args['LLM'].split('/')[0]}.csv")
    else:
        df_results.to_csv(f"./{folder_path}/HD_Finetuning_cross_datasets_results_for_model_{best_args['LLM'].split('/')[0]}.csv")
    logger.info("\n" + "#" * 50)

def get_results_for_HD_cross_Models_seeds(sweep_map, num_epochs=0, lr=0.001, from_scratch=False, num_seeds=5):
    logger = get_logger()
    results = []  # List to store results
    source_llms = list(sweep_map.keys())
    target_llms = list(sweep_map.keys())

    for source_llm in tqdm(sweep_map):
        row_results = []
        for target_llm in sweep_map:
            sweep_name = sweep_map[source_llm]
            target_LLM = target_llm
            # if source_llm == target_llm:
            #     row_results.append((0, 0))  # Diagonal values are irrelevant
            #     continue
            


            best = get_nth_best_run(PROJ, sweep_name, "best_val_AUC", 2, higher_is_better=True, logger=logger)
            logger.info(f'[i] The best run has id: {best.id}')
            best_args = best.config
            
            auc_scores = []
            for seed in range(num_seeds):
                best_args['seed'] = seed  # Change seed for each run
                logger.info(f"Running source LLM {source_llm} VS target LLM {target_llm}, Seed: {seed}...")
                best_test_auc, _ = fine_tune_and_evaluate(
                    best_args,
                    num_epochs=num_epochs,
                    lr=lr,
                    target_LLM=target_LLM,
                    target_dataset=best_args['train_dataset'],
                    from_scratch=from_scratch
                )
                auc_scores.append(best_test_auc)
            
            mean_auc = np.mean(auc_scores)
            std_auc = np.std(auc_scores)
            row_results.append((mean_auc, std_auc))
        results.append(row_results)
    
    logger.info("NOTE! The diagonal values are irrelevant!")
    
    df_results = pd.DataFrame(
        [[f"{mean:.4f} ± {std:.4f}" if isinstance(mean, float) else mean for mean, std in row] for row in results],
        index=source_llms, columns=target_llms
    )
    
    logger.info("RESULTS".center(50))
    logger.info("#" * 50 + "\n")
    print(df_results)
    # Define the folder path
    folder_path = f"./Results/transferability_HD||num_epochs_{num_epochs}_lr_{lr}_num_seeds_{num_seeds}"

    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    if from_scratch:
        df_results.to_csv(f"./{folder_path}/HD_Training_from_scratch_cross_models_results_for_dataset_{best_args['train_dataset']}.csv")
    else:
        df_results.to_csv(f"./{folder_path}/HD_Finetuning_cross_models_results_for_dataset_{best_args['train_dataset']}.csv")
    logger.info("\n" + "#" * 50)


# -- Main -- #
def main():
    # run_tranferability_bookmia()
    num_epochs_list = [10]
    lr_list = [0.0001]
    num_seeds_list = [3]
    for num_epochs in num_epochs_list:
        for lr in lr_list:
            for num_seeds in num_seeds_list:
                run_transferability_cross_LLMs_HD(num_epochs=num_epochs, lr=lr, num_seeds=num_seeds)
                run_transferability_cross_datasets_HD(num_epochs=num_epochs, lr=lr, num_seeds=num_seeds)

    
# -- BOOKMIA -- #
def run_tranferability_bookmia():
    sweep_map_bookmia = {
        # Models: sweep_name
        'EleutherAI/pythia-6.9b': 'nudweghd',
        'EleutherAI/pythia-12b': 'c1xftv4r',
        'huggyllama/llama-13b': 'wtn36occ',
        'huggyllama/llama-30b': 't55706wa'
    }
    get_results_for_bookmia_cross_LLM(sweep_map_bookmia)
    
# -- HD -- #
def run_transferability_cross_LLMs_HD(num_epochs=10, lr=0.0001, num_seeds=5):
    

    imdb_sweep_map_HD_cross_models = {
        # Models: sweep_name
        'meta-llama/Meta-Llama-3-8B-Instruct': 'mbcxaw1d',
        'mistralai/Mistral-7B-Instruct-v0.2': 'h1otb44c',
        'Qwen/Qwen2.5-7B-Instruct': 'fyizkz09'
    }
    
    get_results_for_HD_cross_Models_seeds(imdb_sweep_map_HD_cross_models, num_epochs=num_epochs, lr=lr, from_scratch=True, num_seeds=num_seeds)
    get_results_for_HD_cross_Models_seeds(imdb_sweep_map_HD_cross_models, num_epochs=num_epochs, lr=lr, from_scratch=False, num_seeds=num_seeds)

    
    movies_sweep_map_HD_cross_models = {
        # Models: sweep_name
        'meta-llama/Meta-Llama-3-8B-Instruct': 'nrcys17s',
        'mistralai/Mistral-7B-Instruct-v0.2': 'a5amnezy',
        'Qwen/Qwen2.5-7B-Instruct': 'penzw2rd'
    }
    
    get_results_for_HD_cross_Models_seeds(movies_sweep_map_HD_cross_models, num_epochs=num_epochs, lr=lr, from_scratch=True, num_seeds=num_seeds)
    get_results_for_HD_cross_Models_seeds(movies_sweep_map_HD_cross_models, num_epochs=num_epochs, lr=lr, from_scratch=False, num_seeds=num_seeds)
    
    hotpotqa_sweep_map_HD_cross_models = {
        # Models: sweep_name
        'meta-llama/Meta-Llama-3-8B-Instruct': 'qzxswss4',
        'mistralai/Mistral-7B-Instruct-v0.2': '952ybkat',
        'Qwen/Qwen2.5-7B-Instruct': 'tslop9sa'
    }
    get_results_for_HD_cross_Models_seeds(hotpotqa_sweep_map_HD_cross_models, num_epochs=num_epochs, lr=lr, from_scratch=True, num_seeds=num_seeds)
    get_results_for_HD_cross_Models_seeds(hotpotqa_sweep_map_HD_cross_models, num_epochs=num_epochs, lr=lr, from_scratch=False, num_seeds=num_seeds)

def run_transferability_cross_datasets_HD(num_epochs=10, lr=0.0001, num_seeds=5):
    
    mistral_sweep_map_HD_cross_datasets = {
        # Datasets: sweep_name
        'imdb': 'h1otb44c',
        'movies': 'a5amnezy',
        'hotpotqa': '952ybkat'
    }
    get_results_for_HD_cross_Datasets_seeds(mistral_sweep_map_HD_cross_datasets, num_epochs=num_epochs, lr=lr, from_scratch=True, num_seeds=num_seeds)
    get_results_for_HD_cross_Datasets_seeds(mistral_sweep_map_HD_cross_datasets, num_epochs=num_epochs, lr=lr, from_scratch=False, num_seeds=num_seeds)

    meta_sweep_map_HD_cross_datasets = {
        # Datasets: sweep_name
        'imdb': 'mbcxaw1d',
        'movies': 'nrcys17s',
        'hotpotqa': 'qzxswss4'
    }
    get_results_for_HD_cross_Datasets_seeds(meta_sweep_map_HD_cross_datasets, num_epochs=num_epochs, lr=lr, from_scratch=True, num_seeds=num_seeds)
    get_results_for_HD_cross_Datasets_seeds(meta_sweep_map_HD_cross_datasets, num_epochs=num_epochs, lr=lr, from_scratch=False, num_seeds=num_seeds)

    qwen_sweep_map_HD_cross_datasets = {
        # Datasets: sweep_name
        'imdb': 'fyizkz09',
        'movies': 'penzw2rd',
        'hotpotqa': 'tslop9sa'
    }
    get_results_for_HD_cross_Datasets_seeds(qwen_sweep_map_HD_cross_datasets, num_epochs=num_epochs, lr=lr, from_scratch=True, num_seeds=num_seeds)
    get_results_for_HD_cross_Datasets_seeds(qwen_sweep_map_HD_cross_datasets, num_epochs=num_epochs, lr=lr, from_scratch=False, num_seeds=num_seeds)

    
if __name__ == "__main__":
    main()
