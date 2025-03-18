import pandas as pd
import numpy as np
import wandb
from itertools import product
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constants import LIST_OF_MODELS_DC

def fetch_and_process_wandb_data(project='probing_revisited-src',                   sweep_id='dmetxpy4', 
                        chosen_dataset = 'hotpotqa',
                        chosen_type = 'LOS',
                        chosen_probe_model = 'LOS-Net',
                         chosen_LLM = 'mistralai/Mistral-7B-Instruct-v0.2',
                         prefix_for_file='', seed_based=False):
    """
    Fetch sweep data from wandb, calculate the mean and standard deviation of specified metrics
    for each combination of hyperparameters, and save the results to a CSV file.

    Parameters:
        project (str): Name of the wandb project.
        sweep_id (str): ID of the wandb sweep.
        hyperparameters (list): List of hyperparameter names used in the sweep.
    """# Initialize wandb API
    api = wandb.Api()
    # Fetch the sweep
    sweep = api.sweep(f"{project}/{sweep_id}")
    print(f'[i] Processing sweep {sweep_id}... (status: {sweep.state})')



    

    # Collect all runs in the sweep
    runs = sweep.runs
    print(sweep)

    # Extract data from runs
    data = []
    for run in runs:
        config = run.config
        if seed_based:
            hyp = [key for key in list(config.keys()) if key not in ['base_raw_data_dir', 'best_model_path', 'total_params', 'seed', 'random_number', 'best_val_AUC', 'best_test_AUC', 'best_val_tpr_5_fpr', 'best_test_tpr_5_fpr']]
        else:
            hyp = [key for key in list(config.keys()) if key not in ['base_raw_data_dir', 'best_model_path', 'total_params', 'random_number', 'fold_to_run', 'best_val_AUC', 'best_test_AUC', 'best_val_tpr_5_fpr', 'best_test_tpr_5_fpr']]
        hyperparameters = hyp

        summary = run.summary

        # Extract metrics and hyperparameters
        # hyperparameters = hyp
        row = {key: config.get(key, None) for key in hyperparameters}
        if row['train_dataset'] != chosen_dataset:
            continue
        if row['LLM'] != chosen_LLM:
            continue
        if row['input_type'] != chosen_type:
            continue
        if row['probe_model'] != chosen_probe_model:
            continue
        if seed_based:
            row['seed'] = config.get('seed', None)
        else:
            row['fold_to_run'] = config.get('fold_to_run', None)
        # row['model_number'] = config.get('random_number', None)
        row['best_val_AUC'] = summary.get('best_val_AUC', None)
        row['best_test_AUC'] = summary.get('best_test_AUC', None)
        row['best_val_tpr_5_fpr'] = summary.get('best_val_tpr_5_fpr', None)
        row['best_test_tpr_5_fpr'] = summary.get('best_test_tpr_5_fpr', None)
        
        data.append(row)

    # Convert to DataFrame
    data_df = pd.DataFrame(data)

    # Group by hyperparameters excluding `fold_to_run`
    grouped = data_df.groupby(hyperparameters)

    # Aggregate the results
    results = []
    for params, group in grouped:
        result = dict(zip(hyperparameters, params))
        result['mean_best_val_AUC'] = group['best_val_AUC'].mean()
        result['std_best_val_AUC'] = group['best_val_AUC'].std()
        result['mean_best_test_AUC'] = group['best_test_AUC'].mean()
        result['std_best_test_AUC'] = group['best_test_AUC'].std()
        result['mean_best_val_tpr_5_fpr'] = group['best_val_tpr_5_fpr'].mean()
        result['std_best_val_tpr_5_fpr'] = group['best_val_tpr_5_fpr'].std()
        result['mean_best_test_tpr_5_fpr'] = group['best_test_tpr_5_fpr'].mean()
        result['std_best_test_tpr_5_fpr'] = group['best_test_tpr_5_fpr'].std()
        result['number_of_runs'] = group.shape[0]  # Add the number of runs
        result['run_id'] = f"Project: {project}, Sweep ID: {sweep_id}"

        results.append(result)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by best_test_AUC in descending order
    results_df = results_df.sort_values(by='mean_best_val_AUC', ascending=False)

    # Save to CSV
    output_csv = f"___[{sweep.state}]__sweep_{sweep_id}_{chosen_dataset}_{chosen_LLM.split('/')[0]}_{chosen_LLM.split('/')[1]}.csv"
    output_csv = f"./Results/{chosen_type}/{chosen_LLM}/{chosen_dataset}/{chosen_probe_model}" + prefix_for_file + output_csv
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results_df.to_csv(output_csv, index=False)

    print(f"Results saved to {output_csv}")
    
    

if __name__ == '__main__':
    fetch_and_process_wandb_data(project='LOS-Net', sweep_id='t55706wa', 
                        chosen_dataset = 'BookMIA_128',
                         chosen_probe_model='LOS-Net',
                         chosen_type = 'LOS',
                         chosen_LLM = 'huggyllama/llama-13b', seed_based=True)
    fetch_and_process_wandb_data(project='LOS-Net', sweep_id='t55706wa', 
                        chosen_dataset = 'BookMIA_128',
                         chosen_probe_model='ATP_R_Transf',
                         chosen_type = 'LOS',
                         chosen_LLM = 'huggyllama/llama-13b', seed_based=True)
    fetch_and_process_wandb_data(project='LOS-Net', sweep_id='t55706wa', 
                        chosen_dataset = 'BookMIA_128',
                         chosen_probe_model='ATP_R_MLP',
                         chosen_type = 'LOS',
                         chosen_LLM = 'huggyllama/llama-13b', seed_based=True)
    
    fetch_and_process_wandb_data(project='LOS-Net', sweep_id='wtn36occ', 
                        chosen_dataset = 'BookMIA_128',
                         chosen_probe_model='LOS-Net',
                         chosen_type = 'LOS',
                         chosen_LLM = 'huggyllama/llama-30b', seed_based=True)
    fetch_and_process_wandb_data(project='LOS-Net', sweep_id='wtn36occ', 
                        chosen_dataset = 'BookMIA_128',
                         chosen_probe_model='ATP_R_Transf',
                         chosen_type = 'LOS',
                         chosen_LLM = 'huggyllama/llama-30b', seed_based=True)
    fetch_and_process_wandb_data(project='LOS-Net', sweep_id='wtn36occ', 
                        chosen_dataset = 'BookMIA_128',
                         chosen_probe_model='ATP_R_MLP',
                         chosen_type = 'LOS',
                         chosen_LLM = 'huggyllama/llama-30b', seed_based=True)
    
    fetch_and_process_wandb_data(project='LOS-Net', sweep_id='nudweghd', 
                        chosen_dataset = 'BookMIA_128',
                         chosen_probe_model='LOS-Net',
                         chosen_type = 'LOS',
                         chosen_LLM = 'EleutherAI/pythia-6.9b', seed_based=True)
    fetch_and_process_wandb_data(project='LOS-Net', sweep_id='nudweghd', 
                        chosen_dataset = 'BookMIA_128',
                         chosen_probe_model='ATP_R_Transf',
                         chosen_type = 'LOS',
                         chosen_LLM = 'EleutherAI/pythia-6.9b', seed_based=True)
    fetch_and_process_wandb_data(project='LOS-Net', sweep_id='nudweghd', 
                        chosen_dataset = 'BookMIA_128',
                         chosen_probe_model='ATP_R_MLP',
                         chosen_type = 'LOS',
                         chosen_LLM = 'EleutherAI/pythia-6.9b', seed_based=True)
    
    fetch_and_process_wandb_data(project='LOS-Net', sweep_id='c1xftv4r', 
                        chosen_dataset = 'BookMIA_128',
                         chosen_probe_model='LOS-Net',
                         chosen_type = 'LOS',
                         chosen_LLM = 'EleutherAI/pythia-12b', seed_based=True)
    fetch_and_process_wandb_data(project='LOS-Net', sweep_id='c1xftv4r', 
                        chosen_dataset = 'BookMIA_128',
                         chosen_probe_model='ATP_R_Transf',
                         chosen_type = 'LOS',
                         chosen_LLM = 'EleutherAI/pythia-12b', seed_based=True)
    fetch_and_process_wandb_data(project='LOS-Net', sweep_id='c1xftv4r', 
                        chosen_dataset = 'BookMIA_128',
                         chosen_probe_model='ATP_R_MLP',
                         chosen_type = 'LOS',
                         chosen_LLM = 'EleutherAI/pythia-12b', seed_based=True)