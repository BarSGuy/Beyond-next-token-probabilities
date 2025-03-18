from sklearn.metrics import auc, roc_curve
import wandb
from utils.logger import get_logger
from utils.args import parse_args_main
import os
from pathlib import Path
from utils.dataset_preprocess import *
from utils.constants import LIST_OF_DATASETS_DC
from transformers import set_seed
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from utils.Architectures import get_model
from transformers import get_scheduler
import time


def get_train_test_datasets(args, logger):
    """Preprocesses datasets and loads them based on the task type."""
    
    # Define output directories
    train_data_preprocessed_dir = Path(args.base_pre_processed_data_dir) / args.LLM / args.train_dataset
    test_data_preprocessed_dir = Path(args.base_pre_processed_data_dir) / args.LLM / args.test_dataset

    
    logger.info(f"Starting data preparation for model '{args.LLM}' using training dataset '{args.train_dataset}'.")
    
    if args.train_dataset in LIST_OF_DATASETS_DC and ('BookMIA' not in args.train_dataset):

        dataset_train = CustomSavedDataset(
            preprocessed_dir=train_data_preprocessed_dir,
            topk_preprocess=args.topk_preprocess,
            topk_dim=args.topk_dim,
            input_output_flag=args.input_output_type, 
            input_type = args.input_type
        )
        logger.info("Training dataset loaded successfully.")
        dataset_test = None
        logger.info("Test dataset is not required for this task.")
    elif 'BookMIA' in args.train_dataset:
        def split_bookmia(train_size=0.80, seed=None):
            from datasets import load_dataset
            import random
            raw_bookmia = load_dataset('swj0419/BookMIA')
            labels2ids = {0: set(), 1: set()}
            for item in raw_bookmia['train']:
                labels2ids[item['label']].add(item['book_id'])
            assert len(set(labels2ids[0]) & set(labels2ids[1])) == 0
            cut_0 = int(len(labels2ids[0]) * train_size)
            list_0 = list(labels2ids[0])
            if seed is not None:
                random.Random(seed).shuffle(list_0)
            train_0 = list_0[:cut_0]
            test_0 = list_0[cut_0:]
            cut_1 = int(len(labels2ids[1]) * train_size)
            list_1 = list(labels2ids[1])
            if seed is not None:
                random.Random(seed).shuffle(list_1)
            train_1 = list_1[:cut_1]
            test_1 = list_1[cut_1:]
            train = train_0+train_1
            test = test_0+test_1
            assert len(set(train_0) & set(test_0)) == 0, set(train_0) & set(test_0)
            assert len(set(train_1) & set(test_1)) == 0, set(train_1) & set(test_1)
            assert len(set(train) & set(test)) == 0, set(train) & set(test)
            
            bookmia_train_indices = [i for i in range(len(raw_bookmia['train'])) if raw_bookmia['train'][i]['book_id'] in train]
            bookmia_test_indices = [i for i in range(len(raw_bookmia['train'])) if raw_bookmia['train'][i]['book_id'] in test]

            return bookmia_train_indices, bookmia_test_indices

        dataset = CustomSavedDataset(
            preprocessed_dir=train_data_preprocessed_dir,
            topk_preprocess=args.topk_preprocess,
            topk_dim=args.topk_dim,
            input_output_flag=args.input_output_type, 
            input_type = args.input_type
        )

        
        bookmia_train_indices, bookmia_test_indices = split_bookmia(train_size=0.80, seed=42)
        dataset_train = Subset(dataset, bookmia_train_indices)
        dataset_test = Subset(dataset, bookmia_test_indices)
    else:
        dataset_train = CustomSavedDataset(
            preprocessed_dir=train_data_preprocessed_dir,
            topk_preprocess=args.topk_preprocess,
            topk_dim=args.topk_dim,
            input_output_flag=args.input_output_type, 
            input_type = args.input_type
        )
        logger.info("Training dataset loaded successfully.")
        
        dataset_test = CustomSavedDataset(
            preprocessed_dir=test_data_preprocessed_dir,
            topk_preprocess=args.topk_preprocess,
            topk_dim=args.topk_dim,
            input_output_flag=args.input_output_type, 
            input_type = args.input_type
        )
        logger.info("Test dataset loaded successfully.")
    
    logger.info("Dataset processing pipeline completed successfully.")
    return dataset_train, dataset_test

def get_train_test_val_subsets(args, train_indices, val_indices, test_indices, fold, train_dataset, test_dataset):
    if args.train_dataset in LIST_OF_DATASETS_DC and ('BookMIA' not in args.train_dataset):
        train_data = Subset(train_dataset, train_indices[fold])
        val_data = Subset(train_dataset, val_indices[fold])
        test_data = Subset(train_dataset, test_indices[fold])
    else:
        train_data = Subset(train_dataset, train_indices)
        val_data = Subset(train_dataset, val_indices)
        test_data = test_dataset
    return train_data, val_data, test_data


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, input_type='LOS'):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0
    all_labels, all_predictions = [], []
    
    for batch in tqdm(dataloader, desc="Training Progress"):
        batch = [item.to(device) for item in batch]
        if input_type == 'LOS':
            sorted_TDS_normalized, normalized_ATP, ATP_R, labels = batch
            optimizer.zero_grad()
            predictions = model(sorted_TDS_normalized, normalized_ATP, ATP_R).reshape(-1)
        else:
            raise ValueError("Invalid input type.")
        loss = criterion(predictions, labels.float())
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())
    
    fpr, tpr, _ = roc_curve(np.array(all_labels, dtype=bool), np.array(all_predictions))
    return total_loss / len(dataloader), auc(fpr, tpr)

def evaluate(model, dataloader, criterion, device, desc="Validation", input_type='LOS'):
    """Evaluates the model on validation or test data."""
    model.eval()
    total_loss = 0
    all_labels, all_predictions = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{desc} Progress"):
            batch = [item.to(device) for item in batch]
            if input_type == 'LOS':
                sorted_TDS_normalized, normalized_ATP, ATP_R, labels = batch
                predictions = model(sorted_TDS_normalized, normalized_ATP, ATP_R).reshape(-1)
            else:
                raise ValueError("Invalid input type.")
                
            loss = criterion(predictions, labels.float())
            total_loss += loss.item()
            
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())
    
    fpr, tpr, _ = roc_curve(np.array(all_labels, dtype=bool), np.array(all_predictions))
    auc_score = auc(fpr, tpr)
    tpr_5_fpr = tpr[np.where(fpr < 0.05)[0][-1]] if np.any(fpr < 0.05) else 0
    
    return total_loss / len(dataloader), auc_score, tpr_5_fpr

def save_best_model(logger, model, best_val_auc, best_test_auc, args):
    """Saves the best model state."""
    
    os.makedirs(args.best_model_path, exist_ok=True)
    model_path = os.path.join(args.best_model_path, f"{args.random_number}_best_model.pth")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_val_auc': best_val_auc,
        'best_test_auc': best_test_auc
    }, model_path)
    logger.info(f"Model saved at {model_path}")

def train_model(logger, model, dataloader_train, dataloader_val, dataloader_test, criterion, optimizer, scheduler, args, device):
    """Trains and evaluates the model with early stopping."""
    best_val_auc, best_val_tpr_5_fpr = 0, 0
    best_test_auc, best_test_tpr_5_fpr = 0, 0
    patience, no_improve_count = args.patience, 0
    
    for epoch in range(args.num_epochs):
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}")
        
        # Training
        train_loss, auc_train = train_one_epoch(model, dataloader_train, criterion, optimizer, scheduler, device, input_type=args.input_type)
        logger.info(f"Train Loss: {train_loss:.4f}, Train AUC: {auc_train:.4f}")
        
        # Validation
        val_loss, auc_val, tpr_5_fpr_val = evaluate(model, dataloader_val, criterion, device, desc="Validation", input_type=args.input_type)
        logger.info(f"Val Loss: {val_loss:.4f}, Val AUC: {auc_val:.4f}, Val TPR@5%FPR: {tpr_5_fpr_val:.4f}")
        
        # Test
        test_loss, auc_test, tpr_5_fpr_test = evaluate(model, dataloader_test, criterion, device, desc="Test", input_type=args.input_type)
        logger.info(f"Test Loss: {test_loss:.4f}, Test AUC: {auc_test:.4f}, Test TPR@5%FPR: {tpr_5_fpr_test:.4f}")
        
        # Save the best model if validation AUC improves
        if auc_val > best_val_auc:
            save_best_model(logger, model, auc_val, auc_test, args)
            best_val_auc, best_test_auc = auc_val, auc_test
            no_improve_count = 0  # Reset counter
        else:
            no_improve_count += 1
            logger.info(f"No improvement for {no_improve_count} epochs.")
        
        # Update best TPR@5%FPR
        if tpr_5_fpr_val > best_val_tpr_5_fpr:
            best_val_tpr_5_fpr, best_test_tpr_5_fpr = tpr_5_fpr_val, tpr_5_fpr_test
        
        # Early stopping
        if no_improve_count >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs.")
            break
        
        # Logging to WandB
        wandb.log({
            "train_loss_epoch": train_loss,
            "AUC_train_epoch": auc_train,
            "val_loss_epoch": val_loss,
            "best_val_AUC": best_val_auc,
            "best_val_tpr_5_fpr": best_val_tpr_5_fpr,
            "test_loss_epoch": test_loss,
            "best_test_AUC": best_test_auc,
            "best_test_tpr_5_fpr": best_test_tpr_5_fpr,
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch": epoch + 1,
        })
    
    wandb.finish()
    logger.info("Training complete.")
    
    
def main():
    """Main function to preprocess data and load datasets based on task type."""
    # Initialize logger
    logger = get_logger()
    
    # Parse command-line arguments
    args = parse_args_main()
    logger.info("Starting the data processing pipeline.")
    logger.info(f"Parsed Arguments: {vars(args)}")
    
    if args.input_type == 'LOS':
        assert args.probe_model in ["LOS-Net", "ATP_R_MLP", "ATP_R_Transf"]
    else:
        raise ValueError("Invalid input type.")
    
    logger.info(f"Loading preproccessed data for model '{args.LLM}'")
    # Process datasets
    dataset_train, dataset_test = get_train_test_datasets(args, logger)


    logger.info("Splitting dataset into train, validation, and test indices.")
    assert args.num_folds == 5, "num_folds should be 5."
    splits = stratified_split(dataset_train, percentage=1/args.num_folds, random_state=42)
    train_indices, val_indices, test_indices = get_train_val_test_indices(splits=splits)




    if args.train_dataset in LIST_OF_DATASETS_DC and ('BookMIA' not in args.train_dataset):
        logger.info(f"for {args.train_dataset} splitting to {args.num_folds} folds")
        logger.info(f"Train size: {len(train_indices[0])}, Validation size: {len(val_indices[0])}, Test size: {len(test_indices[0])}")
    else:
        train_indices =  [train_indices[0], val_indices[0]]
        train_indices = [idx for sublist in train_indices for idx in sublist]
        val_indices = [test_indices[0]]
        val_indices = [idx for sublist in val_indices for idx in sublist]
        logger.info(f"Train size: {len(train_indices)}, Validation size: {len(val_indices)}, Test indices: {len(dataset_test)}")
    
    
    set_seed(args.seed)
    device = f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu"
    
    assert args.fold_to_run < args.num_folds, "fold_to_run should be less than num_folds."
        

    logger.info(f"Running fold {args.fold_to_run + 1} of {args.num_folds}.")
    train_data, val_data, test_data = get_train_test_val_subsets(args, train_indices, val_indices, test_indices, args.fold_to_run, dataset_train, dataset_test)
    logger.info("Creating dataloaders for training, validation, and test sets.")    
    dataloader_train = DataLoader(
        train_data,          # Your dataset instance
        batch_size=args.batch_size,     # Number of samples per batch
        shuffle=True,     # Shuffle data for training
        prefetch_factor=2 if args.num_workers > 0 else None,
        num_workers=args.num_workers,    # Number of worker threads for data loading
        pin_memory=True if args.pin_memory==1 else False
    )

    dataloader_val = DataLoader(
        val_data,          # Your dataset instance
        batch_size=args.batch_size,     # Number of samples per batch
        shuffle=False,     # Shuffle data for training
        prefetch_factor=2 if args.num_workers > 0 else None,
        num_workers=args.num_workers,    # Number of worker threads for data loading
        pin_memory=True if args.pin_memory==1 else False
    )
    
    dataloader_test = DataLoader(
        test_data,          # Your dataset instance
        batch_size=args.batch_size,     # Number of samples per batch
        shuffle=False,     # Shuffle data for training
        prefetch_factor=2 if args.num_workers > 0 else None,
        num_workers=args.num_workers,    # Number of worker threads for data loading
        pin_memory=True if args.pin_memory==1 else False 
    )
    
    # NOTE: Assuming max_sequence_length=200 -- this is basically the maximal sequence length we allow
    assert train_data[0][0].shape[-2] <= 200, "max_sequence_length should be 200."
    
    logger.info(f"Creating model for input type: {args.input_type} with sequence length {train_data[0][0].shape[-2]} and feature dimension: {train_data[0][0].shape[-1]}")
    model = get_model(args=args,
                      max_sequence_length=200,
                      actual_sequence_length=train_data[0][0].shape[-2],
                      input_dim=train_data[0][0].shape[-1],
                      input_shape=train_data[0][0].shape).to(device=device)

    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total number of parameters in the model: {total_params}")
    args.total_params = total_params
    
    logger.info("Creating optimizer and scheduler.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Define the number of training steps
    num_training_steps = len(dataloader_train) * args.num_epochs  # Total training steps
    logger.info(f"Total number of training steps: {num_training_steps}, and warm-up steps: {int(0.1 * num_training_steps)}")
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% of steps for warm-up

    # Create the scheduler
    scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    
    criterion = torch.nn.BCELoss()
    
    
    random_number = str(int(time.time() * 1e6) % (10**10))
    args.random_number = random_number

    args.best_model_path = Path(args.best_model_path) / args.LLM / args.train_dataset
    logger.info(f"will save the best model in this folder: {args.best_model_path} with this file name: {args.random_number}.")
    logger.info("Starting wandb, project is LOS-Net.")
    logger.info("Starting training loop.")
    
    wandb.init(project="LOS-Net", config=args)
    
    train_model(logger=logger, model=model, dataloader_train=dataloader_train, dataloader_val=dataloader_val, dataloader_test=dataloader_test, criterion=criterion, optimizer=optimizer, scheduler=scheduler, args=args, device=device)
    
if __name__ == '__main__':
    main()