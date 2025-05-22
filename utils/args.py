import argparse
from utils.constants import LIST_OF_MODELS_DC, LIST_OF_DATASETS_DC, PROBE_MODELS, MAXIMAL_VOCAB_SIZE, LIST_OF_DATASETS_HD, LIST_OF_MODELS_HD, LIST_OF_ALL_MODELS, LIST_OF_ALL_DATASETS


def parse_args_DC():
    """
    Parse command-line arguments for the script.
    
    Returns:
    --------
    argparse.Namespace:
        Parsed command-line arguments with dataset, model, and split details.
    """
    parser = argparse.ArgumentParser(description="Generate model LOS and labels from a specified dataset.")
    
    # Argument for selecting the model
    parser.add_argument(
        "--LLM",
        choices=LIST_OF_MODELS_DC,
        default="huggyllama/llama-13b",
        help="Pretrained model to use for generating LOS."
    )
    
    # Argument for selecting the dataset
    parser.add_argument(
        "--dataset",
        choices=LIST_OF_DATASETS_DC,
        default='BookMIA_128',
        help="Dataset to be processed."
    )
    
    
    parser.add_argument(
        "--take_top_k",
        type=int,
        default=MAXIMAL_VOCAB_SIZE,
        help="Top-K to use when extracting the raw dataset -- should be max over all vocab sizes (default: 1_000_000)."
    )
    
    parser.add_argument(
        "--base_raw_data_dir",
        type=str,
        # default='./raw_data',
        default='/home/guy_b/big-storage/raw_data',
        help="Base directory for saving raw data."
    )
    
    return parser.parse_args()


def parse_args_HD():
    """
    Parse command-line arguments for the script.
    
    Returns:
    --------
    argparse.Namespace:
        Parsed command-line arguments with dataset, model, and split details.
    """
    parser = argparse.ArgumentParser(description="Generate model LOS and labels from a specified dataset.")
    
    # Argument for selecting the model
    parser.add_argument(
        "--LLM",
        choices=LIST_OF_MODELS_HD,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Pretrained model to use for generating LOS."
    )
    
    # Argument for selecting the dataset
    parser.add_argument(
        "--dataset",
        choices=LIST_OF_DATASETS_HD,
        default='hotpotqa',
        help="Dataset to be processed."
    )
    parser.add_argument(
        "--n_samples", 
        type=int, 
        help='number of examples to use', 
        default=10_000
        )
    parser.add_argument(
        "--chunk", 
        type=int, 
        default=6,
        )
    
    parser.add_argument(
        "--base_raw_data_dir",
        type=str,
        # default='./raw_data',
        default='/mnt/storage/guy_b/LOS_NET',
        help="Base directory for saving raw data."
    )
    
    parser.add_argument(
        "--take_top_k",
        type=int,
        default=MAXIMAL_VOCAB_SIZE,
        help="Top-K to use when extracting the raw dataset -- should be max over all vocab sizes (default: 1_000_000)."
    )
    

    
    return parser.parse_args()


def parse_args_main():
    """
    Parses command-line arguments for training, model, dataset, and logging configurations.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Parse arguments for training and evaluation pipeline."
    )
    parser.add_argument(
        "--LLM",
        choices=LIST_OF_ALL_MODELS,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Pretrained model of which its logits are used."
    )
    
    parser.add_argument(
        "--base_raw_data_dir",
        type=str,
        # default='./raw_data',
        default='/home/guy_b/big-storage/raw_data',
        help="Base directory for saving raw data."
    )
    parser.add_argument(
        "--train_dataset", 
        type=str,
        default="imdb",
        help="Train dataset (default: 'WikiMIA_32')."
    )
    
    parser.add_argument(
        "--test_dataset", 
        type=str,
        default="imdb_test",
        help="Test dataset (default: 'WikiMIA_32')."
    )

    
    parser.add_argument(
        "--base_pre_processed_data_dir", 
        type=str, 
        default='/home/guy_b/LOS-Net/pre_processed_data',
        help="Base directory for saving pre processed data."
    )
    
    parser.add_argument(
        "--probe_model", 
        choices=PROBE_MODELS, 
        default="LOS-Net",
        # default="ATP_R_MLP",
        # default="ATP_R_Transf",
        help="The probing model to use (default: 'LOS-Net')."
    )
    
    parser.add_argument(
        "--topk_preprocess", 
        type=int, 
        # default=MAXIMAL_VOCAB_SIZE,
        default=1_000,
        help="Top-K to load the preprocessed dataset -- should be max over all vocab sizes for DCD, or 1_000 for HD (default: 1_000)."
    )
    
    parser.add_argument(
        "--input_output_type", 
        type=str, 
        default="output",
        help="Usage of input or output."
    )
    
    parser.add_argument(
        "--topk_dim", 
        type=int, 
        default=1000,
        help="Top-K dimension to actually use for the model, should be 1000 for HD, or between 10 to 1000 for ablation study, see paper (default: 1,000,000). For DCD should be "
    )
    
    parser.add_argument(
        "--N_max", 
        type=int, 
        default=100,
        help="Maximal sequence length (default: 100)."
    )
    
    parser.add_argument(
        "--num_folds", 
        type=int, 
        default=5,
        help="Number of folds to run (default: 5)."
    )
        
    parser.add_argument(
        "--fold_to_run", 
        type=int, 
        default=0,
        help="fold to run (default: 0)."
    )
    
    parser.add_argument(
        "--input_type", 
        type=str, 
        choices=["LOS"], 
        default="LOS",
        help="Input type to use for the model."
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=0,
        help="seed (default: 0)."
    )
    
    parser.add_argument(
        "--cuda_idx", 
        type=int, 
        default=0,
        help="cuda index (default: 0)."
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=64,
        help="batch size (default: 0)."
    )
    
    parser.add_argument(
        "--hidden_dim", 
        type=int, 
        default=128,
        help="hidden dimension (default: 128)."
    )
    
    parser.add_argument(
        "--heads", 
        type=int, 
        default=8,
        help="number of heads (default: 4)."
    )
    
    parser.add_argument(
        "--dropout", 
        type=float, 
        default=0.3,
        help="dropout to use (default: 0.3)."
    )
    
    parser.add_argument(
        "--num_layers", 
        type=int, 
        default=2,
        help="number of layers to use (default: 1)."
    )
    
    parser.add_argument(
        "--pool", 
        type=str, 
        default='cls',
        help="pooling (default: cls)."
    )
    
    parser.add_argument(
        "--patience", 
        type=int, 
        default=100,
        help="patience (default: 30)."
    )
    
        
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=100,
        help="epochs (default: 100)."
    )
    
    parser.add_argument(
        "--best_model_path", 
        type=str, 
        default="saved_models",
        help="Path to save the best model (default: best_model)."
    )
    
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.0001,
        help="Learning rate for the optimizer (default: 0.0001)."
    )
    
    parser.add_argument(
        "--rank_encoding", 
        type=str, 
        default="scale_encoding",
        choices=["scale_encoding", "one_hot_encoding"], 
        help="The way to use the rank encoding."
    )
    
    
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.001,
        help="Weight decay for regularization (default: 0.001)."
    )
    
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=4,
        help="Number of workers (default: 4)."
    )
    parser.add_argument(
        "--pin_memory", 
        type=int, 
        default=1,
        help="pin memory (default: 4)."
    )
    



    return parser.parse_args()


def parse_args_pre_process():
    """
    Parse command-line arguments for the script.
    
    Returns:
    --------
    argparse.Namespace:
        Parsed command-line arguments with dataset, model, and split details.
    """
    parser = argparse.ArgumentParser(description="Generate model LOS and labels from a specified dataset.")
    
    # Argument for selecting the model
    parser.add_argument(
        "--LLM",
        choices=LIST_OF_ALL_MODELS,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Pretrained model to use for generating LOS."
    )
    
    # Argument for selecting the dataset
    parser.add_argument(
        "--dataset",
        choices=LIST_OF_ALL_DATASETS,
        default='movies',
        help="Dataset to be processed."
    )
    
    parser.add_argument(
        "--base_raw_data_dir",
        type=str,
        default='/home/guy_b/big-storage/raw_data',
        help="Base directory for saving raw data."
    )
    
    parser.add_argument(
        "--base_pre_processed_data_dir", 
        type=str, 
        default='/home/guy_b/LOS-Net/pre_processed_data',
        help="Base directory for saving pre processed data."
    )
    
    
    parser.add_argument(
        "--topk_preprocess", 
        type=int, 
        default=1_000,
        help="Top-K to use when preprocessing the dataset -- should be max for DCD and 1000 for HD (default: 1_000_000)."
    )
    
    parser.add_argument(
        "--input_output_type", 
        type=str, 
        default="output",
        help="Usage of input or output."
    )
    
    
    parser.add_argument(
        "--N_max", 
        type=int, 
        default=100,
        help="Maximal sequence length (default: 100)."
    )
    
    parser.add_argument(
        "--input_type", 
        type=str, 
        choices=["LOS"], 
        default="LOS",
        help="Input type to use for the model."
    )
    


    return parser.parse_args()

