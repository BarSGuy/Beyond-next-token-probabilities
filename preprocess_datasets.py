from utils.dataset_preprocess import preprocess_data
from utils.constants import LIST_OF_DATASETS_DC
from pathlib import Path
from utils.logger import get_logger
from utils.args import parse_args_pre_process


def process_datasets(args, logger):
    """Preprocesses datasets and loads them based on the task type."""
    
    if args.input_type == 'LOS':
        logger.info(f"Initializing dataset processing, loading raw data, and saving only TOP-{args.topk_preprocess} elements of the {args.input_output_type} sorted TDS.")
    else:
        raise ValueError("Invalid input type.")
    logger.info(f"Loading raw data from: {args.base_raw_data_dir}")
    
    # Define output directories
    output_dir = Path(args.base_pre_processed_data_dir) / args.LLM / args.dataset

    logger.info(f"Preprocessed data will be stored at: {output_dir}")
    
    logger.info(f"Configured maximum sequence length (N_max) for '{args.dataset}': {args.N_max}")
    
    logger.info(f"Starting data preparation for model '{args.LLM}' using training dataset '{args.dataset}'.")

    logger.info(f"Preprocessing dataset: {args.dataset}")
    preprocess_data(
        args=args,
        input_type=args.input_type,
        raw_data_dir=args.base_raw_data_dir,
        LLM=args.LLM,
        dataset_name=args.dataset,
        output_dir=output_dir,
        N_max=args.N_max,
        topk_preprocess=args.topk_preprocess,
        input_output_flag=args.input_output_type
    )
    logger.info("Dataset preprocessing completed successfully.")


    



if __name__ == "__main__":
    """Main function to preprocess data and load datasets based on task type."""
    # Initialize logger
    logger = get_logger()
    
    # Parse command-line arguments
    args = parse_args_pre_process()
    logger.info("Starting the data processing pipeline.")
    logger.info(f"Parsed Arguments: {vars(args)}")
    
    logger.info(f"Preparing data for model '{args.LLM}' using training dataset '{args.dataset}'.")
    # Process datasets
    process_datasets(args, logger)
    
    
    