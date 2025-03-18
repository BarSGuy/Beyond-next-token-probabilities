from datasets import load_dataset
from utils.args import parse_args_DC
import torch

from transformers import set_seed
from utils.LLM_helpers import load_model_and_validate_gpu
from utils.file_io import save_raw_data
from utils.constants import *
from tqdm import tqdm 
from utils.logits_handler import compute_logprobs_with_selection_and_ranks
from utils.logger import get_logger




def get_llm_tokenizer(args):
    return load_model_and_validate_gpu(args.LLM)

def whitebox_mri(prompt, llm, tokenizer, device):
    # Tokenize the input text
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids']
    
    # Generate data from the model
    with torch.no_grad():
        output = llm(input_ids.to(device), output_hidden_states=True)
    raw_logits = output.logits.cpu()

    return raw_logits, input_ids, None

def llm_mri(prompt, llm, tokenizer, device):
    raw_logits, input_ids, _ = whitebox_mri(prompt, llm, tokenizer, device)
    return raw_logits, input_ids, None


def convert_dataset_to_list(dataset_name, dataset):
    """
    Convert a dataset to a list of dictionaries for processing.
    
    Args:
    -----
    args: argparse.Namespace
        Command-line arguments containing dataset configuration.
    dataset: datasets.Dataset
        The loaded Hugging Face dataset with 'input' and 'label' features.
    
    Returns:
    --------
    list of dict:
        A list where each element is a dictionary with 'text' and 'label' keys.
    """
    result = []
    if "WikiMIA" in dataset_name:
        for i in range(dataset.num_rows):
            row = dataset[i]
            result.append({
                'text': row['input'],
                'label': row['label']
            })
    elif "BookMIA" in dataset_name:
        for i in range(dataset['train'].num_rows):
            row = dataset['train'][i]
            result.append({
                'text': row['snippet'],
                'label': row['label']
            })
    elif "mimir" in dataset_name:
        for i in range(dataset.num_rows):
            result.append({
                'text': dataset['member'][i],
                'label': 1 # 1 means contaminated
            })
            # 0 means not contaminated
            result.append({
                'text': dataset['nonmember'][i],
                'label': 0
            })
    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' is not implemented.")
    return result


def load_data(dataset_name):
    if "WikiMIA" in dataset_name:
        split = 'WikiMIA_length' + dataset_name.split('_')[-1]
        dataset_name = 'swj0419/WikiMIA'
        dataset = load_dataset(dataset_name, split=split)
        print(f"Loading dataset: {dataset_name}, split: {split}")
    elif dataset_name == "BookMIA" or dataset_name == "BookMIA_128":
        dataset_name = 'swj0419/BookMIA'
        dataset = load_dataset(dataset_name)
        print(f"Loading dataset: {dataset_name}")
    elif "mimir" in dataset_name:
        if 'github' in dataset_name:
            name = 'github'
        elif 'arxiv' in dataset_name:
            name = 'arxiv'
        elif 'dm_mathematics' in dataset_name:
            name = 'dm_mathematics'
        elif 'hackernews' in dataset_name:
            name = 'hackernews'
        elif 'pile_cc' in dataset_name:
            name = 'pile_cc'
        elif 'pubmed_central' in dataset_name:
            name = 'pubmed_central'
        elif 'wikipedia_(en)' in dataset_name:
            name = 'wikipedia_(en)'
        
        dataset = load_dataset("iamgroot42/mimir", name, split="ngram_13_0.8")
    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' is not implemented")
    return dataset
    
    

def main():
    # Get the logger instance
    logger = get_logger()

    """
    Main function to load the model, dataset, and process the data.
    """
    # Parse command-line arguments
    args = parse_args_DC()
    
    logger.info(f"Parsed Arguments: {vars(args)}")

    # Set the random seed for reproducibility
    set_seed(0)
    
    # Load the specified model and tokenizer, ensuring GPU compatibility
    
    logger.info(f"Loading model: {args.LLM}")
    llm, tokenizer = get_llm_tokenizer(args)
    
    # Determine the device to use for computation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    
    # Load the specified dataset
    dataset = load_data(args.dataset)
    
    # Convert the dataset to a list of dictionaries
    logger.info("Converting dataset to list format...")
    
    data = convert_dataset_to_list(args.dataset, dataset)
    logger.info(f"The dataset has {len(data)} rows.")
    
    
    
    logger.info("Processing dataset to extract logits...")
    if llm is not None:
        llm.eval()
    for idx, row in enumerate(tqdm(data, desc="Extracting logits", unit="row")):
        prompt = row['text']
        if 'BookMIA' in args.dataset:
            split = args.dataset.split('_')[-1]
            if idx == 0:
                logger.info(f"BookMIA dataset: Using the first 128 words to create the split {split}.")
            prompt = ' '.join(prompt.split(' ')[:int(split)])
            
        raw_logits, input_ids, _ = llm_mri(prompt, llm, tokenizer, device)

        canonized_logits = compute_logprobs_with_selection_and_ranks(input_ids=input_ids, raw_logits=raw_logits, take_top_k=args.take_top_k)
        
        if idx % 10 == 0:
            logger.info(f"Sample {idx}:\nPrompt: {prompt}\nLogits shape: {raw_logits.shape}\n")

        save_raw_data(LLM=args.LLM, dataset_name=args.dataset, base_dir=args.base_raw_data_dir, 
                  probs_input=canonized_logits, 
                  idx=idx, label=row['label'])


if __name__ == "__main__":
    main()
