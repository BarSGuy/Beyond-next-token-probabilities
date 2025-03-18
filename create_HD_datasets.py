
import torch
from utils.logger import get_logger
from transformers import set_seed
from utils.LLM_helpers import load_model_and_validate_gpu
from utils.args import parse_args_HD
from utils.datasets_HD_helper import *
from utils.generation_utils import *


def process_and_save_model_io(args, data, model, tokenizer, device, model_name, wrong_labels, labels, do_sample=False, output_LOS=True,
                           temperature=1.0,
                           top_p=1.0, max_new_tokens=100, stop_token_id=None, output_ACT=True, logger=False):


    time_so_far = 0
    prompts_so_far = 0
    for index, prompt in tqdm(enumerate(data), desc="Processing Prompts"):
        if index not in args.actual_indices:
            logger.info(f"skiping index {index} as it is not in the actual indices")
            continue
        logger.info(f"Processing index {index}")

        start_time = time.time()

        model_input = tokenize(prompt, tokenizer, model_name).to(device)


        with torch.no_grad():
            model_output = generate(model_input, model, model_name, do_sample, max_new_tokens=max_new_tokens,
                                    top_p=top_p, temperature=temperature, stop_token_id=stop_token_id, tokenizer=tokenizer, output_hidden_states=True)

        answer = tokenizer.decode(model_output['sequences'][0][len(model_input[0]):])

        if output_LOS:
            logger.info(f"Computing canonized_logits for index {index}")
            canonized_logits = extract_scores(model_output=model_output, model_input=model_input, take_top_k=args.take_top_k)
            logger.info(f"Canonized logits shape is {canonized_logits.shape}")


        
        logger.info(f"Computing correctness for index {index}")
        res = compute_correctness([data[index]], args.dataset, args.LLM, [labels[index]], model, [answer], tokenizer, wrong_labels)
        
        correctness = res['correctness'][0]
        logger.info(f"Correctness: {correctness}")
        
        save_raw_data(LLM=args.LLM, dataset_name=args.dataset, base_dir=args.base_raw_data_dir, probs_output=canonized_logits, idx=index, label=correctness)


        end_time = time.time()
        delta_time = end_time - start_time
        time_so_far += delta_time
        prompts_so_far += 1
        time_per_prompt = time_so_far / prompts_so_far
        time_left = time_per_prompt * (len(args.actual_indices) - prompts_so_far - 1)
        
        logger.info(f"\n\n\n Prompt number {prompts_so_far}, Time left: {time_left / 3600} hours \n\n Number of prompts left = {len(args.actual_indices) - prompts_so_far - 1}\n\n\n")





def main():
    # Get the logger instance
    logger = get_logger()
    partition = {
        1: torch.arange(0, 1000),
        2: torch.arange(1000, 2000),
        3: torch.arange(2000, 3000),
        4: torch.arange(3000, 4000),
        5: torch.arange(4000, 5000),
        6: torch.arange(5000, 6000),
        7: torch.arange(6000, 7000),
        8: torch.arange(7000, 8000),
        9: torch.arange(8000, 9000),
        10: torch.arange(9000, 10000),
    }

    """
    Main function to load the model, dataset, and process the data.
    """
    # Parse command-line arguments
    args = parse_args_HD()
    logger.info(f"Parsed Arguments: {vars(args)}")
    if args.chunk != -1:
        logger.info(f"Using chunk {args.chunk} to select the indices.")
        args.actual_indices = partition[args.chunk]
    else:
        logger.info(f"Using all indices to select the indices.")
        args.actual_indices = torch.arange(0, 10000)
    logger.info(f"Working with the indices: {args.actual_indices}")
    # Set the random seed for reproducibility
    set_seed(0)
    
    # Load the specified model and tokenizer, ensuring GPU compatibility
    logger.info(f"Loading model: {args.LLM}")
    llm, tokenizer = load_model_and_validate_gpu(args.LLM)
    
    # Determine the device to use for computation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    stop_token_id = None
    if 'instruct' not in args.LLM.lower():
        stop_token_id = tokenizer.encode('\n', add_special_tokens=False)[-1]
        logger.info(f"The model '{args.LLM}' is not an Instruct model. Generation will stop at the token ID corresponding to a newline ('\\n'): {stop_token_id}.")
    else:
        logger.info(f"The model '{args.LLM}' is identified as an Instruct model. No specific stop token will be used (stop_token_id is set to None).")
        
    all_questions, context, labels, max_new_tokens, origin, preprocess_fn, stereotype, type_, wrong_labels = load_data(args, args.dataset)
    
    
    dataset_size = args.n_samples
    
    if dataset_size:
        logger.info(f"Using a subset of {dataset_size} samples from the dataset .")
        all_questions = all_questions[:dataset_size]
        labels = labels[:dataset_size]
        if 'mnli' in args.dataset:
            origin = origin[:dataset_size]
        if 'winogrande' in args.dataset:
            wrong_labels = wrong_labels[:dataset_size]
    
    if preprocess_fn:
        all_questions = preprocess_fn(args, args.LLM, all_questions, labels)
    logger.info(f"Starting to generate model answers.")
    process_and_save_model_io(args, all_questions, llm, tokenizer, device, args.LLM, max_new_tokens=max_new_tokens, stop_token_id=stop_token_id, wrong_labels=wrong_labels, labels=labels, logger=logger)
    


if __name__ == "__main__":
    main()
