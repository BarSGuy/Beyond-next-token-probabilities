# sample.py
import argparse
import os
import pickle
import time

from sklearn.metrics import auc, roc_curve
from utils.generation_utils import tokenize
from tqdm import tqdm
from transformers import set_seed
import torch
import numpy as np

from semantic_entropy import EntailmentDeberta, EntailmentLlama, get_semantic_ids, logsumexp_by_id, predictive_entropy_rao
from utils.LLM_helpers import load_model_and_validate_gpu
from utils.constants import LIST_OF_DATASETS_HD, LIST_OF_MODELS_HD
from utils.datasets_HD_helper import load_data
from utils.generation_utils import compute_correctness
from utils.logger import get_logger

def parse_args_SE():

    parser = argparse.ArgumentParser(
        description="Parse arguments for Semantic Entropy Evaluation."
    )

    parser.add_argument(
        "--LLM",
        choices=LIST_OF_MODELS_HD,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Pretrained model which will generate multiple candidate responses."
    )
    
    parser.add_argument(
        "--entailment_model",
        default="deberta",
        help="Pretrained entailment model to get semantic clustering."
    )

    parser.add_argument(
        "--strict_entailment",
        action="store_true"
    )

    parser.add_argument(
        "--entail_with_question",
        action="store_true"
    )

    parser.add_argument(
        "--dataset",
        choices=LIST_OF_DATASETS_HD,
        default='movies_test',
        help="Dataset to be used."
    )
    
    parser.add_argument(
        "--n_samples", 
        type=int, 
        help='number of examples to use', 
        default=10_000
    )

    parser.add_argument(
        "--start_index", 
        type=int, 
        help='first index to consider', 
        default=0
    )

    parser.add_argument(
        "--end_index", 
        type=int, 
        help='last index to consider', 
        default=10_000
    )
    
    parser.add_argument(
        "--num_generations",
        type=int,
        help="number of different generations to run on which to then calculate semantic entropy",
        default=10
    )

    parser.add_argument(
        "--seed", 
        type=int, 
        default=0,
        help="seed (default: 0)."
    )

    parser.add_argument(
        "--output_folder", 
        type=str, 
        default='./semantic_entropy_results/',
        help="where to save the results"
    )

    return parser.parse_args()

def generate(
        prompt,
        model,
        model_name,
        tokenizer,
        device,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        max_new_tokens=100,
        stop_token_id=None,
        additional_kwargs=None,
        batch_n=None):

    model_input = tokenize(prompt, tokenizer, model_name).to(device)
    if batch_n is not None and batch_n > 1:
        model_input = model_input.repeat(batch_n, 1)

    if stop_token_id is not None:
        eos_token_id = stop_token_id
    else:
        eos_token_id = None

    with torch.no_grad():
        model_output = model.generate(
                                    model_input,
                                    max_new_tokens=max_new_tokens,
                                    output_hidden_states=False,
                                    output_scores=True,
                                    return_dict_in_generate=True,
                                    do_sample=do_sample,
                                    temperature=temperature,
                                    top_k=top_k,
                                    top_p=top_p,
                                    eos_token_id=eos_token_id,
                                    **(additional_kwargs or {}))
        transition_scores = model.compute_transition_scores(
            model_output.sequences, model_output.scores, normalize_logits=True)

    responses = list()
    for i in range(model_output.sequences.shape[0]):

        full_answer = tokenizer.decode(model_output.sequences[i], skip_special_tokens=True)
        
        # Remove input from answer.
        input_data_offset = len(tokenizer.decode(model_input[i], skip_special_tokens=True))
        answer = full_answer[input_data_offset:]
        
        # Remove stop_words from answer.
        sliced_answer = answer
        stop_at = len(answer)
        stop_sequences = [tokenizer.eos_token]
        if eos_token_id is not None:
            stop_sequences.append(tokenizer.decode([eos_token_id])[0])
        for stop in stop_sequences:
            if answer.endswith(stop):
                stop_at = len(answer) - len(stop)
                sliced_answer = answer[:stop_at]
                break
        if not all([stop not in sliced_answer for stop in stop_sequences]):
            error_msg = '\n\n[!]: Stop words not removed successfully!'
            error_msg += f'Answer: >{answer}< '
            error_msg += f'Sliced Answer: >{sliced_answer}<\n\n'
            print(error_msg)

        # Remove whitespaces from answer (in particular from beginning.)
        sliced_answer = sliced_answer.strip()

        # Get the number of tokens until the stop word comes up.
        # Note: Indexing with `stop_at` already excludes the stop_token.
        # Note: It's important we do this with full answer, since there might be
        # non-trivial interactions between the input_data and generated part
        # in tokenization (particularly around whitespaces.)
        token_stop_index = tokenizer(full_answer[:input_data_offset + stop_at], return_tensors="pt")['input_ids'].shape[1]
        n_input_token = len(model_input[i])
        n_generated = token_stop_index - n_input_token
        if n_generated == 0:
            n_generated = 1

        log_likelihoods = [score.item() for score in transition_scores[i]]
        if len(log_likelihoods) == 1:
            log_likelihoods = log_likelihoods
        else:
            log_likelihoods = log_likelihoods[:n_generated]
        if len(log_likelihoods) == 0:
            raise ValueError
    
        # Also get answer as commonly done in the rest of the project in case we
        # are generating the low temperature one ...
        # NB: This is important for computing correctness
        if not do_sample:
            assert i == 0
            answer = tokenizer.decode(model_output['sequences'][0][len(model_input[i]):])

        prefix = 'GREEDY ' if not do_sample else ''
        print(f'\n{prefix}ANSWER:\n', answer, '\n')

        responses.append((answer,log_likelihoods))

    return responses

def sample_responses(prompt, model, tokenizer, model_name, device, max_new_tokens, stop_token_id, args):
    responses = generate(
        prompt,
        model,
        model_name,
        tokenizer,
        device,
        do_sample=False,
        temperature=0.0,
        top_k=None,
        top_p=None,
        max_new_tokens=max_new_tokens,
        stop_token_id=stop_token_id)
    t_0 = time.time()
    responses += generate(
        prompt,
        model,
        model_name,
        tokenizer,
        device,
        do_sample=True,
        temperature=1.0,  # Stochastic generation parameters as in https://www.nature.com/articles/s41586-024-07421-0
        top_k=50,
        top_p=0.9,
        max_new_tokens=max_new_tokens,
        stop_token_id=stop_token_id,
        batch_n=args.num_generations)
    gen_time = time.time() - t_0
    return responses, gen_time


def process_and_calculate_entropy_scores(
        index,
        logger,
        args,
        data,
        model,
        tokenizer,
        entailment_model,
        device,
        model_name,
        max_new_tokens,
        stop_token_id,
        wrong_labels,
        labels):

    logger.info(f"Processing index {index}")
    prompt = data[index]
    print(f"\nPROMPT [{index}]:\n", prompt, '\n')
    print("\nEXPECTED ANSWER:\n", labels[index], '\n')

    logger.info("Generating possible responses...")
    responses, gen_time = sample_responses(prompt, model, tokenizer, model_name, device, max_new_tokens, stop_token_id, args)
    answers = [r[0] for r in responses]
    log_liks = [r[1] for r in responses]

    logger.info(f"Computing correctness for generated ``greedy`` response...")
    res = compute_correctness([prompt], args.dataset, model_name, [labels[index]], model, [answers[0]], tokenizer, [wrong_labels[index]] if wrong_labels is not None else None)
    label = min(res['correctness'])
    logger.info(f"Label: {label} for answer {answers[0]}")
    
    logger.info("Clustering and calculating entropy...")

    t_0 = time.time()
    semantic_ids = get_semantic_ids(
            answers[1:], model=entailment_model,
            strict_entailment=args.strict_entailment, example=(prompt if args.entail_with_question else None))
    cluster_time = time.time() - t_0
    log_liks_agg = [np.mean(log_lik) for log_lik in log_liks[1:]]  # Length normalization of generation probabilities.
    assert len(semantic_ids) == len(log_liks_agg) == args.num_generations
    log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')
    semantic_entropy = predictive_entropy_rao(log_likelihood_per_semantic_id)
    se_time = time.time() - t_0

    print("\nRUNTIMES\n")
    print(f"  - auxiliary generation:                 {gen_time:.2f}")
    print(f"  - clustering:                           {cluster_time:.2f}")
    print(f"  - SE calculation (includes clust.):     {se_time:.2f}")
    print("-----------------------------------------------------------------------")
    print(f"overall:                                  {gen_time+se_time:.2f}")
    
    return (semantic_entropy, semantic_ids, log_liks_agg, label, answers[0], labels[index], gen_time, cluster_time, se_time)


def main():

    # Parse args
    logger = get_logger()
    args = parse_args_SE()
    logger.info(f"Parsed Arguments: {vars(args)}")
    os.makedirs(args.output_folder, exist_ok=True)
    exp_name = f"semantic_entropy__{args.dataset}__{args.n_samples}__{args.LLM.replace('/', '_')}__{args.num_generations}__{args.entailment_model.replace('/', '_')}__{args.strict_entailment}__{args.entail_with_question}__{args.seed}"
    os.makedirs(os.path.join(args.output_folder, exp_name), exist_ok=True)

    # Set the random seed for reproducibility
    set_seed(args.seed)
    
    # Load the specified model and tokenizer, ensuring GPU compatibility
    logger.info(f"Loading model: {args.LLM}")
    llm, tokenizer = load_model_and_validate_gpu(args.LLM)
    logger.info('Model loading complete.')

    logger.info('Beginning loading for entailment model.')
    if args.entailment_model == 'deberta':
        entailment_model = EntailmentDeberta()
    # TODO: support Llama models
    # elif 'llama' in args.entailment_model.lower():
    #     entailment_model = EntailmentLlama(args.entailment_cache_id, args.entailment_cache_only, args.entailment_model)
    else:
        raise NotImplementedError
    logger.info('Entailment model loading complete.')
    
    # Determine the device to use for computation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    stop_token_id = None
    if 'instruct' not in args.LLM.lower():
        stop_token_id = tokenizer.encode('\n', add_special_tokens=False)[-1]
        logger.info(f"The model '{args.LLM}' is not an Instruct model. Generation will stop at the token ID corresponding to a newline ('\\n'): {stop_token_id}.")
    else:
        logger.info(f"The model '{args.LLM}' is identified as an Instruct model. No specific stop token will be used (stop_token_id is set to None).")
        
    # Load data
    all_questions, context, labels, max_new_tokens, origin, preprocess_fn, stereotype, type_, wrong_labels = load_data(args, args.dataset)
    dataset_size = args.n_samples
    logger.info(f"Using a subset of {dataset_size} samples from the dataset.")
    all_questions = all_questions[:dataset_size]
    labels = labels[:dataset_size]
    if 'mnli' in args.dataset:
        origin = origin[:dataset_size]
    if 'winogrande' in args.dataset:
        wrong_labels = wrong_labels[:dataset_size]
    
    if preprocess_fn:
        logger.info(f"Applying preprocessing to input questions...")
        all_questions = preprocess_fn(args, args.LLM, all_questions, labels)
    
    logger.info(f"Starting to generate model answers.")

    pred_dict = dict()
    for index in tqdm(range(len(all_questions)), desc="Processing Prompts"):
        if args.start_index <= index < args.end_index:
            pred_dict[index] = process_and_calculate_entropy_scores(
                index,
                logger,
                args,
                all_questions,
                llm,
                tokenizer,
                entailment_model,
                device,
                args.LLM,
                max_new_tokens,
                stop_token_id,
                wrong_labels,
                labels)
            # Save ...
            with open(os.path.join(args.output_folder, exp_name, f'{index}.pkl'), 'wb') as handle:
                pickle.dump(pred_dict[index], handle)
    
    all_labels = list()
    all_predictions = list()
    gen_times = list()
    clust_times = list()
    se_times = list()
    times = list()
    for index in pred_dict:
        all_labels.append(pred_dict[index][3])
        all_predictions.append(-pred_dict[index][0])
        gen_times.append(pred_dict[index][6])
        clust_times.append(pred_dict[index][7])
        se_times.append(pred_dict[index][8])
        times.append(pred_dict[index][6]+pred_dict[index][8])
    fpr, tpr, _ = roc_curve(np.array(all_labels, dtype=bool), np.array(all_predictions))
    rocauc = auc(fpr, tpr)
    print(f'\n\nAUROC: {rocauc:.4f}')
    print("RUNTIMES:")
    print(f"  - auxiliary generation:                 {np.mean(gen_times):.2f} ± {np.std(gen_times):.2f}")
    print(f"  - clustering:                           {np.mean(clust_times):.2f} ± {np.std(clust_times):.2f}")
    print(f"  - SE calculation (includes clust.):     {np.mean(se_times):.2f} ± {np.std(se_times):.2f}")
    print("-----------------------------------------------------------------------")
    print(f"overall:                                  {np.mean(times):.2f}±{np.std(times):.2f}")

    exp_name = f"semantic_entropy__{args.dataset}__{args.n_samples}__{args.LLM.replace('/', '_')}__{args.num_generations}__{args.entailment_model.replace('/', '_')}__{args.strict_entailment}__{args.entail_with_question}__{args.seed}__{args.start_index}__{args.end_index}.pkl"
    with open(os.path.join(args.output_folder, exp_name), 'wb') as handle:
        pickle.dump((pred_dict, rocauc), handle)

if __name__ == '__main__':
    main()