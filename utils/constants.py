
############### DC ###############

LIST_OF_MODELS_DC = ['EleutherAI/pythia-6.9b',
                     'EleutherAI/pythia-12b',
                     'huggyllama/llama-13b',
                     'huggyllama/llama-30b', 
                     'state-spaces/mamba-1.4b-hf']

LIST_OF_DATASETS_DC = ['WikiMIA_32', 'WikiMIA_64', 'BookMIA_128']

############### HD ###############

LIST_OF_MODELS_HD = ['mistralai/Mistral-7B-Instruct-v0.2', 'meta-llama/Meta-Llama-3-8B-Instruct', 'Qwen/Qwen2.5-7B-Instruct']


LIST_OF_DATASETS_HD = ['imdb', 'imdb_test',
                       'movies', 'movies_test',
                       'hotpotqa', 'hotpotqa_test',
                        'triviaqa', 'triviaqa_test',
                       'hotpotqa_with_context', 'hotpotqa_with_context_test']

##################################
MODEL_VOCAB_SIZES = {
    "mistralai/Mistral-7B-Instruct-v0.2": 32_000,
    "meta-llama/Meta-Llama-3-8B-Instruct": 128_256,
    "EleutherAI/pythia-6.9b": 50_434,
    "EleutherAI/pythia-12b": 50_690,
    "huggyllama/llama-13b": 32_000,
    "huggyllama/llama-30b": 32_000,
    "state-spaces/mamba-1.4b-hf": 50_280,
    "Qwen/Qwen2.5-7B-Instruct": 152_064,
}

PROBE_MODELS = [
    ## logit-based
    'LOS-Net',
    'ATP_R_MLP',
    'ATP_R_Transf',
    ]

MAXIMAL_VOCAB_SIZE = 1_000_000

LIST_OF_ALL_DATASETS = LIST_OF_DATASETS_DC + LIST_OF_DATASETS_HD
LIST_OF_ALL_MODELS = LIST_OF_MODELS_DC + LIST_OF_MODELS_HD