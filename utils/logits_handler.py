import torch

def compute_logprobs_with_selection_and_ranks(input_ids, raw_logits, take_top_k=1000000, HD=False):
    # NOTE: log_probs[0] gives the probability for the next token -- input_ids[1], therefore:
    # for sequence of length n+1, the following holds: 
    # log_probs[-1] is log prob for first word -- we don't have this!
    ## log_probs[0] -> input_ids[1] (log prob for second word)
    ## log_probs[1] -> input_ids[2] (log prob for third word)
    ## ...
    ## log_probs[n-1] -> input_ids[n] (log prob for last word -- ``end of sentence'')
    # log_probs[n] we don't take! (log prob for the word after ``end of sentence'')
    log_probs = torch.nn.functional.log_softmax(raw_logits, dim=-1)
    if HD:
        log_probs_of_selected_tokens = log_probs[0][range(len(input_ids[0])), input_ids[0]]  
    else:
        log_probs_of_selected_tokens = log_probs[0][range(len(input_ids[0][1:])), input_ids[0][1:]]  
    
    sorted_log_probs, _ = torch.sort(log_probs, dim=-1, descending=True)
    if HD:
        sorted_log_probs = sorted_log_probs[0]
    else:
        sorted_log_probs = sorted_log_probs[0][:-1, :]
    
    
    sorted_log_probs_inverted, _ = torch.sort(log_probs, dim=-1, descending=False)
    sorted_log_probs_inverted = sorted_log_probs_inverted[0][:-1, :]
    
    
    def find_column_indices(sorted_logits, logit_of_selected_tokens):
        """
        Finds the column index in `sorted_logits` where each value of `logit_of_selected_tokens` appears in its row.
        If a value appears multiple times in a row, the first index is returned.
        
        Args:
            sorted_logits (torch.Tensor): A tensor of shape [n, vocab_size].
            logit_of_selected_tokens (torch.Tensor): A tensor of shape [n].
            
        Returns:
            torch.Tensor: A tensor of shape [n], containing the column indices.
        """
        # Expand logit_of_selected_tokens to match the shape of sorted_logits
        expanded_tokens = logit_of_selected_tokens.unsqueeze(1)
        
        # Compare for equality along the rows
        matches = sorted_logits == expanded_tokens
        
        # Convert boolean to int64 before applying argmax
        indices = torch.argmax(matches.to(dtype=torch.int64), dim=1)
        
        return indices
    
    ranks = find_column_indices(sorted_logits=sorted_log_probs, 
                                logit_of_selected_tokens=log_probs_of_selected_tokens)
    sorted_log_probs_with_selected = torch.cat((sorted_log_probs[:, :take_top_k],  
                                             log_probs_of_selected_tokens.unsqueeze(1), ranks.reshape(-1, 1)), dim=1)
    return sorted_log_probs_with_selected

