import torch


def generate_causal_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    # in nn.MultiheadAttention
    # binary mask is used and True means not allowed to attend
    # so we use triu instead of tril
    return mask


def generate_self_only_mask(seq_len):
    # Initialize a matrix with all True values (no attention allowed)
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
    # Set the diagonal to False (allow self-attention)
    mask.fill_diagonal_(False)
    return mask
