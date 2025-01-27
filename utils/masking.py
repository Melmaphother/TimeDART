import torch


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class SelfOnlyMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.ones(mask_shape, dtype=torch.bool).to(device)
            self._mask.fill_diagonal_(False)

    @property
    def mask(self):
        return self._mask


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

def generate_partial_mask(seq_len, mask_ratio):
    """Generate a partial mask with given mask ratio for the lower triangular part.
    
    Args:
        seq_len: Length of the sequence
        mask_ratio: Float between 0 and 1, portion of lower triangular elements to mask
                   0 -> causal mask (no masking in lower triangular)
                   1 -> self-only mask (all lower triangular masked)
    
    Returns:
        torch.Tensor: Boolean mask where True indicates positions to be masked
    """
    # Start with causal mask (upper triangular is True)
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    
    # Get lower triangular indices (excluding diagonal)
    lower_indices = torch.tril_indices(seq_len, seq_len, offset=-1)
    
    # Randomly select positions to mask in lower triangular part
    num_lower = lower_indices.shape[1]
    num_to_mask = int(num_lower * mask_ratio)
    mask_indices = torch.randperm(num_lower)[:num_to_mask]
    
    # Set selected lower triangular positions to True (masked)
    mask[lower_indices[0][mask_indices], lower_indices[1][mask_indices]] = True
    
    return mask

class ProbMask:
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
        ].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
