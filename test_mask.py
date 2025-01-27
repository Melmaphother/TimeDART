from utils.masking import generate_partial_mask, generate_causal_mask, generate_self_only_mask

# Causal mask (mask_ratio = 0)
mask = generate_partial_mask(4, 0.0)
causal_mask = generate_causal_mask(4)
print(causal_mask)

# Self-only mask (mask_ratio = 1)
mask = generate_partial_mask(4, 1.0)
self_only_mask = generate_self_only_mask(4)
print(self_only_mask)

# Partial mask (e.g., mask_ratio = 0.5)
mask = generate_partial_mask(4, 0.5)
print(mask)