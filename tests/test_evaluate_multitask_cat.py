"""Quick test: evaluate_multitask force_atom cat with variable N_max."""

import torch

# Simulate the bug: cat padded [B, N, 2] with different N
batch1_logits = torch.randn(4, 50, 2)
batch2_logits = torch.randn(3, 100, 2)
try:
    torch.cat([batch1_logits, batch2_logits])
    raise AssertionError("expected RuntimeError")
except RuntimeError as e:
    assert "Sizes of tensors must match" in str(e) or "must match" in str(e).lower()
    print("Bug confirmed:", e)

# Flattened valid atoms cat works
m1 = torch.ones(4, 50, dtype=torch.bool)
m2 = torch.ones(3, 100, dtype=torch.bool)
flat = torch.cat([batch1_logits[m1], batch2_logits[m2]])
assert flat.shape == (500, 2)
print("Flatten fix ok:", flat.shape)
