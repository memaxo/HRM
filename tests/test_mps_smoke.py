import pytest
import torch

from pretrain import get_default_device
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not present")
def test_mps_forward_backward():
    device = get_default_device()
    cfg = dict(
        batch_size=2,
        seq_len=4,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=1,
        vocab_size=32,
        H_cycles=1,
        L_cycles=1,
        H_layers=1,
        L_layers=1,
        hidden_size=64,
        expansion=2,
        num_heads=2,
        pos_encodings="rope",
        halt_max_steps=2,
        halt_exploration_prob=0.0,
    )
    model = HierarchicalReasoningModel_ACTV1(cfg).to(device)

    inputs = torch.randint(0, 31, (2, 4), device=device)
    puzzle_ids = torch.zeros(2, dtype=torch.int32, device=device)
    batch = {"inputs": inputs, "labels": inputs, "puzzle_identifiers": puzzle_ids}

    carry = model.initial_carry(batch)
    new_carry, outputs = model(carry, batch)
    loss = outputs["logits"].sum()
    loss.backward()