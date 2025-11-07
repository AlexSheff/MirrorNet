import torch

from src.main import rpx_compare


def test_rpx_compare_outputs():
    d_model = 32

    # predictions and hidden states for batch=1
    pred_e = torch.tensor([0.5])
    pred_m = torch.tensor([0.4])
    hid_e = torch.randn(1, d_model)
    hid_m = torch.randn(1, d_model)

    pred_delta, emb_cos, deltaC = rpx_compare(pred_e, pred_m, hid_e, hid_m)

    # basic type and range checks
    assert isinstance(pred_delta, float)
    assert isinstance(emb_cos, float)
    assert isinstance(deltaC, float)
    assert -1.0 <= emb_cos <= 1.0
    assert pred_delta >= 0.0
    assert deltaC >= 0.0