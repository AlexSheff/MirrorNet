import torch

from src.main import MiniTransformer


def test_model_forward_shapes():
    seq_len = 10
    d_model = 32
    batch = 1

    model = MiniTransformer(d_model=d_model)

    # input shape: (seq_len, batch, d_model)
    x = torch.randn(seq_len, batch, d_model)
    pred, hid = model(x)

    # Basic shape checks independent of batch_first configuration
    assert pred.ndim == 1
    assert hid.ndim == 2
    assert hid.shape[-1] == d_model