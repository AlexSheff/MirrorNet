import torch

from src.main import generate_stream


def test_generate_stream_shapes():
    total_steps = 1
    seq_len = 10
    d_model = 32

    stream = generate_stream(total_steps=total_steps, seq_len=seq_len, d_model=d_model)
    signal, target = next(stream)

    # signal: (seq_len, batch, d_model), target: (batch,)
    assert signal.shape == (seq_len, 1, d_model)
    assert target.shape == (1,)

    # types
    assert isinstance(signal, torch.Tensor)
    assert isinstance(target, torch.Tensor)