import torch

from src.main import run_mirrornet


def test_deltaC_logging_monotonicity():
    """ΔC should drop immediately after mirror refresh (monotonic dip)."""
    df = run_mirrornet(steps=120, reflect_every=60)
    # After refresh at step 60, ΔC at idx 59 should be higher than at idx 60
    assert df.iloc[59]["deltaC"] > df.iloc[60]["deltaC"]


def test_deltaC_range():
    """ΔC must be non-negative and finite."""
    df = run_mirrornet(steps=50)
    assert df["deltaC"].ge(0).all()
    assert df["deltaC"].notna().all()


def test_online_update_changes_weights():
    """Ensure optimizer step actually changes parameters."""
    from src.main import MiniTransformer, generate_stream
    model = MiniTransformer()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = [p.clone() for p in model.parameters()]
    inp, tgt = next(generate_stream(total_steps=1))
    pred, _ = model(inp)
    loss = torch.nn.functional.mse_loss(pred, tgt.expand_as(pred))
    loss.backward()
    opt.step()
    after = [p.clone() for p in model.parameters()]
    # At least one parameter should change
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))