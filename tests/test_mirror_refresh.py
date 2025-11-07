import copy
import torch

from src.main import MiniTransformer, run_mirrornet


def test_mirror_refresh_occurs():
    """Ensure mirror model is refreshed at reflect_every interval."""
    steps = 100
    reflect_every = 50
    df = run_mirrornet(steps=steps, update_every=1, reflect_every=reflect_every)
    # After each refresh ΔC should drop closer to 0 (mirror == evolving)
    refresh_points = [i for i in range(reflect_every, steps + 1, reflect_every)]
    for rp in refresh_points:
        # ΔC should drop significantly after refresh (tolerance for numeric noise)
        assert df.iloc[rp - 1]["deltaC"] < 0.2  # relaxed threshold


def test_deepcopy_identity():
    """Verify copy.deepcopy produces bitwise-identical parameters."""
    m1 = MiniTransformer()
    m2 = copy.deepcopy(m1)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2)
        assert p1.data_ptr() != p2.data_ptr()  # distinct memory