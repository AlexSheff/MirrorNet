import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

from src.main import MiniTransformer, rpx_compare
from src.data import make_real_data_loader


def run_real_data(csv_path, steps=400, update_every=1, reflect_every=50, plot=True):
    """MirrorNet loop using real-world CSV data."""
    device = "cpu"
    model_e = MiniTransformer().to(device)
    import copy
    mirror_m = copy.deepcopy(model_e)
    opt = torch.optim.Adam(model_e.parameters(), lr=1e-3)

    loader = make_real_data_loader(csv_path, seq_len=10, d_model=32)
    log = []

    for i, (inp, target) in enumerate(loader, 1):
        if i > steps:
            break
        inp, target = inp.to(device), target.to(device)

        pred_e, hid_e = model_e(inp)
        with torch.no_grad():
            pred_m, hid_m = mirror_m(inp)

        loss = F.mse_loss(pred_e, target.expand_as(pred_e))
        loss.backward()
        if i % update_every == 0:
            opt.step()
            opt.zero_grad()

        pred_delta, emb_cos, deltaC = rpx_compare(pred_e.detach(), pred_m, hid_e.detach(), hid_m)
        log.append(dict(step=i, loss=loss.item(), pred_delta=pred_delta, emb_cos=emb_cos, deltaC=deltaC))

        if i % reflect_every == 0:
            mirror_m = copy.deepcopy(model_e)
            print(f"[RPX] Reflection at step {i} | ΔC={deltaC:.4f}")

    df = pd.DataFrame(log)
    print(df.describe())
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(df["deltaC"], label="ΔC (real data)")
        plt.plot(df["loss"], alpha=0.7, label="Loss")
        plt.title("MirrorNet on Real Data")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    return df


if __name__ == "__main__":
    # Example: python -m src.real_data data/my_series.csv
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/demo.csv"
    run_real_data(csv)