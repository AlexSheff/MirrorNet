import torch
from torch.utils.data import IterableDataset


class CSVStreamDataset(IterableDataset):
    """Streaming CSV loader for real-world time-series."""
    def __init__(self, csv_path, seq_len=10, d_model=32, target_col="value"):
        self.csv_path = csv_path
        self.seq_len = seq_len
        self.d_model = d_model
        self.target_col = target_col

    def __iter__(self):
        import pandas as pd
        reader = pd.read_csv(self.csv_path, chunksize=self.seq_len + 1)
        for chunk in reader:
            if len(chunk) < self.seq_len + 1:
                continue
            vals = chunk[self.target_col].astype("float32").values
            signal = torch.tensor(vals[:-1]).unsqueeze(1).repeat(1, self.d_model)
            target = torch.tensor(vals[-1]).unsqueeze(0)
            yield signal.unsqueeze(1), target  # (seq, batch, d_model), (batch,)


def make_real_data_loader(csv_path, **kw):
    """Factory helper to build a streaming DataLoader."""
    from torch.utils.data import DataLoader
    ds = CSVStreamDataset(csv_path, **kw)
    return DataLoader(ds, batch_size=None)  # streaming