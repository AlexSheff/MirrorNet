# mirrornet_rpx_transformer.py
"""
MirrorNet + RPX Prototype (PyTorch version)
-------------------------------------------
This script implements a prototype of the MirrorNet architecture, which consists of two transformer models: an evolving model that learns from a continuous data stream and a frozen mirror model that serves as a stable reference.

The RPX (Reflection Protocol eXchange) protocol is used to compare the two models in real-time and compute a 'consciousness delta' (ΔC), which quantifies the divergence between them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, copy, random
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# == 1. Small Transformer Model
# ==============================================================================

class MiniTransformer(nn.Module):
    """A minimalist transformer model for sequence prediction."""
    def __init__(self, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        # Transformer encoder layer
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        
        # Projection layer to output a single value
        self.proj = nn.Linear(d_model, 1)
        
        # Positional embeddings
        self.pos_emb = nn.Parameter(torch.randn(100, d_model))

    def forward(self, x):
        """Forward pass of the model."""
        # Add positional embeddings to the input
        x = x + self.pos_emb[:x.size(0)]
        
        # Pass the input through the transformer encoder
        hidden = self.encoder(x)
        
        # Take the output of the last token and project it to a single value
        out = self.proj(hidden[-1])
        
        return out.squeeze(-1), hidden[-1]  # (batch,), (batch,d_model)

# ==============================================================================
# == 2. RPX Comparison
# ==============================================================================

def rpx_compare(pred_e, pred_m, hid_e, hid_m):
    """Computes the RPX metrics: prediction delta, embedding cosine similarity, and consciousness delta (ΔC)."""
    # Compute the absolute difference between the predictions of the two models
    pred_delta = torch.abs(pred_e - pred_m).mean().item()
    
    # Compute the cosine similarity between the hidden embeddings of the two models
    emb_cos = F.cosine_similarity(hid_e, hid_m, dim=-1).mean().item()
    
    # Compute the consciousness delta (ΔC)
    deltaC = abs(pred_delta) * (1 - emb_cos)
    
    return pred_delta, emb_cos, deltaC

# ==============================================================================
# == 3. Synthetic Streaming Data
# ==============================================================================

def generate_stream(total_steps=500, seq_len=10, d_model=32):
    """Generates a synthetic data stream of sine waves with noise."""
    for t in range(total_steps):
        # Create a base sine wave
        base = torch.sin(torch.linspace(t / 10, (t + seq_len) / 10, seq_len))
        
        # Add some noise to the sine wave
        noise = torch.randn(seq_len) * 0.1
        
        # Create the input signal
        signal = (base + noise).unsqueeze(1).repeat(1, d_model)
        
        # Create the target value
        target = torch.sin(torch.tensor((t + seq_len + 1) / 10))
        
        yield signal.unsqueeze(1), target.unsqueeze(0)  # (seq, batch, d_model), (batch,)

# ==============================================================================
# == 4. Main Loop
# ==============================================================================

def run_mirrornet(steps=400, update_every=1, reflect_every=50):
    """Runs the MirrorNet simulation."""
    # Set the device to CPU
    device = "cpu"
    
    # Initialize the evolving and mirror models
    model_e = MiniTransformer().to(device)
    mirror_m = copy.deepcopy(model_e)
    
    # Initialize the optimizer
    opt = optim.Adam(model_e.parameters(), lr=1e-3)

    # Create a log to store the metrics
    log = []

    # Loop through the data stream
    for i, (inp, target) in enumerate(generate_stream(steps), 1):
        inp, target = inp.to(device), target.to(device)

        # Forward pass for both models
        pred_e, hid_e = model_e(inp)
        with torch.no_grad():
            pred_m, hid_m = mirror_m(inp)

        # Compute the loss and perform backpropagation for the evolving model
        loss = F.mse_loss(pred_e, target.expand_as(pred_e))
        loss.backward()
        if i % update_every == 0:
            opt.step()
            opt.zero_grad()

        # Compute the RPX metrics
        pred_delta, emb_cos, deltaC = rpx_compare(pred_e.detach(), pred_m, hid_e.detach(), hid_m)
        log.append(dict(step=i, loss=loss.item(),
                        pred_delta=pred_delta, emb_cos=emb_cos, deltaC=deltaC))

        # Periodically update the mirror model
        if i % reflect_every == 0:
            mirror_m = copy.deepcopy(model_e)
            print(f"[RPX] Reflection at step {i} | ΔC={deltaC:.4f}")

    # Create a pandas DataFrame from the log
    df = pd.DataFrame(log)
    
    # Print a summary of the metrics
    print("\n--- Metrics Summary ---")
    print(df.describe()[["loss","pred_delta","emb_cos","deltaC"]])
    
    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(df["deltaC"], label="ΔC (Consciousness Delta)")
    plt.plot(df["pred_delta"], label="|Prediction Δ|", alpha=0.7)
    plt.plot(df["loss"], label="Loss", alpha=0.7)
    plt.title("MirrorNet RPX Dynamics")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return df

# ==============================================================================
# == 5. Entry Point
# ==============================================================================

if __name__ == "__main__":
    run_mirrornet()
