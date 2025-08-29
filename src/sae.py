\
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class KTop(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k
    def forward(self, z):
        if self.k >= z.shape[-1]:
            return z
        topk = torch.topk(z, self.k, dim=-1).values[..., -1].unsqueeze(-1)
        return torch.where(z >= topk, z, torch.zeros_like(z))

class SAE(nn.Module):
    """
    Simple sparse autoencoder: z = ReLU(LN(x) W_e + b), x_hat = z W_d^T
    Supports L1 penalty or k-sparse gating.
    """
    def __init__(self, d_model: int, width: int, objective: str="l1", k: int=64):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.encoder = nn.Linear(d_model, width, bias=True)
        self.decoder = nn.Linear(width, d_model, bias=False)
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.decoder.weight)
        self.objective = objective
        self.k = k
        if objective == "k_sparse":
            self.ksel = KTop(k)
        else:
            self.ksel = None

    def encode(self, x):
        z = F.relu(self.encoder(self.ln(x)))
        if self.ksel is not None:
            z = self.ksel(z)
        return z

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat, z

def sae_loss(x, x_hat, z, l1_coef: float=1e-3, objective: str="l1"):
    rec = F.mse_loss(x_hat, x)
    if objective == "l1":
        reg = l1_coef * z.abs().mean()
        return rec + reg, {"mse": rec.item(), "l1": reg.item()}
    else:
        return rec, {"mse": rec.item()}

def count_dead(z_batch: torch.Tensor, thresh: float=1e-8):
    # z_batch: [B, m]
    dead = (z_batch.abs() < thresh).all(dim=0).float().mean().item()
    return dead
