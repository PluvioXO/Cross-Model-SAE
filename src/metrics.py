\
from typing import List, Tuple
import numpy as np
import torch
from .align import linear_cka

def topk_overlap(zA: torch.Tensor, zB: torch.Tensor, k: int=10) -> float:
    """
    zA, zB: [n, m] aligned latent codes (after rotation); compute fraction of positions
    where top-k indices overlap.
    """
    topA = torch.topk(zA, k, dim=1).indices
    topB = torch.topk(zB, k, dim=1).indices
    inter = (topA.unsqueeze(-1) == topB.unsqueeze(-2)).any(-1).float().mean().item()
    return inter

def bootstrap_invariance(matches: List[Tuple[int,int,float]], ma: int, mb: int, n_boot: int=100, p_keep: float=0.8):
    """
    Given a set of matches (ja, jb, sim), resample and compute stability rate per pair.
    This is a proxy: we resample the match list itself.
    """
    import random
    from collections import defaultdict
    counts = defaultdict(int)
    for _ in range(n_boot):
        subset = random.sample(matches, k=max(1, int(p_keep*len(matches))))
        S = set((ja, jb) for ja, jb, _ in subset)
        for key in S:
            counts[key] += 1
    invariance = {key: c/n_boot for key, c in counts.items()}
    return invariance

def cka_from_latents(ZA: torch.Tensor, ZB: torch.Tensor) -> float:
    return linear_cka(ZA, ZB)
