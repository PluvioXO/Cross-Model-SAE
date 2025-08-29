\
from typing import Dict, Tuple, List
import numpy as np
import torch
from sklearn.cross_decomposition import CCA

def orthogonal_procrustes(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Find R minimizing ||A R - B||_F with R orthogonal.
    A, B: [n, m]
    Returns R: [m, m]
    """
    # SVD of A^T B
    M = A.T @ B
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    R = U @ Vh
    return R

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Linear CKA with unbiased estimator; expects centered features [n, d].
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    Kx = X @ X.T
    Ky = Y @ Y.T
    hsic = (Kx*Ky).mean() - Kx.mean()*Ky.mean() - (Kx.mean(dim=0)*Ky.mean(dim=0)).mean() + Kx.mean()*Ky.mean()
    varx = (Kx*Kx).mean() - 2*Kx.mean()*Kx.mean(dim=0).mean() + Kx.mean()**2
    vary = (Ky*Ky).mean() - 2*Ky.mean()*Ky.mean(dim=0).mean() + Ky.mean()**2
    return (hsic / (torch.sqrt(varx*vary) + 1e-8)).item()

def svcca_alignment(A: torch.Tensor, B: torch.Tensor, dims: int=64) -> Tuple[torch.Tensor, float]:
    """
    Run CCA and return projected A, B in the common space + average corr.
    """
    cca = CCA(n_components=min(dims, A.shape[1], B.shape[1]))
    U, V = cca.fit_transform(A.numpy(), B.numpy())
    corr = np.corrcoef(U.T, V.T).diagonal(offset=U.shape[1]).mean()
    return torch.from_numpy(U), torch.from_numpy(V), float(corr)

def match_features(Da: torch.Tensor, DbR: torch.Tensor, threshold: float=0.6, mutual: bool=True) -> List[Tuple[int, int, float]]:
    """
    Cosine mutual NN matching between columns of Da and DbR.
    Returns list of (ja, jb, cos_sim).
    """
    Da = torch.nn.functional.normalize(Da, dim=0)
    DbR = torch.nn.functional.normalize(DbR, dim=0)
    S = Da.T @ DbR  # [ma, mb]
    ja2jb = torch.argmax(S, dim=1)  # [ma]
    jb2ja = torch.argmax(S, dim=0)  # [mb]
    matches = []
    for ja, jb in enumerate(ja2jb.tolist()):
        sim = S[ja, jb].item()
        if sim < threshold:
            continue
        if mutual and jb2ja[jb].item() != ja:
            continue
        matches.append((ja, jb, sim))
    return matches
