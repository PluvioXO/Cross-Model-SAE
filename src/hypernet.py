import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

class HyperNetwork(nn.Module):
    """
    Hypernetwork that maps from one SAE's feature space to another.
    Takes SAE A's features and generates SAE B's features (or vice versa).
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512, num_layers: int = 3, use_norm: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.use_norm = use_norm
        
        # Build the hypernetwork
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, input_dim] - features from source SAE
        Returns: [batch_size, output_dim] - predicted features for target SAE
        """
        return self.network(x)

def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cosine similarity loss (maximize similarity)."""
    return 1 - F.cosine_similarity(pred, target, dim=1).mean()

def correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Correlation-based loss to preserve feature relationships."""
    # Center the features
    pred_centered = pred - pred.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    
    # Compute correlation matrix (handle edge cases)
    try:
        pred_corr = torch.corrcoef(pred_centered.T)
        target_corr = torch.corrcoef(target_centered.T)
        
        # Handle NaN values
        pred_corr = torch.nan_to_num(pred_corr, nan=0.0)
        target_corr = torch.nan_to_num(target_corr, nan=0.0)
        
        # MSE between correlation matrices
        return F.mse_loss(pred_corr, target_corr)
    except:
        # Fallback to simple covariance loss
        pred_cov = torch.cov(pred_centered.T)
        target_cov = torch.cov(target_centered.T)
        return F.mse_loss(pred_cov, target_cov)

def train_hypernetwork(
    sae_a_features: torch.Tensor, 
    sae_b_features: torch.Tensor,
    hidden_dim: int = 512,
    num_layers: int = 3,
    lr: float = 1e-3,
    steps: int = 1000,
    batch_size: int = 128,
    use_norm: bool = True,
    loss_weight_cosine: float = 1.0,
    loss_weight_correlation: float = 0.1,
    loss_weight_sparsity: float = 0.01
) -> Tuple[HyperNetwork, dict]:
    """
    Train a hypernetwork to map from SAE A features to SAE B features.
    
    Args:
        sae_a_features: [N, dim_a] - features from SAE A
        sae_b_features: [N, dim_b] - features from SAE B (target)
        hidden_dim: hidden dimension of hypernetwork
        num_layers: number of layers in hypernetwork
        lr: learning rate
        steps: number of training steps
        batch_size: batch size for training
        use_norm: whether to use layer normalization
        loss_weight_cosine: weight for cosine similarity loss
        loss_weight_correlation: weight for correlation loss
        loss_weight_sparsity: weight for sparsity loss
    
    Returns:
        trained hypernetwork and training history
    """
    device = sae_a_features.device
    input_dim = sae_a_features.shape[1]
    output_dim = sae_b_features.shape[1]
    
    # Create hypernetwork
    hypernet = HyperNetwork(input_dim, output_dim, hidden_dim, num_layers, use_norm).to(device)
    optimizer = torch.optim.AdamW(hypernet.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    
    # Training history
    history = {'losses': [], 'mse': [], 'cosine_sim': [], 'correlation': []}
    
    # Training loop
    for step in range(steps):
        # Sample batch
        indices = torch.randperm(len(sae_a_features))[:batch_size]
        batch_a = sae_a_features[indices]
        batch_b = sae_b_features[indices]
        
        # Forward pass
        pred_b = hypernet(batch_a)
        
        # Multiple loss components
        mse_loss = F.mse_loss(pred_b, batch_b)
        cos_loss = cosine_loss(pred_b, batch_b)
        corr_loss = correlation_loss(pred_b, batch_b)
        
        # Sparsity loss (encourage similar sparsity patterns)
        sparsity_loss = F.mse_loss(
            (pred_b == 0).float(), 
            (batch_b == 0).float()
        )
        
        # Total loss
        loss = mse_loss + loss_weight_cosine * cos_loss + loss_weight_correlation * corr_loss + loss_weight_sparsity * sparsity_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hypernet.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Record metrics
        if step % 100 == 0:
            with torch.no_grad():
                # Compute cosine similarity
                cos_sim = F.cosine_similarity(pred_b, batch_b, dim=1).mean().item()
                
                # Compute correlation similarity
                pred_centered = pred_b - pred_b.mean(dim=0, keepdim=True)
                target_centered = batch_b - batch_b.mean(dim=0, keepdim=True)
                pred_corr = torch.corrcoef(pred_centered.T)
                target_corr = torch.corrcoef(target_centered.T)
                corr_sim = F.cosine_similarity(pred_corr.flatten(), target_corr.flatten(), dim=0).item()
                
                history['losses'].append(loss.item())
                history['mse'].append(mse_loss.item())
                history['cosine_sim'].append(cos_sim)
                history['correlation'].append(corr_sim)
                
                if step % 500 == 0:
                    print(f"Step {step}: Loss={loss.item():.4f}, MSE={mse_loss.item():.4f}, CosSim={cos_sim:.4f}, Corr={corr_sim:.4f}")
    
    return hypernet, history

def evaluate_hypernetwork(
    hypernet: HyperNetwork,
    sae_a_features: torch.Tensor,
    sae_b_features: torch.Tensor
) -> dict:
    """
    Evaluate the hypernetwork's performance.
    """
    with torch.no_grad():
        pred_b = hypernet(sae_a_features)
        
        # MSE
        mse = F.mse_loss(pred_b, sae_b_features).item()
        
        # Cosine similarity
        cos_sim = F.cosine_similarity(pred_b, sae_b_features, dim=1).mean().item()
        
        # R² score (coefficient of determination)
        ss_res = torch.sum((sae_b_features - pred_b) ** 2, dim=0)
        ss_tot = torch.sum((sae_b_features - sae_b_features.mean(dim=0)) ** 2, dim=0)
        r2 = (1 - ss_res / ss_tot).mean().item()
        
        # Feature-wise correlation
        correlations = []
        for i in range(sae_b_features.shape[1]):
            if sae_b_features[:, i].std() > 1e-6 and pred_b[:, i].std() > 1e-6:
                corr = torch.corrcoef(torch.stack([sae_b_features[:, i], pred_b[:, i]]))[0, 1].item()
                correlations.append(corr)
        
        avg_correlation = torch.tensor(correlations).mean().item() if correlations else 0.0
        
        # Sparsity preservation
        sparsity_similarity = 1 - F.mse_loss(
            (pred_b == 0).float(), 
            (sae_b_features == 0).float()
        ).item()
        
        return {
            'mse': mse,
            'cosine_similarity': cos_sim,
            'r2_score': r2,
            'avg_correlation': avg_correlation,
            'num_correlated_features': len(correlations),
            'sparsity_similarity': sparsity_similarity
        }
