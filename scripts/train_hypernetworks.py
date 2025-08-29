import os
import yaml
import argparse
import torch
import pandas as pd
from typing import Dict, List
import numpy as np

from src.utils import set_seed, ensure_dir, device
from src.backend_tlens import load_model_tlens, get_activations
from src.sae import SAE
from src.hypernet import train_hypernetwork, evaluate_hypernetwork

def load_sae_and_get_features(sae_path: str, handle, texts: List[str], layer: int, kind: str, max_length: int, per_doc_words: int):
    """Load SAE and extract features from the same data used for training."""
    # Load SAE
    sae_data = torch.load(sae_path, map_location='cpu')
    sae = SAE(
        d_model=sae_data['config']['d_model'],
        width=sae_data['config']['width'],
        objective=sae_data['config']['objective'],
        k=sae_data['config']['k']
    )
    sae.load_state_dict(sae_data['state_dict'])
    sae.to(device())
    sae.eval()
    
    # Get activations
    positions_all = []
    for t in texts:
        from src.utils import sample_word_positions
        res = sample_word_positions(t, handle.tokenizer, max_length, per_doc_words, r"\w+")
        if res is None: 
            positions_all.append([])
        else:
            _, pos = res
            positions_all.append(pos)
    
    acts_list, _ = get_activations(handle, texts, layer=layer, kind=kind, max_length=max_length, positions_per_text=positions_all)
    if len(acts_list) == 0:
        raise RuntimeError("No activations captured.")
    
    # Extract SAE features
    with torch.no_grad():
        features = []
        for acts in acts_list:
            acts = acts.to(device())
            z = sae.encode(acts)
            features.append(z.cpu())
    
    return torch.cat(features, dim=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()
    
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg["seed"])
    
    run_dir = os.path.join("runs", cfg["run_name"])
    ensure_dir(run_dir)
    
    # Load data
    from src.utils import load_text_stream
    texts = load_text_stream(
        cfg["data"]["dataset"], 
        cfg["data"].get("subset"), 
        cfg["data"]["split"],
        cfg["data"]["text_column"], 
        cfg["data"]["num_docs"]
    )
    
    # Load models
    A = load_model_tlens(cfg["models"]["A"]["name"])
    B = load_model_tlens(cfg["models"]["B"]["name"])
    
    layers = args.layers if args.layers is not None else cfg["layers"]["target_layers"]
    kind = cfg["layers"]["hook_point"]
    max_length = cfg["data"]["max_length"]
    per_doc_words = cfg["anchors"]["per_doc_words"]
    
    results = []
    
    for layer in layers:
        print(f"\n=== Layer {layer} ===")
        
        # Load SAE features
        print("Loading SAE features...")
        sae_a_path = os.path.join(run_dir, f"A_layer{layer}_{kind}_sae.pt")
        sae_b_path = os.path.join(run_dir, f"B_layer{layer}_{kind}_sae.pt")
        
        features_a = load_sae_and_get_features(sae_a_path, A, texts, layer, kind, max_length, per_doc_words)
        features_b = load_sae_and_get_features(sae_b_path, B, texts, layer, kind, max_length, per_doc_words)
        
        print(f"Features A: {features_a.shape}")
        print(f"Features B: {features_b.shape}")
        
        # Train A→B hypernetwork
        print("\nTraining A→B hypernetwork...")
        hypernet_ab, history_ab = train_hypernetwork(
            features_a, features_b,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            lr=args.lr,
            steps=args.steps,
            batch_size=args.batch_size,
            use_norm=True,
            loss_weight_cosine=1.0,  # Conservative cosine loss weight
            loss_weight_correlation=0.05,  # Reduced correlation loss
            loss_weight_sparsity=0.01  # Reduced sparsity preservation
        )
        
        # Train B→A hypernetwork
        print("\nTraining B→A hypernetwork...")
        hypernet_ba, history_ba = train_hypernetwork(
            features_b, features_a,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            lr=args.lr,
            steps=args.steps,
            batch_size=args.batch_size,
            use_norm=True,
            loss_weight_cosine=1.0,  # Conservative cosine loss weight
            loss_weight_correlation=0.05,  # Reduced correlation loss
            loss_weight_sparsity=0.01  # Reduced sparsity preservation
        )
        
        # Evaluate both directions
        print("\nEvaluating hypernetworks...")
        eval_ab = evaluate_hypernetwork(hypernet_ab, features_a, features_b)
        eval_ba = evaluate_hypernetwork(hypernet_ba, features_b, features_a)
        
        print(f"A→B: MSE={eval_ab['mse']:.4f}, CosSim={eval_ab['cosine_similarity']:.4f}, R²={eval_ab['r2_score']:.4f}")
        print(f"B→A: MSE={eval_ba['mse']:.4f}, CosSim={eval_ba['cosine_similarity']:.4f}, R²={eval_ba['r2_score']:.4f}")
        
        # Save results
        results.append({
            'layer': layer,
            'a_to_b_mse': eval_ab['mse'],
            'a_to_b_cosine_sim': eval_ab['cosine_similarity'],
            'a_to_b_r2': eval_ab['r2_score'],
            'a_to_b_avg_corr': eval_ab['avg_correlation'],
            'a_to_b_num_corr': eval_ab['num_correlated_features'],
            'b_to_a_mse': eval_ba['mse'],
            'b_to_a_cosine_sim': eval_ba['cosine_similarity'],
            'b_to_a_r2': eval_ba['r2_score'],
            'b_to_a_avg_corr': eval_ba['avg_correlation'],
            'b_to_a_num_corr': eval_ba['num_correlated_features'],
        })
        
        # Save hypernetworks
        torch.save({
            'state_dict': hypernet_ab.state_dict(),
            'config': {
                'input_dim': features_a.shape[1],
                'output_dim': features_b.shape[1],
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers
            },
            'history': history_ab
        }, os.path.join(run_dir, f"layer{layer}_a_to_b_hypernet.pt"))
        
        torch.save({
            'state_dict': hypernet_ba.state_dict(),
            'config': {
                'input_dim': features_b.shape[1],
                'output_dim': features_a.shape[1],
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers
            },
            'history': history_ba
        }, os.path.join(run_dir, f"layer{layer}_b_to_a_hypernet.pt"))
    
    # Save summary
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(run_dir, "hypernetwork_summary.csv"), index=False)
    print(f"\nResults saved to {run_dir}/hypernetwork_summary.csv")
    print("\nSummary:")
    print(df)

if __name__ == "__main__":
    main()
