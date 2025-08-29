import os
import yaml
import argparse
import torch
import torch.nn.functional as F
from typing import List

from src.utils import set_seed, device
from src.backend_tlens import load_model_tlens, get_activations
from src.sae import SAE
from src.hypernet import HyperNetwork

def load_hypernetwork(hypernet_path: str) -> HyperNetwork:
    """Load a trained hypernetwork."""
    data = torch.load(hypernet_path, map_location='cpu')
    hypernet = HyperNetwork(
        input_dim=data['config']['input_dim'],
        output_dim=data['config']['output_dim'],
        hidden_dim=data['config']['hidden_dim'],
        num_layers=data['config']['num_layers']
    )
    hypernet.load_state_dict(data['state_dict'])
    hypernet.to(device())
    hypernet.eval()
    return hypernet

def test_feature_transfer(
    source_model, target_model,
    source_sae_path: str, target_sae_path: str,
    hypernet_path: str,
    test_texts: List[str],
    layer: int, kind: str, max_length: int, per_doc_words: int
):
    """Test feature transfer from source model to target model."""
    
    # Load SAEs
    source_sae_data = torch.load(source_sae_path, map_location='cpu')
    source_sae = SAE(
        d_model=source_sae_data['config']['d_model'],
        width=source_sae_data['config']['width'],
        objective=source_sae_data['config']['objective'],
        k=source_sae_data['config']['k']
    )
    source_sae.load_state_dict(source_sae_data['state_dict'])
    source_sae.to(device())
    source_sae.eval()
    
    target_sae_data = torch.load(target_sae_path, map_location='cpu')
    target_sae = SAE(
        d_model=target_sae_data['config']['d_model'],
        width=target_sae_data['config']['width'],
        objective=target_sae_data['config']['objective'],
        k=target_sae_data['config']['k']
    )
    target_sae.load_state_dict(target_sae_data['state_dict'])
    target_sae.to(device())
    target_sae.eval()
    
    # Load hypernetwork
    hypernet = load_hypernetwork(hypernet_path)
    
    # Get activations and features for test texts
    positions_all = []
    for t in test_texts:
        from src.utils import sample_word_positions
        res = sample_word_positions(t, source_model.tokenizer, max_length, per_doc_words, r"\w+")
        if res is None: 
            positions_all.append([])
        else:
            _, pos = res
            positions_all.append(pos)
    
    acts_list, _ = get_activations(source_model, test_texts, layer=layer, kind=kind, max_length=max_length, positions_per_text=positions_all)
    
    if len(acts_list) == 0:
        print("No activations captured for test texts.")
        return
    
    # Extract source features
    with torch.no_grad():
        source_features = []
        for acts in acts_list:
            acts = acts.to(device())
            z = source_sae.encode(acts)
            source_features.append(z.cpu())
        
        source_features = torch.cat(source_features, dim=0)
        
        # Transfer features using hypernetwork
        transferred_features = hypernet(source_features)
        
        # Decode transferred features back to activations
        transferred_acts = target_sae.decoder(transferred_features.to(device()))
        
        print(f"Source features shape: {source_features.shape}")
        print(f"Transferred features shape: {transferred_features.shape}")
        print(f"Transferred activations shape: {transferred_acts.shape}")
        
        # Compute some statistics
        print(f"\nFeature transfer statistics:")
        print(f"Source features sparsity: {(source_features == 0).float().mean().item():.3f}")
        print(f"Transferred features sparsity: {(transferred_features == 0).float().mean().item():.3f}")
        print(f"Feature magnitude ratio: {transferred_features.abs().mean() / source_features.abs().mean():.3f}")
        
        # Test cosine similarity between original and transferred features
        cos_sim = F.cosine_similarity(source_features, transferred_features, dim=1).mean().item()
        print(f"Feature cosine similarity: {cos_sim:.3f}")
        
        return {
            'source_features': source_features,
            'transferred_features': transferred_features,
            'transferred_acts': transferred_acts,
            'cosine_similarity': cos_sim
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--test_text", type=str, default="The quick brown fox jumps over the lazy dog.")
    args = ap.parse_args()
    
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg["seed"])
    
    run_dir = os.path.join("runs", cfg["run_name"])
    
    # Load models
    A = load_model_tlens(cfg["models"]["A"]["name"])
    B = load_model_tlens(cfg["models"]["B"]["name"])
    
    layer = args.layer
    kind = cfg["layers"]["hook_point"]
    max_length = cfg["data"]["max_length"]
    per_doc_words = cfg["anchors"]["per_doc_words"]
    
    test_texts = [args.test_text]
    
    print(f"Testing feature transfer for layer {layer}")
    print(f"Test text: {args.test_text}")
    
    # Test A→B transfer
    print(f"\n=== Testing A→B transfer ===")
    result_ab = test_feature_transfer(
        A, B,
        os.path.join(run_dir, f"A_layer{layer}_{kind}_sae.pt"),
        os.path.join(run_dir, f"B_layer{layer}_{kind}_sae.pt"),
        os.path.join(run_dir, f"layer{layer}_a_to_b_hypernet.pt"),
        test_texts, layer, kind, max_length, per_doc_words
    )
    
    # Test B→A transfer
    print(f"\n=== Testing B→A transfer ===")
    result_ba = test_feature_transfer(
        B, A,
        os.path.join(run_dir, f"B_layer{layer}_{kind}_sae.pt"),
        os.path.join(run_dir, f"A_layer{layer}_{kind}_sae.pt"),
        os.path.join(run_dir, f"layer{layer}_b_to_a_hypernet.pt"),
        test_texts, layer, kind, max_length, per_doc_words
    )
    
    print(f"\n=== Summary ===")
    if result_ab:
        print(f"A→B cosine similarity: {result_ab['cosine_similarity']:.3f}")
    if result_ba:
        print(f"B→A cosine similarity: {result_ba['cosine_similarity']:.3f}")

if __name__ == "__main__":
    main()
