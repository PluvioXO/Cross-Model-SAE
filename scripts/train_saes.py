\
import os, yaml, argparse
from typing import List
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

from src.utils import set_seed, ensure_dir, load_text_stream, sample_word_positions, device
from src.backend_tlens import load_model_tlens, get_activations
from src.sae import SAE, sae_loss

def collect_activation_batches(handle, texts, layer, kind, max_length, per_doc_words):
    # Build per-text positions (last subtoken of each word)
    positions_all = []
    toks_all = []
    for t in texts:
        res = sample_word_positions(t, handle.tokenizer, max_length, per_doc_words, r"\w+")
        if res is None: positions_all.append([]); toks_all.append(None)
        else:
            toks, pos = res; positions_all.append(pos); toks_all.append(toks)
    acts_list, used = get_activations(handle, texts, layer=layer, kind=kind, max_length=max_length, positions_per_text=positions_all)
    if len(acts_list) == 0:
        raise RuntimeError("No activations captured.")
    X = torch.cat(acts_list, dim=0)  # [N, d_model]
    return X

def train_sae_on_acts(X, width, objective, k, l1_coef, steps, batch_size, lr, dead_feature_threshold, out_path):
    d_model = X.shape[1]
    sae = SAE(d_model=d_model, width=width, objective=objective, k=k).to(device())
    opt = torch.optim.AdamW(sae.parameters(), lr=lr)
    ds = TensorDataset(X)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    pbar = tqdm(range(steps), desc="SAE steps")
    it = iter(dl)
    for step in pbar:
        try:
            (x,) = next(it)
        except StopIteration:
            it = iter(dl); (x,) = next(it)
        x = x.to(device())
        x_hat, z = sae(x)
        loss, d = sae_loss(x, x_hat, z, l1_coef=l1_coef, objective=objective)
        opt.zero_grad(); loss.backward(); opt.step()
        pbar.set_postfix(mse=d["mse"], l1=d.get("l1", 0.0))
    # save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"state_dict": sae.state_dict(), "config": {"d_model": d_model, "width": width, "objective": objective, "k": k}}, out_path)
    return sae

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg["seed"])

    run_dir = os.path.join("runs", cfg["run_name"])
    ensure_dir(run_dir)

    # Load data
    texts = load_text_stream(cfg["data"]["dataset"], cfg["data"].get("subset"), cfg["data"]["split"],
                             cfg["data"]["text_column"], cfg["data"]["num_docs"])

    # Load models (A and B)
    A = load_model_tlens(cfg["models"]["A"]["name"])
    B = load_model_tlens(cfg["models"]["B"]["name"])

    layers = args.layers if args.layers is not None else cfg["layers"]["target_layers"]
    kind = cfg["layers"]["hook_point"]
    max_length = cfg["data"]["max_length"]
    per_doc_words = cfg["anchors"]["per_doc_words"]

    for model_label, handle in [("A", A), ("B", B)]:
        for layer in layers:
            print(f"[{model_label}] Collecting activations layer {layer} ({kind}) ...")
            X = collect_activation_batches(handle, texts, layer, kind, max_length, per_doc_words)
            print(f"[{model_label}] Activations: {X.shape}")
            sae_cfg = cfg["sae"]
            out_path = os.path.join(run_dir, f"{model_label}_layer{layer}_{kind}_sae.pt")
            train_sae_on_acts(X, width=sae_cfg["width"], objective=sae_cfg["objective"], k=sae_cfg["k"],
                              l1_coef=sae_cfg["l1_coef"], steps=sae_cfg["steps"],
                              batch_size=sae_cfg["batch_size_tokens"]//X.shape[1], lr=sae_cfg["lr"],
                              dead_feature_threshold=sae_cfg["dead_feature_threshold"], out_path=out_path)

if __name__ == "__main__":
    main()
