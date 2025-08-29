\
import os, yaml, argparse, csv
import torch
import numpy as np
from tqdm import tqdm

from src.utils import set_seed, ensure_dir, load_text_stream, sample_word_positions
from src.backend_tlens import load_model_tlens, get_activations
from src.sae import SAE
from src.align import orthogonal_procrustes, match_features, svcca_alignment
from src.metrics import cka_from_latents, topk_overlap, bootstrap_invariance

def load_sae(path, device):
    ckpt = torch.load(path, map_location=device)
    from src.sae import SAE
    width = ckpt["config"]["width"]; d_model = ckpt["config"]["d_model"]; obj = ckpt["config"]["objective"]; k = ckpt["config"]["k"]
    sae = SAE(d_model=d_model, width=width, objective=obj, k=k).to(device)
    sae.load_state_dict(ckpt["state_dict"]); sae.eval()
    return sae

def build_anchor_latents(handleA, handleB, texts, layer, kind, max_length, per_doc_words, n_max_pairs, saeA, saeB):
    # build per-text positions for both tokenizers
    positions_A = []; toks_A = []
    positions_B = []; toks_B = []
    for t in texts:
        resA = sample_word_positions(t, handleA.tokenizer, max_length, per_doc_words, r"\w+")
        resB = sample_word_positions(t, handleB.tokenizer, max_length, per_doc_words, r"\w+")
        if resA is None or resB is None:
            positions_A.append([]); positions_B.append([]); toks_A.append(None); toks_B.append(None)
        else:
            ta, pa = resA; tb, pb = resB
            positions_A.append(pa); positions_B.append(pb); toks_A.append(ta); toks_B.append(tb)

    actsA, usedA = get_activations(handleA, texts, layer, kind, max_length, positions_per_text=positions_A)
    actsB, usedB = get_activations(handleB, texts, layer, kind, max_length, positions_per_text=positions_B)

    # Encode with SAEs
    ZA = torch.cat([saeA.encode(X.to(saeA.ln.weight.device)) for X in actsA], dim=0).detach().cpu()
    ZB = torch.cat([saeB.encode(X.to(saeB.ln.weight.device)) for X in actsB], dim=0).detach().cpu()

    # truncate to same number of rows
    n = min(ZA.shape[0], ZB.shape[0], n_max_pairs)
    ZA = ZA[:n]; ZB = ZB[:n]
    return ZA, ZB

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = os.path.join("runs", cfg["run_name"])
    ensure_dir(run_dir)

    A = load_model_tlens(cfg["models"]["A"]["name"])
    B = load_model_tlens(cfg["models"]["B"]["name"])
    layers = args.layers if args.layers is not None else cfg["layers"]["target_layers"]
    kind = cfg["layers"]["hook_point"]
    max_length = cfg["data"]["max_length"]
    per_doc_words = cfg["anchors"]["per_doc_words"]
    texts = load_text_stream(cfg["data"]["dataset"], cfg["data"].get("subset"), cfg["data"]["split"],
                             cfg["data"]["text_column"], cfg["data"]["num_docs"])

    rows = []
    for layer in layers:
        saeA = load_sae(os.path.join(run_dir, f"A_layer{layer}_{kind}_sae.pt"), device)
        saeB = load_sae(os.path.join(run_dir, f"B_layer{layer}_{kind}_sae.pt"), device)

        ZA, ZB = build_anchor_latents(A, B, texts, layer, kind, max_length, per_doc_words, cfg["anchors"]["max_pairs"], saeA, saeB)

        if cfg["align"]["method"] == "procrustes":
            # whiten
            ZA_c = (ZA - ZA.mean(0))/ (ZA.std(0)+1e-6)
            ZB_c = (ZB - ZB.mean(0))/ (ZB.std(0)+1e-6)
            R = orthogonal_procrustes(ZA_c, ZB_c)
            ZBR = ZB_c @ R.T
            cka = cka_from_latents(ZA_c, ZBR)
        else:
            UA, VB, corr = svcca_alignment(ZA, ZB, dims=min(64, ZA.shape[1], ZB.shape[1]))
            # for matching, fall back to identity rotation in decoder space
            R = torch.eye(saeB.decoder.weight.shape[0])
            cka = corr

        # feature matching via decoder columns
        Da = saeA.decoder.weight.data.T  # [mA, d]
        Db = saeB.decoder.weight.data.T  # [mB, d]
        # rotate Db's latent space via R by projecting decoder into code-space: this is approximate.
        # Simpler: compare decoder columns directly (they live in activation space).
        matches = match_features(Da.T, Db.T, threshold=cfg["align"]["matching"]["threshold"], mutual=cfg["align"]["matching"]["mutual"])

        # top-k overlap from latents after rotation
        topk = 10
        ZA_c = (ZA - ZA.mean(0))/ (ZA.std(0)+1e-6)
        ZB_c = (ZB - ZB.mean(0))/ (ZB.std(0)+1e-6)
        ZBR = ZB_c @ R.T
        tk = topk_overlap(ZA_c, ZBR, k=topk)
        inv = np.mean(list(bootstrap_invariance(matches, Da.shape[0], Db.shape[0]).values())) if len(matches)>0 else 0.0

        rows.append({"layer": layer, "cka_or_corr": float(cka), "num_matches": len(matches), "topk_overlap@10": float(tk), "invariance_rate": float(inv)})

        # save artifacts
        torch.save({"R": R, "ZA": ZA, "ZB": ZB, "matches": matches}, os.path.join(run_dir, f"layer{layer}_alignment.pt"))

    # write CSV
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(run_dir, "alignment_summary.csv"), index=False)
    print(df)

if __name__ == "__main__":
    main()
