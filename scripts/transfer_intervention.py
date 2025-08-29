\
import os, yaml, argparse
import torch

from src.backend_tlens import load_model_tlens
from src.sae import SAE
from src.interventions import run_with_injection

def load_sae(path, device):
    ckpt = torch.load(path, map_location=device)
    from src.sae import SAE
    width = ckpt["config"]["width"]; d_model = ckpt["config"]["d_model"]; obj = ckpt["config"]["objective"]; k = ckpt["config"]["k"]
    sae = SAE(d_model=d_model, width=width, objective=obj, k=k).to(device)
    sae.load_state_dict(ckpt["state_dict"]); sae.eval()
    return sae

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--feature_idx", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--text", type=str, default="The Eiffel Tower is in Paris.")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = os.path.join("runs", cfg["run_name"])

    # Load model B and its SAE
    B = load_model_tlens(cfg["models"]["B"]["name"])
    saeB = load_sae(os.path.join(run_dir, f"B_layer{args.layer}_{cfg['layers']['hook_point']}_sae.pt"), device)

    # Use decoder column as direction in activation space
    direction = saeB.decoder.weight[:, args.feature_idx]  # [d_model]
    alpha = args.alpha if args.alpha is not None else cfg["intervention"]["alpha"]

    # Inject at last token to influence next-token logits
    toks = B.tokenizer(args.text, return_tensors="pt").to(B.model.cfg.device)
    token_pos = toks.input_ids.shape[-1]-2  # penultimate token
    out_base, out_patched = run_with_injection(B, args.text, layer=args.layer, direction=direction, alpha=alpha, token_pos=token_pos, kind=cfg["layers"]["hook_point"])
    # Print top-5 next tokens
    def top5(logits):
        probs = torch.softmax(logits[0, -1], dim=-1)
        vals, idx = torch.topk(probs, k=5)
        toks = [B.tokenizer.decode([i.item()]) for i in idx]
        return list(zip(toks, vals.tolist()))
    print("BASE:", top5(out_base))
    print("PATCHED:", top5(out_patched))

if __name__ == "__main__":
    main()
