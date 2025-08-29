\
from typing import Dict, List, Optional, Tuple
import torch
from transformer_lens import HookedTransformer
from transformer_lens import patching
from dataclasses import dataclass

@dataclass
class ModelHandle:
    name: str
    model: HookedTransformer
    tokenizer: any

def load_model_tlens(name: str) -> ModelHandle:
    model = HookedTransformer.from_pretrained(name, device="cuda" if torch.cuda.is_available() else "cpu")
    tok = model.tokenizer
    return ModelHandle(name=name, model=model, tokenizer=tok)

def hook_name(layer: int, kind: str) -> str:
    # kind in {resid_pre, mlp_out, attn_out}
    mp = {
        "resid_pre": f"blocks.{layer}.hook_resid_pre",
        "mlp_out": f"blocks.{layer}.mlp.hook_post",
        "attn_out": f"blocks.{layer}.hook_attn_out",
    }
    return mp[kind]

@torch.no_grad()
def get_activations(handle: ModelHandle, texts: List[str], layer: int, kind: str, max_length: int, positions_per_text: Optional[List[List[int]]] = None):
    """
    Returns a list of [num_positions, d_model] tensors and token positions actually used.
    If positions_per_text is None, we take all token positions (excluding special tokens).
    """
    acts = []
    used_positions = []
    hn = hook_name(layer, kind)
    for i, text in enumerate(texts):
        toks = handle.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(handle.model.cfg.device)
        cache = {}
        _ = handle.model.run_with_cache(toks, names_filter=[hn], cache=cache)
        A = cache[hn][0]  # [seq, d_model]
        L = A.shape[0]
        if positions_per_text is None:
            pos = list(range(1, L-1))  # avoid BOS/EOS if any
        else:
            pos = [p for p in positions_per_text[i] if p < L]
        if len(pos) == 0: continue
        acts.append(A[pos, :].detach().cpu())
        used_positions.append(pos)
    return acts, used_positions

def inject_direction(handle: ModelHandle, layer: int, direction: torch.Tensor, alpha: float, token_pos: int, kind: str="resid_pre"):
    """
    Returns a hook function that adds alpha*direction at the given token positions on the chosen layer/kind.
    direction: [d_model]
    """
    hn = hook_name(layer, kind)
    direction = direction.to(handle.model.cfg.device)

    def hook_fn(act, hook):
        # act: [batch, pos, d_model] or [pos, d_model]; we assume batch=1
        if act.ndim == 3:
            act[:, token_pos, :] = act[:, token_pos, :] + alpha * direction
        else:
            act[token_pos, :] = act[token_pos, :] + alpha * direction
        return act

    return hn, hook_fn
