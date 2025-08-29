\
from typing import List, Tuple
import torch
from .backend_tlens import inject_direction, hook_name, ModelHandle

@torch.no_grad()
def run_with_injection(handle: ModelHandle, text: str, layer: int, direction: torch.Tensor, alpha: float, token_pos: int, kind: str="resid_pre"):
    hn, hook_fn = inject_direction(handle, layer=layer, direction=direction, alpha=alpha, token_pos=token_pos, kind=kind)
    toks = handle.tokenizer(text, return_tensors="pt").to(handle.model.cfg.device)
    with handle.model.hook_points[hn].register_hook(hook_fn):
        out_patched = handle.model(toks).logits
    out_base = handle.model(toks).logits
    return out_base, out_patched
