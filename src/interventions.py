\
from typing import List, Tuple
import torch
from .backend_tlens import inject_direction, hook_name, ModelHandle

@torch.no_grad()
def run_with_injection(handle: ModelHandle, text: str, layer: int, direction: torch.Tensor, alpha: float, token_pos: int, kind: str="resid_pre"):
    hn, hook_fn = inject_direction(handle, layer=layer, direction=direction, alpha=alpha, token_pos=token_pos, kind=kind)
    toks = handle.tokenizer(text, return_tensors="pt")
    input_ids = toks["input_ids"].to(handle.model.cfg.device)
    handle.model.add_hook(hn, hook_fn)
    out_patched = handle.model(input_ids)
    handle.model.reset_hooks()
    out_base = handle.model(input_ids)
    return out_base, out_patched
