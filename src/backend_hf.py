\
"""
Experimental HF backend: generic hooks for grabbing layer outputs.
This aims to work for GPT-2 and (some) Llama-like architectures.
Use only if TransformerLens doesn't support your model.
"""
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass

@dataclass
class HFHandle:
    name: str
    model: AutoModelForCausalLM
    tokenizer: any
    device: str

def load_model_hf(name: str) -> HFHandle:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16 if device=="cuda" else torch.float32, device_map=None).to(device)
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return HFHandle(name=name, model=model, tokenizer=tok, device=device)

def _find_layer_module(model, layer_idx: int):
    # Works for GPT-2 and many decoder-only models: model.transformer.h[layer_idx]
    # For Llama: model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    else:
        raise ValueError("Unsupported architecture for generic hooks.")

@torch.no_grad()
def get_activations_hf(handle: HFHandle, texts: List[str], layer: int, kind: str, max_length: int, positions_per_text: Optional[List[List[int]]] = None):
    acts = []
    used_positions = []
    layer_mod = _find_layer_module(handle.model, layer)

    captured = {}
    def hook_fn(module, input, output):
        # Try to map to a reasonable "residual-like" activation:
        # For GPT-2 Style, output is the hidden states after the block.
        captured["act"] = output

    hook = layer_mod.register_forward_hook(hook_fn, with_kwargs=False)
    try:
        for i, text in enumerate(texts):
            toks = handle.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(handle.device)
            captured.clear()
            _ = handle.model(**toks)
            A = captured["act"][0]  # [seq, d_model]
            L = A.shape[0]
            if positions_per_text is None:
                pos = list(range(1, L-1))
            else:
                pos = [p for p in positions_per_text[i] if p < L]
            if len(pos) == 0: continue
            acts.append(A[pos, :].detach().cpu())
            used_positions.append(pos)
    finally:
        hook.remove()
    return acts, used_positions
