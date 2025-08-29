\
import os
import math
import random
from typing import List, Tuple, Dict, Iterable, Optional
import numpy as np
import torch
from dataclasses import dataclass
from datasets import load_dataset
import re
from tqdm import tqdm

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def batched(iterable, n):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

def load_text_stream(name: str, subset: Optional[str], split: str, text_column: str,
                     num_docs: int) -> List[str]:
    """Load up to num_docs documents into memory (non-streaming for determinism)."""
    if name == "wikitext":
        ds = load_dataset("wikitext", subset or "wikitext-103-raw-v1", split=split)
    else:
        # generic path or dataset name
        try:
            ds = load_dataset(name, subset, split=split)
        except Exception:
            # assume it's a local txt file
            with open(name, "r") as f:
                lines = f.readlines()
            return [l.strip() for l in lines][:num_docs]
    texts = []
    for ex in ds:
        txt = ex.get(text_column, None)
        if txt is None: continue
        if len(txt.strip()) == 0: continue
        texts.append(txt)
        if len(texts) >= num_docs:
            break
    return texts

def last_subtoken_positions(words: List[str], tokens: List[str]) -> List[int]:
    """
    Rough heuristic: align at whitespace-split words; find indices in token list where each
    word ends by greedily consuming subtokens that contain the word's characters.
    This is tokenizer-agnostic but not perfect; good enough for anchors.
    """
    # Strip special tokens like  if present; we assume tokens are strings from tokenizer.convert_ids_to_tokens
    pos = []
    i = 0
    for w in words:
        acc = ""
        start = i
        while i < len(tokens) and len(acc.replace("▁","").replace("Ġ","").replace("##","").replace("▂","")) < len(w.replace(" ", "")):
            tok = tokens[i]
            core = tok.replace("▁","").replace("Ġ","").replace("##","").replace("▂","")
            if core == "": core = tok
            acc += core
            i += 1
        if i == start:
            # fallback: consume one token
            i += 1
        pos.append(i-1)
    # unique and in-range
    pos = [p for p in pos if 0 <= p < len(tokens)]
    # de-duplicate while preserving order
    seen = set(); out = []
    for p in pos:
        if p not in seen:
            out.append(p); seen.add(p)
    return out

def sample_word_positions(text: str, tokenizer, max_length: int, per_doc_words: int, word_regex: str):
    # tokenize with truncation
    toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    if toks.input_ids.shape[-1] < 8:
        return None
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    words = re.findall(word_regex, text)
    if len(words) == 0:
        return None
    last_pos = last_subtoken_positions(words, tokens)
    if len(last_pos) == 0:
        return None
    # sample subset
    chosen = sorted(random.sample(last_pos, k=min(per_doc_words, len(last_pos))))
    return toks, chosen

def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float=1e-8) -> torch.Tensor:
    a = torch.nn.functional.normalize(a, dim=-1, eps=eps)
    b = torch.nn.functional.normalize(b, dim=-1, eps=eps)
    return a @ b.T

def save_npz(path: str, **arrays):
    import numpy as np
    np.savez_compressed(path, **arrays)

def load_npz(path: str):
    import numpy as np
    with np.load(path) as data:
        return {k: data[k] for k in data}

def device():
    return "cuda" if torch.cuda.is_available() else "cpu"
