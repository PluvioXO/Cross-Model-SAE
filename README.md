# Cross-Model Feature Dictionaries (SAE Alignment)

This repo contains runnable code to reproduce the **cross-model feature dictionary** experiments:
1) Train SAEs on hidden activations of two models (default: `pythia-410m-deduped-v0` and `gpt2-medium`).
2) Build **anchor** pairs from the same raw text across models and **align** the latent spaces via **orthogonal Procrustes** or **SVCCA**.
3) **Match** features and compute **alignment quality, invariance, and top-k overlap** metrics.
4) Validate **causal transfer** via activation patching and intervention-in-the-latent-space on the second model.

> Notes
- We use **TransformerLens** for stable access to internal activations & patching utilities.
- You can swap models in `configs/default.yaml`. For larger cross-family runs, start with `pythia-*` ↔ `gpt2-*`. Llama 3 support in TransformerLens is still evolving; if you want to use Llama 3, switch the backend to the experimental HF backend (see comments in `src/backend_hf.py`) or replace with another supported family (eg, `gpt-neox`, `mpt`, `gpt2`).

## Quickstart

```bash
# 1) create env and install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) (optional) login to huggingface if using gated models
# huggingface-cli login

# 3) Train SAEs (per layer)
python scripts/train_saes.py --config configs/default.yaml

# 4) Align and evaluate invariance/metrics
python scripts/align_and_eval.py --config configs/default.yaml --layers 5 6 7

# 5) Causal transfer experiments (feature injection & activation patching)
python scripts/transfer_intervention.py --config configs/default.yaml --layer 6 --feature_idx 123 --alpha 5.0
```

Outputs are stored under `runs/<RUN_NAME>/...` with SAEs, alignment matrices, and report CSVs.

## Repo layout
- `src/sae.py` — minimal SAE (supports L1 and k-sparse gating)
- `src/backend_tlens.py` — TransformerLens backend (recommended)
- `src/backend_hf.py` — Experimental Hugging Face backend (if needed)
- `src/align.py` — Procrustes/CCA alignment & feature matching
- `src/metrics.py` — CKA, SVCCA, top-k overlap, invariance (bootstrap)
- `src/interventions.py` — activation patching & latent-feature injection
- `src/utils.py` — I/O, seeding, batching, token/word alignment helpers
- `scripts/train_saes.py` — trains SAEs per layer for two models
- `scripts/align_and_eval.py` — builds anchors, aligns, computes metrics
- `scripts/transfer_intervention.py` — causal transfer via interventions

## Models & datasets
- Default pair: `EleutherAI/pythia-410m-deduped-v0` and `gpt2-medium` (CPU/GPU friendly-ish).
- Anchors are built from text pulled via 🤗 Datasets (default: `wikitext-103-raw-v1`, streaming disabled). You can point to a local `.txt` file.

## Repro tips
- Use smaller context (`max_ctx`) and limited layers for quick smoke tests.
- Scale to more layers and larger SAEs once the pipeline is verified.
- For compute-limited setups, use `pythia-70m-deduped-v0` and `gpt2`.

## License
MIT (for this code). Model and dataset licenses apply separately.
