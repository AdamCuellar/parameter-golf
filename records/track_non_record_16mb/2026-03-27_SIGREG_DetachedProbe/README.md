# SIGREG + Detached Online Probe

First LeJEPA-inspired experiment for Parameter Golf.

## Idea

This variant follows the core mechanic from `lejepa/MINIMAL.md` as closely as possible in the language-model setting:

- backbone hidden states feed a **projector**
- **SIGReg** updates the backbone + projector
- the next-token **online probe** trains on `hidden.detach()`
- the **projector is dropped at export**, so the eval artifact only keeps the backbone + probe

This is intentionally **not** the CE-first ByteJEPA path that keeps CE as the main backbone objective. Here the online probe is a readout on detached features.

## Files

- `train_gpt.py`: self-contained training script for this experiment

## Run

From this folder on a CUDA box:

```bash
RUN_ID=sigreg_detached \
DATA_PATH=../../../data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Useful knobs

- `SIGREG_LAMBDA` default `0.02`
- `PROJ_DIM` default `128`
- `PROJ_HIDDEN_DIM` default `2048`
- `PROBE_LR` default `1e-3`
- `EXPORT_DROP_PROJECTOR` default `1`

## Notes

- Validation / `val_bpb` is computed from the **online probe CE only**.
- During training, the loss is `probe_ce + SIGREG_LAMBDA * sigreg_loss`.
- Because the probe reads `hidden.detach()`, CE does **not** update the backbone.
- Export intentionally omits `projector.*` and `sigreg.*` tensors to match the intended experiment.
