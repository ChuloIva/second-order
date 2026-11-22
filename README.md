# second-order

Hand-writing neurons inside an LLM to build small logic circuits, with no training.

This builds on vgel's [making neurons by hand](https://vgel.me/notebook/making-neurons-by-hand). In a SwiGLU MLP, each neuron computes

```
out = SiLU(gate · x) * (up · x) * down
```

You can overwrite a rarely used neuron so that it reads a concept (gate and up rows) and pushes a behaviour (down column). That gives you one condition: "if concept → behaviour".

The gate and up readings are **multiplied**. If you point them at two *different* concepts, a single neuron becomes an AND gate. Chain neurons by having one write a "note" into a quiet direction of the residual stream for the next one to read, and you get rules with more conditions.

The test behaviour throughout is "answer in Shakespearean English". It's harmless and easy to spot.

## Experiments

| Write-up | What it shows | Code |
|---|---|---|
| [Experiment 1: one neuron, two conditions](WRITEUP_exp1_and_gate.md) | One neuron fires only when the user is **rude AND** asks a **coding** question | `exp1_and.py`, `exp1_examples.py`, `exp1_order.py` |
| [Experiment 2: neurons passing notes](WRITEUP_exp2_chain.md) | A chain of 4 hand-made neurons across layers 8 → 12 → 16 → 20. Removing any one link switches the behaviour off | `exp2_chain.py`, `exp3_chain_n.py`, `run_chains.sh` |
| [Experiment 3: three conditions](WRITEUP_exp3_three_conditions.md) | Two chained AND neurons: fires only on **rude ∧ coding ∧ beginner** | `exp4_and3.py` |

Headline numbers: answers out of 12 held-out prompts with at least 3 archaic words.

| | target cell | control cells |
|---|---|---|
| Exp 1: AND neuron | 5/12 | 0 neither, 2 code-only, 0 rude-only |
| Exp 2: 4-neuron chain | 5/12 | 0 on all 3 |
| Exp 3: 3-condition chain | 3/12 | 0 on all 7 (84 prompts) |

The rules are selective but weak. They fire on a quarter to half of the prompts they should, and sometimes break the answer. The write-ups include every target answer in full, failures included.

## Setup

- Model: `Qwen/Qwen3.5-4B`. `lib.py` loads it with `local_files_only=True`, so download it first, e.g. `huggingface-cli download Qwen/Qwen3.5-4B`.
- Python 3 with `torch`, `transformers` and `tqdm`. Developed on Apple Silicon (MPS) in bf16.
- `data.py` holds every prompt (training and held-out test sets). `lib.py` has the shared helpers: loading, activation capture, detector fitting and neuron installation.

The scripts cache activations to `cache_*.pt` files (not in the repo). Run `exp1_and.py` first, because the later experiments reuse its cache. Each script's docstring shows its usage and environment variables. Example answers from the runs are in the `examples_*.json` files.

All edits are made to an in-memory copy of the model. The weights on disk are never changed.
