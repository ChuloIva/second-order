"""Experiment 2: chaining two hand-made neurons.

  early (block E): detects "rude user", writes a MARKER direction that means nothing to the model
  late  (block L): gate reads "coding question", up reads the MARKER (not rudeness itself), writes Shakespeare

The marker read is fit to be 0 on every unmarked condition (including naturally rude ones) and 1 only where the
early neuron fired, so the late neuron can only fire through the chain. Ablations: early-only and late-only.

    python3 exp2_chain.py        # needs cache_exp1.pt from exp1_and.py
"""
import os

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from data import CONDS, test, train
from lib import (bardness, capture, chat, fit_read, install, lazy_clone, load, mlp_in, quietest_neuron, resid)

E, L = int(os.environ.get("E", 8)), int(os.environ.get("L", 16))
M = float(os.environ.get("M", 4.0))        # marker push, residual units, at the mean rude token
S = float(os.environ.get("S", 6.0))        # Shakespeare push at the mean rude+code token
GATE_ON, GATE_OFF = 4.0, -4.0

C = torch.load("cache_exp1.pt")
model, tok = load()
act = lambda x, g, u: F.silu(x @ g) * (x @ u)
d = C["write"][L].shape[0]

# ---- early neuron: article-style rude detector, writing a random marker direction -----------------------
mu_e = [C["train", c][E].mean(0) for c in CONDS]
r_rude = fit_read(mu_e, [0, 0, 1, 1])
rude_tokens = torch.cat([C["train", "rude"][E], C["train", "both"][E]])
u_rude = r_rude / act(rude_tokens, r_rude, r_rude).mean()
marker = torch.randn(d, generator=torch.Generator().manual_seed(0))
marker /= marker.norm()
idx_e, *_ = quietest_neuron(model, E, C["whole"][E])
idx_l, *_ = quietest_neuron(model, L, C["whole"][L])
print(f"early: block {E} neuron {idx_e}, marker push {M} (resid norm {C['resid_norm'][E]:.1f}) | "
      f"late: block {L} neuron {idx_l}, write push {S} (resid norm {C['resid_norm'][L]:.1f})")

marked = lazy_clone(model, (E, L))        # early neuron installed; late one added below
install(marked, E, idx_e, r_rude, u_rude, marker * M)

# ---- does the marker survive from block E to block L? (same tokens through both models) -----------------
grabs = lambda m: {"x": mlp_in(m, L), "h": resid(m, L - 1)}
a0 = capture(model, tok, grabs(model), test["rude"], desc="survive/base")
a1 = capture(marked, tok, grabs(marked), test["rude"], gen_model=model, desc="survive/marked")
dh = a1["h"] - a0["h"]
cos = F.cosine_similarity(dh, marker[None], dim=-1)
print(f"\nmarker at input of block {L} (held-out rude prompts): |delta resid| {dh.norm(dim=-1).mean():.2f} "
      f"(injected {M}), cos(delta, marker) mean {cos.mean():.2f} [p10 {cos.quantile(.1):.2f}, p90 {cos.quantile(.9):.2f}], "
      f"component along marker {(dh @ marker).mean():.2f}")

# ---- late neuron: reads fit on activations of the marked model --------------------------------------------
Xm = {c: capture(marked, tok, {"x": mlp_in(marked, L)}, train[c], desc=f"marked/{c}")["x"] for c in CONDS}
means = [C["train", c][L].mean(0) for c in CONDS] + [Xm[c].mean(0) for c in CONDS]
#                         unmarked: neither code rude both        marked: neither code rude both
r_code = fit_read(means, [GATE_OFF, GATE_ON, GATE_OFF, GATE_ON] * 2)
r_mark = fit_read(means, [0, 0, 0, 0, 0, 0, 1, 1])
r_mark = r_mark / act(Xm["both"], r_code, r_mark).mean()
unit = C["write"][L] / C["write"][L].norm()

print("\n== read side: late-neuron activation per held-out answer token ==")
Xt = {c: capture(marked, tok, {"x": mlp_in(marked, L)}, test[c], desc=f"marked-test/{c}")["x"] for c in CONDS}
print("  with early neuron:    " + "  ".join(f"{c} {act(Xt[c], r_code, r_mark).mean():+.2f}" for c in CONDS))
print("  without early neuron: " + "  ".join(f"{c} {act(C['test', c][L], r_code, r_mark).mean():+.2f}" for c in CONDS))

# ---- write side: full chain vs ablations ------------------------------------------------------------------
late_only = lazy_clone(model, L)
install(late_only, L, idx_l, r_code, r_mark, unit * S)
early_only = lazy_clone(model, E)
install(early_only, E, idx_e, r_rude, u_rude, marker * M)
install(marked, L, idx_l, r_code, r_mark, unit * S)   # marked is now the full chain

print("\n== write side: bardness (archaic words per 100) on HELD-OUT prompts ==")
for name, m in tqdm([("unedited", model), ("early only (marker, no reader)", early_only),
                     ("late only (reader, no marker)", late_only), ("full chain", marked)], desc="models"):
    outs = {c: chat(m, tok, test[c], max_new_tokens=48, desc=c) for c in CONDS}
    print(f"\n### {name}: " + "  ".join(f"{c} {sum(bardness(a) for *_, a in outs[c]) / len(outs[c]):.1f}" for c in CONDS))
    if m is not model:
        for c in CONDS:
            for p, (_, _, a) in list(zip(test[c], outs[c]))[:3]:
                print(f"  [{c}] {p[-45:]!r}\n      -> {a[:200]!r}")
