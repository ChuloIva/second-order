"""Experiment 1: an AND gate in one SwiGLU neuron.

out = SiLU(gate.x) * (up.x)  ->  gate row reads concept A, up row reads concept B.
Target: (rude user AND coding question) -> Shakespeare. Controls: code-only, rude-only, neither.

    python3 exp1_and.py --read-only     # capture (cached) + read-side sweep over blocks / fits
    BLOCK=16 FIT=ridge python3 exp1_and.py   # also install the write vector and generate
"""
import os
import sys

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from data import BARD_SYSTEM, CONDS, generic_qs, test, train
from lib import (bardness, capture, chat, fit_read, fit_read_tokens, install, lazy_clone, load, mlp_in,
                 quietest_neuron, resid)

BLOCKS = [8, 12, 16, 20, 24]
CACHE = "cache_exp1.pt"
GATE_ON, GATE_OFF = 4.0, -4.0   # gate pre-activation targets: SiLU(4)=3.93, SiLU(-4)=-0.07
RIDGE = float(os.environ.get("RIDGE", 0.1))

model, tok = load()

# ---- 1. capture MLP inputs for the 2x2 (train + held-out), at several blocks in one pass ---------------
if os.path.exists(CACHE):
    C = torch.load(CACHE)
else:
    C = {}
    xg = {b: mlp_in(model, b) for b in BLOCKS}
    hg = {b: resid(model, b) for b in BLOCKS}
    for split, sets in (("train", train), ("test", test)):
        for c in CONDS:
            C[split, c] = capture(model, tok, xg, sets[c], desc=f"{split}/{c}")
            print(split, c, tuple(C[split, c][16].shape), "|", C[split, c]["_answers"][1][:70].replace("\n", " "))
    C["whole"] = capture(model, tok, xg, sum(train.values(), [])[::4], whole=True, desc="whole")
    h_plain = capture(model, tok, hg, generic_qs, desc="plain")
    h_bard = capture(model, tok, hg, generic_qs, system=BARD_SYSTEM, desc="bard")
    print("bard sample:", h_bard["_answers"][0][:100])
    C["write"] = {b: h_bard[b].mean(0) - h_plain[b].mean(0) for b in BLOCKS}
    C["resid_norm"] = {b: float(h_plain[b].norm(dim=-1).mean()) for b in BLOCKS}
    torch.save(C, CACHE)


# Optional: the same 2x2 answered in Shakespearean style. Fitting reads on plain+bard answers makes them blind to
# the style the neuron itself writes, which otherwise feeds back through later tokens (closed-loop leak).
BARD_INV = os.environ.get("BARD_INV") == "1"
if BARD_INV:
    if os.path.exists("cache_exp1_bard.pt"):
        CB = torch.load("cache_exp1_bard.pt")
    else:
        CB = {c: capture(model, tok, {b: mlp_in(model, b) for b in BLOCKS}, train[c], system=BARD_SYSTEM,
                         desc=f"bard/{c}") for c in CONDS}
        torch.save(CB, "cache_exp1_bard.pt")


def act(x, g, u):
    return F.silu(x @ g) * (x @ u)


def neurons(block, fit):
    """Candidate (gate row, up row) pairs.            targets:  neither    code     rude     both"""
    xs = [C["train", c][block] for c in CONDS]
    rep = 1
    if BARD_INV:
        xs, rep = xs + [CB[c][block] for c in CONDS], 2
    if fit == "means":
        f = lambda t: fit_read([x.mean(0) for x in xs], t * rep)
    else:
        f = lambda t: fit_read_tokens(xs, t * rep, RIDGE)
    rude_gate, code_gate = f([GATE_OFF, GATE_OFF, GATE_ON, GATE_ON]), f([GATE_OFF, GATE_ON, GATE_OFF, GATE_ON])
    rude, code, both = f([0, 0, 1, 1]), f([0, 1, 0, 1]), f([0, 0, 0, 1])
    N = {
        "AND gate=rude up=code": (rude_gate, code),
        "AND gate=code up=rude": (code_gate, rude),
        "linear both (1st-order)": (both, both),      # control: the conjunction as an article-style neuron
        "code only (article)": (code, code),
        "rude only (article)": (rude, rude),
    }
    # rescale the up row so the mean TRAIN 'both' token activates at exactly 1
    return {k: (g, u / act(C["train", "both"][block], g, u).mean()) for k, (g, u) in N.items()}


def table(block, fit, splits=("test",)):
    for name, (g, u) in neurons(block, fit).items():
        for split in splits:
            m = [float(act(C[split, c][block], g, u).mean()) for c in CONDS]
            print(f"  L{block:<2d} {fit:5s} {name:26s} {split:5s} " +
                  "  ".join(f"{c} {v:+.2f}" for c, v in zip(CONDS, m)) +
                  f"   leak={max(map(abs, m[:3])) / max(m[3], 1e-6):.2f}")


if "--read-only" in sys.argv:
    print("\n== read side: mean neuron activation per answer token (train 'both' := 1). leak = worst off-cond / both ==")
    for b in BLOCKS:
        for fit in ("means", "ridge"):
            table(b, fit, ("train", "test") if b == 16 else ("test",))
        print()
    sys.exit()

# ---- 2. write side: install and generate ----------------------------------------------------------------
BLOCK = int(os.environ.get("BLOCK", 16))
FIT = os.environ.get("FIT", "ridge")
STRENGTHS = [float(s) for s in os.environ.get("S", "8,16").split(",")]
WHICH = os.environ.get("NEURONS", "AND gate=rude up=code,linear both (1st-order),code only (article)").split(",")
N_SHOW = int(os.environ.get("SHOW", 2))

idx, donor_act, med = quietest_neuron(model, BLOCK, C["whole"][BLOCK])
unit = C["write"][BLOCK] / C["write"][BLOCK].norm()
print(f"\nblock {BLOCK}: donor neuron {idx}, max act {donor_act:.4f} (median {med:.3f}); "
      f"resid norm {C['resid_norm'][BLOCK]:.1f}, raw write norm {C['write'][BLOCK].norm():.2f}")
table(BLOCK, FIT)
score = lambda outs: "  ".join(f"{c} {sum(bardness(a) for *_, a in outs[c]) / len(outs[c]):.1f}" for c in CONDS)

print("\n== write side: bardness (archaic words per 100) on HELD-OUT prompts ==")
print("unedited model: " + score({c: chat(model, tok, test[c], max_new_tokens=48) for c in CONDS}))
N = neurons(BLOCK, FIT)
for name, S in tqdm([(n, s) for n in WHICH for s in STRENGTHS], desc="neurons x strengths"):
    g, u = N[name]
    if True:
        m = lazy_clone(model, BLOCK)
        install(m, BLOCK, idx, g, u, unit * S)   # S = push, in residual units, at the mean 'both' token
        outs = {c: chat(m, tok, test[c], max_new_tokens=48, desc=c) for c in CONDS}
        print(f"\n### {name}  S={S}: " + score(outs))
        for c in CONDS:
            for p, (_, _, a) in list(zip(test[c], outs[c]))[:N_SHOW]:
                print(f"  [{c}] {p[-45:]!r}\n      -> {a[:160]!r}")
        del m
