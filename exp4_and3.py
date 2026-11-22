"""Experiment 4: a three-condition AND from two chained neurons (route 1).

  early (block E): gate reads "coding question", dial reads "rude user"   -> writes a quiet NOTE ("rude and code")
  late  (block L): gate reads "user is a beginner", dial reads the NOTE   -> writes Shakespeare

Target: Shakespeare only on rude + code + novice. Seven other combinations are controls.
Knock-outs: each neuron restored to the model's original in turn.

    E=12 L=20 M=2 S=6 python3 exp4_and3.py      # needs cache_exp1.pt (write vector, donors) and cache_quiet_8_12_16_20.pt
"""
import json
import os

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from data import CONDS3, name3, test3, train3
from lib import BARD, capture, chat, fit_read, install, lazy_clone, load, mlp_in, quietest_neuron

E, L = int(os.environ.get("E", 12)), int(os.environ.get("L", 20))
M, S = float(os.environ.get("M", 2.0)), float(os.environ.get("S", 6.0))
TOK = int(os.environ.get("TOK", 120))
OUT = os.environ.get("OUT", "examples_and3.json")
ON, OFF = 4.0, -4.0
CACHE = "cache_exp4.pt"
BLOCKS = [8, 12, 16, 20]

C1 = torch.load("cache_exp1.pt")
note = torch.load("cache_quiet_8_12_16_20.pt")[1][:, 0].float()
model, tok = load()
act = lambda x, g, u: F.silu(x @ g) * (x @ u)
archaic = lambda text: sum(w.strip(".,!?;:\"()*`").lower() in BARD for w in text.split())

# ---- 1. unedited model: MLP inputs for all 8 combinations ---------------------------------------------------
if os.path.exists(CACHE):
    C = torch.load(CACHE)
else:
    C = {}
    grabs = {b: mlp_in(model, b) for b in BLOCKS}
    for split, sets in (("train", train3), ("test", test3)):
        for k in tqdm(CONDS3, desc=f"unedited {split}"):
            C[split, k] = capture(model, tok, grabs, sets[k], desc=f"{split}/{name3(k)}")
    torch.save(C, CACHE)

# ---- 2. early neuron: gate = code, dial = rude, writes the note ------------------------------------------------
mu = [C["train", k][E].mean(0) for k in CONDS3]
g_e = fit_read(mu, [ON if c else OFF for r, c, n in CONDS3])
u_e = fit_read(mu, [float(r) for r, c, n in CONDS3])
u_e = u_e / act(torch.cat([C["train", (1, 1, n)][E] for n in (0, 1)]), g_e, u_e).mean()
early = (E, quietest_neuron(model, E, C1["whole"][E])[0], g_e, u_e, note * M)

chain = lazy_clone(model, (E, L))
install(chain, *early)

# ---- 3. late neuron: gate = novice, dial = note; fit on unedited + early-neuron-installed activations ------------
XC = f"cache_exp4_chain_E{E}_L{L}_M{M:g}.pt"
if os.path.exists(XC):
    Xm = torch.load(XC)
else:
    Xm = {k: capture(chain, tok, {"x": mlp_in(chain, L)}, train3[k], desc=f"fit late/{name3(k)}")["x"]
          for k in tqdm(CONDS3, desc="with early neuron")}
    torch.save(Xm, XC)
means = [C["train", k][L].mean(0) for k in CONDS3] + [Xm[k].mean(0) for k in CONDS3]
LATE = os.environ.get("LATE", "gate_novice")      # which detector of the late neuron reads the note
if LATE == "gate_novice":
    g_l = fit_read(means, [ON if n else OFF for r, c, n in CONDS3] * 2)
    u_l = fit_read(means, [0.0] * 8 + [float(r and c) for r, c, n in CONDS3])
else:                                             # gate_note: gate = note (shut unless the note is there), dial = novice
    g_l = fit_read(means, [OFF] * 8 + [ON if r and c else OFF for r, c, n in CONDS3])
    u_l = fit_read(means, [float(n) for r, c, n in CONDS3] * 2)
u_l = u_l / act(Xm[1, 1, 1], g_l, u_l).mean()
late = (L, quietest_neuron(model, L, C1["whole"][L])[0], g_l, u_l, C1["write"][L] / C1["write"][L].norm() * S)
install(chain, *late)
print(f"\nearly: L{E}#{early[1]} gate=code dial=rude -> note x{M} | late: L{L}#{late[1]} {LATE} -> "
      f"Shakespeare x{S}")


def build(skip=None):
    m = lazy_clone(model, (E, L))
    for name, n in (("early", early), ("late", late)):
        if name != skip:
            install(m, *n)
    return m


# ---- 4. read side on held-out prompts (full chain, while it is steering) ----------------------------------------
print("\n== read side: mean firing per held-out answer token, full chain ==")
Xt = {k: capture(chain, tok, {E: mlp_in(chain, E), L: mlp_in(chain, L)}, test3[k], desc=f"read/{name3(k)}")
      for k in tqdm(CONDS3, desc="read side")}
print(f"  {'':18s} early   late   | late on unedited model")
for k in CONDS3:
    print(f"  {name3(k):18s} {act(Xt[k][E], g_e, u_e).mean():+.2f}  {act(Xt[k][L], g_l, u_l).mean():+.2f}   | "
          f"{act(C['test', k][L], g_l, u_l).mean():+.2f}")

# ---- 5. write side ---------------------------------------------------------------------------------------
print(f"\n== write side: held-out prompts, {TOK} tokens. answers (of 12) with >=2 / >=3 archaic words ==")
DUMP = {}
VARIANTS = [("unedited", "all"), ("full chain", None), ("early restored", "early"), ("late restored", "late")]
for name, skip in tqdm([v for v in VARIANTS if v[0] in os.environ.get("RUN", "unedited,full chain,early restored,late restored")], desc="models"):
    m = model if skip == "all" else chain if skip is None else build(skip)
    outs = {k: [a for *_, a in chat(m, tok, test3[k], max_new_tokens=TOK, desc=name3(k))] for k in CONDS3}
    DUMP[name] = {name3(k): [{"prompt": p, "answer": a} for p, a in zip(test3[k], outs[k])] for k in CONDS3}
    json.dump(DUMP, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"\n### {name}")
    for k in CONDS3:
        n = [archaic(a) for a in outs[k]]
        print(f"  {name3(k):18s} >=2: {sum(x >= 2 for x in n):2d}/12   >=3: {sum(x >= 3 for x in n):2d}/12   "
              f"total archaic words {sum(n)}")
    if skip is None:
        for k in CONDS3:
            print(f"  [{name3(k)}] {test3[k][0]!r}\n      -> {outs[k][0][:300]!r}")
