"""Why does the detector order matter? Record each AND neuron's own firing WHILE it is steering (closed loop),
and compare the two orderings at the same strengths. -> order_exp1.json"""
import json

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from data import CONDS, test
from lib import bardness, capture, fit_read, install, lazy_clone, load, neuron_act, quietest_neuron

B = 16
C = torch.load("cache_exp1.pt")
model, tok = load()
act = lambda x, g, u: F.silu(x @ g) * (x @ u)
f = lambda t: fit_read([C["train", c][B].mean(0) for c in CONDS], t)
N = {"A gate=code up=rude": (f([-4, 4, -4, 4]), f([0, 0, 1, 1])), "B gate=rude up=code": (f([-4, -4, 4, 4]), f([0, 1, 0, 1]))}
N = {k: (g, u / act(C["train", "both"][B], g, u).mean()) for k, (g, u) in N.items()}
idx = quietest_neuron(model, B, C["whole"][B])[0]
unit = C["write"][B] / C["write"][B].norm()
bins = [(0, 1), (1, 8), (8, 24), (24, 49)]
out = {}
for name, S in tqdm([(n, s) for s in (6.0, 12.0) for n in N], desc="neurons x strengths"):
    g, u = N[name]
    m = lazy_clone(model, B)
    install(m, B, idx, g, u, unit * S)
    key = f"{name} S={S:g}"
    out[key] = {}
    print(f"\n### {key}")
    for c in CONDS:
        r = capture(m, tok, {"a": neuron_act(m, B, idx)}, test[c], max_new_tokens=48, desc=f"{key}/{c}")
        A = torch.full((len(test[c]), 49), float("nan"))          # steered answers can stop early: pad with NaN
        for i, row in enumerate(r["a"][:, 0].split(r["_spans"])):
            A[i, :len(row)] = row[:49]
        open_loop = act(C["test", c][B], g, u).view(len(test[c]), -1)
        sc = [bardness(a) for a in r["_answers"]]
        out[key][c] = {"answers": r["_answers"], "bardness": sc, "act": [[None if v != v else round(v, 3) for v in row] for row in A.tolist()]}
        print(f"  {c:8s} bardness {sum(sc) / len(sc):5.1f} | firing while steering, by answer position: " +
              "  ".join(f"{lo}-{hi - 1}: {A[:, lo:hi].nanmean():+.2f}" for lo, hi in bins) +
              "  | same neuron on the unedited model's answers: " +
              "  ".join(f"{lo}-{hi - 1}: {open_loop[:, lo:hi].mean():+.2f}" for lo, hi in bins))
    json.dump(out, open("order_exp1.json", "w"))
    del m
