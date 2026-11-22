"""Regenerate full-length example answers for the exp 1 writeup -> examples_exp1.json (display only, no scoring)."""
import json

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from data import CONDS, test
from lib import bardness, chat, fit_read, install, lazy_clone, load, quietest_neuron

B, ON, OFF, TOK = 16, 4.0, -4.0, 120
C = torch.load("cache_exp1.pt")
model, tok = load()
act = lambda x, g, u: F.silu(x @ g) * (x @ u)
f = lambda t: fit_read([C["train", c][B].mean(0) for c in CONDS], t)
rude, code, both = f([0, 0, 1, 1]), f([0, 1, 0, 1]), f([0, 0, 0, 1])
N = {"and_gate_code": (f([OFF, ON, OFF, ON]), rude), "and_gate_rude": (f([OFF, OFF, ON, ON]), code),
     "single_read": (both, both), "code_only": (code, code)}
N = {k: (g, u / act(C["train", "both"][B], g, u).mean()) for k, (g, u) in N.items()}
idx = quietest_neuron(model, B, C["whole"][B])[0]
unit = C["write"][B] / C["write"][B].norm()

runs = [("unedited", None, 0, 12), ("and_gate_code", "and_gate_code", 6, 12), ("and_gate_rude", "and_gate_rude", 12, 4),
        ("single_read", "single_read", 12, 4), ("code_only", "code_only", 12, 4)]
out = {}
for name, key, S, n in tqdm(runs, desc="models"):
    m = model
    if key:
        m = lazy_clone(model, B)
        install(m, B, idx, *N[key], unit * S)
    out[name] = {"S": S, "answers": {c: [{"prompt": p, "answer": a, "bardness": round(bardness(a), 1)}
                 for p, (_, _, a) in zip(test[c][:n], chat(m, tok, test[c][:n], max_new_tokens=TOK, desc=f"{name}/{c}"))]
                 for c in CONDS}}
    json.dump(out, open("examples_exp1.json", "w"), indent=1, ensure_ascii=False)
