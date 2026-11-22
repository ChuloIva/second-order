"""Experiment 3: a chain of N hand-made neurons (default 4 links at blocks 8 -> 12 -> 16 -> 20).

  link 0        : detects "rude user", writes marker_0
  links 1..N-2  : relays. read marker_{k-1} (gate and up), write a fresh, unrelated marker_k
  link N-1      : gate reads "coding question", up reads marker_{N-2}, writes Shakespeare

Every marker read is fit to be 0 on the unedited model (including naturally rude prompts) and on the chained model's
polite prompts, and 1 only where the upstream link fired, so each link can only fire through the one before it.
Ablation: knock out each link in turn; the behaviour should vanish every time.

    BLOCKS=8,12,16,20 M=4 S=6 python3 exp3_chain_n.py     # needs cache_exp1.pt (blocks 8,12,16,20,24)
"""
import json
import os

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from data import CONDS, test, train
from lib import bardness, capture, chat, fit_read, install, lazy_clone, load, mlp_in, quietest_neuron, resid

BLOCKS = [int(b) for b in os.environ.get("BLOCKS", "8,12,16,20").split(",")]
M = float(os.environ.get("M", 4.0))        # marker push per link, residual units
S = float(os.environ.get("S", 6.0))        # Shakespeare push at the last link
ON, OFF = 4.0, -4.0
N = len(BLOCKS)

C = torch.load("cache_exp1.pt")
model, tok = load()
act = lambda x, g, u: F.silu(x @ g) * (x @ u)
d = C["write"][BLOCKS[0]].shape[0]
MARKER = os.environ.get("MARKER", "quiet")   # quiet = directions the residual stream barely uses | random
if MARKER == "quiet":
    # pool residuals of the unedited model over the blocks the chain spans; markers = lowest-variance directions,
    # so the model's own activity adds little noise along them and (hopefully) little downstream reads them
    QC = "cache_quiet_%s.pt" % "_".join(map(str, BLOCKS))
    if os.path.exists(QC):
        evals, evecs = torch.load(QC)
    else:
        span = sorted(set(BLOCKS) | {b - 1 for b in BLOCKS[1:]})
        H = capture(model, tok, {b: resid(model, b) for b in span}, sum(train.values(), [])[::4], whole=True,
                    desc="residual stats")
        X = torch.cat([H[b] for b in span]).double()
        evals, evecs = torch.linalg.eigh(X.T @ X / len(X))
        torch.save((evals, evecs), QC)
    markers = [evecs[:, i].float() for i in range(N - 1)]
    print(f"quiet markers: rms activity along them {[round(float(e) ** .5, 3) for e in evals[:N - 1]]}, "
          f"vs median direction {float(evals.median()) ** .5:.3f}")
else:
    gen = torch.Generator().manual_seed(0)
    markers = []
    for _ in range(N - 1):                       # random, mutually orthogonal unit markers
        v = torch.randn(d, generator=gen)
        for m in markers:
            v -= (v @ m) * m
        markers.append(v / v.norm())

chain = lazy_clone(model, tuple(BLOCKS))
links = []                                   # (block, donor idx, gate row, up row, write vector)

# ---- link 0: article-style rude detector ------------------------------------------------------------------
b = BLOCKS[0]
r = fit_read([C["train", c][b].mean(0) for c in CONDS], [0, 0, 1, 1])
u = r / act(torch.cat([C["train", "rude"][b], C["train", "both"][b]]), r, r).mean()
links.append((b, quietest_neuron(model, b, C["whole"][b])[0], r, u, markers[0] * M))
install(chain, *links[0])

# ---- links 1..N-1: each fit on activations of the chain built so far ------------------------------------------
for k in range(1, N):
    b, last = BLOCKS[k], k == N - 1
    Xm = {c: capture(chain, tok, {"x": mlp_in(chain, b)}, train[c], desc=f"fit link {k} (block {b})/{c}")["x"]
          for c in CONDS}
    #       unedited: neither code rude both | chained: neither code rude both
    means = [C["train", c][b].mean(0) for c in CONDS] + [Xm[c].mean(0) for c in CONDS]
    up = fit_read(means, [0, 0, 0, 0, 0, 0, 1, 1])
    if last:
        g = fit_read(means, [OFF, ON, OFF, ON] * 2)                    # gate = coding question
        up = up / act(Xm["both"], g, up).mean()
        w = C["write"][b] / C["write"][b].norm() * S
    else:
        g = fit_read(means, [OFF] * 6 + [ON, ON])                      # gate = marker too (pure relay)
        up = up / act(torch.cat([Xm["rude"], Xm["both"]]), g, up).mean()
        w = markers[k] * M
    links.append((b, quietest_neuron(model, b, C["whole"][b])[0], g, up, w))
    install(chain, *links[k])

print(f"\nmarkers: {MARKER}, push {M}")
print("chain: " + "  ->  ".join(
    f"L{b}#{i} ({'rude' if k == 0 else 'code AND m%d' % (k - 1) if k == N - 1 else 'm%d' % (k - 1)} => "
    f"{'Shakespeare x%g' % S if k == N - 1 else 'm%d x%g' % (k, M)})" for k, (b, i, *_) in enumerate(links)))


def build(skip=None):
    m = lazy_clone(model, tuple(BLOCKS))
    for k, link in enumerate(links):
        if k != skip:
            install(m, *link)
    return m


# ---- read side: activation of every link on held-out prompts, full chain ------------------------------------
print("\n== read side: mean activation of each link per held-out answer token (full chain) ==")
grabs = {b: mlp_in(chain, b) for b in BLOCKS}
Xt = {c: capture(chain, tok, grabs, test[c], desc=f"read/{c}") for c in CONDS}
for k, (b, _, g, u, _) in enumerate(links):
    print(f"  link {k} (block {b:2d}): " + "  ".join(f"{c} {act(Xt[c][b], g, u).mean():+.2f}" for c in CONDS))
print("  same rows on the UNEDITED model (should be ~0 for links >= 1):")
for k, (b, _, g, u, _) in enumerate(links):
    print(f"  link {k} (block {b:2d}): " + "  ".join(f"{c} {act(C['test', c][b], g, u).mean():+.2f}" for c in CONDS))

# ---- write side: full chain and one knockout per link -------------------------------------------------------
print("\n== write side: bardness (archaic words per 100) on HELD-OUT prompts ==")
TOK, DUMP = int(os.environ.get("TOK", 48)), {}
variants = [("full chain", chain)] + [(f"knock out link {k} (block {BLOCKS[k]})", k) for k in range(N)]
for name, m in tqdm(variants, desc="models"):
    show = m is chain
    if not show:
        m = build(skip=m)
    outs = {c: chat(m, tok, test[c], max_new_tokens=TOK, desc=c) for c in CONDS}
    DUMP[name] = {c: [{"prompt": p, "answer": a} for p, (_, _, a) in zip(test[c], outs[c])] for c in CONDS}
    json.dump(DUMP, open(os.environ.get("OUT", "examples_chain.json"), "w"), indent=1, ensure_ascii=False)
    print(f"\n### {name}: " + "  ".join(f"{c} {sum(bardness(a) for *_, a in outs[c]) / len(outs[c]):.1f}" for c in CONDS))
    for c in CONDS if show else ("both",):
        for p, (_, _, a) in list(zip(test[c], outs[c]))[:3]:
            print(f"  [{c}] {p[-45:]!r}\n      -> {a[:200]!r}")
    if not show:
        del m
