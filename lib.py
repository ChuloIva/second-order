"""Hand-made neurons (after vgel.me/notebook/making-neurons-by-hand), ported to Qwen3.5-4B on MPS.

The HF cache is only ever read: all edits happen on in-memory clones of one MLP (lazy_clone).
"""
import copy

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.5-4B"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load(dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, local_files_only=True).to(DEVICE).eval()
    model.requires_grad_(False)
    return model, tok


def layers(model):
    """The decoder block list, wherever this architecture keeps it."""
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers"):
        m = model
        try:
            for p in path.split("."):
                m = getattr(m, p)
            return m
        except AttributeError:
            continue
    raise AttributeError("no decoder layers found")


def lazy_clone(model, except_mlp=()):
    """Shallow clone sharing every weight except the MLPs of the listed blocks, which get private copies."""
    if isinstance(except_mlp, int):
        except_mlp = (except_mlp,)

    def clone(mod):
        new = copy.copy(mod)
        for k, v in mod.__dict__.items():
            if isinstance(v, dict):
                new.__dict__[k] = copy.copy(v)   # keeps OrderedDict (hooks need weakref-able dicts)
        new._modules = {k: (clone(c) if c is not None else None) for k, c in mod._modules.items()}
        return new

    new = clone(model)
    for b in except_mlp:
        mlp = layers(new)[b].mlp
        for name in ("gate_proj", "up_proj", "down_proj"):
            lin = copy.copy(getattr(mlp, name))
            lin._parameters = dict(lin._parameters)
            lin.weight = torch.nn.Parameter(lin.weight.data.clone(), requires_grad=False)
            setattr(mlp, name, lin)
    return new


@torch.no_grad()
def chat(model, tok, prompts, system=None, max_new_tokens=64, batch=12, desc="generate"):
    """Greedy answers. Returns a list of (ids of prompt+answer, prompt length, answer text)."""
    out = []
    for i in tqdm(range(0, len(prompts), batch), desc=desc, leave=False):
        msgs = [([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": p}]
                for p in prompts[i:i + batch]]
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                 for m in msgs]
        enc = tok(texts, return_tensors="pt", padding=True).to(DEVICE)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        width = enc.input_ids.shape[1]
        for j in range(len(texts)):
            n_prompt = int(enc.attention_mask[j].sum())
            ans = gen[j, width:]
            ans = ans[~torch.isin(ans, torch.tensor(tok.all_special_ids, device=ans.device))]
            ids = torch.cat([enc.input_ids[j, width - n_prompt:], ans])
            out.append((ids, n_prompt, tok.decode(ans, skip_special_tokens=True).strip()))
    return out


@torch.no_grad()
def capture(model, tok, grabs, prompts, system=None, whole=False, max_new_tokens=48, gen_model=None, desc=""):
    """Run prompt+answer through `model`, capturing at several sites in one pass.

    grabs: {name: (module, fn(args, out) -> tensor[1, seq, d])}
    Returns {name: tensor[n_tokens, d]} over answer positions (or all positions if whole), plus
    "_spans": list of token counts per prompt so results can be split per prompt.
    """
    gens = chat(gen_model or model, tok, prompts, system, max_new_tokens, desc=f"{desc} generate")
    seen = {k: [] for k in grabs}
    cur = {}
    hooks = [mod.register_forward_hook(lambda m, a, o, k=k, fn=fn: cur.__setitem__(k, fn(a, o)[0].float().cpu()))
             for k, (mod, fn) in grabs.items()]
    spans = []
    try:
        for ids, n_prompt, _ in tqdm(gens, desc=f"{desc} capture", leave=False):
            model(input_ids=ids[None])
            start = 1 if whole else n_prompt - 1
            for k in grabs:
                seen[k].append(cur[k][start:])
            spans.append(len(ids) - start)
    finally:
        for h in hooks:
            h.remove()
    res = {k: torch.cat(v) for k, v in seen.items()}
    res["_spans"] = spans
    res["_answers"] = [g[2] for g in gens]
    return res


def mlp_in(model, block):
    return layers(model)[block].mlp, lambda a, o: a[0]


def resid(model, block):
    return layers(model)[block], lambda a, o: o[0] if isinstance(o, tuple) else o


def neuron_act(model, block, idx):
    """Pre-down_proj activation of one neuron: SiLU(gate.x) * (up.x)."""
    return layers(model)[block].mlp.down_proj, lambda a, o: a[0][..., idx:idx + 1]


@torch.no_grad()
def quietest_neuron(model, block, x):
    """Donor neuron: the one whose largest contribution over the MLP inputs x is smallest."""
    mlp = layers(model)[block].mlp
    x = x.to(DEVICE, mlp.gate_proj.weight.dtype)
    act = (F.silu(x @ mlp.gate_proj.weight.T) * (x @ mlp.up_proj.weight.T)).abs().amax(0).float()
    act = act * mlp.down_proj.weight.float().norm(dim=0)
    idx = int(act.argmin())
    return idx, float(act[idx]), float(act.median())


def gate(off, on):
    """Article's read vector: ~0 on the mean off token, 1 on the mean on token."""
    d = on - off
    d = d - off * (d @ off) / off.norm() ** 2
    return d / (d @ on)


def fit_read(means, targets, ridge=0.0):
    """Min-norm read vector r with r @ means[i] == targets[i] (the gate() idea, for any number of conditions)."""
    M = torch.stack(means).double()
    t = torch.tensor(targets, dtype=torch.double)
    G = M @ M.T + ridge * torch.eye(len(means), dtype=torch.double)
    return (M.T @ torch.linalg.solve(G, t)).float()


@torch.no_grad()
def install(model, block, idx, gate_row, up_row, write):
    mlp = layers(model)[block].mlp
    w = mlp.gate_proj.weight
    mlp.gate_proj.weight[idx] = gate_row.to(w.device, w.dtype)
    mlp.up_proj.weight[idx] = up_row.to(w.device, w.dtype)
    mlp.down_proj.weight[:, idx] = write.to(w.device, w.dtype)


BARD = ("thee", "thou", "thy", "thine", "'tis", "tis", "prithee", "hath", "doth", "dost", "art", "ye", "hast",
        "wherefore", "methinks", "forsooth", "shalt", "nay", "verily", "o'er", "ere", "'twas", "wilt", "canst")


def bardness(text):
    """Crude Shakespeare score: archaic-marker words per 100 words."""
    words = [w.strip(".,!?;:\"()*`").lower() for w in text.split()]
    return 100 * sum(w in BARD for w in words) / max(len(words), 1)


def fit_read_tokens(xs, targets, ridge=0.1):
    """Ridge-regress a read vector on tokens: xs[i] is [n_i, d] and every token of xs[i] should read targets[i].

    Unlike fit_read (which only matches condition means) this also shrinks within-condition token variance.
    Conditions are weighted equally; ridge is relative to the mean eigenvalue of the token covariance.
    """
    d = xs[0].shape[1]
    A = torch.zeros(d, d, dtype=torch.double)
    b = torch.zeros(d, dtype=torch.double)
    for x, t in zip(xs, targets):
        x = x.double()
        A += x.T @ x / len(x)
        b += x.sum(0) * t / len(x)
    A += ridge * A.trace() / d * torch.eye(d, dtype=torch.double)
    return torch.linalg.solve(A, b).float()
