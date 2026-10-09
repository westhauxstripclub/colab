"""E7 - When does routing settle in a real, large MoE? (Colab / GPU; needs Hugging Face access)

Loads a public MoE's intermediate pretraining checkpoints (default: OLMoE-1B-7B, which publishes them as branches),
runs a fixed text sample from this repo's corpus through each one, and reports per checkpoint and layer:
  * load balance        effective experts, dead experts, max/mean load
  * router saturation   overlap of each token's top-k experts with the final checkpoint's top-k
                        (1 = routing already identical to the end of training)
  * specialisation      MI between the top-1 expert and the text domain

This tests the round-1 claim that routing outcomes are fixed early. Caveat: OLMoE was trained *with* load
balancing, so it shows when routing settles under balancing; it cannot show unbalanced collapse at scale.

STATUS: tested in this repo only against tiny randomly initialised OLMoE-architecture models (--local_dirs); it has
not been run on the real checkpoints here because huggingface.co is not reachable from the build environment.
Memory: OLMoE-1B-7B is ~14 GB in bf16; use a >= 24 GB GPU (A100/L4) or --device_map auto with CPU offload.

Usage:
  pip install -r requirements-colab.txt
  python e7_real_checkpoints.py --repo allenai/OLMoE-1B-7B-0924 --max_checkpoints 10 --device cuda
  python e7_real_checkpoints.py --local_dirs ckpt_a ckpt_b --tokenizer bytes          # offline test
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moe_lab.data import load_text_corpus  # noqa: E402
from moe_lab.metrics import load_stats, specialization  # noqa: E402

OUT_DIR = ROOT / "results" / "e7_real_checkpoints"


def parse_step(name: str) -> int | None:
    m = re.search(r"step(\d+)", name)
    return int(m.group(1)) if m else None


def pick_checkpoints(names: list[str], k: int) -> list[str]:
    """Up to k step-named branches spread log-uniformly over training (first and last always included)."""
    steps = sorted({(parse_step(n), n) for n in names if parse_step(n) is not None})
    if len(steps) <= k:
        return [n for _, n in steps]
    lo, hi = math.log1p(steps[0][0]), math.log1p(steps[-1][0])
    targets = np.expm1(np.linspace(lo, hi, k))
    chosen = []
    for t in targets:
        s, n = min((sn for sn in steps if sn[1] not in chosen), key=lambda sn: abs(math.log1p(sn[0]) - math.log1p(t)))
        chosen.append(n)
    return sorted(chosen, key=parse_step)


def text_sample(n_chunks: int, chunk_chars: int):
    """Fixed chunks of each domain's held-out text (same for every checkpoint)."""
    corpus = load_text_corpus(str(ROOT / "data"))
    rng = np.random.default_rng(0)
    texts, domains = [], []
    for d, t in enumerate(corpus.val):
        raw = bytes(t.tolist())
        for _ in range(n_chunks):
            o = int(rng.integers(0, len(raw) - chunk_chars - 1))
            texts.append(raw[o : o + chunk_chars].decode("utf-8", errors="ignore"))
            domains.append(d)
    return texts, domains, corpus.domains


def tokenize(texts, tokenizer, seq_len, vocab_size):
    if tokenizer == "bytes":
        ids = [torch.tensor(list(t.encode("utf-8"))[:seq_len]) % vocab_size for t in texts]
    else:
        ids = [torch.tensor(tokenizer(t, add_special_tokens=False)["input_ids"][:seq_len]) for t in texts]
    n = min(len(i) for i in ids)
    return torch.stack([i[:n] for i in ids])


@torch.no_grad()
def route(model, ids, batch_size, device):
    """Per layer: [N_tokens, E] router logits (concatenated over batches)."""
    per_layer = None
    for i in range(0, len(ids), batch_size):
        out = model(input_ids=ids[i : i + batch_size].to(device), output_router_logits=True)
        logits = [l.float().cpu() for l in out.router_logits]
        per_layer = logits if per_layer is None else [torch.cat([a, b]) for a, b in zip(per_layer, logits)]
    return per_layer


def analyse(per_layer, k, token_domains):
    layers, topks = [], []
    for logits in per_layer:
        top = logits.topk(k, dim=-1).indices
        M = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, top, True)
        st = load_stats(M)
        st.pop("load")
        st["domain_info_recovered"] = specialization(top[:, 0].numpy(), token_domains)["label_info_recovered"]
        layers.append(st)
        topks.append(top)
    return layers, topks


def saturation(top_t, top_final, k):
    eq = (top_t.unsqueeze(-1) == top_final.unsqueeze(-2)).any(-1)  # [N, k]: is each chosen expert also chosen at the end?
    return float(eq.float().sum(-1).mean() / k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--local_dirs", nargs="*", default=None, help="analyse saved checkpoints instead of a hub repo")
    ap.add_argument("--max_checkpoints", type=int, default=10)
    ap.add_argument("--tokenizer", default=None, help="'bytes' for byte ids (offline tests); default: the repo's")
    ap.add_argument("--n_chunks", type=int, default=32, help="text chunks per domain")
    ap.add_argument("--chunk_chars", type=int, default=1500)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, a.dtype)
    if a.local_dirs:
        sources = [(Path(d).name, dict(pretrained_model_name_or_path=d)) for d in a.local_dirs]
        slug = "local"
    else:
        from huggingface_hub import list_repo_refs

        branches = [b.name for b in list_repo_refs(a.repo).branches]
        chosen = pick_checkpoints(branches, a.max_checkpoints)
        sources = [(n, dict(pretrained_model_name_or_path=a.repo, revision=n)) for n in chosen]
        sources.append(("main", dict(pretrained_model_name_or_path=a.repo)))
        slug = a.repo.replace("/", "__")
    tokenizer = a.tokenizer if a.tokenizer == "bytes" else AutoTokenizer.from_pretrained(
        a.local_dirs[-1] if a.local_dirs else a.repo)

    texts, domains, domain_names = text_sample(a.n_chunks, a.chunk_chars)
    ids = None
    results, topks_all = [], []
    for label, kw in sources:
        print(f"loading {label} ...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(**kw, dtype=dtype).to(a.device).eval()
        if ids is None:
            ids = tokenize(texts, tokenizer, a.seq_len, model.config.vocab_size)
            token_domains = np.repeat(np.array(domains), ids.shape[1])
        k = model.config.num_experts_per_tok
        layers, topks = analyse(route(model, ids, a.batch_size, a.device), k, token_domains)
        results.append({"checkpoint": label, "step": parse_step(label), "layers": layers})
        topks_all.append(topks)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for res, topks in zip(results, topks_all):
        for st, t, tf in zip(res["layers"], topks, topks_all[-1]):
            st["saturation_vs_final"] = saturation(t, tf, t.shape[1])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(a.out) if a.out else OUT_DIR / f"{slug}.json"
    out.write_text(json.dumps({"repo": None if a.local_dirs else a.repo, "domains": list(domain_names),
                               "n_tokens": int(ids.numel()), "checkpoints": results}, indent=1))
    print(f"wrote {out}")
    for r in results:
        L = r["layers"]
        print(f"{r['checkpoint']:>28s}  eff.experts mean {np.mean([l['effective_experts'] for l in L]):6.2f}  "
              f"dead {np.mean([l['dead_experts'] for l in L]):5.2f}  "
              f"saturation {np.mean([l['saturation_vs_final'] for l in L]):.3f}")


if __name__ == "__main__":
    main()
