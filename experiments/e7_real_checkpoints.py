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
not yet completed on the real checkpoints.
Memory: OLMoE-1B-7B is 13.8 GB in 16-bit. One >= 24 GB GPU (A100/L4) holds it; on two 15 GB cards (Kaggle T4 x2) use
--device_map auto to split the layers across both. T4/P100 have no native bf16, so use --dtype float16 there; the
script stops if any router logit is non-finite (a float16 overflow) rather than reporting metrics from it.

Usage:
  pip install -r requirements-colab.txt
  python e7_real_checkpoints.py --repo allenai/OLMoE-1B-7B-0924 --max_checkpoints 10 --device cuda
  python e7_real_checkpoints.py --max_checkpoints 10 --dtype float16 --device_map auto     # Kaggle T4 x2
  python e7_real_checkpoints.py --local_dirs ckpt_a ckpt_b --tokenizer bytes          # offline test
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # read at the first CUDA allocation
import numpy as np  # noqa: E402
import torch  # noqa: E402

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
        bad = [li for li, l in enumerate(logits) if not torch.isfinite(l).all()]
        if bad:
            raise FloatingPointError(f"non-finite router logits in layers {bad} (dtype {model.dtype}); "
                                     "float16 can overflow on a bf16-trained model -- try --dtype bfloat16 or float32")
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


def gpu_mem() -> str:
    return ", ".join(f"cuda:{i} {torch.cuda.memory_allocated(i) / 2**30:.1f} GiB" for i in range(torch.cuda.device_count()))


def free_model():
    """Release the previous checkpoint. A model dispatched over several GPUs can survive `del` via reference cycles
    (accelerate hooks), which made the second OLMoE checkpoint run out of memory on Kaggle T4 x2."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        left = max(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count()))
        print(f"  freed: {gpu_mem()}", flush=True)
        if left > 2**30:
            raise RuntimeError(f"previous checkpoint still holds GPU memory after freeing ({gpu_mem()})")


def max_memory(headroom_gib: float):
    """Per-GPU budget for device_map, leaving room for transformers' load-time conversion buffers (the fused expert
    weights are concatenated on the GPU, 512 MiB at a time for OLMoE)."""
    return {i: f"{max(1.0, torch.cuda.get_device_properties(i).total_memory / 2**30 - headroom_gib):.1f}GiB"
            for i in range(torch.cuda.device_count())}


def load_cached(cache: Path, label: str, params: dict):
    j, z = cache / f"{label}.json", cache / f"{label}.npz"
    if not (j.exists() and z.exists()):
        return None
    res = json.loads(j.read_text())
    if res.pop("params") != params:
        return None
    arr = np.load(z)["topk"]
    return res, [torch.from_numpy(a.astype(np.int64)) for a in arr]


def save_cached(cache: Path, label: str, params: dict, res: dict, topks):
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache / f"{label}.npz", topk=np.stack([t.numpy().astype(np.uint8) for t in topks]))
    (cache / f"{label}.json").write_text(json.dumps({**res, "params": params}))  # written last: marks it complete


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
    ap.add_argument("--device_map", default=None,
                    help="e.g. 'auto': let accelerate split the model over all visible GPUs (overrides --device)")
    ap.add_argument("--headroom_gib", type=float, default=3.0,
                    help="with --device_map: GPU memory per card left free for load-time buffers")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    wants_cuda = str(a.device).startswith("cuda") if a.device_map is None else a.device_map not in ("cpu",)
    if wants_cuda and not torch.cuda.is_available():  # before any multi-GB download
        asked = f"--device_map {a.device_map}" if a.device_map else f"--device {a.device}"
        sys.exit(f"{asked} needs a CUDA GPU but none is available "
                 "(on Kaggle: Settings -> Accelerator -> GPU T4 x2, then restart).")

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

    out = Path(a.out) if a.out else OUT_DIR / f"{slug}.json"
    cache = out.parent / out.stem  # one file pair per finished checkpoint, so a crash or timeout loses nothing
    params = dict(n_chunks=a.n_chunks, chunk_chars=a.chunk_chars, seq_len=a.seq_len, tokenizer=str(a.tokenizer))
    texts, domains, domain_names = text_sample(a.n_chunks, a.chunk_chars)
    ids = None
    results, topks_all = [], []
    for label, kw in sources:
        cached = load_cached(cache, label, params)
        if cached:
            print(f"cached {label}", flush=True)
            results.append(cached[0])
            topks_all.append(cached[1])
            continue
        print(f"loading {label} ...", flush=True)
        if a.device_map is None:
            model = AutoModelForCausalLM.from_pretrained(**kw, dtype=dtype).to(a.device).eval()
            in_device = a.device
        else:
            extra = dict(max_memory=max_memory(a.headroom_gib)) if a.device_map != "cpu" else {}
            model = AutoModelForCausalLM.from_pretrained(**kw, dtype=dtype, device_map=a.device_map, **extra).eval()
            in_device = model.device  # where the embedding layer lives; accelerate moves activations between cards
        if torch.cuda.is_available():
            print(f"  loaded: {gpu_mem()}", flush=True)
        if ids is None:
            ids = tokenize(texts, tokenizer, a.seq_len, model.config.vocab_size)
            token_domains = np.repeat(np.array(domains), ids.shape[1])
        k = model.config.num_experts_per_tok
        layers, topks = analyse(route(model, ids, a.batch_size, in_device), k, token_domains)
        res = {"checkpoint": label, "step": parse_step(label), "layers": layers, "n_tokens": int(ids.numel())}
        save_cached(cache, label, params, res, topks)
        results.append(res)
        topks_all.append(topks)
        del model
        free_model()
    for res, topks in zip(results, topks_all):
        for st, t, tf in zip(res["layers"], topks, topks_all[-1]):
            st["saturation_vs_final"] = saturation(t, tf, t.shape[1])

    out.parent.mkdir(parents=True, exist_ok=True)
    n_tokens = results[0].pop("n_tokens")
    for r in results[1:]:
        r.pop("n_tokens")
    out.write_text(json.dumps({"repo": None if a.local_dirs else a.repo, "domains": list(domain_names),
                               "n_tokens": n_tokens, "checkpoints": results}, indent=1))
    print(f"wrote {out}")
    for r in results:
        L = r["layers"]
        print(f"{r['checkpoint']:>28s}  eff.experts mean {np.mean([l['effective_experts'] for l in L]):6.2f}  "
              f"dead {np.mean([l['dead_experts'] for l in L]):5.2f}  "
              f"saturation {np.mean([l['saturation_vs_final'] for l in L]):.3f}")


if __name__ == "__main__":
    main()
