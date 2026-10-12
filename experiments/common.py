"""Shared experiment plumbing: config grids, parallel execution with resume, result loading."""
from __future__ import annotations

import copy
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"


def merge(base: dict, *overrides: dict) -> dict:
    """Merge nested dicts one level deep (enough for the moe/task sub-dicts)."""
    out = copy.deepcopy(base)
    for o in overrides:
        for k, v in o.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = {**out[k], **v}
            else:
                out[k] = v
    return out


def _worker(kind: str, cfg: dict, path: str, threads: int):
    import torch

    torch.set_num_threads(threads)
    cfg = {**cfg, "device": assign_device(cfg.get("device", "cpu"), _WORKER_GPU)} if "device" in cfg else cfg
    try:
        if kind == "synthetic":
            from moe_lab.train_synthetic import SynthRunConfig, run_synthetic

            res = run_synthetic(SynthRunConfig(**cfg))
        else:
            from moe_lab.train_lm import LMRunConfig, run_lm

            res = run_lm(LMRunConfig(**cfg))
    except Exception:
        return path, traceback.format_exc()
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, path)
    return path, None


def require_cuda(device: str):
    """Exit with one clear message (instead of one traceback per run) when a CUDA device is asked for but absent."""
    import torch

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        sys.exit(f"--device {device} requested but no CUDA GPU is available in this session "
                 "(on Kaggle: Settings -> Accelerator -> GPU T4 x2, then restart).")


_WORKER_GPU = None  # set per worker process by _init_worker when a grid runs on CUDA


def _init_worker(gpu_ids):
    global _WORKER_GPU
    _WORKER_GPU = gpu_ids.get()


def assign_device(device: str, gpu: int | None) -> str:
    """Plain 'cuda' becomes this worker's own GPU (cuda:0, cuda:1, ...); an explicit 'cuda:N' or 'cpu' is kept."""
    if device == "cuda" and gpu is not None:
        return f"cuda:{gpu}"
    return device


def run_grid(kind: str, exp: str, configs: list[dict], workers: int = 4, threads: int = 1):
    out_dir = RESULTS / exp
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for cfg in configs:
        path = out_dir / f"{cfg['name']}__s{cfg['seed']}.json"
        if not path.exists():
            todo.append((cfg, str(path)))
    print(f"[{exp}] {len(configs)} runs, {len(configs) - len(todo)} cached, {len(todo)} to run", flush=True)
    pool = dict(max_workers=workers)
    if any(str(cfg.get("device", "cpu")).startswith("cuda") for cfg, _ in todo):
        import multiprocessing

        import torch

        # One GPU per worker process (so --workers 2 on T4 x2 uses both cards, one run each); spawn because CUDA is
        # not fork-safe once the parent has queried it.
        ctx = multiprocessing.get_context("spawn")
        gpu_ids = ctx.Queue()
        n_gpus = max(1, torch.cuda.device_count())
        for i in range(workers):
            gpu_ids.put(i % n_gpus)
        pool.update(mp_context=ctx, initializer=_init_worker, initargs=(gpu_ids,))
        print(f"[{exp}] {workers} worker(s) over {n_gpus} GPU(s)", flush=True)
    with ProcessPoolExecutor(**pool) as ex:
        futs = [ex.submit(_worker, kind, cfg, path, threads) for cfg, path in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            path, err = fut.result()
            status = "FAILED\n" + err if err else "ok"
            print(f"[{exp}] {i}/{len(todo)} {Path(path).name}: {status}", flush=True)


def load_results(exp: str) -> list[dict]:
    out = []
    for p in sorted((RESULTS / exp).glob("*.json")):
        with open(p) as f:
            out.append(json.load(f))
    return out
