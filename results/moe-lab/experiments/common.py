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


def run_grid(kind: str, exp: str, configs: list[dict], workers: int = 4, threads: int = 1):
    out_dir = RESULTS / exp
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for cfg in configs:
        path = out_dir / f"{cfg['name']}__s{cfg['seed']}.json"
        if not path.exists():
            todo.append((cfg, str(path)))
    print(f"[{exp}] {len(configs)} runs, {len(configs) - len(todo)} cached, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
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
