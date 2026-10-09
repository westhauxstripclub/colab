"""Datasets with *known* structure, so specialisation can be measured against ground truth.

1. Synthetic "mixture of tasks": inputs come from C Gaussian clusters and each cluster has its own random
   teacher network. A single small expert cannot fit all C teachers; C specialised experts can. The
   cluster id is the ground-truth "skill" an ideal router would discover.
2. Multi-domain byte-level text: Shakespeare, Python source code and English technical documentation.
   The domain id is the ground truth for "does any expert specialise in code?".
"""
from __future__ import annotations

import glob
import os
import random
import sysconfig
import urllib.request
from dataclasses import dataclass

import numpy as np
import torch

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DOMAINS = ("shakespeare", "python_code", "tech_docs")


# ---------------------------------------------------------------------------------------- synthetic
@dataclass
class ClusterTask:
    x_train: torch.Tensor
    y_train: torch.Tensor
    c_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    c_val: torch.Tensor
    n_clusters: int
    freqs: np.ndarray


def make_cluster_task(
    n_clusters: int = 8,
    d_in: int = 16,
    d_out: int = 8,
    teacher_hidden: int = 32,
    n_train: int = 8000,
    n_val: int = 4000,
    separation: float = 3.0,
    freq: str = "uniform",  # "uniform" | "zipf"
    zipf_s: float = 1.0,
    seed: int = 0,
) -> ClusterTask:
    g = torch.Generator().manual_seed(10_000 + seed)
    means = torch.randn(n_clusters, d_in, generator=g) * separation
    w1 = torch.randn(n_clusters, d_in, teacher_hidden, generator=g) / d_in**0.5
    b1 = torch.randn(n_clusters, teacher_hidden, generator=g) * 0.5
    w2 = torch.randn(n_clusters, teacher_hidden, d_out, generator=g) / teacher_hidden**0.5

    if freq == "uniform":
        freqs = np.full(n_clusters, 1.0 / n_clusters)
    elif freq == "zipf":
        freqs = 1.0 / np.arange(1, n_clusters + 1) ** zipf_s
        freqs = freqs / freqs.sum()
    else:
        raise ValueError(freq)

    def sample(n):
        c = torch.multinomial(torch.tensor(freqs, dtype=torch.float), n, replacement=True, generator=g)
        z = torch.randn(n, d_in, generator=g)  # local coordinates inside the cluster
        x = means[c] + z
        h = torch.tanh(torch.einsum("nd,ndh->nh", z, w1[c]) + b1[c])
        y = torch.einsum("nh,nho->no", h, w2[c])
        return x, y, c

    x_tr, y_tr, c_tr = sample(n_train)
    x_va, y_va, c_va = sample(n_val)
    mu, sd = x_tr.mean(0), x_tr.std(0)
    ymu, ysd = y_tr.mean(0), y_tr.std(0)
    return ClusterTask(
        (x_tr - mu) / sd, (y_tr - ymu) / ysd, c_tr,
        (x_va - mu) / sd, (y_va - ymu) / ysd, c_va,
        n_clusters, freqs,
    )


# ---------------------------------------------------------------------------------------- text
@dataclass
class TextCorpus:
    train: list[torch.Tensor]  # one uint8 tensor per domain
    val: list[torch.Tensor]
    domains: tuple[str, ...] = DOMAINS


def _load_shakespeare(data_dir: str) -> bytes:
    path = os.path.join(data_dir, "shakespeare.txt")
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    with open(path, "rb") as f:
        return f.read()


def _load_python_code(max_bytes: int, seed: int = 0) -> bytes:
    files = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "**", "*.py"), recursive=True))
    random.Random(seed).shuffle(files)
    chunks, total = [], 0
    for path in files:
        try:
            with open(path, "rb") as f:
                b = f.read()
        except OSError:
            continue
        chunks.append(b)
        total += len(b)
        if total >= max_bytes:
            break
    return b"\n".join(chunks)[:max_bytes]


def _load_tech_docs() -> bytes:
    import pydoc_data.topics as topics  # English reference documentation shipped with CPython

    return "\n\n".join(topics.topics[k] for k in sorted(topics.topics)).encode("utf-8")


def load_text_corpus(data_dir: str = "data", bytes_per_domain: int = 550_000, val_frac: float = 0.1) -> TextCorpus:
    raw = [_load_shakespeare(data_dir), _load_python_code(bytes_per_domain), _load_tech_docs()]
    train, val = [], []
    for b in raw:
        b = b[:bytes_per_domain]
        t = torch.frombuffer(bytearray(b), dtype=torch.uint8)
        cut = int(len(t) * (1 - val_frac))
        train.append(t[:cut].clone())
        val.append(t[cut:].clone())
    return TextCorpus(train, val)


def sample_text_batch(corpus: TextCorpus, split: str, batch_size: int, seq_len: int, gen: torch.Generator):
    """Each sequence comes from one domain (chosen uniformly). Returns x, y [B, T] and domain [B]."""
    data = corpus.train if split == "train" else corpus.val
    dom = torch.randint(0, len(data), (batch_size,), generator=gen)
    xs, ys = [], []
    for d in dom.tolist():
        t = data[d]
        o = int(torch.randint(0, len(t) - seq_len - 1, (1,), generator=gen))
        chunk = t[o : o + seq_len + 1].long()
        xs.append(chunk[:-1])
        ys.append(chunk[1:])
    return torch.stack(xs), torch.stack(ys), dom
