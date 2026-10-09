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
    x_test: torch.Tensor | None = None  # held out from both training and model/hyperparameter selection
    y_test: torch.Tensor | None = None
    c_test: torch.Tensor | None = None
    loss: str = "mse"  # "mse": y is a real vector; "ce": y is a class index


def make_cluster_task(
    n_clusters: int = 8,
    d_in: int = 16,
    d_out: int = 8,
    teacher_hidden: int = 32,
    n_train: int = 8000,
    n_val: int = 4000,
    n_test: int = 4000,
    separation: float = 3.0,
    freq: str = "uniform",  # "uniform" | "zipf"
    zipf_s: float = 1.0,
    routing: str = "clusters",  # "clusters": task = Gaussian cluster (linearly separable); "xor": task = XOR bits
    loss: str = "mse",  # "mse": regress the teacher's output; "ce": classify the teacher's argmax over d_out classes
    seed: int = 0,
) -> ClusterTask:
    """Each latent task has its own random teacher network.

    routing="clusters": x = mean_c + z, task c drawn by frequency; a linear router can separate the tasks.
    routing="xor": x ~ N(0, I) and bit j of the task id is sign(x[2j] * x[2j+1]); no linear function of x carries
    any information about any bit, so routing by task requires learned (nonlinear) features upstream of the router.
    """
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
    if routing == "xor":
        n_bits = int(round(np.log2(n_clusters)))
        assert 2**n_bits == n_clusters and 2 * n_bits <= d_in and freq == "uniform", "xor needs C = 2^b, d_in >= 2b"

    def sample(n):
        if routing == "clusters":
            c = torch.multinomial(torch.tensor(freqs, dtype=torch.float), n, replacement=True, generator=g)
            z = torch.randn(n, d_in, generator=g)  # local coordinates inside the cluster
            x = means[c] + z
        elif routing == "xor":
            z = x = torch.randn(n, d_in, generator=g)
            bits = (x[:, 0 : 2 * n_bits : 2] * x[:, 1 : 2 * n_bits : 2] > 0).long()
            c = (bits * (2 ** torch.arange(n_bits))).sum(-1)
        else:
            raise ValueError(routing)
        h = torch.tanh(torch.einsum("nd,ndh->nh", z, w1[c]) + b1[c])
        y = torch.einsum("nh,nho->no", h, w2[c])
        return x, y, c

    x_tr, y_tr, c_tr = sample(n_train)
    x_va, y_va, c_va = sample(n_val)
    x_te, y_te, c_te = sample(n_test)  # drawn last so train/val are identical to earlier versions
    mu, sd = x_tr.mean(0), x_tr.std(0)
    if loss == "mse":
        ymu, ysd = y_tr.mean(0), y_tr.std(0)
        ys = [(y - ymu) / ysd for y in (y_tr, y_va, y_te)]
    elif loss == "ce":
        ys = [y.argmax(-1) for y in (y_tr, y_va, y_te)]
    else:
        raise ValueError(loss)
    return ClusterTask(
        (x_tr - mu) / sd, ys[0], c_tr,
        (x_va - mu) / sd, ys[1], c_va,
        n_clusters, freqs,
        (x_te - mu) / sd, ys[2], c_te,
        loss,
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
