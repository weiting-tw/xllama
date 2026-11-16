# SPDX-License-Identifier: MIT
"""Map a quantized GGUF's weights back onto the HF checkpoint layout, as a bf16 patch.

For each quantized tensor (Q4_K / Q6_K / ...): dequantize it, find the HF tensor it came from via gguf's name map, and
recover the converter's layout change (V-head reorders) by matching the f16 GGUF tensor against the HF tensor exactly
(identity, row permutation or column permutation). Only tensors whose mapping is verified exactly are written.

usage: python dequant_patch.py <stage_dir> <f16.gguf> <quant.gguf> <out_patch.safetensors>
"""
import hashlib
import json
import sys

import gguf
import numpy as np
import torch
from gguf.quants import dequantize
from safetensors import safe_open
from safetensors.torch import save_file

stage, f16_path, q_path, out_path = sys.argv[1:5]
cfg = json.load(open(f"{stage}/config.json"))
idx = json.load(open(f"{stage}/model.safetensors.index.json"))["weight_map"]
tmap = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN35, cfg["num_hidden_layers"])
gg2hf = {}
for hf in idx:
    g = tmap.get_name(hf, try_suffixes=(".weight", ".bias"))
    if g:
        gg2hf[g] = hf

f16 = {t.name: t for t in gguf.GGUFReader(f16_path).tensors}
quant = gguf.GGUFReader(q_path).tensors
handles = {}


def hf_tensor(name):
    shard = idx[name]
    if shard not in handles:
        handles[shard] = safe_open(f"{stage}/{shard}", "pt")
    return handles[shard].get_tensor(name).float()


def as_np(t):
    return dequantize(t.data, t.tensor_type).astype(np.float32).reshape(tuple(reversed([int(x) for x in t.shape])))


def row_perm(src, dst):
    """perm with dst[i] == src[perm[i]] (rows compared as fp16 bytes), or None."""
    key = lambda a: [hashlib.blake2b(r.tobytes(), digest_size=16).digest() for r in a.astype(np.float16)]
    where = {}
    for i, h in enumerate(key(src)):
        where.setdefault(h, []).append(i)
    perm = []
    for h in key(dst):
        if not where.get(h):
            return None
        perm.append(where[h].pop())
    return np.array(perm)


patch, report = {}, {"identity": 0, "rows": 0, "cols": 0, "skipped": []}
for t in quant:
    if t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16):
        continue
    hf = gg2hf.get(t.name) or gg2hf.get(t.name.removesuffix(".weight"))
    if hf is None:
        report["skipped"].append((t.name, "no hf name"));
        continue
    H = hf_tensor(hf).numpy()
    F = as_np(f16[t.name])
    Q = as_np(t)
    if F.shape != H.shape:
        if F.size != H.size:
            report["skipped"].append((t.name, f"shape {F.shape} vs {H.shape}"));
            continue
        F, Q = F.reshape(H.shape), Q.reshape(H.shape)
    H16 = H.astype(np.float16).astype(np.float32)
    if np.array_equal(H16, F):
        out, how = Q, "identity"
    elif (p := row_perm(H, F)) is not None and np.array_equal(H16[p], F):
        out = np.empty_like(Q); out[p] = Q; how = "rows"
    elif (p := row_perm(H.T, F.T)) is not None and np.array_equal(H16[:, p], F):
        out = np.empty_like(Q); out[:, p] = Q; how = "cols"
    else:
        report["skipped"].append((t.name, "layout not recovered"));
        continue
    report[how] += 1
    err = float(np.abs(out - H).mean() / (np.abs(H).mean() + 1e-12))
    patch["language_model." + hf[len("model."):]] = torch.from_numpy(out).to(torch.bfloat16)
    print(f"{t.name:40} {t.tensor_type.name:5} {how:8} rel.err {err:.4f}", flush=True)

save_file(patch, out_path, metadata={"source": q_path})
print(json.dumps({k: (v if k != "skipped" else v[:20]) for k, v in report.items()}, indent=1))
print("patched tensors:", len(patch), "skipped:", len(report["skipped"]))
