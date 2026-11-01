"""Route B check: feed llama.cpp encoder hidden states through the Laya head (PyTorch) and compare
encoder states and final calibrated probabilities with the full PyTorch reference.

usage: head_from_hidden.py <model_dir> <out_dir> <hidden.bin>
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from laya_ref import load, self_attn, softmax  # noqa: E402

model_dir, out_dir, hidden_bin = sys.argv[1:4]
cfg, tok, model = load(model_dir, "eager")
d = model.encoder.config.hidden_size
raw = np.fromfile(hidden_bin, dtype=np.float32)
items = json.load(open(os.path.join(out_dir, "cases.json")))["items"]
off, worst_h, worst_p, argmax_ok = 0, 0.0, 0.0, 0
for it in items:
    n, k = len(it["ids"]), len(it["markers"])
    h_lc = torch.from_numpy(raw[off:off + n * d].reshape(1, n, d).copy())
    off += n * d
    ids = torch.tensor([it["ids"]])
    att = torch.ones(1, n, dtype=torch.long)
    with torch.no_grad():
        h_hf = model.encoder(input_ids=ids, attention_mask=att).last_hidden_state
        rel = float((h_lc - h_hf).norm() / h_hf.norm())
        h = h_lc + model.type_emb(torch.tensor([it["qtype"]]))[:, None, :]
        neg = torch.zeros(1, 1, 1, n)
        for layer in model.head.layers:
            h = h + self_attn(layer, layer.norm1(h), neg)
            h = h + layer.linear2(torch.relu(layer.linear1(layer.norm2(h))))
        lg = model.scorer(h[0, it["markers"]]).squeeze(-1).numpy()
    p = softmax(lg / it["temperature"])
    ref = np.array(it["torch_probs"])
    worst_h, worst_p = max(worst_h, rel), max(worst_p, float(np.abs(p - ref).max()))
    argmax_ok += int(p.argmax() == ref.argmax())
    print("%3d tok  hidden rel.err %.2e  p=%s ref=%s" % (n, rel, np.round(p, 4), np.round(ref, 4)))
print("hidden rel.err max %.2e, max |dp| %.2e, argmax agree %d/%d" % (worst_h, worst_p, argmax_ok, len(items)))
