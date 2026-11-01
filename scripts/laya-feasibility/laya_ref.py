"""Laya feasibility: PyTorch reference, ONNX export, ORT parity, token dump for the C++ harness.

Re-implements the inference path of convaiinnovations/laya (rl_common.py / rl_agent_api.py,
read as text) so no code from the model repo is imported or executed.

usage: laya_ref.py <model_dir> <out_dir> [--fp16]
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QNAMES = {v: k for k, v in QTYPES.items()}


def render_options(q):
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [k if not v else "%s: %s" % (k, v) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, c) for i, c in enumerate(crit)]
    crit = crit or {}
    return ["false: " + (crit.get("false") or "no, the statement does not hold"),
            "true: " + (crit.get("true") or "yes, the statement holds")]


def build_sequence(tok, state, q, max_len, head_max_len):
    mask_tok = tok.mask_token
    opts = render_options(q)
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = tok("%s question: %s" % (q["t"], ins), add_special_tokens=False)["input_ids"]
    opt_ids = [[tok.mask_token_id] + tok(" " + o.replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:48]
               for o in opts]
    budget = head_max_len - sum(len(o) for o in opt_ids)
    if budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[:max(8, budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    st_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    st = tok(st_text.replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:room]
    ids = ids + st + [tok.sep_token_id]
    return ids[:max_len], [m for m in markers if m < max_len]


def self_attn(layer, x, neg):
    """nn.MultiheadAttention (batch_first, eval) written out so the export keeps a dynamic length."""
    mha = layer.self_attn
    d, nh = mha.embed_dim, mha.num_heads
    b, n = x.shape[0], x.shape[1]
    qkv = torch.nn.functional.linear(x, mha.in_proj_weight, mha.in_proj_bias)
    qkv = qkv.reshape(b, n, 3, nh, d // nh).permute(2, 0, 3, 1, 4)
    att = torch.matmul(qkv[0], qkv[1].transpose(-1, -2)) / (d // nh) ** 0.5 + neg
    o = torch.matmul(torch.softmax(att, -1), qkv[2]).transpose(1, 2).reshape(b, n, d)
    return mha.out_proj(o)


class DecisionModel(nn.Module):
    def __init__(self, encoder, head_layers=2, n_act=2):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        layer = nn.TransformerEncoderLayer(d, max(1, d // 64), 4 * d, 0.1, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.register_buffer("temperature", torch.ones(3))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :]
        neg = (1.0 - attention_mask[:, None, None, :].to(h.dtype)) * -1e9
        for layer in self.head.layers:
            h = h + self_attn(layer, layer.norm1(h), neg)
            h = h + layer.linear2(torch.relu(layer.linear1(layer.norm2(h))))
        idx = marker_pos[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        # act head (escalate) dropped from the export: xllama would only need the answer distribution
        return logits


def load(model_dir, attn):
    cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    ecfg = AutoConfig.from_pretrained(os.path.join(model_dir, "encoder"))
    enc = AutoModel.from_config(ecfg, attn_implementation=attn)
    m = DecisionModel(enc, cfg["head_layers"], len(cfg["act_costs"]) + 1)
    m.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    return cfg, tok, m.eval()


def temp_for(cfg, qt, k):
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return cfg.get("temperature_by_options", {}).get("%s:%s" % (QNAMES[qt], size), cfg["temperature"][qt])


CASES = [
    {"state": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
     "q": {"t": "choice", "ins": "Which department should handle this?",
           "crit": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, system errors",
                    "other": "everything else"}}},
    {"state": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
     "q": {"t": "score", "ins": "How urgent is this?", "crit": ["not urgent", "soon", "blocking"]}},
    {"state": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
     "q": {"t": "noul", "ins": "Does the user threaten to cancel or leave?", "crit": None}},
    {"state": "The app crashes every time I open the settings page since the last update.",
     "q": {"t": "choice", "ins": "Which department should handle this?",
           "crit": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, system errors",
                    "other": "everything else"}}},
    {"state": "您好，三月份的帳單被重複扣款兩次，請今天退還重複的金額，否則我們會取消訂閱。",
     "q": {"t": "choice", "ins": "Which department should handle this?",
           "crit": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, system errors",
                    "other": "everything else"}}},
    {"state": "每次打開設定頁面 App 就會閃退，從上次更新之後就這樣。",
     "q": {"t": "choice", "ins": "Which department should handle this?",
           "crit": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, system errors",
                    "other": "everything else"}}},
    {"state": "每次打開設定頁面 App 就會閃退，從上次更新之後就這樣。",
     "q": {"t": "noul", "ins": "Does the user threaten to cancel or leave?", "crit": None}},
]


def softmax(z):
    p = np.exp(z - z.max())
    return p / p.sum()


def main():
    model_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_num_threads(os.cpu_count())
    cfg, tok, model = load(model_dir, "eager")

    items, ref = [], []
    for c in CASES:
        ids, markers = build_sequence(tok, c["state"], c["q"], cfg["max_len"], cfg["head_max_len"])
        k, qt = len(markers), QTYPES[c["q"]["t"]]
        a = lambda x, dt=torch.long: torch.tensor([x], dtype=dt)
        t0 = time.perf_counter()
        with torch.no_grad():
            lg = model(a(ids), a([1] * len(ids)), a(markers), a([True] * k, torch.bool), a(qt))[0].numpy()
        ms = (time.perf_counter() - t0) * 1000
        T = temp_for(cfg, qt, k)
        p = softmax(lg[:k] / T)
        opts = render_options(c["q"])
        print("[torch %6.1f ms, %3d tok] %-5s %-48s -> %s" % (
            ms, len(ids), c["q"]["t"], c["state"][:46],
            ", ".join("%s=%.3f" % (o.split(":")[0], v) for o, v in zip(opts, p))))
        items.append({"ids": ids, "markers": markers, "qtype": qt, "temperature": T,
                      "options": [o.split(":")[0] for o in opts], "torch_probs": p.tolist()})
        ref.append(p)

    # export: batch 1, dynamic sequence length and option count
    L = len(items[0]["ids"])
    k = len(items[0]["markers"])
    dummy = (torch.tensor([items[0]["ids"]]), torch.ones(1, L, dtype=torch.long),
             torch.tensor([items[0]["markers"]]), torch.ones(1, k, dtype=torch.bool),
             torch.tensor([items[0]["qtype"]]))
    onnx_path = os.path.join(out_dir, "laya.onnx")
    t0 = time.perf_counter()
    torch.onnx.export(model, dummy, onnx_path, dynamo=False, opset_version=17,
                      input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
                      output_names=["logits"],
                      dynamic_axes={"input_ids": {1: "seq"}, "attention_mask": {1: "seq"},
                                    "marker_pos": {1: "k"}, "marker_mask": {1: "k"}, "logits": {1: "k"}})
    print("exported %s (%.0f MB) in %.0fs" % (onnx_path, os.path.getsize(onnx_path) / 1e6, time.perf_counter() - t0))

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    worst = 0.0
    for it, p_ref in zip(items, ref):
        k = len(it["markers"])
        feed = {"input_ids": np.array([it["ids"]], np.int64),
                "attention_mask": np.ones((1, len(it["ids"])), np.int64),
                "marker_pos": np.array([it["markers"]], np.int64),
                "marker_mask": np.ones((1, k), bool), "qtype": np.array([it["qtype"]], np.int64)}
        lg = sess.run(None, feed)[0][0]
        p = softmax(lg[:k] / it["temperature"])
        worst = max(worst, float(np.abs(p - p_ref).max()))
        it["ort_py_probs"] = p.tolist()
    print("ORT(python) vs torch: max |dp| = %.2e" % worst)
    json.dump({"pad_id": tok.pad_token_id, "items": items}, open(os.path.join(out_dir, "cases.json"), "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
