#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Score a protocol-v4 decision model (pplx-decider) on question suites through one of three backends.

Prompt building and the readout are the reference ``pplx_decider`` code shipped inside the checkpoint
(``<ckpt>/source/src``); a backend only supplies the final hidden state at the readout marker.

Backends
  torch     reference PyTorch backbone (bf16 on CUDA). ``--patch FILE`` swaps in dequantized weights from
            dequant_patch.py, which measures weight-only quantization error.
  llamacpp  a llama.cpp server started with ``--embeddings --pooling none`` on a GGUF prepared by
            set_noncausal.py. Token ids are sent directly, so tokenization is exactly the reference one.
  xbox      xllama's existing ``/api/embed`` on the console (the GGUF registered with role "embedding" through
            a LocalState manifest.json override). The prompt is sent as text and xllama returns the
            L2-normalized last-token state: the top answer is exact, the scale is not. With ``--ref-url`` (a
            llamacpp server on the same GGUF) probabilities are rebuilt with the reference norm and the cosine
            between the two states is recorded.

Suites are Kev-format JSONL records ({"state", "questions": {qid: {..., "label"}}}); every question is asked on
its own (the checkpoint's evaluated "separate" layout). Records are shuffled with seed 0 and the first N
questions of each suite are kept, so runs on different backends score the same questions.

usage:
  decider_check.py --ckpt DIR --backend {torch,llamacpp,xbox} --out FILE.jsonl
                   --suite NAME=PATH[:N] [--suite ...] [--url URL] [--ref-url URL] [--patch FILE]
                   [--model pplx-decider-q5] [--ctx 512]
"""

import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, type=Path, help="pplx-decider checkpoint directory (HF snapshot)")
    p.add_argument("--backend", required=True, choices=("torch", "llamacpp", "xbox"))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--suite", action="append", required=True, help="NAME=PATH[:N] (N questions kept; default all)")
    p.add_argument("--url", help="llamacpp: server URL (http://127.0.0.1:8790); xbox: http://<console>:11434")
    p.add_argument("--ref-url", help="xbox only: llama.cpp server on the same GGUF, for norms and cosines")
    p.add_argument("--patch", type=Path, help="torch only: dequantized weight patch (dequant_patch.py)")
    p.add_argument("--model", default="pplx-decider-q5", help="xbox only: model id on the console")
    p.add_argument("--ctx", type=int, default=512, help="xbox only: options.num_ctx; longer prompts are skipped")
    return p.parse_args()


def questions(spec):
    name, _, rest = spec.partition("=")
    path, limit = rest, None
    head, sep, tail = rest.rpartition(":")
    if sep and tail.isdigit():
        path, limit = head, int(tail)
    recs = [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]
    random.Random(0).shuffle(recs)
    k = 0
    for r in recs:
        for qid, q in r["questions"].items():
            if limit is not None and k >= limit:
                return
            q = dict(q)
            label = q.pop("label")
            q.pop("src", None)
            yield name, r.get("_meta", {}).get("id", ""), qid, {"state": r["state"], "questions": {qid: q}}, label
            k += 1


def correct(kind, label, ans):
    if kind == "choice":
        return ans["choice"] == label
    if kind == "noul":
        return (ans["noul"] >= 0.5) == bool(label)
    return round(ans["score"]) == int(label)


def post(url, body, timeout=900):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.load(res)


def main():
    a = parse_args()
    sys.path.insert(0, str(a.ckpt / "source" / "src"))
    import torch
    from safetensors.torch import load_file
    from transformers import AutoProcessor
    from pplx_decider.protocol import MARKER, SEPARATOR, Encoder, answer

    cfg = json.loads((a.ckpt / "decision_config.json").read_text())
    W = load_file(str(a.ckpt / "readout.safetensors"))["weight"].float()
    T = float(cfg["temperature"])
    enc = Encoder(AutoProcessor.from_pretrained(str(a.ckpt)), cfg["codes"])

    def llamacpp_hidden(url, ids):
        rows = post(url + "/embedding", {"content": ids})[0]["embedding"]
        assert len(rows) == len(ids), (len(rows), len(ids))
        return torch.tensor(rows[-1])  # the readout marker is the last token

    decider = None
    if a.backend == "torch":
        from pplx_decider import Decider

        decider = Decider.from_pretrained(a.ckpt, device="cuda")
        if a.patch:
            params = dict(decider.backbone.named_parameters())
            with torch.no_grad():
                for k, v in load_file(str(a.patch)).items():
                    name = next(c for c in (k, k.removeprefix("language_model."), "model." + k) if c in params)
                    params[name].copy_(v.to(params[name].dtype))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    items = [it for spec in a.suite for it in questions(spec)]
    with a.out.open("w", encoding="utf-8") as f:
        for i, (suite, rid, qid, req, label) in enumerate(items):
            q = req["questions"][qid]
            (e,) = enc.plan(req, None, "separate")
            row = {"suite": suite, "id": rid, "qid": qid, "type": q["type"], "label": label, "tokens": e.length}
            t = time.perf_counter()
            try:
                n = e.counts[0]
                if a.backend == "torch":
                    ans = decider.decide(req)["answers"][qid]
                elif a.backend == "llamacpp":
                    h = llamacpp_hidden(a.url, e.input_ids.tolist())
                    ans = answer(q, ((W[:n] @ h) / T).softmax(-1).tolist())
                else:
                    if e.length > a.ctx:
                        row["skipped"] = f"{e.length} tokens > num_ctx {a.ctx}"
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        continue
                    text = enc.context_text(req) + SEPARATOR + enc.question_text(q) + MARKER
                    res = post(
                        a.url + "/api/embed",
                        {"model": a.model, "input": text, "truncate": False, "options": {"num_ctx": a.ctx}},
                    )
                    hx = torch.tensor(res["embeddings"][0])
                    row["top_index"] = int((W[:n] @ hx).argmax())
                    row["console_ms"] = round(res.get("total_duration", 0) / 1e6)
                    if a.ref_url:
                        hp = llamacpp_hidden(a.ref_url, e.input_ids.tolist())
                        row["cos"] = round(float(torch.nn.functional.cosine_similarity(hx, hp, dim=0)), 5)
                        row["ref_top_index"] = int((W[:n] @ hp).argmax())
                        hx = hx * hp.norm()
                    ans = answer(q, ((W[:n] @ hx) / T).softmax(-1).tolist())
                row.update(ok=correct(q["type"], label, ans), answer=ans, error=None)
            except Exception as ex:  # keep going; the failure is part of the record
                row.update(ok=False, answer=None, error=repr(ex)[:300])
            row["ms"] = round((time.perf_counter() - t) * 1000, 1)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0:
                print(i, suite, row.get("ok"), row.get("error") or "", flush=True)


if __name__ == "__main__":
    main()
