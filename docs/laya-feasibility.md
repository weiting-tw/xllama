# Laya decision model — host feasibility (2026-11-01)

Dated research note. It records a **host-only** feasibility test of the Laya
"System One" decision models. Nothing here is catalogued or shipped, and no
figure below is a Series S measurement. Status lives in
[model-matrix.md § J](model-matrix.md#j-decision-models-system-one-host-feasibility).

## What Laya is

[`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)
(Apache-2.0) is not a generative model. It reads a _state_ (text or JSON) and
typed questions (`choice`, `score`, `noul`) and returns a calibrated
probability per option in one forward pass. The model card and its
`rl_common.py` (read as text, not executed) give the layout:

```text
[CLS] "<type> question: <instructions>" [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP]
  → bidirectional encoder → + type embedding[qtype]
  → 2 × pre-norm TransformerEncoderLayer (d_ff = 4d, ReLU)
  → LayerNorm → Linear → GELU → Linear(1) at each [MASK] → softmax(logits / T)
```

`T` is a per-type, per-option-count temperature from `rl_agent_config.json`.
An `act_head` (escalate) exists and was left out; xllama would only need the
answer distribution.

| Checkpoint                    | Encoder          | Params | Context | Weights (fp32) |
| ----------------------------- | ---------------- | -----: | ------: | -------------: |
| `laya` (root, English)        | ModernBERT-large |   421M |     512 |         843 MB |
| `laya/multilingual` (100+ lg) | mmBERT-base      |   322M |    1024 |         644 MB |

## Method

Seven cases: the model card's billing e-mail (`choice` / `score` / `noul`), an
English crash report, and the same billing and crash texts in Traditional
Chinese. Reference = a PyTorch re-implementation of the inference path
(`laya_ref.py`), fp32, eager attention. The `laya` pip package was not
installed or run. Host: Linux x86-64 container, 4 threads for ORT and
llama.cpp, ONNX Runtime 1.22.0 C++ release, llama.cpp at the submodule pin.

Two routes were measured:

- **Route A — ONNX Runtime.** Export the whole model (encoder + head) to ONNX
  and run it from C++ (`laya_ort.cpp`). `nn.MultiheadAttention` in the head
  is written out by hand: the legacy exporter otherwise freezes the sequence
  length into a `Reshape`.
- **Route B — llama.cpp.** Strip the encoder into a `ModernBertModel`
  directory (`split_encoder.py`), convert with the pin's
  `convert_hf_to_gguf.py` (arch `modern-bert`), run it through `libllama`
  with `pooling NONE` (`laya_llama_enc.cpp`), and apply the head to those
  hidden states (`head_from_hidden.py`).

## Results

Max |Δp| is the largest absolute difference of any calibrated option
probability against the PyTorch fp32 reference. "argmax" counts cases whose
top option matches.

**Route A — ORT C++**

| Model        | Variant                         |   File | Peak RSS | Mean latency | Max \|Δp\| | argmax |
| ------------ | ------------------------------- | -----: | -------: | -----------: | ---------: | -----: |
| multilingual | fp32                            | 1287MB |  2048 MB |       ~86 ms |    2.7e-07 |    7/7 |
| multilingual | int8 dynamic, per-channel (all) |  324MB |   541 MB |       ~52 ms |      0.069 |    7/7 |
| English      | fp32                            | 1685MB |  3117 MB |      ~209 ms |    1.0e-06 |    7/7 |
| English      | int8 dynamic, per-tensor MatMul |  577MB |   978 MB |      ~116 ms |      0.354 |    7/7 |
| English      | int8 dynamic, per-channel (all) |  424MB |   817 MB |            — |      0.442 |      — |

The fp16 conversion (`onnxconverter_common`, `keep_io_types`) produced a graph
ORT rejects (`Cast` type mismatch inside ModernBERT attention); not pursued.

**Route B — llama.cpp encoder + head**

| Model        | Encoder GGUF | File | Peak RSS | Mean encode | Hidden rel. err | Max \|Δp\| | argmax |
| ------------ | ------------ | ---: | -------: | ----------: | --------------: | ---------: | -----: |
| multilingual | F32          | 1.2G |  1346 MB |       96 ms |          3.4e-3 |    2.7e-04 |    7/7 |
| multilingual | Q8_0         | 341M |   486 MB |      121 ms |          8.3e-2 |      0.022 |    7/7 |
| English      | F32          | 1.6G |  1559 MB |      182 ms |          1.4e-2 |    6.4e-03 |    7/7 |
| English      | Q8_0         | 421M |   453 MB |      361 ms |          1.2e-1 |      0.047 |    7/7 |

The head ran in PyTorch for route B; peak RSS is the encoder process only.

**Tokenizer.** The pin's `llama-tokenize` on the converted GGUFs gives ids
identical to the Hugging Face tokenizer for **20/20** strings per checkpoint
(the exact pieces `build_sequence` tokenizes plus CJK, Korean, Greek, Cyrillic,
emoji, accents, whitespace, digits and JSON). ModernBERT uses the
`modern-bert` pre-tokenizer; mmBERT is a Metaspace/byte-fallback BPE.

## Reading

- **Feasible.** A C++ process reproduces Laya's calibrated probabilities to
  1e-6 (ORT fp32) or 3e-4 (llama.cpp F32 multilingual), and the tokenizer xllama
  already links is exact. Nothing needs Python at run time.
- **Route B fits xllama better.** The Linux build is llama.cpp-only and
  host-testable, so the encoder, tokenizer and `ModernBertModel` conversion are
  already on the pin. The new code is the head (two small transformer layers
  and an MLP, about 15M parameters for mmBERT-base) and the sequence builder.
  Route A would need plain ORT sessions in the text path, which today exist
  only in UWP (`uwp/diffuse.cpp`).
- **Quantization is the open question, not feasibility.** Multilingual stays
  usable at 8 bits (Q8_0 Δp 0.022, ORT int8 per-channel 0.069). The English
  checkpoint loses calibration under ORT int8 (Δp 0.35–0.44; the billing case
  drops from 0.99 to 0.45 while staying top),
  and Q8_0 gives Δp 0.047. Calibration is the product, so an 8-bit build needs
  a re-fitted temperature or a labelled ECE check before any claim.
- **Memory looks admissible.** Under 0.5 GB peak at 8 bits on the host. The
  Series S budget is [uwp-constraints.md](uwp-constraints.md); no console run
  has been made.
- **Quality caveat.** These are seven smoke cases, not an accuracy benchmark.
  The multilingual checkpoint answered the English "will we cancel" `noul`
  question with p(yes) = 0.045 where the English checkpoint gives 0.879.

## What a product path would still need

1. A WinRT-free header (`include/xllama/decision.h`) with the sequence builder,
   the temperature table and the head, plus doctest cases against
   `cases.json`-style fixtures.
2. The head in C++ (ggml graph or plain loops) and a way to ship its weights
   next to the encoder GGUF.
3. Calibration check of the chosen quant on labelled data (ECE / Brier).
4. A Series S bench (peak, latency), then a catalogue entry. This is the usual
   host smoke → console bench → manifest ladder.
5. A surface: a LAN route compatible with `/v1/systemone`, decided in
   [api-endpoint.md](api-endpoint.md) before code.

## Reproduce

Scripts are in [`../scripts/laya-feasibility/`](../scripts/laya-feasibility/).
They need `torch` (CPU), `transformers`, `safetensors`, `onnx`,
`onnxruntime`, `onnxconverter-common`, the ONNX Runtime C++ release tarball
for `laya_ort.cpp`, and a build of the `llama.cpp` submodule
(`llama-tokenize`, `llama-quantize`, `libllama`) for route B. Download
`model.safetensors`, `rl_agent_config.json`, `encoder/config.json` and
`tokenizer/` (or the `multilingual/` subfolder) from the model repository.

```bash
python3 laya_ref.py <model_dir> <out>          # torch reference, ONNX export, ORT parity, cases.json
python3 laya_quant.py <out>                    # cases.txt + int8 per-tensor
python3 laya_quant2.py <out>                   # int8 per-channel (+ fp16 attempt)
./laya_ort <out>/laya.onnx <out>/cases.txt 4   # route A
python3 split_encoder.py <model_dir> <enc_dir>
python3 llama.cpp/convert_hf_to_gguf.py <enc_dir> --outtype f32 --outfile enc.gguf
python3 tok_parity.py <model_dir>/tokenizer enc.gguf <bin>/llama-tokenize
./laya_llama_enc enc.gguf <out>/cases.txt hidden.bin
python3 head_from_hidden.py <model_dir> <out> hidden.bin   # route B
```
