# scripts/decider/

Tooling for the pplx-decider experiment in
[docs/series-x.md](../../docs/series-x.md#decision-model-pplx-decider). It is
an experiment, not a shipped feature: xllama has no decision endpoint, and the
console run reuses `/api/embed`.

[pplx-decider-flash-v1.3](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.3)
(Apache-2.0, derived from Qwen3.5-4B) is a classifier. A request is a `state`
and typed questions (`choice`, `noul`, `score`); the answer is a probability
per option. The model reads the hidden state at a marker token after each
question through a separate 255 x 2560 readout matrix, and its full-attention
layers run without a causal mask. Nothing is generated.

| File                 | Purpose                                                                                                                                                                                                                                                   |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `set_noncausal.py`   | Copies a converted GGUF with the metadata llama.cpp needs: `qwen35.attention.causal = false`, and the MTP layer removed (the converter declares `block_count` 33 and `nextn_predict_layers` 1, but the checkpoint has no MTP weights, so the load fails). |
| `dequant_patch.py`   | Dequantizes a quantized GGUF back onto the Hugging Face tensor layout (recovering the converter's V-head reorders exactly) as a bf16 patch, to measure weight-only quantization error in the reference PyTorch model.                                     |
| `decider_check.py`   | Scores the model on Kev-format question suites through the reference PyTorch model, a llama.cpp server, or the console's `/api/embed`. Prompt building and readout always come from the checkpoint's own `pplx_decider` code.                             |
| `zh-tw-sample.jsonl` | 25 Traditional Chinese questions (home control, ticket routing, approvals, ratings) written for this experiment.                                                                                                                                          |

The other suites come from [jaredpalmer/kev](https://github.com/jaredpalmer/kev)
(Apache-2.0, commit `2074cc4`): `evals/external/ekzhang-mmlupro-v1`,
`evals/external/semif-v1` (development split), and the `test` splits of
`documents-v1`, `hard-v1` and `devtools-v1`. They are not copied here.

## Reproduce

Python environment: `torch`, `transformers`, `safetensors`, `torchvision`
(the checkpoint's processor needs it) and the `gguf` package from this
repository's `llama.cpp/gguf-py`. Use the llama.cpp converter and
`llama-quantize` from the same llama.cpp build for the GGUF.

```bash
CK=<snapshot of perplexity-ai/pplx-decider-flash-v1.3>
# 1. convert (text model only) and quantize, then fix the metadata
python llama.cpp/convert_hf_to_gguf.py <text-only staging dir> --outtype f16 --outfile flash-f16.gguf
llama-quantize flash-f16.gguf flash-Q5_K_M.gguf Q5_K_M
python scripts/decider/set_noncausal.py flash-Q5_K_M.gguf flash-Q5_K_M-nc.gguf

# 2. reference and llama.cpp scores on the PC
SUITES=(--suite zh-tw=scripts/decider/zh-tw-sample.jsonl --suite mmlu-pro=<kev>/evals/external/ekzhang-mmlupro-v1/records.jsonl:60 ...)
python scripts/decider/decider_check.py --ckpt "$CK" --backend torch --out bf16.jsonl "${SUITES[@]}"
llama-server -m flash-Q5_K_M-nc.gguf --port 8790 --embeddings --pooling none -c 8192 -b 8192 -ub 8192 -np 1
python scripts/decider/decider_check.py --ckpt "$CK" --backend llamacpp --url http://127.0.0.1:8790 --out q5.jsonl "${SUITES[@]}"

# 3. console: upload the GGUF to LocalState\models\pplx-decider-q5\ and a
#    LocalState\manifest.json override that registers it (kind gguf, role
#    embedding), then
python scripts/decider/decider_check.py --ckpt "$CK" --backend xbox --url http://<console>:11434 \
  --ref-url http://127.0.0.1:8790 --ctx 512 --out console.jsonl "${SUITES[@]}"
```

The staging directory for step 1 holds the checkpoint's language-model tensors
renamed from `model.language_model.*` to `model.*`, without the vision tower,
and a flat text config (`Qwen3_5ForCausalLM`, `model_type` `qwen3_5_text`).
