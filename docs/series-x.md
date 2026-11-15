# Series X: independent campaign with a device receipt (2026-11-15)

A second independent run of xllama on an Xbox **Series X** in Developer
Mode. It follows the field report on which Series X support was accepted
([positioning.md](./positioning.md#series-x)) and covers what that report
did not: memory limits, nine catalogue GGUF models, the d3d12 GPU path, and
`/api/embed` (through a decision-model test). Unlike those runs, it comes with
a Device Portal receipt of the device model.

This is one console and one session from an independent tester. It is not a
project measurement and does not complete the
[independent Series X acceptance protocol](./console-validation-runbook.md#independent-series-x-acceptance-374)
([what is missing](#against-the-acceptance-protocol)). Nothing here changes a
Series S claim, the catalogue, or the comparison policy in
[benchmarks.md](./benchmarks.md).

Raw evidence: [`bench/results/2026-11-15-series-x/`](../bench/results/2026-11-15-series-x/).

## Setup

| Item              | Value                                                                                                                                                                                                      |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Console           | Xbox Series X (observed, not only declared), OS `26100.9662.amd64fre.xb_flt_2610ge.261104-2200` (Device Portal receipt: [`device-receipt.json`](../bench/results/2026-11-15-series-x/device-receipt.json)) |
| Package           | `GianlucaMazza.xllama_1.7.0.1495_x64` (v1.7.0.0 release MSIX, SHA-256 in [`provenance.json`](../bench/results/2026-11-15-series-x/provenance.json)), designated **Game** in Dev Home                       |
| Harness           | `scripts/run-xab.sh --profile series-x-devmode --include text` (host column `xbox-series-x-declared`), plus `bench-xbox-ort.sh --hardware xbox-series-x --gpu-layers 99`                                   |
| Prompt / settings | `bench/prompts/standard-512.txt` (298 tokens), 6 threads, `n_ctx` 2048, decode to EOG or 512 tokens, 4 runs with run 1 discarded                                                                           |
| Host              | Windows 11 with Git Bash (see [Windows host notes](#windows-host-notes))                                                                                                                                   |

The hardware identity is an operator declaration backed by the Device Portal
receipt above, as [bench/README.md](../bench/README.md#hardware-identity-for-new-text-campaigns)
requires. No power or thermal data was recorded, so none of these rows is a
headline throughput claim.

## Memory: App vs Game designation

A fresh sideload on this console came up designated **App**: its
`DefaultUWPContentTypeToGame` setting is false, so every reinstall starts as
App until it is switched in Dev Home (tile, View details, App type).
[uwp-constraints.md §5](./uwp-constraints.md#5-continued--disk-availability-and-the-appgame-lever)
describes the same lever on Series S.

Measured with `scripts/bench-ramceil.sh` (128 MB commit steps):

| Designation | Largest commit that succeeded | Available physical (first step) | GPU budget (`QueryVideoMemoryInfo`) | Evidence                                                                                            |
| ----------- | ----------------------------- | ------------------------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------- |
| App         | 768 MB                        | 767 MB                          | 1209 MB                             | [`ramceil-app-designation.csv`](../bench/results/2026-11-15-series-x/ramceil-app-designation.csv)   |
| Game        | 4864 MB                       | 4985 MB                         | 4147 MB                             | [`ramceil-game-designation.csv`](../bench/results/2026-11-15-series-x/ramceil-game-designation.csv) |

The probe writes `xbox-series-s` in its host column on every console; these
two files come from the Series X in the receipt.

Under the App designation, `qwen3-1.7b` failed to allocate its 224 MB KV cache
over the LAN API (HTTP 500). Every result below was measured as Game. Under
Game the app is not listed in Device Portal's process list, but
`deploy.sh start-app` launches it normally.

## Text throughput

Medians of three recorded runs. Decode spread is min to max. Generation
stopped at EOG, so `n_gen_tok` differs between runs; a run with very few
generated tokens (6 for `lfm25-350m`) reads high.

| Model                 | Quant    | Backend | Prefill tok/s | Decode tok/s        | Peak WS MB | Load ms |
| --------------------- | -------- | ------- | ------------: | ------------------- | ---------: | ------: |
| `lfm25-350m`          | Q4_0     | cpu     |         396.2 | 104.6 (102.5–123.9) |        312 |     978 |
| `qwen35-0.8b`         | Q4_K_M   | cpu     |         185.3 | 36.4 (36.1–36.6)    |        715 |    2405 |
| `lfm25-1.2b-instruct` | Q4_0     | cpu     |         116.0 | 41.2 (41.2–41.3)    |        786 |    3203 |
| `qwen3-1.7b`          | Q4_K_M   | cpu     |         101.3 | 22.4 (22.3–22.5)    |       1397 |    4893 |
| `lfm2-2.6b`           | Q4_K_M   | cpu     |          61.3 | 18.3 (18.3–18.4)    |       1629 |    7657 |
| `gemma4-e2b`          | Q3_K_S   | cpu     |          54.5 | 16.2 (15.9–16.8)    |       2731 |    8186 |
| `llama32-3b`          | Q3_K_S   | cpu     |          41.4 | 14.9 (14.9–14.9)    |       1832 |    9356 |
| `phi4-mini`           | Q4_K_M   | cpu     |          44.0 | 11.5 (11.5–11.5)    |       2772 |   10385 |
| `lfm25-8b-a1b`        | UD-IQ3_S | cpu     |          64.7 | 15.2 (15.2–15.2)    |       3560 |    8443 |
| `lfm25-350m`          | Q4_0     | d3d12   |        1691.0 | 129.2 (127.3–132.2) |        162 |     369 |
| `lfm25-1.2b-instruct` | Q4_0     | d3d12   |         436.8 | 92.7 (92.3–93.5)    |        860 |    1223 |

All nine models loaded and generated, including `lfm25-8b-a1b` (3.56 GB
peak working set). The other catalogue GGUFs (`lfm25-230m`, `gemma3-270m`,
the two Qwen2.5-Coder models and `lfm25-1.2b-thinking`) were not run. The d3d12 rows report 224 MB and 678 MB of GPU
memory in use against the 4147 MB budget.

Against the Series S rows in [benchmarks.md](./benchmarks.md) (other builds
and campaigns, so read these as direction only):

- **CPU decode** is 2 to 7 percent higher for all nine models. That is about
  the CPU clock difference between the two consoles (both 8-core Zen 2); the
  larger memory bandwidth of Series X does not show up on the CPU path. This
  agrees with the on-par finding in
  [positioning.md](./positioning.md#series-x).
- **CPU prefill** is not compared: several Series S rows predate the prefill
  boundaries listed in [bench/README.md](../bench/README.md), and the gaps
  (up to 3.4 times for `lfm25-8b-a1b`) track those builds rather than the
  hardware.
- **d3d12** gains more: `lfm25-1.2b-instruct` decode is about 24 percent
  higher and prefill about 70 percent higher than its Series S d3d12 row.

## Decision model (pplx-decider)

The question was whether a "decision model" can run on the console next to the
chat models. The model tested is
[pplx-decider-flash-v1.3](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.3)
(Apache-2.0, Qwen3.5-4B based). It is a classifier, not a chat model: the
input is a state plus typed questions, and the output is a calibrated
probability per option, read from the final hidden state at a marker token
through a separate 255 x 2560 readout matrix. Its full-attention layers are
non-causal. Tooling and the reproduction steps are in
[`scripts/decider/`](../scripts/decider/README.md).

### Quantization (PC)

Questions: 25 Traditional Chinese questions written for this test plus 60
each from five [Kev](https://github.com/jaredpalmer/kev) suites, 325 in
total, each asked on its own. The quantized rows put dequantized GGUF weights
into the reference PyTorch model, so they isolate weight error. The
llama.cpp row runs the real Q5_K_M GGUF through `llama-server` (build 11146).

| Run              | GGUF size | zh-TW | MMLU-Pro | SemIf | documents |  hard | devtools | Overall | Top-1 same as bf16 | Mean max prob. drift |
| ---------------- | --------: | ----: | -------: | ----: | --------: | ----: | -------: | ------: | -----------------: | -------------------: |
| bf16 (reference) |   8.42 GB | 0.960 |    0.567 | 0.950 |     0.867 | 0.733 |    0.783 |   0.794 |                  — |                    — |
| Q4_K_M weights   |   2.71 GB | 0.960 |    0.533 | 0.917 |     0.833 | 0.700 |    0.817 |   0.775 |              93.5% |                0.042 |
| Q5_K_M weights   |   3.07 GB | 0.960 |    0.583 | 0.967 |     0.867 | 0.717 |    0.817 |   0.803 |              97.5% |                0.021 |
| Q6_K weights     |   3.46 GB | 0.960 |    0.550 | 0.917 |     0.833 | 0.733 |    0.817 |   0.785 |              96.3% |                0.014 |
| Q5_K_M llama.cpp |   3.07 GB | 0.960 |    0.583 | 0.950 |     0.850 | 0.717 |    0.783 |   0.791 |              96.6% |                0.025 |

Q5_K_M is the size chosen for the console: within noise of bf16, and it fits
the Game budget. Q4_K_M loses measurably (11 bf16-correct answers turn wrong,
5 the other way).

Two GGUF fixes were needed, both in `scripts/decider/set_noncausal.py`:

- The converter declared an MTP layer from `mtp_num_hidden_layers`
  (`block_count` 33, `nextn_predict_layers` 1) although the checkpoint has no
  MTP weights, so llama.cpp failed with `tensor 'blk.32.attn_norm.weight' not found`.
- `qwen35.attention.causal = false`. llama.cpp then masks only the
  full-attention layers as non-causal; the recurrent layers stay causal, which
  is what the model expects. No llama.cpp change was needed.

### On the console

There is no decision endpoint in xllama. The test reuses `/api/embed`: the
GGUF sits in `LocalState\models\pplx-decider-q5\`, a `LocalState\manifest.json`
override registers it with role `embedding`
([`localstate-manifest-override.json`](../bench/results/2026-11-15-series-x/decider/localstate-manifest-override.json)),
the PC builds the prompt text, and the PC applies the readout to the returned
vector. Sending the prompt as text is safe: llama.cpp's tokenizer reproduced
the reference token ids on 75 of 75 sampled prompts.

`/api/embed` L2-normalizes its output, which keeps the top answer exact but
loses the logit scale. The script therefore also asks a PC `llama-server` on
the same GGUF for the reference state, records the cosine between the two, and
rebuilds probabilities with the reference norm.

| Suite     | Questions |  Accuracy | Top-1 same as PC llama.cpp | Median tokens | Median console ms |
| --------- | --------: | --------: | -------------------------: | ------------: | ----------------: |
| zh-TW     |        25 |     0.960 |                      25/25 |           102 |              4507 |
| MMLU-Pro  |         8 |     0.500 |                        7/8 |           350 |             14336 |
| SemIf     |         8 |     1.000 |                        8/8 |           124 |              4979 |
| documents |         5 |     1.000 |                        5/5 |           302 |             12223 |
| hard      |         4 |     0.750 |                        4/4 |           306 |             12802 |
| devtools  |         6 |     0.833 |                        6/6 |           258 |             10160 |
| **All**   |    **56** | **0.875** |                  **55/56** |           132 |              5699 |

- Correctness matched the PC llama.cpp run on all 56 questions; the one
  different top answer is an MMLU-Pro question both get wrong. Cosine between
  console and PC states: median 0.9994, minimum 0.9927.
- Prefill runs on the CPU at a median 23.7 tok/s: about 4.5 s for a
  100-token question and 12 to 14 s for 300 tokens. The same GGUF answers in
  about 0.3 s on a desktop GPU.
- Nine questions longer than 512 tokens were skipped (560 to 3460 tokens).

The 512-token cap comes from memory, not from the model. The weights take
2922 MB. A non-causal graph must hold the whole sequence in one ubatch, and
the compute buffer grows with `n_ctx`: 510 MB at 512, 1022 MB at 1024, and
2.1 GB at 2048, which failed to allocate. In addition, `/api/embed` runs with
pooling `none`, so llama.cpp marks every token as an output and reserves full
vocabulary logits for each (about 1 MB per token): a 149-token prompt failed
at `n_ctx` 1024 for that reason. Peak working set at `n_ctx` 512 was 4143 MB.

A native decision endpoint would remove both limits on the output side: read
only the marker position, return the raw state, and apply the readout and
softmax on the console. It would not change the speed, which is CPU prefill.

## Against the acceptance protocol

| Protocol step                                                                | This campaign                                                                                                                                             |
| ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Fixed package and device receipt                                          | Release MSIX 1.7.0.1495 with its SHA-256 (not a CI `xllama-appx` of a candidate SHA); Device Portal OS/device receipt; no serial or network address       |
| 2. Developer Mode setup, `XLLAMA_EXPECTED_PFN`                               | Installed and designated Game by hand; the harness resolved the single installed PFN                                                                      |
| 3. `validate-console.sh all`, `validate-api.sh all`                          | **Not run**                                                                                                                                               |
| 4. Model and prompt hashes, settings, cold and warm repeats, prefill, decode | Bundled-catalogue SHA-256 per model, prompt SHA-256 and settings in `provenance.json`; per-run rows with load, prefill, decode, TTFT and peak working set |
| 5. Campaign directory with raw results and hashes                            | `bench/results/2026-11-15-series-x/` with `SHA256SUMS`; CSVs pass `validate-benchmark.py`; no sidecars                                                    |
| 6. Review under #374                                                         | Pending                                                                                                                                                   |

The speed finding in [positioning.md](./positioning.md#series-x) holds here:
designated Game, CPU text generation is on par with a Series S (see
[Text throughput](#text-throughput)). The memory table above is a possible
lead for its App-designation slowdown: the App designation leaves at most
768 MB of commit and a 1209 MB GPU budget on this console. It was not tested
as a cause.

## Windows host notes

Running the console scripts from Git Bash on Windows needed two fixes, now in
`scripts/wdp-file-upload.sh`, `scripts/bench-xbox-ort.sh` and
`scripts/bench-ramceil.sh`: `curl.exe` cannot open an MSYS path such as
`/e/...` in `-F file=@`, and cannot write to the MSYS `/tmp` from `mktemp -d`
(both fail with curl exit code 26). Paths are converted with `cygpath -m` when
it exists; Linux and macOS are unaffected. A checkout with
`core.autocrlf=true` also breaks the scripts (`$'\r'`); use
`core.autocrlf=input`.

## Not covered

- H8 in [phase7-hypotheses.md](./phase7-hypotheses.md#h8--series-x-gpu-budget-for-1b-fp16-dml)
  (1B fp16 DirectML): only the Game GPU budget (4147 MB) was measured; no 1B
  fp16 DirectML asset was run.
- ORT GenAI / DirectML models, diffusion, training, KV and H9 workloads (the
  XAB Series X profile is text only).
- Power, thermals, and repeated campaigns. One console, one session.
