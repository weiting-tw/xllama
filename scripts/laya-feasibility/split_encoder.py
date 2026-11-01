"""Write an HF ModernBertModel directory (encoder weights only) for convert_hf_to_gguf.py."""
import json, os, shutil, sys
from safetensors.torch import load_file, save_file
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
sd = load_file(os.path.join(src, "model.safetensors"))
enc = {k[len("encoder."):]: v.contiguous() for k, v in sd.items() if k.startswith("encoder.")}
save_file(enc, os.path.join(dst, "model.safetensors"))
cfg = json.load(open(os.path.join(src, "encoder", "config.json")))
cfg["architectures"] = ["ModernBertModel"]
json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=1)
for f in os.listdir(os.path.join(src, "tokenizer")):
    shutil.copy(os.path.join(src, "tokenizer", f), dst)
print(len(enc), "encoder tensors; head tensors:", sorted({k.split(".")[0] for k in sd if not k.startswith("encoder.")}))
