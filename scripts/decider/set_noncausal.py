# SPDX-License-Identifier: MIT
"""Copy a pplx-decider GGUF with the metadata llama.cpp needs to run it as a decider:

- <arch>.attention.causal = false  (the full-attention layers are non-causal; linear-attention layers stay causal)
- drop the MTP layer the converter declared from config.mtp_num_hidden_layers (the checkpoint has no MTP weights):
  block_count 33 -> 32, nextn_predict_layers -> 0, recurrent_layers truncated to the real layers.

usage: python set_noncausal.py <in.gguf> <out.gguf>
"""
import sys
import gguf
from gguf.scripts.gguf_new_metadata import MetadataDetails, copy_with_new_metadata

src, dst = sys.argv[1:3]
reader = gguf.GGUFReader(src, "r")
arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
layers = 1 + max(int(t.name.split(".")[1]) for t in reader.tensors if t.name.startswith("blk."))
K = lambda k: k.format(arch=arch)
new = {
    K(gguf.Keys.Attention.CAUSAL): MetadataDetails(gguf.GGUFValueType.BOOL, False),
    K(gguf.Keys.LLM.BLOCK_COUNT): MetadataDetails(gguf.GGUFValueType.UINT32, layers),
}
remove = []
nextn = K(gguf.Keys.LLM.NEXTN_PREDICT_LAYERS)
if reader.get_field(nextn) is not None:
    new[nextn] = MetadataDetails(gguf.GGUFValueType.UINT32, 0)
rec = f"{arch}.attention.recurrent_layers"
if (f := reader.get_field(rec)) is not None:
    vals = f.contents()[:layers]
    new[rec] = MetadataDetails(gguf.GGUFValueType.ARRAY, vals)
summary = {k: (v.value if not isinstance(v.value, list) else len(v.value)) for k, v in new.items()}
writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)
copy_with_new_metadata(reader, writer, new, remove)
print("wrote", dst, summary)
