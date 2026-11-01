"""Write cases.txt for the C++ harness and an int8 (dynamic, weight-only MatMul) copy of the ONNX."""
import json
import os
import sys

from onnxruntime.quantization import QuantType, quantize_dynamic

out = sys.argv[1]
cases = json.load(open(os.path.join(out, "cases.json")))
with open(os.path.join(out, "cases.txt"), "w") as f:
    for it in cases["items"]:
        row = [it["qtype"], it["temperature"], len(it["markers"]), *it["markers"], len(it["ids"]), *it["ids"],
               *it["torch_probs"]]
        f.write(" ".join(str(x) for x in row) + "\n")
quantize_dynamic(os.path.join(out, "laya.onnx"), os.path.join(out, "laya-int8.onnx"),
                 weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
print("int8: %.0f MB" % (os.path.getsize(os.path.join(out, "laya-int8.onnx")) / 1e6))
