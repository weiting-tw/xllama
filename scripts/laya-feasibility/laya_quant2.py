"""Variants: per-channel int8 (MatMul + Gather) and fp16 weights with fp32 I/O."""
import os, sys
import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxconverter_common import float16
out = sys.argv[1]
src = os.path.join(out, "laya.onnx")
quantize_dynamic(src, os.path.join(out, "laya-int8pc.onnx"), weight_type=QuantType.QInt8, per_channel=True,
                 op_types_to_quantize=["MatMul", "Gather"])
m = float16.convert_float_to_float16(onnx.load(src), keep_io_types=True)
onnx.save(m, os.path.join(out, "laya-fp16.onnx"))
for f in ("laya-int8pc.onnx", "laya-fp16.onnx"):
    print("%s: %.0f MB" % (f, os.path.getsize(os.path.join(out, f)) / 1e6))
