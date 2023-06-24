import sys
import onnx
from onnxsim import simplify
# output_file = "dlo2_uisee_368x208_modify_1_20230208.onnx"
# output_file = "dlo2_uisee_368x208_modify_20230208.onnx"
output_file = sys.argv[-1]
print("output_file:",output_file)
onnx_model = onnx.load(output_file)# load onnx model
onnx_model_sim_file = output_file.split('.')[0] + "_sim.onnx"
model_simp, check_ok = simplify(onnx_model)
if check_ok:
    print("check_ok:",check_ok)
    onnx.save(model_simp, onnx_model_sim_file)
    print(f'Successfully simplified ONNX model: {onnx_model_sim_file}')

