import onnx
import onnx.helper as helper
import sys
import os

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", type=str,default="",
                    help="yolov9 onnx model path.")
    return parser.parse_args()

def main():
    args = parse_args()
    prefix, suffix = os.path.splitext(args.onnx_path)
    save_onnx_path = prefix + "_transd" + suffix

    old_output = ["output0", "1889"]

    model = onnx.load(args.onnx_path)

    for node_ in model.graph.node:
        if node_.output[0] == old_output[0]:
            node_.output[0] = "pre_transpose0"
        if node_.output[0] == old_output[1]:
            node_.output[0] = "pre_transpose1"

    for specout in model.graph.output:
        if specout.name == old_output[0] or old_output[1]:        
            shape0 = specout.type.tensor_type.shape.dim[0]
            shape1 = specout.type.tensor_type.shape.dim[1]
            shape2 = specout.type.tensor_type.shape.dim[2]
            new_out = helper.make_tensor_value_info(
                specout.name,
                specout.type.tensor_type.elem_type,
                [0, 0, 0]
            )
            new_out.type.tensor_type.shape.dim[0].CopyFrom(shape0)
            new_out.type.tensor_type.shape.dim[2].CopyFrom(shape1)
            new_out.type.tensor_type.shape.dim[1].CopyFrom(shape2)
            specout.CopyFrom(new_out)

    model.graph.node.append(
        helper.make_node("Transpose", ["pre_transpose0"], [old_output[0]], perm=[0, 2, 1])
    )
    model.graph.node.append(
        helper.make_node("Transpose", ["pre_transpose1"], [old_output[1]], perm=[0, 2, 1])
    )

    print(f"Model save to {save_onnx_path}")
    onnx.save(model, save_onnx_path)
    return 0

if __name__ == "__main__":
    sys.exit(main())
