trtexec --onnx=./yolop-640.onnx --saveEngine=./yolop-640.trt --buildOnly --fp16

trtexec --onnx=./yolopv2-480x640.onnx --saveEngine=./yolopv2-480x640.trt --buildOnly --fp16
