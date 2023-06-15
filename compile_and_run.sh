rm -rf build && mkdir -p build && cd build && cmake ..  && make -j9 && cd ..
# mkdir -p build && cd build && cmake ..  && make -j48 && cd ..
cd workspaces

rm -rf cuda_res/*

# ./infer -f rtdetr_r50vd_6x_coco.trt -i res/zidane.jpg -b 1 -c 10 -o cuda_res
# ./infer -f rtdetr_r50vd_6x_coco.trt -i res/bus.jpg -b 1 -c 10 -o cuda_res
# ./infer -f rtdetr_r50vd_6x_coco.trt -i res/dog.jpg -b 1 -c 10 -o cuda_res

# ./infer -f rtdetr_r50vd_6x_coco_dynamic_fp16.trt -i res/dog.jpg -b 16 -c 10 -o cuda_res -t rt_detr_det

# ./infer -f yolov8n.transd.trt -i res/dog.jpg -b 16 -c 10 -o cuda_res -t yolov8_det
# ./infer -f yolov8n.transd.trt -i res/bus.jpg -b 16 -c 10 -o cuda_res -t yolov8_det
./infer -f yolov8n.transd.trt -i res/dog.jpg -b 16 -c 10 -o cuda_res -t yolov8_det -p true

# ./infer -f yolov8n-seg.b1.transd.trt -i res/dog.jpg -b 1 -c 10 -o cuda_res -t yolov8_seg

# ./infer -f yolov8s_pose_fp16.trt -i res/bus.jpg -b 1 -c 10 -o cuda_res -t yolov8_pose
# ./infer -f yolov8s-pose_transd_fp16.trt -i res/bus.jpg -b 1 -c 10 -s 0.5 -o cuda_res -t yolov8_pose

# ./infer -f yolov8n.transd.trt -i res/example.mp4  -s 0.25 -o cuda_res -t yolov8_det_track

# ./infer -f yolov7_640.trt -i res/dog.jpg -b 1 -c 10 -o cuda_res -t yolov7_det
# ./infer -f yolov7_640.trt -i res/bus.jpg -b 1 -c 10 -o cuda_res -t yolov7_det
# ./infer -f yolov7_640.trt -i res/zidane.jpg -b 1 -c 10 -o cuda_res -t yolov7_det

# ./infer -f yolox_s_fp16.trt -i res/dog.jpg -b 1 -c 1 -w 1 -o cuda_res -t yolox_det
# ./infer -f yolox_s_fp16.trt -i res/bus.jpg -b 1 -c 1 -w 1 -o cuda_res -t yolox_det
# ./infer -f yolox_s_fp16.trt -i res/zidane.jpg -b 1 -c 1 -w 1 -o cuda_res -t yolox_det

# ./infer -f yolov5s_fp16.trt -i res/dog.jpg -b 1 -c 2 -w 1 -o cuda_res -t yolov5_det
# ./infer -f yolov5s_fp16.trt -i res/bus.jpg -b 1 -c 1 -w 2 -o cuda_res -s 0.45 -t yolov5_det
# ./infer -f yolov5s_fp16.trt -i res/zidane.jpg -b 1 -c 2 -w 1 -o cuda_res -t yolov5_det

# rm -rf pointpillar_infer_res/*
# ./infer -c 80 -t pointpillar_det

# ./infer -f yolov7_cutoff_fp16.trt -i res/dog.jpg -b 1 -c 10 -o cuda_res -t yolov7_cutoff_det
# ./infer -f yolov7_cutoff_fp16.trt -i res/bus.jpg -b 1 -c 1 -o cuda_res -t yolov7_cutoff_det
# ./infer -f yolov7_cutoff_fp16.trt -i res/zidane.jpg -b 1 -c 10 -o cuda_res -t yolov7_cutoff_det

# ./infer -f yolov7-w6-pose_sim.trt -i res/bus.jpg -b 1 -c 2 -o cuda_res -t yolov7_pose
# ./infer -f yolov7-w6-pose_sim.trt -i res/person.jpg -b 1 -c 2 -o cuda_res -t yolov7_pose

cd ..