#include "post_process.h"
namespace ai
{
    namespace postprocess
    {
        // keepflag主要是用来进行nms时候判断是否将该框抛弃
        // const int NUM_BOX_ELEMENT = 7; // left, top, right, bottom, confidence, class, keepflag
        static __device__ void affine_project(float *matrix, float x, float y, float *ox, float *oy)
        {
            *ox = matrix[0] * x + matrix[1] * y + matrix[2];
            *oy = matrix[3] * x + matrix[4] * y + matrix[5];
        }

        static __global__ void decode_kernel_common(float *predict, int num_bboxes, int num_classes,
                                                    int output_cdim, float confidence_threshold,
                                                    float *invert_affine_matrix, float *parray,
                                                    int MAX_IMAGE_BOXES, int NUM_BOX_ELEMENT)
        {
            int position = blockDim.x * blockIdx.x + threadIdx.x;
            if (position >= num_bboxes)
                return;

            // pitem就获取了每个box的首地址
            // output_cdim是指每个box中有几个元素，可以根据onnx最后的输出定
            // 每个box的元素一般是(根据你的模型可修改,objectness是yolo系列的是否有物体得分)：left,top,right,bottom,objectness,class0,class1,...,classn
            float *pitem = predict + output_cdim * position;
            float objectness = pitem[4];
            if (objectness < confidence_threshold)
                return;

            // 从多个类别得分中，找出最大类别的class_score+label
            float *class_confidence = pitem + 5;
            float confidence = *class_confidence++; // 取class1给confidence并且class_confidence自增1
            int label = 0;
            // ++class_confidence和class_confidence++在循环中执行的结果是一样的，都是执行完循环主体后再加一
            for (int i = 1; i < num_classes; ++i, ++class_confidence)
            {
                if (*class_confidence > confidence)
                {
                    confidence = *class_confidence;
                    label = i;
                }
            }

            confidence *= objectness; // yolo系列的最终得分是两者相乘
            if (confidence < confidence_threshold)
                return;

            // cuda的原子操作：int atomicAdd(int *M,int V); 它们把一个内存位置M和一个数值V作为输入。
            // 与原子函数相关的操作在V上执行，数值V早已存储在内存地址*M中了，然后将相加的结果写到同样的内存位置中。
            int index = atomicAdd(parray, 1); // 所以这段代码意思是用parray[0]来计算boxes的总个数
            if (index >= MAX_IMAGE_BOXES)
                return;

            float cx = *pitem++;
            float cy = *pitem++;
            float width = *pitem++;
            float height = *pitem++;
            float left = cx - width * 0.5f;
            float top = cy - height * 0.5f;
            float right = cx + width * 0.5f;
            float bottom = cy + height * 0.5f;
            // boxes映射回相对于真实图片的尺寸
            affine_project(invert_affine_matrix, left, top, &left, &top);
            affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

            // parray+1之后的值全部用来存储boxes元素，每个框有NUM_BOX_ELEMENT个元素
            float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
            *pout_item++ = left;
            *pout_item++ = top;
            *pout_item++ = right;
            *pout_item++ = bottom;
            *pout_item++ = confidence;
            *pout_item++ = label;
            *pout_item++ = 1; // 1 = keep, 0 = ignore
        }

        static __device__ float box_iou(
            float aleft, float atop, float aright, float abottom,
            float bleft, float btop, float bright, float bbottom)
        {

            float cleft = max(aleft, bleft);
            float ctop = max(atop, btop);
            float cright = min(aright, bright);
            float cbottom = min(abottom, bbottom);

            float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
            if (c_area == 0.0f)
                return 0.0f;

            float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
            float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
            return c_area / (a_area + b_area - c_area);
        }

        static __global__ void nms_kernel(float *bboxes, int max_objects, float threshold, int NUM_BOX_ELEMENT)
        {

            int position = (blockDim.x * blockIdx.x + threadIdx.x);
            int count = min((int)*bboxes, max_objects);
            if (position >= count)
                return;

            // left, top, right, bottom, confidence, class, keepflag
            float *pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
            for (int i = 0; i < count; ++i)
            {
                float *pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
                if (i == position || pcurrent[5] != pitem[5])
                    continue;

                if (pitem[4] >= pcurrent[4])
                {
                    if (pitem[4] == pcurrent[4] && i < position)
                        continue;

                    float iou = box_iou(
                        pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                        pitem[0], pitem[1], pitem[2], pitem[3]);

                    if (iou > threshold)
                    {
                        pcurrent[6] = 0; // 1=keep, 0=ignore
                        return;
                    }
                }
            }
        }

        static __global__ void decode_kernel_v8_trans(float *predict, int num_bboxes, int num_classes,
                                                      int output_cdim, float confidence_threshold,
                                                      float *invert_affine_matrix, float *parray,
                                                      int MAX_IMAGE_BOXES, int NUM_BOX_ELEMENT)
        {
            int position = blockDim.x * blockIdx.x + threadIdx.x;
            if (position >= num_bboxes)
                return;

            float *pitem = predict + output_cdim * position;
            float *class_confidence = pitem + 4;
            float confidence = *class_confidence++;
            int label = 0;
            for (int i = 1; i < num_classes; ++i, ++class_confidence)
            {
                if (*class_confidence > confidence)
                {
                    confidence = *class_confidence;
                    label = i;
                }
            }
            if (confidence < confidence_threshold)
                return;

            int index = atomicAdd(parray, 1);
            if (index >= MAX_IMAGE_BOXES)
                return;

            float cx = *pitem++;
            float cy = *pitem++;
            float width = *pitem++;
            float height = *pitem++;
            float left = cx - width * 0.5f;
            float top = cy - height * 0.5f;
            float right = cx + width * 0.5f;
            float bottom = cy + height * 0.5f;
            affine_project(invert_affine_matrix, left, top, &left, &top);
            affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

            float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
            *pout_item++ = left;
            *pout_item++ = top;
            *pout_item++ = right;
            *pout_item++ = bottom;
            *pout_item++ = confidence;
            *pout_item++ = label;
            *pout_item++ = 1; // 1 = keep, 0 = ignore
            if (NUM_BOX_ELEMENT == 8)
                *pout_item++ = position;
        }

        static __global__ void decode_kernel_v8_pose_trans(float *predict, int num_bboxes, int pose_num,
                                                           int output_cdim, float confidence_threshold,
                                                           float *invert_affine_matrix, float *parray,
                                                           int MAX_IMAGE_BOXES, int NUM_BOX_ELEMENT)
        {
            int position = blockDim.x * blockIdx.x + threadIdx.x;
            if (position >= num_bboxes)
                return;

            float *pitem = predict + output_cdim * position;
            float confidence = *(pitem + 4);
            int label = 0;
            if (confidence < confidence_threshold)
                return;

            int index = atomicAdd(parray, 1);
            if (index >= MAX_IMAGE_BOXES)
                return;

            float cx = *pitem++;
            float cy = *pitem++;
            float width = *pitem++;
            float height = *pitem++;
            float box_score = *pitem++; // 这句其实没起什么作用，只是简单的让pitem自增
            float left = cx - width * 0.5f;
            float top = cy - height * 0.5f;
            float right = cx + width * 0.5f;
            float bottom = cy + height * 0.5f;
            affine_project(invert_affine_matrix, left, top, &left, &top);
            affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

            float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
            *pout_item++ = left;
            *pout_item++ = top;
            *pout_item++ = right;
            *pout_item++ = bottom;
            *pout_item++ = confidence;
            *pout_item++ = label;
            *pout_item++ = 1; // 1 = keep, 0 = ignore
            for (int i = 0; i < pose_num; i++)
            {
                affine_project(invert_affine_matrix, *pitem++, *pitem++, pout_item++, pout_item++);
                *pout_item++ = *pitem++; // pose score 赋值
            }
        }

        static __global__ void decode_single_mask_kernel(int left, int top, float *mask_weights,
                                                         float *mask_predict, int mask_width,
                                                         int mask_height, unsigned char *mask_out,
                                                         int mask_dim, int out_width, int out_height)
        {
            // mask_predict to mask_out
            // mask_weights @ mask_predict
            int dx = blockDim.x * blockIdx.x + threadIdx.x;
            int dy = blockDim.y * blockIdx.y + threadIdx.y;
            if (dx >= out_width || dy >= out_height)
                return;

            int sx = left + dx;
            int sy = top + dy;
            if (sx < 0 || sx >= mask_width || sy < 0 || sy >= mask_height)
            {
                mask_out[dy * out_width + dx] = 0;
                return;
            }

            float cumprod = 0;
            for (int ic = 0; ic < mask_dim; ++ic)
            {
                float cval = mask_predict[(ic * mask_height + sy) * mask_width + sx];
                float wval = mask_weights[ic];
                cumprod += cval * wval;
            }

            float alpha = 1.0f / (1.0f + exp(-cumprod));
            // mask_out[dy * out_width + dx] = alpha;
            if (alpha > 0.5)
                mask_out[dy * out_width + dx] = 1;
            else
                mask_out[dy * out_width + dx] = 0;
        }

        static __global__ void decode_kernel_rtdetr(float *predict, int num_bboxes, int num_classes,
                                                    int output_cdim, float confidence_threshold,
                                                    int scale_expand, float *parray,
                                                    int MAX_IMAGE_BOXES, int NUM_BOX_ELEMENT)
        {
            int position = blockDim.x * blockIdx.x + threadIdx.x;
            if (position >= num_bboxes)
                return;

            float *pitem = predict + output_cdim * position;

            // 从多个类别得分中，找出最大类别的class_score+label
            float *class_confidence = pitem + 4;
            float confidence = *class_confidence++; // 取class1给confidence并且class_confidence自增1
            int label = 0;
            // ++class_confidence和class_confidence++在循环中执行的结果是一样的，都是执行完循环主体后再加一
            for (int i = 1; i < num_classes; ++i, ++class_confidence)
            {
                if (*class_confidence > confidence)
                {
                    confidence = *class_confidence;
                    label = i;
                }
            }

            if (confidence < confidence_threshold)
                return;

            // cuda的原子操作：int atomicAdd(int *M,int V); 它们把一个内存位置M和一个数值V作为输入。
            // 与原子函数相关的操作在V上执行，数值V早已存储在内存地址*M中了，然后将相加的结果写到同样的内存位置中。
            int index = atomicAdd(parray, 1); // 所以这段代码意思是用parray[0]来计算boxes的总个数
            if (index >= MAX_IMAGE_BOXES)
                return;

            float cx = *pitem++;
            float cy = *pitem++;
            float width = *pitem++;
            float height = *pitem++;
            float left = (cx - width * 0.5f) * scale_expand;
            float top = (cy - height * 0.5f) * scale_expand;
            float right = (cx + width * 0.5f) * scale_expand;
            float bottom = (cy + height * 0.5f) * scale_expand;

            // parray+1之后的值全部用来存储boxes元素，每个框有NUM_BOX_ELEMENT个元素
            float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
            *pout_item++ = left;
            *pout_item++ = top;
            *pout_item++ = right;
            *pout_item++ = bottom;
            *pout_item++ = confidence;
            *pout_item++ = label;
            *pout_item++ = 1; // 1 = keep, 0 = ignore
        }

        static __global__ void decode_kernel_yolox(const float* cls_data, const float* obj_data, const float* bbox_data,
                        const int batchsize, const int det_obj_len, const int det_bbox_len, const int det_cls_len,
                        const int input_h, const int input_w, const int stage_w, const int stage_h,
                        const int MAX_IMAGE_BOXES, int NUM_BOX_ELEMENT,
                        const int stride, const int numThreads, const float confidence_threshold,
                        float *invert_affine_matrix, float* parray)
        {
            int idx = threadIdx.x + blockDim.x * blockIdx.x;
            if (idx >= numThreads)
                return;

            int row = (idx / stage_w) % stage_h;
            int col = idx % stage_w;
            int area = stage_w * stage_h;
            int label = idx / stage_w / stage_h;

            float obj = obj_data[row * stage_w + col];
            obj = 1 / (1 + expf(-obj));
            // if (obj < confidence_threshold) return;
            
            float x_feat = bbox_data[row * stage_w + col];
            float y_feat = bbox_data[area + (row * stage_w + col)];
            float w_feat = bbox_data[area * 2 + (row * stage_w + col)];
            float h_feat = bbox_data[area * 3 + (row * stage_w + col)];

            float x_center = (x_feat + col) * stride;
            float y_center = (y_feat + row) * stride;
            float w = expf(w_feat) * stride;
            float h = expf(h_feat) * stride;

            float cls_feat = cls_data[idx];
            cls_feat = 1 / (1 + expf(-cls_feat));
            float confidence = cls_feat * obj;
            if(confidence < confidence_threshold)
                return;

            float left = x_center - 0.5 * w;
            float top = y_center - 0.5 * h;
            float right = x_center + 0.5 * w;
            float bottom = y_center + 0.5 * h;
            affine_project(invert_affine_matrix, left, top, &left, &top);
            affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

            int index = (int)atomicAdd(parray, 1);
            if(index >= MAX_IMAGE_BOXES)
                return;

            // parray+1之后的值全部用来存储boxes元素，每个框有NUM_BOX_ELEMENT个元素
            float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
            *pout_item++ = left;
            *pout_item++ = top;
            *pout_item++ = right;
            *pout_item++ = bottom;
            *pout_item++ = confidence;
            *pout_item++ = label;
            *pout_item++ = 1; // 1 = keep, 0 = ignore
        }

        void decode_kernel_yolox_invoker(const float* cls_data, const float* obj_data, const float* bbox_data,
                        const int batchsize, const int det_obj_len, const int det_bbox_len, const int det_cls_len,
                        const int MAX_IMAGE_BOXES, int NUM_BOX_ELEMENT,
                        const int input_h, const int input_w, const int stride,
                        const float confThreshold, const float nmsThreshold,
                        float *invert_affine_matrix, float* output, cudaStream_t stream)
        {
            int stage_h = (int) input_h / stride;
            int stage_w = (int) input_w / stride;
            const int num_bboxes = batchsize * stage_h * stage_w * det_cls_len;
            auto grid = CUDATools::grid_dims(num_bboxes);
            auto block = CUDATools::block_dims(num_bboxes);

            decode_kernel_yolox<<<grid, block, 0, stream>>>(
                                cls_data, obj_data, bbox_data,
                                batchsize,  det_obj_len, det_bbox_len, det_cls_len,
                                input_h, input_w, stage_w, stage_h,
                                MAX_IMAGE_BOXES, NUM_BOX_ELEMENT,
                                stride, num_bboxes, confThreshold,
                                invert_affine_matrix, output
            );
        }

        void decode_detect_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                          float confidence_threshold, float *invert_affine_matrix,
                                          float *parray, int MAX_IMAGE_BOXES, int NUM_BOX_ELEMENT, cudaStream_t stream)
        {
            auto grid = CUDATools::grid_dims(num_bboxes);
            auto block = CUDATools::block_dims(num_bboxes);

            checkCudaKernel(decode_kernel_common<<<grid, block, 0, stream>>>(
                predict, num_bboxes, num_classes, output_cdim, confidence_threshold, invert_affine_matrix,
                parray, MAX_IMAGE_BOXES, NUM_BOX_ELEMENT));
        }

        void nms_kernel_invoker(float *parray, float nms_threshold, int max_objects, int NUM_BOX_ELEMENT, cudaStream_t stream)
        {

            auto grid = CUDATools::grid_dims(max_objects);
            auto block = CUDATools::block_dims(max_objects);
            checkCudaKernel(nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold, NUM_BOX_ELEMENT));
        }

        void decode_detect_yolov8_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                                 float confidence_threshold, float *invert_affine_matrix,
                                                 float *parray, int MAX_IMAGE_BOXES, int NUM_BOX_ELEMENT, cudaStream_t stream)
        {
            auto grid = CUDATools::grid_dims(num_bboxes);
            auto block = CUDATools::block_dims(num_bboxes);
            checkCudaKernel(decode_kernel_v8_trans<<<grid, block, 0, stream>>>(
                predict, num_bboxes, num_classes, output_cdim, confidence_threshold, invert_affine_matrix,
                parray, MAX_IMAGE_BOXES, NUM_BOX_ELEMENT));
        }

        void decode_pose_yolov8_kernel_invoker(float *predict, int num_bboxes, int pose_num, int output_cdim,
                                               float confidence_threshold, float *invert_affine_matrix,
                                               float *parray, int MAX_IMAGE_BOXES, int NUM_BOX_ELEMENT, cudaStream_t stream)
        {
            auto grid = CUDATools::grid_dims(num_bboxes);
            auto block = CUDATools::block_dims(num_bboxes);
            checkCudaKernel(decode_kernel_v8_pose_trans<<<grid, block, 0, stream>>>(
                predict, num_bboxes, pose_num, output_cdim, confidence_threshold, invert_affine_matrix,
                parray, MAX_IMAGE_BOXES, NUM_BOX_ELEMENT));
        }

        void decode_single_mask(float left, float top, float *mask_weights, float *mask_predict,
                                int mask_width, int mask_height, unsigned char *mask_out,
                                int mask_dim, int out_width, int out_height, cudaStream_t stream)
        {
            // mask_weights is mask_dim(32 element) gpu pointer
            dim3 grid((out_width + 31) / 32, (out_height + 31) / 32);
            dim3 block(32, 32);

            checkCudaKernel(decode_single_mask_kernel<<<grid, block, 0, stream>>>(
                left, top, mask_weights, mask_predict, mask_width, mask_height, mask_out, mask_dim, out_width,
                out_height));
        }

        void decode_detect_rtdetr_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                                 float confidence_threshold, int scale_expand, float *parray, int MAX_IMAGE_BOXES,
                                                 int NUM_BOX_ELEMENT, cudaStream_t stream)
        {
            auto grid = CUDATools::grid_dims(num_bboxes);
            auto block = CUDATools::block_dims(num_bboxes);
            checkCudaKernel(decode_kernel_rtdetr<<<grid, block, 0, stream>>>(
                predict, num_bboxes, num_classes, output_cdim, confidence_threshold, scale_expand,
                parray, MAX_IMAGE_BOXES, NUM_BOX_ELEMENT));
        }
    }
}