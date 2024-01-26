#include "post_process.h"
namespace ai
{
    namespace postprocess
    {
        // static _device__ float sigmoid(const float x)
        // { 
        //     return 1.0f / (1.0f + expf(-x));
        // }
        __device__ float sigmoid_gpu(const float x) { return 1.0f / (1.0f + expf(-x)); }

        __device__ float Clamp(const float x)
        {
            float x_clamp = x;

            if (x_clamp > 1.0f)
            {
                x_clamp = 1.0f;
            }
            else if (x_clamp < 0.0f)
            {
                x_clamp = 0.0f;
            }

            return x_clamp;
        }

        // keepflag主要是用来进行nms时候判断是否将该框抛弃
        // const int NUM_BOX_ELEMENT = 7; // left, top, right, bottom, confidence, class, keepflag
        static __device__ void affine_project(float *matrix, float x, float y, float *ox, float *oy)
        {
            *ox = matrix[0] * x + matrix[1] * y + matrix[2];
            *oy = matrix[3] * x + matrix[4] * y + matrix[5];
        }

        static __device__ void affine_project_(float* matrix, float x, float y, float w, float h, float* ox, float* oy, float* ow, float* oh)
        {
            *ox = matrix[0] * x + matrix[1] * y + matrix[2];
            *oy = matrix[3] * x + matrix[4] * y + matrix[5];
            *ow = matrix[0] * w;
            *oh = matrix[0] * h;
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

        static __device__ void convariance_matrix(float w, float h, float r, float& a, float& b, float& c)
        {
            float a_val = w * w / 12.0f;
            float b_val = h * h / 12.0f;
            float cos_r = cosf(r); 
            float sin_r = sinf(r);

            a = a_val * cos_r * cos_r + b_val * sin_r * sin_r;
            b = a_val * sin_r * sin_r + b_val * cos_r * cos_r;
            c = (a_val - b_val) * sin_r * cos_r;
        }

        static __device__ float box_probiou(
            float cx1, float cy1, float w1, float h1, float r1,
            float cx2, float cy2, float w2, float h2, float r2,
            float eps = 1e-7
        )
        {

            // Calculate the prob iou between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.
            float a1, b1, c1, a2, b2, c2;
            convariance_matrix(w1, h1, r1, a1, b1, c1);
            convariance_matrix(w2, h2, r2, a2, b2, c2);

            float t1 = ((a1 + a2) * powf(cy1 - cy2, 2) + (b1 + b2) * powf(cx1 - cx2, 2)) / ((a1 + a2) * (b1 + b2) - powf(c1 + c2, 2) + eps);
            float t2 = ((c1 + c2) * (cx2 - cx1) * (cy1 - cy2)) / ((a1 + a2) * (b1 + b2) - powf(c1 + c2, 2) + eps);
            float t3 = logf(((a1 + a2) * (b1 + b2) - powf(c1 + c2, 2)) / (4 * sqrtf(fmaxf(a1 * b1 - c1 * c1, 0.0f)) * sqrtf(fmaxf(a2 * b2 - c2 * c2, 0.0f)) + eps) + eps); 
            float bd = 0.25f * t1 + 0.5f * t2 + 0.5f * t3;
            bd = fmaxf(fminf(bd, 100.0f), eps);
            float hd = sqrtf(1.0f - expf(-bd) + eps);
            return 1 - hd;    
        }

        static __global__ void rotatebbox_nms_kernel(float* bboxes, int max_objects, float threshold, int NUM_BOX_ELEMENT)
        {
            int position = (blockDim.x * blockIdx.x + threadIdx.x);
            int count = min((int)*bboxes, max_objects);
            if (position >= count) 
                return;
            
            // cx, cy, w, h, angle, confidence, class_label, keepflag
            float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
            for(int i = 0; i < count; ++i)
            {
                float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
                if(i == position || pcurrent[6] != pitem[6])
                    continue;

                if(pitem[5] >= pcurrent[5])
                {
                    if(pitem[5] == pcurrent[5] && i < position)
                        continue;

                    float iou = box_probiou(
                        pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pcurrent[4],
                        pitem[0],    pitem[1],    pitem[2],    pitem[3],    pitem[4]
                    );

                    if(iou > threshold)
                    {
                        pcurrent[7] = 0;  // 1=keep, 0=ignore
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

        static __global__ void decode_yolov8_obb_kernel(float* predict, int num_bboxes, int num_classes,
                                                        float confidence_threshold, float* invert_affine_matrix,
                                                        float* parray, int MAX_IMAGE_BOXES, int NUM_ROTATEBOX_ELEMENT)
        {  
            // cx, cy, w, h, cls, angle
            int position = blockDim.x * blockIdx.x + threadIdx.x;
            if (position >= num_bboxes)
                return;

            float* pitem            = predict + (5 + num_classes) * position;
            float* class_confidence = pitem + 4;
            float confidence        = *class_confidence++;
            int label               = 0;
            for(int i = 1; i < num_classes; ++i, ++class_confidence)
            {
                if(*class_confidence > confidence)
                {
                    confidence = *class_confidence;
                    label      = i;
                }
            }

            if(confidence < confidence_threshold)
                return;

            int index = atomicAdd(parray, 1);
            if(index >= MAX_IMAGE_BOXES)
                return;

            float cx         = *pitem++;
            float cy         = *pitem++;
            float width      = *pitem++;
            float height     = *pitem++;
            float angle      = *(pitem + num_classes);
            affine_project_(invert_affine_matrix, cx, cy, width, height, &cx, &cy, &width, &height);

            float* pout_item = parray + 1 + index * NUM_ROTATEBOX_ELEMENT;
            *pout_item++ = cx;
            *pout_item++ = cy;
            *pout_item++ = width;
            *pout_item++ = height;
            *pout_item++ = angle;
            *pout_item++ = confidence;
            *pout_item++ = label;
            *pout_item++ = 1; // 1 = keep, 0 = ignore
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

         static __global__ void decode_kernel_detr(float *bbox_predict, float *label_predict,
                                                int input_h, int input_w,
                                                int num_bboxes, float confidence_threshold,
                                                float *parray,
                                                int MAX_IMAGE_BOXES, int NUM_BOX_ELEMENT)
        {
            int position = blockDim.x * blockIdx.x + threadIdx.x;
            if (position >= num_bboxes)
                return;

            float *pbbox = bbox_predict + 5 * position;
            float *plabel = label_predict + 1 * position;

            float *class_confidence = pbbox + 4;
            float confidence = *class_confidence;
            int label = (int)(*plabel);

            if (confidence < confidence_threshold)
                return;

            int index = atomicAdd(parray, 1);
            if (index >= MAX_IMAGE_BOXES)
                return;

            float x1 = *pbbox;
            float y1 = *(pbbox + 1);
            float x2 = *(pbbox + 2);
            float y2 = *(pbbox + 3);

            float left   =  static_cast<float>(input_w) * Clamp(x1);
            float top    =  static_cast<float>(input_h) * Clamp(y1);
            float right  =  static_cast<float>(input_w) * Clamp(x2);
            float bottom =  static_cast<float>(input_h) * Clamp(y2);

            float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
            *pout_item++ = left;
            *pout_item++ = top;
            *pout_item++ = right;
            *pout_item++ = bottom;
            *pout_item++ = confidence;
            *pout_item++ = label;
            *pout_item++ = 1;
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

        void rotatebbox_nms_kernel_invoker(float* parray, float nms_threshold, int max_objects, int NUM_BOX_ELEMENT, cudaStream_t stream)
        {
            auto grid = CUDATools::grid_dims(max_objects);
            auto block = CUDATools::block_dims(max_objects);
            checkCudaKernel(rotatebbox_nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold, NUM_BOX_ELEMENT));
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

        void decode_yolov8_obb_kernel_invoker(float* predict, int num_bboxes, int num_classes,
                                              float confidence_threshold, float* invert_affine_matrix,
                                              float* parray, int MAX_IMAGE_BOXES, int NUM_ROTATEBOX_ELEMENT, cudaStream_t stream)
        {
            auto grid = CUDATools::grid_dims(num_bboxes);
            auto block = CUDATools::block_dims(num_bboxes);
            checkCudaKernel(decode_yolov8_obb_kernel<<<grid, block, 0, stream>>>(
                predict, num_bboxes, num_classes, confidence_threshold, invert_affine_matrix,
                parray, MAX_IMAGE_BOXES, NUM_ROTATEBOX_ELEMENT));
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

        void decode_detect_detr_kernel_invoker(float *bbox_predict, float *label_predict,
                                            int input_h, int input_w, int num_bboxes,
                                            float confidence_threshold, float *parray, int MAX_IMAGE_BOXES,
                                            int NUM_BOX_ELEMENT, cudaStream_t stream)
        {
            auto grid = CUDATools::grid_dims(num_bboxes);
            auto block = CUDATools::block_dims(num_bboxes);
            checkCudaKernel(decode_kernel_detr<<<grid, block, 0, stream>>>(
                bbox_predict, label_predict, input_h, input_w, num_bboxes, confidence_threshold,
                parray, MAX_IMAGE_BOXES, NUM_BOX_ELEMENT));
        }

        /**********bev postprocess*************/
        __global__ void BEVDecodeObjectKernel(const int map_size,         // 40000
                                    const float score_thresh,   // 0.1
                                    //    const int nms_pre_max_size, // 4096
                                    const float x_start,
                                    const float y_start,
                                    const float x_step,
                                    const float y_step,
                                    const int output_h,
                                    const int output_w,
                                    const int downsample_size,
                                    const int num_class_in_task,
                                    const int cls_range,
                                    const float* reg,
                                    const float* hei,
                                    const float* dim,
                                    const float* rot,
                                    const float* vel,
                                    const float* cls,
                                    float* res_box,
                                    float* res_conf,
                                    int* res_cls,
                                    int* res_box_num,
                                    float* rescale_factor) // 根据置信度，初筛，筛选后有res_box_num个box，不超过nms_pre_max_size 4096
        {
            int idx = threadIdx.x + blockDim.x * blockIdx.x;
            if (idx >= map_size) return;

            float max_score = cls[idx]; // 初始化为task的第一个类
            int label = cls_range;      // 初始化为task的第一个类
            for (int i = 1; i < num_class_in_task; ++i)
            {
                float cur_score = cls[idx + i * map_size];
                if (cur_score > max_score)
                {
                    max_score = cur_score;
                    label = i + cls_range;
                }
            }

            int coor_x = idx % output_h;  //
            int coor_y = idx / output_w;  //

            float conf = sigmoid_gpu(max_score); // 计算置信度
            if (conf > score_thresh)
            {
                int cur_valid_box_id = atomicAdd(res_box_num, 1);
                res_box[cur_valid_box_id * kBoxBlockSize + 0] = 
                    (reg[idx + 0 * map_size] + coor_x) * x_step + x_start;
                res_box[cur_valid_box_id * kBoxBlockSize + 1] = 
                    (reg[idx + 1 * map_size] + coor_y) * y_step + y_start;
                res_box[cur_valid_box_id * kBoxBlockSize + 2] = hei[idx];
                res_box[cur_valid_box_id * kBoxBlockSize + 3] = 
                                        expf(dim[idx + 0 * map_size]) * rescale_factor[label]; // nms scale
                res_box[cur_valid_box_id * kBoxBlockSize + 4] = 
                                        expf(dim[idx + 1 * map_size]) * rescale_factor[label];
                res_box[cur_valid_box_id * kBoxBlockSize + 5] = 
                                        expf(dim[idx + 2 * map_size]) * rescale_factor[label];
                res_box[cur_valid_box_id * kBoxBlockSize + 6] = atan2f(rot[idx], rot[idx + map_size]);
                res_box[cur_valid_box_id * kBoxBlockSize + 7] = vel[idx];
                res_box[cur_valid_box_id * kBoxBlockSize + 8] = vel[idx + map_size];


                res_conf[cur_valid_box_id] = conf;
                res_cls[cur_valid_box_id] = label;
            }
        }

        BEVDetPostprocessGPU::BEVDetPostprocessGPU(const int _class_num, 
                                    const float _score_thresh,
                                    const float _nms_thresh,
                                    const int _nms_pre_maxnum,
                                    const int _nms_post_maxnum,
                                    const int _down_sample,
                                    const int _output_h,
                                    const int _output_w,
                                    const float _x_step,
                                    const float _y_step,
                                    const float _x_start,
                                    const float _y_start,
                                    const std::vector<int>& _class_num_pre_task,
                                    const std::vector<float>& _nms_rescale_factor) :
                                    class_num(_class_num),
                                    score_thresh(_score_thresh),
                                    nms_thresh(_nms_thresh),
                                    nms_pre_maxnum(_nms_pre_maxnum),
                                    nms_post_maxnum(_nms_post_maxnum), 
                                    down_sample(_down_sample),
                                    output_h(_output_h),
                                    output_w(_output_w),
                                    x_step(_x_step),
                                    y_step(_y_step), 
                                    x_start(_x_start), 
                                    y_start(_y_start),
                                    map_size(output_h * output_w),
                                    class_num_pre_task(_class_num_pre_task),
                                    nms_rescale_factor(_nms_rescale_factor),
                                    task_num(_class_num_pre_task.size())
        {
            CHECK(cudaMalloc((void**)&boxes_dev, kBoxBlockSize * map_size * sizeof(float)));
            CHECK(cudaMalloc((void**)&score_dev, map_size * sizeof(float)));
            CHECK(cudaMalloc((void**)&cls_dev, map_size * sizeof(int)));
            CHECK(cudaMalloc((void**)&sorted_indices_dev, map_size * sizeof(int)));
            CHECK(cudaMalloc((void**)&valid_box_num, sizeof(int)));
            CHECK(cudaMalloc((void**)&nms_rescale_factor_dev, class_num * sizeof(float)));

            CHECK(cudaMallocHost((void**)&boxes_host, kBoxBlockSize * map_size * sizeof(float)));
            CHECK(cudaMallocHost((void**)&score_host, nms_pre_maxnum * sizeof(float)));
            CHECK(cudaMallocHost((void**)&cls_host, map_size * sizeof(float)));
            CHECK(cudaMallocHost((void**)&sorted_indices_host, nms_pre_maxnum * sizeof(int)));
            CHECK(cudaMallocHost((void**)&keep_data_host, nms_pre_maxnum * sizeof(long)));

            CHECK(cudaMemcpy(nms_rescale_factor_dev, nms_rescale_factor.data(),
                            class_num * sizeof(float), cudaMemcpyHostToDevice));

            iou3d_nms.reset(new Iou3dNmsCuda(output_h, output_w, nms_thresh));

            for(auto i = 0; i < nms_rescale_factor.size(); i++)
            {
                printf("%.2f%c", nms_rescale_factor[i], i == nms_rescale_factor.size() - 1 ? '\n' : ' ');
            }

        }
        BEVDetPostprocessGPU::~BEVDetPostprocessGPU()
        {
            CHECK(cudaFree(boxes_dev));
            CHECK(cudaFree(score_dev));
            CHECK(cudaFree(cls_dev));
            CHECK(cudaFree(sorted_indices_dev));
            CHECK(cudaFree(valid_box_num));
            CHECK(cudaFree(nms_rescale_factor_dev));

            CHECK(cudaFreeHost(boxes_host));
            CHECK(cudaFreeHost(score_host));
            CHECK(cudaFreeHost(cls_host));
            CHECK(cudaFreeHost(sorted_indices_host));
            CHECK(cudaFreeHost(keep_data_host));
        }

        void BEVDetPostprocessGPU::DoPostprocess(std::vector<void*> bev_buffer, std::vector<bevBox>& out_detections)
        {
            // bev_buffer : BEV_feat, reg_0, hei_0, dim_0, rot_0, vel_0, heatmap_0, reg_1 ...
            const int task_num = class_num_pre_task.size();
            std::cout << "task_num: " << task_num << std::endl;
            int cur_start_label = 0;
            for(int i = 0; i < task_num; i++)
            {
                float* reg = (float*)bev_buffer[i * 6 + 1];     // 2 x 128 x 128
                float* hei = (float*)bev_buffer[i * 6 + 2];     // 1 x 128 x 128
                float* dim = (float*)bev_buffer[i * 6 + 3];     // 3 x 128 x 128
                float* rot = (float*)bev_buffer[i * 6 + 4];     // 2 x 128 x 128
                float* vel = (float*)bev_buffer[i * 6 + 5];     // 2 x 128 x 128
                float* heatmap = (float*)bev_buffer[i * 6 + 6]; // c x 128 x 128

                dim3 grid(DIVUP(map_size, NUM_THREADS));
                CHECK(cudaMemset(valid_box_num, 0, sizeof(int)));
                BEVDecodeObjectKernel<<<grid, NUM_THREADS>>>(map_size, score_thresh, 
                                                x_start, y_start, x_step, y_step, output_h,
                                                output_w, down_sample, class_num_pre_task[i],
                                                cur_start_label, reg, hei, dim, rot, 
                                                vel, 
                                                heatmap,
                                                boxes_dev, score_dev, cls_dev, valid_box_num,
                                                nms_rescale_factor_dev);

                /*
                此时 boxes_dev, score_dev, cls_dev 有 valid_box_num 个元素，可能大于nms_pre_maxnum, 而且是无序排列的
                */ 
                int box_num_pre = 0;
                CHECK(cudaMemcpy(&box_num_pre, valid_box_num, sizeof(int), cudaMemcpyDeviceToHost));

                thrust::sequence(thrust::device, sorted_indices_dev, sorted_indices_dev + box_num_pre);
                thrust::sort_by_key(thrust::device, score_dev, score_dev + box_num_pre, 
                                    sorted_indices_dev, thrust::greater<float>());
                // 此时 score_dev 是降序排列的，而 sorted_indices_dev 索引着原顺序，
                // 即 sorted_indices_dev[i] = j; i:现在的位置，j:原索引;  j:[0, map_size)

                box_num_pre = std::min(box_num_pre, nms_pre_maxnum);

                int box_num_post = iou3d_nms->DoIou3dNms(box_num_pre, boxes_dev, 
                                                                sorted_indices_dev, keep_data_host);

                box_num_post = std::min(box_num_post, nms_post_maxnum);


                CHECK(cudaMemcpy(sorted_indices_host, sorted_indices_dev, box_num_pre * sizeof(int),
                                                                            cudaMemcpyDeviceToHost));
                CHECK(cudaMemcpy(boxes_host, boxes_dev, kBoxBlockSize * map_size * sizeof(float),
                                                                            cudaMemcpyDeviceToHost));
                CHECK(cudaMemcpy(score_host, score_dev, box_num_pre * sizeof(float), 
                                                                            cudaMemcpyDeviceToHost));
                CHECK(cudaMemcpy(cls_host, cls_dev, map_size * sizeof(float), 
                                                                            cudaMemcpyDeviceToHost));


                for (auto j = 0; j < box_num_post; j++)
                {
                    int k = keep_data_host[j];
                    int idx = sorted_indices_host[k];
                    bevBox box;
                    box.x = boxes_host[idx * kBoxBlockSize + 0];
                    box.y = boxes_host[idx * kBoxBlockSize + 1];
                    box.z = boxes_host[idx * kBoxBlockSize + 2];
                    box.l = boxes_host[idx * kBoxBlockSize + 3] / nms_rescale_factor[cls_host[idx]];
                    box.w = boxes_host[idx * kBoxBlockSize + 4] / nms_rescale_factor[cls_host[idx]];
                    box.h = boxes_host[idx * kBoxBlockSize + 5] / nms_rescale_factor[cls_host[idx]];
                    box.r = boxes_host[idx * kBoxBlockSize + 6];
                    box.vx = boxes_host[idx * kBoxBlockSize + 7];
                    box.vy = boxes_host[idx * kBoxBlockSize + 8];

                    box.label = cls_host[idx];
                    box.score = score_host[k];
                    box.z -= box.h * 0.5; // bottom height
                    out_detections.push_back(box);
                }

                cur_start_label += class_num_pre_task[i];
            }
        }
        /**********bev postprocess*************/
    }
}