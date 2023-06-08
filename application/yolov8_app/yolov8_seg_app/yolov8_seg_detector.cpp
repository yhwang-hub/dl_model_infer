#include "yolov8_seg_detctor.h"

namespace tensorrt_infer
{
    namespace yolov8_infer
    {
        void yolov8_seg_detector::initParameters(const std::string& engine_file, float score_thr, float nms_thr)
        {
            if (!file_exist(engine_file))
            {
                INFO("Error: engine_file is not exist!!!");
                exit(0);
            }
            
            this->model_info = std::make_shared<ModelInfo>();
            // 传入配置参数
            model_info->m_modelPath = engine_file;
            model_info->m_postProcCfg.confidence_threshold_ = score_thr;
            model_info->m_postProcCfg.nms_threshold_ = nms_thr;

            this->model_ = trt::infer::load(engine_file); // 加载infer对象
            this->model_->print(); // 打印engine的一些基本信息

            // 获取输入的尺寸信息
            auto input_dim = this->model_->get_network_dims(0); // 获取输入维度信息
            model_info->m_preProcCfg.infer_batch_size = input_dim[0];
            model_info->m_preProcCfg.network_input_channels_ = input_dim[1];
            model_info->m_preProcCfg.network_input_height_ = input_dim[2];
            model_info->m_preProcCfg.network_input_width_ = input_dim[3];
            model_info->m_preProcCfg.network_input_numel = input_dim[1] * input_dim[2] * input_dim[3];
            model_info->m_preProcCfg.isdynamic_model_ = this->model_->has_dynamic_dim();
            model_info->m_preProcCfg.normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::RGB);

            // seg分支的输出尺寸
            auto output_seg_dim = this->model_->get_network_dims(1);
            model_info->m_postProcCfg.seg_head_dims_ = output_seg_dim;
            model_info->m_postProcCfg.seg_head_dims_output_numel_ = output_seg_dim[1] * output_seg_dim[2] * output_seg_dim[3];

            // box分支的输出尺寸
            auto output_box_dim = this->model_->get_network_dims(2);
            model_info->m_postProcCfg.bbox_head_dims_ = output_box_dim;
            model_info->m_postProcCfg.bbox_head_dims_output_numel_ = output_box_dim[1] * output_box_dim[2];

            // 分类num
            if (model_info->m_postProcCfg.num_classes_ == 0)
            {
                model_info->m_postProcCfg.num_classes_ = output_box_dim[2] - 4 - output_seg_dim[1]; // yolov8
            }
            
            // 框的最大数量和每个框的元素数
            model_info->m_postProcCfg.NUM_BOX_ELEMENT += 1;
            model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT =\
                model_info->m_postProcCfg.MAX_IMAGE_BOXES * model_info->m_postProcCfg.NUM_BOX_ELEMENT;

            // 初始化seg后处理的scale系数
            this->scale_to_predict_x = output_seg_dim[3] / (float)input_dim[3];
            this->scale_to_predict_y = output_seg_dim[2] / (float)input_dim[2];

            CHECK(cudaStreamCreate(&cu_stream));
        }

        yolov8_seg_detector::~yolov8_seg_detector()
        {
            CHECK(cudaStreamDestroy(cu_stream));
        }

        void yolov8_seg_detector::adjust_memory(int batch_size)
        {
            input_buffer_.gpu(batch_size * model_info->m_preProcCfg.network_input_numel);
            bbox_predict_.gpu(batch_size * model_info->m_postProcCfg.bbox_head_dims_output_numel_);

            output_boxarray_.gpu(batch_size * (32 + model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT));
            output_boxarray_.cpu(batch_size * (32 + model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT));

            segment_predict_.gpu(batch_size * model_info->m_postProcCfg.seg_head_dims_output_numel_);

            if ((int)preprocess_buffers_.size() < batch_size)
            {
                for (int i = preprocess_buffers_.size(); i < batch_size; ++i)
                {
                    preprocess_buffers_.push_back(std::make_shared<Memory<unsigned char>>());
                }
            }

            if ((int)affine_matrixs.size() < batch_size)
            {
                for (int i = affine_matrixs.size(); i < batch_size; i++)
                {
                    affine_matrixs.push_back(AffineMatrix());
                }
            }
        }

        void yolov8_seg_detector::preprocess_gpu(
                int ibatach, const Image& image,
                std::shared_ptr<Memory<unsigned char>> preprocess_buffer, AffineMatrix& affine,
                cudaStream_t stream_)
        {
            if (image.channels != model_info->m_preProcCfg.network_input_channels_)
            {
                INFO("Warning : Number of channels wanted differs from number of channels in the actual image \n");
                exit(-1);
            }
            
            affine.compute(
                std::make_tuple(image.width, image.height),
                std::make_tuple(model_info->m_preProcCfg.network_input_width_, model_info->m_preProcCfg.network_input_height_),
                DetectorType::V8Seg
            );
            
            float* input_device = input_buffer_.gpu() + ibatach * model_info->m_preProcCfg.network_input_numel; // 获取当前batch的gpu内存指针
            
            size_t size_image = image.width * image.height * image.channels;
            size_t size_matrix = upbound(sizeof(affine.d2i), 32);  // 向上取整
            uint8_t* gpu_workspace = preprocess_buffer->gpu(size_matrix + size_image); // 这里把仿射矩阵+image_size放在一起申请gpu内存
            float* affine_matrix_device = (float*)gpu_workspace;
            uint8_t* image_device = gpu_workspace + size_matrix;

            // 同上，只不过申请的是cpu内存
            uint8_t* cpu_workspace = preprocess_buffer->cpu(size_matrix + size_image);
            float* affine_matrix_host = (float*)cpu_workspace;
            uint8_t* image_host = cpu_workspace + size_matrix;

            // 赋值这一步并不是多余的，这个是从分页内存到固定页内存的数据传输，可以加速向gpu内存进行数据传输
            memcpy(image_host, image.bgrptr, size_image);
            memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));

            // 从cpu-->gpu，其中image_host也可以替换为image.bgrptr然后删除上面几行，但是会慢个0.02ms左右
            checkRuntime(
                cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_)
            );
            checkRuntime(
                cudaMemcpyAsync(
                    affine_matrix_device, affine_matrix_host, sizeof(affine.d2i), cudaMemcpyHostToDevice, stream_
                )
            );

            // 执行resize + fill
            warp_affine_bilinear_and_normalize_plane(
                image_device, image.width * image.channels,
                image.width, image.height, input_device,
                model_info->m_preProcCfg.network_input_width_, model_info->m_preProcCfg.network_input_height_,
                affine_matrix_device, const_value,
                model_info->m_preProcCfg.normalize_, stream_
            );
        }

        void yolov8_seg_detector::postprocess_gpu(int ibatch, cudaStream_t stream_)
        {
            // boxarray_device: 对推理结果进行解析后要存储的gpu指针
            float* boxarray_device = output_boxarray_.gpu() + ibatch * (32 + model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT);
            // affine_matrix_device: 获取仿射变换矩阵+size_image的gpu指针，主要是用于将归一化的框尺寸缩放至图片尺寸
            float* affine_matrix_device = (float*)preprocess_buffers_[ibatch]->gpu();
            // image_based_bbox_output: 推理结果产生的所有预测框的gpu指针
            float* image_based_bbox_output = bbox_predict_.gpu() + ibatch * model_info->m_postProcCfg.bbox_head_dims_output_numel_;

            checkRuntime(
                cudaMemsetAsync(boxarray_device, 0, sizeof(int), stream_)
            );

            decode_detect_yolov8_kernel_invoker(
                image_based_bbox_output, model_info->m_postProcCfg.bbox_head_dims_[1], model_info->m_postProcCfg.num_classes_,
                model_info->m_postProcCfg.bbox_head_dims_[2], model_info->m_postProcCfg.confidence_threshold_,
                affine_matrix_device, boxarray_device, model_info->m_postProcCfg.MAX_IMAGE_BOXES,
                model_info->m_postProcCfg.NUM_BOX_ELEMENT, stream_
            );

            nms_kernel_invoker(
                boxarray_device,
                model_info->m_postProcCfg.nms_threshold_,
                model_info->m_postProcCfg.MAX_IMAGE_BOXES,
                model_info->m_postProcCfg.NUM_BOX_ELEMENT,
                stream_
            );
        }

        BatchSegBoxArray yolov8_seg_detector::parser_box(int num_image)
        {
            BatchSegBoxArray arrout(num_image);
            for (int ib = 0; ib < num_image; ib++)
            {
                float* parray = output_boxarray_.cpu() + ib * (32 + model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT);
                int count = std::min(model_info->m_postProcCfg.MAX_IMAGE_BOXES, (int)*parray);
                SegBoxArray& output = arrout[ib];
                output.reserve(count); // 增加vector的容量大于或等于count的值

                float* batch_mask_weights = bbox_predict_.gpu() + ib * model_info->m_postProcCfg.bbox_head_dims_output_numel_;
                float* batch_mask_head_predict = segment_predict_.gpu() + ib * model_info->m_postProcCfg.seg_head_dims_output_numel_;

                for (int i = 0; i < count; i++)
                {
                    float* pbox = parray + 1 + i * model_info->m_postProcCfg.NUM_BOX_ELEMENT;
                    int label = pbox[5];
                    int keepflag = pbox[6];
                    if (keepflag == 1)
                    {
                        SegBox result_object_box(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label);
                        int row_index = pbox[7];
                        // 获取筛选的框中对应的32个mask_weight
                        float* mask_weights = batch_mask_weights + row_index * model_info->m_postProcCfg.bbox_head_dims_[2] +
                                            4 + model_info->m_postProcCfg.num_classes_;

                        /*
                        此时pbox是相对于原图片的，然而mask分支是相对于160×160的
                        想要获取缩放比例，还需要把pbox进行缩放 
                        */
                        float left, top, right, bottom;
                        float* i2d = affine_matrixs[ib].i2d;
                        affine_project(i2d, pbox[0], pbox[1], &left, &top);
                        affine_project(i2d, pbox[2], pbox[3], &right, &bottom);

                        // 此时box_width和box_height是相对于640 × 640的
                        float box_width = right - left;
                        float box_height = bottom - top;

                        // 此时的mask_out是当前框相对于160×160的mask宽高
                        int mask_out_width = box_width * scale_to_predict_x + 0.5f;
                        int mask_out_height = box_height * scale_to_predict_y + 0.5f;

                        if (mask_out_width > 0 && mask_out_height > 0)
                        {
                            // box_mask的cpu内存申请
                            result_object_box.seg = std::make_shared<InstanceSegmentMap>(mask_out_width, mask_out_height);
                            unsigned char* mask_out_host = result_object_box.seg->data;

                            // box_mask的gpu内存申请
                            box_segment_predict_.release_gpu();
                            int bytes_of_mask_out = mask_out_width * mask_out_height;
                            unsigned char* mask_out_device = box_segment_predict_.gpu(bytes_of_mask_out);

                            // 解析mask， 并存入mask_out_host中
                            decode_single_mask(
                                left * scale_to_predict_x, top * scale_to_predict_y, mask_weights,
                                batch_mask_head_predict, model_info->m_postProcCfg.seg_head_dims_[3],
                                model_info->m_postProcCfg.seg_head_dims_[2], mask_out_device,
                                model_info->m_postProcCfg.seg_head_dims_[1], mask_out_width, mask_out_height, cu_stream
                            );

                            checkRuntime(
                                cudaMemcpyAsync(
                                    mask_out_host, mask_out_device,
                                    box_segment_predict_.gpu_bytes(),
                                    cudaMemcpyDeviceToHost, cu_stream
                                )
                            );
                        }
                        output.emplace_back(result_object_box);
                    }
                }
            }
            checkRuntime(cudaStreamSynchronize(cu_stream)); // 阻塞cuda流等待其所有任务完成

            return arrout;
        }

        SegBoxArray yolov8_seg_detector::forward(const Image& image)
        {
            auto output = forwards({image});
            if (output.empty())
            {
                return {};
            }
            
            return output[0];
        }

        BatchSegBoxArray yolov8_seg_detector::forwards(const std::vector<Image>& images)
        {
            int num_image = images.size();
            if (num_image == 0)
            {
                return {};
            }
            
            // 动态设置batch_size
            auto input_dims = this->model_->get_network_dims(0);
            if (model_info->m_preProcCfg.infer_batch_size != num_image)
            {
                if (model_info->m_preProcCfg.isdynamic_model_)
                {
                    model_info->m_preProcCfg.infer_batch_size = num_image;
                    input_dims[0] = num_image;
                    if (!model_->set_network_dims(0, input_dims)) // 重新绑定输入batch，返回值类型为bool
                    {
                        return {};
                    }
                }
                else
                {
                    if (model_info->m_preProcCfg.infer_batch_size < num_image)
                    {
                        INFO(
                            "When using static shape model, number of images[%d] must be "
                            "less than or equal to the maximum batch[%d].",
                            num_image, model_info->m_preProcCfg.infer_batch_size);
                        return {};
                    }
                }
            }
            
            // 由于batch_size是动态的，所以需要对gpu/cpu内存进行动态的申请
            adjust_memory(model_info->m_preProcCfg.infer_batch_size);

            // 对图片进行预处理
            for (int i = 0; i < num_image; i++)
            {
                preprocess_gpu(i, images[i], preprocess_buffers_[i], affine_matrixs[i], cu_stream);
            }
            
            // 推理模型
            float* bbox_output_device = bbox_predict_.gpu();
            std::vector<void*> bindings{input_buffer_.gpu(), segment_predict_.gpu(), bbox_output_device};

            if (!model_->forward(bindings, cu_stream))
            {
                INFO("Failed to tensorRT forward.");
                return {};
            }

            for (int ib = 0; ib < num_image; ib++)
            {
                postprocess_gpu(ib, cu_stream);
            }
            
            checkRuntime(
                cudaMemcpyAsync(
                    output_boxarray_.cpu(), output_boxarray_.gpu(),
                    output_boxarray_.gpu_bytes(), cudaMemcpyDeviceToHost, cu_stream
                )
            );
            checkRuntime(cudaStreamSynchronize(cu_stream)); // 阻塞异步流，等流中所有操作执行完成才会继续执行

            return parser_box(num_image);
        }
    }
}