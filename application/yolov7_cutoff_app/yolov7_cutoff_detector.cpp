#include "yolov7_cutoff_detector.h"

namespace tensorrt_infer
{
    namespace yolov7_cutoff_infer
    {
        void yolov7_cutoff_detector::initParameters(
            const std::string& engine_file,
            float score_thr,
            float nms_thr)
        {
            if (!file_exist(engine_file))
            {
                INFO("Error: engine_file is not exist!!!");
                exit(0);
            }

            this->model_info = std::make_shared<ModelInfo>();

            model_info->m_modelPath = engine_file;
            model_info->m_postProcCfg.confidence_threshold_ = score_thr;
            model_info->m_postProcCfg.nms_threshold_ = nms_thr;

            this->model_ = trt::infer::load(engine_file);
            this->model_->print();

            auto input_dim = this->model_->get_network_dims(0);
            model_info->m_preProcCfg.infer_batch_size = input_dim[0];
            model_info->m_preProcCfg.network_input_channels_ = input_dim[1];
            model_info->m_preProcCfg.network_input_height_ = input_dim[2];
            model_info->m_preProcCfg.network_input_width_ = input_dim[3];
            model_info->m_preProcCfg.network_input_numel = input_dim[1] * input_dim[2] * input_dim[3];
            model_info->m_preProcCfg.isdynamic_model_ = this->model_->has_dynamic_dim();

            model_info->m_preProcCfg.normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::RGB);

            for (int i = 0; i < num_stages; i++)
            {
                auto det_output_dim = this->model_->get_network_dims(i + 1);
                det_output_buffer_size[i] = det_output_dim[1] * det_output_dim[2] * det_output_dim[3];
            }

            model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT =\
                model_info->m_postProcCfg.MAX_IMAGE_BOXES * model_info->m_postProcCfg.NUM_BOX_ELEMENT;

            CHECK(cudaStreamCreate(&cu_stream));
        }

        yolov7_cutoff_detector::~yolov7_cutoff_detector()
        {
            CHECK(cudaStreamDestroy(cu_stream));
        }

        void yolov7_cutoff_detector::adjust_memory(int batch_size)
        {
            input_buffer_.gpu(batch_size * model_info->m_preProcCfg.network_input_numel);
            
            for (int i = 0; i < num_stages; i++)
            {
                det_output_predicts_[i].gpu(batch_size * det_output_buffer_size[i]);
                det_output_predicts_[i].cpu(batch_size * det_output_buffer_size[i]);
            }

            output_boxarray_.gpu(batch_size * (32 + model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT));
            output_boxarray_.cpu(batch_size * (32 + model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT));

            if ((int)preprocess_buffers_.size() < batch_size)
            {
                for (int i = preprocess_buffers_.size(); i < batch_size; i++)
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

        void yolov7_cutoff_detector::preprocess_gpu(
            int ibatch, const Image& image,
            std::shared_ptr<Memory<unsigned char>> preprocess_buffer,
            AffineMatrix& affine,
            cudaStream_t stream_)
        {
            if (image.channels != model_info->m_preProcCfg.network_input_channels_)
            {
                INFO("Warning : Number of channels wanted differs from number of channels in the actual image \n");
                exit(-1);
            }

            affine.compute(std::make_tuple(image.width, image.height),
                        std::make_tuple(model_info->m_preProcCfg.network_input_width_, model_info->m_preProcCfg.network_input_height_),
                        DetectorType::V7);
            float* input_device = input_buffer_.gpu() + ibatch * model_info->m_preProcCfg.network_input_numel;
            size_t size_image = image.width * image.height * image.channels;
            size_t size_matrix = upbound(sizeof(affine.d2i), 32);
            uint8_t* gpu_workspace = preprocess_buffer->gpu(size_matrix + size_image);
            float* affine_matrix_device = (float*)gpu_workspace;
            uint8_t* image_device = gpu_workspace + size_matrix;

            uint8_t* cpu_workspace = preprocess_buffer->cpu(size_matrix + size_image);
            float* affine_matrix_host = (float*)cpu_workspace;
            uint8_t* image_host = cpu_workspace + size_matrix;

            memcpy(image_host, image.bgrptr, size_image);
            memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));

            checkRuntime(
                cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_)
            );
            checkRuntime(
                cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i),
                                cudaMemcpyHostToDevice, stream_)
            );

            warp_affine_bilinear_and_normalize_plane(
                image_device, image.width * image.channels, image.width,
                image.height, input_device, model_info->m_preProcCfg.network_input_width_,
                model_info->m_preProcCfg.network_input_height_, affine_matrix_device, const_value,
                model_info->m_preProcCfg.normalize_, stream_
            );
        }

        void yolov7_cutoff_detector::postprocess_gpu(int ibatch, cudaStream_t stream_)
        {
            float* boxarray_cpu = output_boxarray_.cpu() + \
                            ibatch * (32 + model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT);
            float* affine_matrix_host = (float*)preprocess_buffers_[ibatch]->cpu();
            for (int stride = 0; stride < num_stages; stride++)
            {
                checkRuntime(
                    cudaMemcpyAsync(
                        det_output_predicts_[stride].cpu(), det_output_predicts_[stride].gpu(),
                        det_output_predicts_[stride].gpu_bytes(), cudaMemcpyDeviceToHost, stream_
                    )
                );
                checkRuntime(cudaStreamSynchronize(stream_));

                float* det_output_cpu = det_output_predicts_[stride].cpu() + ibatch * det_output_buffer_size[stride];
                checkRuntime(cudaStreamSynchronize(stream_));

                int num_grid_x = (int)(model_info->m_preProcCfg.network_input_width_ / strides[stride]);
                int num_grid_y = (int)(model_info->m_preProcCfg.network_input_height_ / strides[stride]);

                for(int anchor = 0; anchor < 3; ++anchor)
                {
                    const float anchor_w = netAnchors[stride][anchor * 2];
                    const float anchor_h = netAnchors[stride][anchor * 2 + 1];

                    for(int i = 0; i < num_grid_x * num_grid_y; ++i)
                    {
                        int obj_index = i + 4 * num_grid_x * num_grid_y + anchor * 85 * num_grid_x * num_grid_y;
                        float objness = sigmoid_x(det_output_cpu[obj_index]);

                        if(objness < model_info->m_postProcCfg.confidence_threshold_)
                            continue;

                        int label = 0;
                        float prob = 0.0;
                        for (int index = 5; index < 85; index++)
                        {
                            int class_index = i + index * num_grid_x * num_grid_y + anchor * 85 * num_grid_x * num_grid_y;
                            if (sigmoid_x(det_output_cpu[class_index]) > prob)
                            {
                                label = index - 5;
                                prob = sigmoid_x(det_output_cpu[class_index]);
                            }
                        }

                        float confidence = prob * objness;
                        if(confidence < model_info->m_postProcCfg.confidence_threshold_)
                            continue;

                        int grid_y = i / num_grid_x;
                        int grid_x = i - grid_y * num_grid_x;

                        int x_index = i + 0 * num_grid_x * num_grid_y + anchor * 85 * num_grid_x * num_grid_y;
                        float x_data = sigmoid_x(det_output_cpu[x_index]);
                        x_data = x_data * 2.0f * strides[stride] + strides[stride] * (grid_x- 0.5);

                        int y_index = i + 1 * num_grid_x * num_grid_y + anchor * 85 * num_grid_x * num_grid_y;
                        float y_data = sigmoid_x(det_output_cpu[y_index]);
                        y_data = y_data * 2.0f * strides[stride] + strides[stride] * (grid_y - 0.5);

                        int w_index = i + 2 * num_grid_x * num_grid_y + anchor * 85 * num_grid_x * num_grid_y;
                        float w_data = sigmoid_x(det_output_cpu[w_index]);
                        w_data = w_data * w_data * (4 * anchor_w);

                        int h_index = i + 3 * num_grid_x * num_grid_y + anchor * 85 * num_grid_x * num_grid_y;
                        float h_data = sigmoid_x(det_output_cpu[h_index]);
                        h_data = h_data * h_data * (4 * anchor_h);

                        float x_center     = x_data;
                        float y_center     = y_data;
                        float width  = w_data;
                        float height = h_data;

                        float left   = x_center - width * 0.5f;
                        float top    = y_center - height * 0.5f;
                        float right  = x_center + width * 0.5f;
                        float bottom = y_center + height * 0.5f;

                        left    = affine_matrix_host[0] * left  + affine_matrix_host[1] * top,    + affine_matrix_host[2];
                        top     = affine_matrix_host[3] * left  + affine_matrix_host[4] * top,    + affine_matrix_host[5];
                        right   = affine_matrix_host[0] * right + affine_matrix_host[1] * bottom, + affine_matrix_host[2];
                        bottom  = affine_matrix_host[3] * right + affine_matrix_host[4] * bottom, + affine_matrix_host[5];

                        *boxarray_cpu += 1;
                        int index = *boxarray_cpu;

                        float* pout_item = boxarray_cpu + 1 + index * model_info->m_postProcCfg.NUM_BOX_ELEMENT;
                        *pout_item++ = left;
                        *pout_item++ = top;
                        *pout_item++ = right;
                        *pout_item++ = bottom;
                        *pout_item++ = confidence;
                        *pout_item++ = label;
                        *pout_item++ = 1; // 1 = keep, 0 = ignore
                    }
                }
            }

            fast_nms_cpu(
                boxarray_cpu,
                model_info->m_postProcCfg.nms_threshold_,
                model_info->m_postProcCfg.MAX_IMAGE_BOXES,
                model_info->m_postProcCfg.NUM_BOX_ELEMENT
            );
        }

        BatchBoxArray yolov7_cutoff_detector::parser_box(int num_image)
        {
            BatchBoxArray arrout(num_image);
            for (int ib = 0; ib < num_image; ib++)
            {
                float* parray = output_boxarray_.cpu() + ib * (32 + model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT);
                int count = std::min(model_info->m_postProcCfg.MAX_IMAGE_BOXES, (int)*parray);
                BoxArray& output = arrout[ib];
                output.reserve(count);

                for (int i = 0; i < count; i++)
                {
                    float* pbox = parray + 1 + i * model_info->m_postProcCfg.NUM_BOX_ELEMENT;
                    int label = pbox[5];
                    int keepflag = pbox[6];
                    if (keepflag == 1)
                    {
                        output.emplace_back(
                            pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label
                        );
                    }   
                }               
            }
            
            return arrout;
        }

        BoxArray yolov7_cutoff_detector::forward(const Image& image)
        {
            auto output = forwards({image});
            if (output.empty())
            {
                return {};
            }
            
            return output[0];
        }

        BatchBoxArray yolov7_cutoff_detector::forwards(const std::vector<Image>& images)
        {
            int num_image = images.size();
            if (num_image == 0)
            {
                return {};
            }

            auto input_dims = this->model_->get_network_dims(0);
            if (model_info->m_preProcCfg.infer_batch_size != num_image)
            {
                if (model_info->m_preProcCfg.isdynamic_model_)
                {
                    model_info->m_preProcCfg.infer_batch_size = num_image;
                    input_dims[0] = num_image;
                    if (!this->model_->set_network_dims(0, input_dims))
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
                            num_image, model_info->m_preProcCfg.infer_batch_size
                        );
                        return {};
                    }
                }
            }

            adjust_memory(model_info->m_preProcCfg.infer_batch_size);

            for (int i = 0; i < num_image; i++)
            {
                preprocess_gpu(
                    i, images[i], preprocess_buffers_[i],
                    affine_matrixs[i], cu_stream
                );
            }

            std::vector<void*> bindings{input_buffer_.gpu()};
            for (int i = 0; i < num_stages; i++)
            {
                float* det_output_device = det_output_predicts_[i].gpu();
                bindings.push_back(det_output_device);
            }

            if (!this->model_->forward(bindings, cu_stream))
            {
                INFO("Failed to tensorRT forward.");
                return {};
            }

            for (int ib = 0; ib < num_image; ib++)
            {
                postprocess_gpu(ib, cu_stream);
            }

            return parser_box(num_image);
        }
    }
}