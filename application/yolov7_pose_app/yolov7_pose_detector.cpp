#include "yolov7_pose_detector.h"

namespace tensorrt_infer
{
    namespace yolov7_pose_infer
    {
        void yolov7_pose_detector::initParameters(
            const std::string& engine_file,
            float score_thr,
            float nms_thr)
        {
            if (!file_exist(engine_file))
            {
                INFO("Error: engine_file is not exist!!!");
                exit(0);
            }
            
            this->model_info =std::make_shared<ModelInfo>();

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

            if (model_info->m_postProcCfg.pose_num_ == 0)
            {
                model_info->m_postProcCfg.pose_num_ = det_info_len_kpt / 3;
            }

            model_info->m_postProcCfg.NUM_BOX_ELEMENT += model_info->m_postProcCfg.pose_num_ * 3;
            model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT = \
                            model_info->m_postProcCfg.MAX_IMAGE_BOXES * \
                            model_info->m_postProcCfg.NUM_BOX_ELEMENT;

            CHECK(cudaStreamCreate(&cu_stream));
        }

        yolov7_pose_detector::~yolov7_pose_detector()
        {
            CHECK(cudaStreamDestroy(cu_stream));
        }

        void yolov7_pose_detector::adjust_memory(int batch_size)
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

        void yolov7_pose_detector::preprocess_gpu(
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

            affine.compute(
                std::make_tuple(image.width, image.height),
                std::make_tuple(model_info->m_preProcCfg.network_input_width_, model_info->m_preProcCfg.network_input_height_),
                DetectorType::V7Pose
            );

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
                cudaMemcpyAsync(
                    image_device, image_host, size_image,
                    cudaMemcpyHostToDevice, stream_
                )
            );
            checkRuntime(
                cudaMemcpyAsync(
                    affine_matrix_device, affine_matrix_host, sizeof(affine.d2i),
                    cudaMemcpyHostToDevice, stream_
                )
            );

            warp_affine_bilinear_and_normalize_plane(
                image_device, image.width * image.channels, image.width,
                image.height, input_device, model_info->m_preProcCfg.network_input_width_,
                model_info->m_preProcCfg.network_input_height_, affine_matrix_device, const_value,
                model_info->m_preProcCfg.normalize_, stream_
            );
        }

        void yolov7_pose_detector::postprocess_gpu(int ibatch, cudaStream_t stream_)
        {
            float* boxarray_host = output_boxarray_.cpu() + \
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

                float* det_output_host = det_output_predicts_[stride].cpu() + ibatch * det_output_buffer_size[stride];

                int num_grid_x = (int)(model_info->m_preProcCfg.network_input_width_ / strides[stride]);
                int num_grid_y = (int)(model_info->m_preProcCfg.network_input_height_ / strides[stride]);
                int total_grid = num_grid_x * num_grid_y;

                for (int anchor = 0; anchor < 3; anchor++)
                {
                    const float anchor_w = netAnchors[stride][anchor * 2];
                    const float anchor_h = netAnchors[stride][anchor * 2 + 1];

                    for (int i = 0; i < total_grid; i++)
                    {
                        int obj_index = i + anchor * (det_info_len_i + det_info_len_kpt) * total_grid + det_bbox_len * total_grid;
                        float objness = sigmoid_x(det_output_host[obj_index]);
                        if (objness < model_info->m_postProcCfg.confidence_threshold_)
                            continue;

                        int label = 0;
                        int class_index = i + anchor * (det_info_len_i + det_info_len_kpt) * total_grid + (det_bbox_len + det_cls_len) * total_grid;
                        float prob = sigmoid_x(det_output_host[class_index]);

                        float confidence = objness * prob;
                        if (confidence < model_info->m_postProcCfg.confidence_threshold_)
                            continue;

                        int grid_y = i / num_grid_x;
                        int grid_x = i % num_grid_x;

                        int x_index = i + anchor * (det_info_len_i + det_info_len_kpt) * total_grid + 0 * total_grid;
                        float x_data = sigmoid_x(det_output_host[x_index]);
                        x_data = (grid_x - 0.5f + 2.0f * x_data) * strides[stride];

                        int y_index = i + anchor * (det_info_len_i + det_info_len_kpt) * total_grid + 1 * total_grid;
                        float y_data = sigmoid_x(det_output_host[y_index]);
                        y_data = (grid_y - 0.5f + 2.0f * y_data) * strides[stride];

                        int w_index = i + anchor * det_info_len_i * total_grid + 2 * total_grid;
                        float w_data = 2.0f * sigmoid_x(det_output_host[w_index]);
                        w_data = w_data * w_data * anchor_w;

                        int h_index = i + anchor * det_info_len_i * total_grid + 3 * total_grid;
                        float h_data = 2.0f * sigmoid_x(det_output_host[h_index]);
                        h_data = h_data * h_data * anchor_h;

                        float x_center = x_data;
                        float y_center = y_data;
                        float width    = w_data;
                        float height   = h_data;

                        float left   = x_center - width * 0.5f;
                        float top    = y_center - height * 0.5f;
                        float right  = x_center + width * 0.5f;
                        float bottom = y_center + height * 0.5f;

                        left    = affine_matrix_host[0] * left  + affine_matrix_host[1] * top    + affine_matrix_host[2];
                        top     = affine_matrix_host[3] * left  + affine_matrix_host[4] * top    + affine_matrix_host[5];
                        right   = affine_matrix_host[0] * right + affine_matrix_host[1] * bottom + affine_matrix_host[2];
                        bottom  = affine_matrix_host[3] * right + affine_matrix_host[4] * bottom + affine_matrix_host[5];

                        *boxarray_host += 1;
                        int index = *boxarray_host;

                        float* pout_item = boxarray_host + 1 + index * model_info->m_postProcCfg.NUM_BOX_ELEMENT;
                        *pout_item++ = left;
                        *pout_item++ = top;
                        *pout_item++ = right;
                        *pout_item++ = bottom;
                        *pout_item++ = confidence;
                        *pout_item++ = label;
                        *pout_item++ = 1; // 1 = keep, 0 = ignore

                        for (int kpt_idx = 0; kpt_idx < model_info->m_postProcCfg.pose_num_; kpt_idx++)
                        {
                            int key_point_x_index    = i + anchor * (det_info_len_i + det_info_len_kpt) * total_grid + (6 + kpt_idx * 3) * total_grid;
                            int key_point_y_index    = i + anchor * (det_info_len_i + det_info_len_kpt) * total_grid + (7 + kpt_idx * 3) * total_grid;
                            int key_point_conf_index = i + anchor * (det_info_len_i + det_info_len_kpt) * total_grid + (8 + kpt_idx * 3) * total_grid;

                            float key_point_x = (grid_x - 0.5f + 2.0f * (det_output_host[key_point_x_index])) * strides[stride];
                            float key_point_y = (grid_y - 0.5f + 2.0f * (det_output_host[key_point_y_index])) * strides[stride];
                            float key_point_conf = sigmoid_x(det_output_host[key_point_conf_index]);

                            *pout_item++ = affine_matrix_host[0] * key_point_x  + affine_matrix_host[1] * key_point_y + affine_matrix_host[2];
                            *pout_item++ = affine_matrix_host[3] * key_point_x  + affine_matrix_host[4] * key_point_y + affine_matrix_host[5];
                            *pout_item++ = key_point_conf;
                        }
                    }
                }
            }

            fast_nms_cpu(
                boxarray_host,
                model_info->m_postProcCfg.nms_threshold_,
                model_info->m_postProcCfg.MAX_IMAGE_BOXES,
                model_info->m_postProcCfg.NUM_BOX_ELEMENT
            );
        }

        BatchPoseBoxArray yolov7_pose_detector::parser_box(int num_image)
        {
            BatchPoseBoxArray arrout(num_image);
            for (int ib = 0; ib < num_image; ib++)
            {
                float* parray = output_boxarray_.cpu() + ib * (32 + model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT);
                int count = std::min(model_info->m_postProcCfg.MAX_IMAGE_BOXES, (int)*parray);
                PoseBoxArray& output = arrout[ib];
                output.reserve(count);
                for (int i = 0; i < count; i++)
                {
                    float* pbox = parray + 1 + i * model_info->m_postProcCfg.NUM_BOX_ELEMENT;
                    int label = pbox[5];
                    int keepflag = pbox[6];
                    if (keepflag == 1)
                    {
                        PoseBox result_object_box(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label);
                        result_object_box.pose = std::make_shared<InstancePose>();
                        for (int pindex = 7; pindex < model_info->m_postProcCfg.NUM_BOX_ELEMENT; pindex += 3)
                        {
                            result_object_box.pose->pose_data.push_back({pbox[pindex], pbox[pindex + 1], pbox[pindex + 2]});
                        }

                        output.emplace_back(result_object_box);
                    }   
                }
            }

            return arrout;
        }

        PoseBoxArray yolov7_pose_detector::forward(const Image& image)
        {
            auto output = forwards({image});
            if (output.empty())
            {
                return {};
            }
            
            return output[0];
        }

        BatchPoseBoxArray yolov7_pose_detector::forwards(const std::vector<Image>& images)
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
                    if (!model_->set_network_dims(0, input_dims))
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

            adjust_memory(model_info->m_preProcCfg.infer_batch_size);

            for (int i = 0; i < num_image; i++)
            {
                preprocess_gpu(i, images[i], preprocess_buffers_[i], affine_matrixs[i], cu_stream);
            }
            
            // float* bbox_output_device = bbox_predict_.gpu();
            // std::vector<void*> bindings{input_buffer_.gpu(), bbox_output_device};
            std::vector<void*> bindings{input_buffer_.gpu()};
            for (int i = 0; i < num_stages; i++)
            {
                float* det_output_device = det_output_predicts_[i].gpu();
                bindings.push_back(det_output_device);
            }

            if (!model_->forward(bindings, cu_stream))
            {
                INFO("Failed to tensorRT forward.");
                return {};
            }
            
            for (int ib = 0; ib < num_image; ib++)
            {
                postprocess_gpu(ib, cu_stream);
            }
            
            // checkRuntime(
            //     cudaMemcpyAsync(
            //         output_boxarray_.cpu(), output_boxarray_.gpu(),
            //         output_boxarray_.gpu_bytes(), cudaMemcpyDeviceToHost, cu_stream
            //     )
            // );
            // checkRuntime(cudaStreamSynchronize(cu_stream));

            return parser_box(num_image);
        }
    }
}