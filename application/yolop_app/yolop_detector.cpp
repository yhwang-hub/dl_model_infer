#include "yolop_detector.h"

namespace tensorrt_infer
{
    namespace yolop_infer
    {
        void yolop_detector::initParameters(
                const std::string& engine_file,
                const DetectorType type,
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
            model_info->detectortype_ = type;

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

            std::vector<int> output_box_dim, output_seg_dim, output_lane_dim;

            if(model_info->detectortype_ == DetectorType::YOLOPV1)
            {
                output_box_dim = this->model_->get_network_dims(1);
                output_seg_dim = this->model_->get_network_dims(2);
                output_lane_dim = this->model_->get_network_dims(3);
            }
            else if(model_info->detectortype_ == DetectorType::YOLOPV2)
            {
                output_box_dim = this->model_->get_network_dims(3);
                output_seg_dim = this->model_->get_network_dims(1);
                output_lane_dim = this->model_->get_network_dims(2);
            }
            model_info->m_postProcCfg.bbox_head_dims_ = output_box_dim;
            model_info->m_postProcCfg.bbox_head_dims_output_numel_ = output_box_dim[1] * output_box_dim[2];
            
            model_info->m_postProcCfg.seg_head_dims_ = output_seg_dim;
            model_info->m_postProcCfg.seg_head_dims_output_numel_  = output_seg_dim[1] * output_seg_dim[2] * output_seg_dim[3];

            model_info->m_postProcCfg.lane_head_dims_ = output_lane_dim;
            model_info->m_postProcCfg.lane_head_dims_output_numel_ = output_lane_dim[1] * output_lane_dim[2] * output_lane_dim[3];

            if(model_info->detectortype_ == DetectorType::YOLOPV1)
            {
                model_info->m_postProcCfg.num_classes_ = 1;
            }
            else if(model_info->detectortype_ == DetectorType::YOLOPV2)
            {
                model_info->m_postProcCfg.num_classes_ = 80;
            }

            model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT =\
                model_info->m_postProcCfg.MAX_IMAGE_BOXES * model_info->m_postProcCfg.NUM_BOX_ELEMENT;

            CHECK(cudaStreamCreate(&cu_stream));
        }

        yolop_detector::~yolop_detector()
        {
            CHECK(cudaStreamDestroy(cu_stream));
        }

        void yolop_detector::adjust_memory(int batch_size, const std::vector<Image>& images)
        {
            if ((int)drive_lane_mat_.size() < batch_size)
            {
                for (int i = drive_lane_mat_.size(); i < batch_size; ++i)
                {
                    drive_lane_mat_.push_back(std::make_shared<Memory<uint8_t>>());
                }
            }
            if ((int)drive_mask_mat_.size() < batch_size)
            {
                for (int i = drive_mask_mat_.size(); i < batch_size; ++i)
                {
                    drive_mask_mat_.push_back(std::make_shared<Memory<uint8_t>>());
                }
            }
            if ((int)lane_mask_mat_.size() < batch_size)
            {
                for (int i = lane_mask_mat_.size(); i < batch_size; ++i)
                {
                    lane_mask_mat_.push_back(std::make_shared<Memory<uint8_t>>());
                }
            }

            input_buffer_.gpu(batch_size * model_info->m_preProcCfg.network_input_numel);
            bbox_predicts_.gpu(batch_size * model_info->m_postProcCfg.bbox_head_dims_output_numel_);
            drive_area_seg_predicts_.gpu(batch_size * model_info->m_postProcCfg.seg_head_dims_output_numel_);
            lane_seg_predicts_.gpu(batch_size * model_info->m_postProcCfg.lane_head_dims_output_numel_);

            output_boxarray_.gpu(batch_size * (32 + model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT));
            output_boxarray_.cpu(batch_size * (32 + model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT));

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

        void yolop_detector::preprocess_gpu(
                int ibatch, const Image& image,
                std::shared_ptr<Memory<unsigned char>> preprocess_buffer, AffineMatrix& affine,
                cudaStream_t stream_)
        {
            if (image.channels != model_info->m_preProcCfg.network_input_channels_)
            {
                INFO("Warning : Number of channels wanted differs from number of channels in the actual image \n");
                exit(-1);
            }

            size_t size_image = image.width * image.height * image.channels;

            affine.compute(
                std::make_tuple(image.width, image.height),
                std::make_tuple(model_info->m_preProcCfg.network_input_width_, model_info->m_preProcCfg.network_input_height_),
                model_info->detectortype_
            );

            uint8_t* drive_lane_mat_device = drive_lane_mat_[ibatch]->gpu(size_image);
            uint8_t* drive_mask_mat_device = drive_mask_mat_[ibatch]->gpu(size_image / image.channels);
            uint8_t* lane_mask_mat_device  = lane_mask_mat_[ibatch]->gpu(size_image / image.channels);

            checkRuntime(cudaMemcpyAsync(drive_lane_mat_device, image.bgrptr, size_image, cudaMemcpyHostToDevice, cu_stream));
            checkRuntime(cudaMemsetAsync(drive_mask_mat_device, 0, sizeof(drive_mask_mat_device), cu_stream));
            checkRuntime(cudaMemsetAsync(lane_mask_mat_device, 0, sizeof(lane_mask_mat_device), cu_stream));

            float* input_device = input_buffer_.gpu() + ibatch * model_info->m_preProcCfg.network_input_numel;

            size_t size_matrix = upbound(sizeof(affine.d2i), 32);
            uint8_t* gpu_workspace = preprocess_buffer->gpu(size_matrix + size_image);
            float* affine_matrix_device = (float*)gpu_workspace;
            uint8_t* image_device = gpu_workspace + size_matrix;

            uint8_t* cpu_workspace = preprocess_buffer->cpu(size_matrix + size_image);
            float* affine_matrix_host = (float*)cpu_workspace;
            uint8_t* image_host = cpu_workspace + size_matrix;

            memcpy(image_host, image.bgrptr, size_image);
            memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));

            checkRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));
            checkRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i), cudaMemcpyHostToDevice, stream_));

            warp_affine_bilinear_and_normalize_plane(
                image_device, image.width * image.channels,
                image.width, image.height, input_device,
                model_info->m_preProcCfg.network_input_width_, model_info->m_preProcCfg.network_input_height_,
                affine_matrix_device, const_value,
                model_info->m_preProcCfg.normalize_, stream_
            );
        }

        void yolop_detector::postprocess_gpu(int ibatch, const Image& image, cudaStream_t stream_)
        {
            int image_width_    = image.width;
            int image_height_   = image.height;
            int image_channels_ = image.channels;
            int size_image = image_height_ * image_width_ * image_channels_;

            uint8_t* drive_lane_mat_device  = drive_lane_mat_[ibatch]->gpu();
            uint8_t* drive_mask_mat_device  = drive_mask_mat_[ibatch]->gpu();
            uint8_t* lane_mask_mat_device   = lane_mask_mat_[ibatch]->gpu();

            float* boxarray_device = output_boxarray_.gpu() + ibatch * (32 + model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT);
            float* affine_matrix_device = (float*)preprocess_buffers_[ibatch]->gpu();
            float* image_based_bbox_output = bbox_predicts_.gpu() + ibatch * model_info->m_postProcCfg.bbox_head_dims_output_numel_;

            float* drive_area_seg_device = drive_area_seg_predicts_.gpu() + ibatch * (model_info->m_postProcCfg.seg_head_dims_output_numel_);
            float* lane_seg_device       = lane_seg_predicts_.gpu()       + ibatch * (model_info->m_postProcCfg.lane_head_dims_output_numel_);

            checkRuntime(cudaMemsetAsync(boxarray_device, 0, sizeof(int), stream_));

            decode_detect_kernel_invoker(
                image_based_bbox_output,
                model_info->m_postProcCfg.bbox_head_dims_[1],
                model_info->m_postProcCfg.num_classes_,
                model_info->m_postProcCfg.bbox_head_dims_[2],
                model_info->m_postProcCfg.confidence_threshold_,
                affine_matrix_device, boxarray_device,
                model_info->m_postProcCfg.MAX_IMAGE_BOXES,
                model_info->m_postProcCfg.NUM_BOX_ELEMENT,
                stream_
            );

            nms_kernel_invoker(
                boxarray_device,
                model_info->m_postProcCfg.nms_threshold_,
                model_info->m_postProcCfg.MAX_IMAGE_BOXES,
                model_info->m_postProcCfg.NUM_BOX_ELEMENT,
                stream_
            );

            size_t size_matrix = upbound(sizeof(affine_matrixs[ibatch].i2d), 32);
            uint8_t* gpu_workspace_ = preprocess_buffers_[ibatch]->gpu(size_matrix + size_image);
            float* affine_matrix_device_ = (float*)gpu_workspace_;
            uint8_t* cpu_workspace_ = preprocess_buffers_[ibatch]->cpu(size_matrix + size_image);
            float* affine_matrix_host_ = (float*)cpu_workspace_;
            memcpy(affine_matrix_host_, affine_matrixs[ibatch].i2d, sizeof(affine_matrixs[ibatch].i2d));
            checkRuntime(cudaMemcpyAsync(affine_matrix_device_, affine_matrix_host_, sizeof(affine_matrixs[ibatch].i2d), cudaMemcpyHostToDevice, stream_));

            decode_yolop_mask_kernel_invoker(
                drive_area_seg_device, lane_seg_device,
                drive_lane_mat_device, drive_mask_mat_device, lane_mask_mat_device,
                model_info->m_preProcCfg.network_input_width_,
                model_info->m_preProcCfg.network_input_height_,
                affine_matrix_device, image_width_, image_height_,
                model_info->detectortype_, stream_
            );
        }

        BatchPTMM yolop_detector::parser_box(int num_image, const std::vector<Image>& images)
        {
            BatchPTMM arrout;
            arrout.resize(num_image);
            for (int ib = 0; ib < num_image; ib++)
            {
                float* parray = output_boxarray_.cpu() + ib * (32 + model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT);
                int count = std::min(model_info->m_postProcCfg.MAX_IMAGE_BOXES, (int)*parray);
                BoxArray image_based_boxes;
                image_based_boxes.reserve(count);

                for (int i = 0; i < count; i++)
                {
                    float* pbox = parray + 1 + i * model_info->m_postProcCfg.NUM_BOX_ELEMENT;
                    int label = pbox[5];
                    int keepflag = pbox[6];
                    if (keepflag == 1)
                    {
                        image_based_boxes.emplace_back(
                            pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label
                        );
                    }
                }

                int image_width_  = images[ib].width;
                int image_height_ = images[ib].height;
                int image_channel_= images[ib].channels;
                cv::Mat drive_lane_mat(cv::Size(image_width_, image_height_), CV_8UC3);
                cv::Mat drive_mask_mat(cv::Size(image_width_, image_height_), CV_8UC1);
                cv::Mat lane_mask_mat(cv::Size(image_width_,  image_height_), CV_8UC1);
                uint8_t* drive_lane_mat_device = drive_lane_mat_[ib]->gpu();
                uint8_t* drive_mask_mat_device = drive_mask_mat_[ib]->gpu();
                uint8_t* lane_mask_mat_device  = lane_mask_mat_[ib]->gpu();

                checkRuntime(cudaMemcpyAsync(drive_lane_mat.data, drive_lane_mat_device, image_width_ * image_height_ * image_channel_, cudaMemcpyDeviceToHost, cu_stream));
                checkRuntime(cudaMemcpyAsync(drive_mask_mat.data, drive_mask_mat_device, image_width_ * image_height_, cudaMemcpyDeviceToHost, cu_stream));
                checkRuntime(cudaMemcpyAsync(lane_mask_mat.data, lane_mask_mat_device,   image_width_ * image_height_, cudaMemcpyDeviceToHost, cu_stream));

                arrout[ib] = std::make_tuple(image_based_boxes, drive_lane_mat, drive_mask_mat, lane_mask_mat);
            }

            checkRuntime(cudaStreamSynchronize(cu_stream));

            return arrout;
        }

        PTMM yolop_detector::forward(const Image& image)
        {
            auto output = forwards({image});
            if (output.empty())
            {
                return {};
            }
            
            return output[0];
        }

        BatchPTMM yolop_detector::forwards(const std::vector<Image>& images)
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

            adjust_memory(model_info->m_preProcCfg.infer_batch_size, images);

            for (int i = 0; i < num_image; i++)
            {
                preprocess_gpu(i, images[i], preprocess_buffers_[i], affine_matrixs[i], cu_stream);
            }

            std::vector<void*> bindings;

            if(model_info->detectortype_ == DetectorType::YOLOPV1)
            {
                bindings = {input_buffer_.gpu(),
                            bbox_predicts_.gpu(),
                            drive_area_seg_predicts_.gpu(),
                            lane_seg_predicts_.gpu()};
            }
            else if (model_info->detectortype_ == DetectorType::YOLOPV2)
            {
                bindings = {input_buffer_.gpu(),
                            drive_area_seg_predicts_.gpu(),
                            lane_seg_predicts_.gpu(),
                            bbox_predicts_.gpu()};
            }

            if (!model_->forward(bindings, cu_stream))
            {
                INFO("Failed to tensorRT forward.");
                return {};
            }

            for (int ib = 0; ib < num_image; ib++)
            {
                postprocess_gpu(ib, images[ib], cu_stream);
            }

            checkRuntime(
                cudaMemcpyAsync(
                    output_boxarray_.cpu(), output_boxarray_.gpu(),
                    output_boxarray_.gpu_bytes(), cudaMemcpyDeviceToHost, cu_stream
                )
            );
            checkRuntime(cudaStreamSynchronize(cu_stream));

            return parser_box(num_image, images);
        }
    }
}