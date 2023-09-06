#include "detr_detector.h"

namespace tensorrt_infer
{
    namespace detr_infer
    {
        void detr_detector::initParameters(const std::string &engine_file, float score_thr)
        {
            if (!file_exist(engine_file))
            {
                INFO("Error: engine_file is not exist!!!");
                exit(0);
            }

            this->model_info = std::make_shared<ModelInfo>();

            model_info->m_modelPath = engine_file;
            model_info->m_postProcCfg.confidence_threshold_ = score_thr;

            this->model_ = trt::infer::load(engine_file);
            this->model_->print();

            auto input_dim = this->model_->get_network_dims(0);
            model_info->m_preProcCfg.infer_batch_size = input_dim[0];
            model_info->m_preProcCfg.network_input_channels_ = input_dim[1];
            model_info->m_preProcCfg.network_input_height_ = input_dim[2];
            model_info->m_preProcCfg.network_input_width_ = input_dim[3];
            model_info->m_preProcCfg.network_input_numel = input_dim[1] * input_dim[2] * input_dim[3];
            model_info->m_preProcCfg.isdynamic_model_ = this->model_->has_dynamic_dim();

            model_info->m_preProcCfg.normalize_ = Norm::mean_std(mean_rgb, std_rgb, 1.0f, ChannelType::RGB);

            model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT = model_info->m_postProcCfg.MAX_IMAGE_BOXES * model_info->m_postProcCfg.NUM_BOX_ELEMENT;

            CHECK(cudaStreamCreate(&cu_stream));
        }

        detr_detector::~detr_detector()
        {
            CHECK(cudaStreamDestroy(cu_stream)); // 销毁cuda流
        }

        void detr_detector::adjust_memory(int batch_size)
        {
            input_buffer_.gpu(batch_size * model_info->m_preProcCfg.network_input_numel);
            input_buffer_.cpu(batch_size * model_info->m_preProcCfg.network_input_numel);
            det_bboxes_predicts_.gpu(batch_size * NUM_QUERY * 5);
            det_bboxes_predicts_.cpu(batch_size * NUM_QUERY * 5);
            det_labels_predicts_.gpu(batch_size * NUM_QUERY);
            det_labels_predicts_.cpu(batch_size * NUM_QUERY);

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

        void detr_detector::preprocess_gpu(int ibatch, const Image& image,
                                std::shared_ptr<Memory<unsigned char>> preprocess_buffer, AffineMatrix& affine,
                                cudaStream_t stream_)
        {
            if (image.channels != model_info->m_preProcCfg.network_input_channels_)
            {
                INFO("Warning : Number of channels wanted differs from number of channels in the actual image \n");
                exit(-1);
            }

            size_t size_image = image.width * image.height * image.channels;
            float *input_device = input_buffer_.gpu() + ibatch * model_info->m_preProcCfg.network_input_numel;

            uint8_t *image_device = preprocess_buffer->gpu(size_image);
            uint8_t *image_host = preprocess_buffer->cpu(size_image);

            memcpy(image_host, image.bgrptr, size_image);
            checkRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));

            resize_bilinear_and_normalize(
                image_device,
                image.width * image.channels, image.width, image.height,
                input_device,
                model_info->m_preProcCfg.network_input_width_,
                model_info->m_preProcCfg.network_input_height_,
                model_info->m_preProcCfg.normalize_, stream_
            );
        }

        void detr_detector::postprocess_gpu(int ibatch, cudaStream_t stream_)
        {
            float *boxarray_device = output_boxarray_.gpu() + ibatch * (32 + model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT);
            float *image_based_bbox_device = det_bboxes_predicts_.gpu() + ibatch * NUM_QUERY * 5;
            float *image_based_labels_device = det_labels_predicts_.gpu() + ibatch * NUM_QUERY;

            checkRuntime(cudaMemsetAsync(boxarray_device, 0, sizeof(int), stream_));
            decode_detect_detr_kernel_invoker(image_based_bbox_device, image_based_labels_device,
                                            model_info->m_preProcCfg.network_input_height_,
                                            model_info->m_preProcCfg.network_input_width_,
                                            NUM_QUERY,
                                            model_info->m_postProcCfg.confidence_threshold_,
                                            boxarray_device,
                                            model_info->m_postProcCfg.MAX_IMAGE_BOXES,
                                            model_info->m_postProcCfg.NUM_BOX_ELEMENT, stream_);
        }

        BatchBoxArray detr_detector::parser_box(const std::vector<Image> &images)
        {
            int num_image = images.size();
            BatchBoxArray arrout(num_image);
            for (int ib = 0; ib < num_image; ib++)
            {
                int input_h = model_info->m_preProcCfg.network_input_height_;
                int input_w = model_info->m_preProcCfg.network_input_width_;
                float ratio_h = model_info->m_preProcCfg.network_input_height_ * 1.0f / images[ib].height;
                float ratio_w = model_info->m_preProcCfg.network_input_width_ * 1.0f / images[ib].width;

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
                        int left   = std::min(std::max(0, (int)(pbox[0] / ratio_w)), images[ib].width  - 1);
                        int top    = std::min(std::max(0, (int)(pbox[1] / ratio_h)), images[ib].height - 1);
                        int right  = std::min(std::max(0, (int)(pbox[2] / ratio_w)), images[ib].width  - 1);
                        int bottom = std::min(std::max(0, (int)(pbox[3] / ratio_h)), images[ib].height - 1);

                        output.emplace_back(left, top, right, bottom, pbox[4], label);
                    }   
                }               
            }

            return arrout;
        }

        BoxArray detr_detector::forward(const Image &image)
        {
            auto output = forwards({image});
            if (output.empty())
                return {};
            return output[0];
        }

        BatchBoxArray detr_detector::forwards(const std::vector<Image> &images)
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

            float* input_buffer_device = input_buffer_.gpu();
            float* bbox_output_device = det_bboxes_predicts_.gpu();
            float* label_output_device = det_labels_predicts_.gpu();
            std::vector<void*> bindings{input_buffer_device, bbox_output_device, label_output_device};

            if (!this->model_->forward(bindings, cu_stream))
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
            checkRuntime(cudaStreamSynchronize(cu_stream));

            return parser_box(images);
        }
    }
}