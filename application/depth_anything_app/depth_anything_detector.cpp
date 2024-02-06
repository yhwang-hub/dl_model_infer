#include "depth_anything_detector.h"

namespace tensorrt_infer
{
    namespace depth_anything_infer
    {
        bool depth_anything_detector::initParameters(const std::string& engine_file)
        {
            if (!file_exist(engine_file))
            {
                INFO("Error: engine_file is not exist!!!");
                exit(0);
            }

            this->model_info = std::make_shared<ModelInfo>();
            model_info->m_modelPath = engine_file;

            this->model_ = trt::infer::load(engine_file);
            this->model_->print();

            auto input_dim = this->model_->get_network_dims(0);
            model_info->m_preProcCfg.infer_batch_size = input_dim[0];
            model_info->m_preProcCfg.network_input_channels_ = input_dim[1];
            model_info->m_preProcCfg.network_input_height_ = input_dim[2];
            model_info->m_preProcCfg.network_input_width_ = input_dim[3];
            model_info->m_preProcCfg.network_input_numel = input_dim[1] * input_dim[2] * input_dim[3];
            model_info->m_preProcCfg.isdynamic_model_ = this->model_->has_dynamic_dim();

            model_info->m_preProcCfg.normalize_ = Norm::mean_std(norm_mean, norm_std, 1.0f, ChannelType::RGB);

            auto output_dim = this->model_->get_network_dims(1);
            output_height_ = output_dim[1];
            output_width_  = output_dim[2];
            model_info->m_postProcCfg.depth_head_dims_ = output_dim;
            model_info->m_postProcCfg.depth_head_dims_output_numel_ = output_dim[1] * output_dim[2];

            model_info->m_postProcCfg.num_classes_ = 3;

            CHECK(cudaStreamCreate(&cu_stream));

            return true;
        }

        depth_anything_detector::~depth_anything_detector()
        {
            CHECK(cudaStreamDestroy(cu_stream));
        }

        void depth_anything_detector::adjust_memory(int batch_size)
        {
            input_buffer_.gpu(batch_size * model_info->m_preProcCfg.network_input_numel);
            depth_predict_.gpu(batch_size * model_info->m_postProcCfg.depth_head_dims_output_numel_);
            depth_predict_.cpu(batch_size * model_info->m_postProcCfg.depth_head_dims_output_numel_);

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

        void depth_anything_detector::preprocess_gpu(int ibatch, const Image& image,
                                std::shared_ptr<Memory<unsigned char>> preprocess_buffer, AffineMatrix& affine,
                                cudaStream_t stream_)
        {
            if (image.channels != model_info->m_preProcCfg.network_input_channels_)
            {
                INFO("Warning : Number of channels wanted differs from number of channels in the actual image \n");
                exit(-1);
            }

            affine.compute(std::make_tuple(image.width, image.height),
                        std::make_tuple(model_info->m_preProcCfg.network_input_width_, model_info->m_preProcCfg.network_input_height_),
                        DetectorType::DEPTH_ANYTHING);
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

            resize_bilinear_and_normalize(
                image_device, image.width * image.channels, image.width, image.height,
                input_device,
                model_info->m_preProcCfg.network_input_width_,
                model_info->m_preProcCfg.network_input_height_,
                model_info->m_preProcCfg.normalize_, stream_
            );
        }

        void depth_anything_detector::postprocess_gpu(int ibatch, cudaStream_t stream_)
        {

        }

        std::vector<cv::Mat> depth_anything_detector::parser_depthvalue(int num_image)
        {
            std::vector<cv::Mat> depthout(num_image);
            for(int ib = 0; ib < num_image; ++ib)
            {
                int* pdeptharray = depth_predict_.cpu() + \
                    ib * model_info->m_postProcCfg.depth_head_dims_output_numel_;

                depthout[ib] = cv::Mat(
                    output_height_,
                    output_width_,
                    CV_32F,
                    pdeptharray
                );
            }
            return depthout;
        }

        cv::Mat depth_anything_detector::forward(const Image& image)
        {
            auto output = forwards({image});
            if (output.empty())
            {
                return {};
            }
            
            return output[0];
        }

        std::vector<cv::Mat> depth_anything_detector::forwards(const std::vector<Image>& images)
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

            int* depth_output_device = depth_predict_.gpu();
            std::vector<void*> bindings{input_buffer_.gpu(), depth_output_device};
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
                    depth_predict_.cpu(), depth_predict_.gpu(),
                    depth_predict_.gpu_bytes(), cudaMemcpyDeviceToHost, cu_stream
                )
            );
            checkRuntime(cudaStreamSynchronize(cu_stream));
            
            return parser_depthvalue(num_image);
        }
    }
}