#include "smoke_detector.h"

namespace tensorrt_infer
{
    namespace smoke_det_infer
    {
        bool smoke_detector::onnxToTRTModel(const std::string& modelFile, const std::string& engine_file)
        {
            std::cout << "Building TRT engine."<<std::endl;
            // define builder
            auto builder = (nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));

            // define network
            const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
            auto network = (builder->createNetworkV2(explicitBatch));

            // define onnxparser
            auto parser = (nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
            if (!parser->parseFromFile(modelFile.data(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
            {
                std::cerr << ": failed to parse onnx model file, please check the onnx version and trt support op!"
                        << std::endl;
                exit(-1);
            }

            // define config
            auto networkConfig = builder->createBuilderConfig();
#if defined (__arm64__) || defined (__aarch64__) 
            networkConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
            std::cout << "Enable fp16!" << std::endl;
#endif
            // set max batch size
            builder->setMaxBatchSize(1);
            // set max workspace
            networkConfig->setMaxWorkspaceSize(size_t(1) << 30);

            nvinfer1::ICudaEngine *engine_ = (builder->buildEngineWithConfig(*network, *networkConfig));

            if (engine_ == nullptr)
            {
                std::cerr << ": engine init null!" << std::endl;
                exit(-1);
            }

            // serialize the engine, then close everything down
            auto trtModelStream = (engine_->serialize());
            std::fstream trtOut(engine_file, std::ifstream::out);
            if (!trtOut.is_open())
            {
                std::cout << "Can't store trt cache.\n";
                exit(-1);
            }

            trtOut.write((char*)trtModelStream->data(), trtModelStream->size());
            trtOut.close();
            trtModelStream->destroy();

            engine_->destroy();
            networkConfig->destroy();
            parser->destroy();
            network->destroy();
            builder->destroy();

            return true;
        }

        void smoke_detector::initParameters(const std::string& modelFile, float score_thr)
        {
            std::string engine_file = modelFile + ".trt";
            if (!file_exist(engine_file))
            {
                INFO("engine_file is not exist, start building......");
                if (onnxToTRTModel(modelFile, engine_file))
                {
                    INFO("end build......");
                }
                else
                {
                    INFO("Error: engine_file builded failed!!!");
                    exit(0);
                }
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

            bbox_preds_buffer_size_ = topk * 8;
            topk_scores_buffer_size_ = topk;
            topk_indices_buffer_size_ = topk * 1;

            model_info->m_postProcCfg.IMAGE_MAX_CUBES_ADD_ELEMENT =\
                model_info->m_postProcCfg.MAX_IMAGE_CUBES * model_info->m_postProcCfg.NUM_CUBE_ELEMENT;

            base_depth = {28.01f, 16.32f};
            base_dims.resize(3);  //pedestrian, cyclist, car
            base_dims[0].x = 0.88f;
            base_dims[0].y = 1.73f;
            base_dims[0].z = 0.67f;
            base_dims[1].x = 1.78f;
            base_dims[1].y = 1.70f;
            base_dims[1].z = 0.58f;
            base_dims[2].x = 3.88f;
            base_dims[2].y = 1.63f;
            base_dims[2].z = 1.53f;


            CHECK(cudaStreamCreate(&cu_stream));
        }

        smoke_detector::~smoke_detector()
        {
            CHECK(cudaStreamDestroy(cu_stream));
        }

        void smoke_detector::adjust_memory(int batch_size)
        {
            input_buffer_.gpu(batch_size * model_info->m_preProcCfg.network_input_numel);

            det_bbox_predicts_.gpu(batch_size * bbox_preds_buffer_size_);
            det_bbox_predicts_.cpu(batch_size * bbox_preds_buffer_size_);
            det_scores_predicts_.gpu(batch_size * topk_scores_buffer_size_);
            det_scores_predicts_.cpu(batch_size * topk_scores_buffer_size_);
            det_indices_predicts_.gpu(batch_size * topk_indices_buffer_size_);
            det_indices_predicts_.cpu(batch_size * topk_indices_buffer_size_);

            output_cubearray_.gpu(batch_size * (32 + model_info->m_postProcCfg.IMAGE_MAX_CUBES_ADD_ELEMENT));
            output_cubearray_.cpu(batch_size * (32 + model_info->m_postProcCfg.IMAGE_MAX_CUBES_ADD_ELEMENT));

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

        void smoke_detector::preprocess_gpu(
            int ibatch, const Image& image,
            std::shared_ptr<Memory<unsigned char>> preprocess_buffer,
            AffineMatrix& affine,
            cudaStream_t stream_)
        {
            intrinsic_ = (cv::Mat_<float>(3, 3) << 
                                    721.5377, 0.0, 609.5593,
                                    0.0, 721.5377, 172.854,
                                    0.0, 0.0, 1.0);
            intrinsic_.at<float>(0, 0) *= static_cast<float>(model_info->m_preProcCfg.network_input_width_) / image.width;
            intrinsic_.at<float>(0, 2) *= static_cast<float>(model_info->m_preProcCfg.network_input_width_) / image.width;
            intrinsic_.at<float>(1, 1) *= static_cast<float>(model_info->m_preProcCfg.network_input_height_) / image.height;
            intrinsic_.at<float>(1, 2) *= static_cast<float>(model_info->m_preProcCfg.network_input_height_) / image.height;

            if (image.channels != model_info->m_preProcCfg.network_input_channels_)
            {
                INFO("Warning : Number of channels wanted differs from number of channels in the actual image \n");
                exit(-1);
            }

            affine.compute(std::make_tuple(image.width, image.height),
                        std::make_tuple(model_info->m_preProcCfg.network_input_width_, model_info->m_preProcCfg.network_input_height_),
                        DetectorType::SMOKE);
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
                image_device,
                image.width * image.channels, image.width,
                image.height, input_device,
                model_info->m_preProcCfg.network_input_width_,
                model_info->m_preProcCfg.network_input_height_,
                affine_matrix_device,
                const_value,
                model_info->m_preProcCfg.normalize_, stream_
            );
        }

        void smoke_detector::postprocess_gpu(int ibatch, cudaStream_t stream_)
        {
            float* cubearray_host = output_cubearray_.cpu() + \
                            ibatch * (32 + model_info->m_postProcCfg.IMAGE_MAX_CUBES_ADD_ELEMENT);
            float* affine_matrix_host = (float*)preprocess_buffers_[ibatch]->cpu();
            checkRuntime(
                cudaMemcpyAsync(
                    det_bbox_predicts_.cpu(), det_bbox_predicts_.gpu(),
                    det_bbox_predicts_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_
                )
            );
            float* output_bbox_preds = det_bbox_predicts_.cpu() + ibatch * bbox_preds_buffer_size_;

            checkRuntime(
                cudaMemcpyAsync(
                    det_scores_predicts_.cpu(), det_scores_predicts_.gpu(),
                    det_scores_predicts_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_
                )
            );
            float* output_topk_scores = det_scores_predicts_.cpu() + ibatch * topk_scores_buffer_size_;

            checkRuntime(
                cudaMemcpyAsync(
                    det_indices_predicts_.cpu(), det_indices_predicts_.gpu(),
                    det_indices_predicts_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_
                )
            );
            float* output_topk_indices = det_indices_predicts_.cpu() + ibatch * topk_indices_buffer_size_;

            int output_w = model_info->m_preProcCfg.network_input_width_ / 4;
            int output_h = model_info->m_preProcCfg.network_input_height_ / 4;
            for (int i = 0; i < topk; i++)
            {
                float score = output_topk_scores[i];
                if (score < model_info->m_postProcCfg.confidence_threshold_)
                    continue;
                
                // https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/coders/smoke_bbox_coder.py#L52
                int class_id = static_cast<int>(output_topk_indices[i] / output_h / output_w);
                int location = static_cast<int>(output_topk_indices[i]) % (output_h * output_w);
                int img_x = location % output_w;
                int img_y = location / output_w;

                // Depth  bbox_preds_预测的是相对偏移.
                float z = base_depth[0] + output_bbox_preds[8 * i] * base_depth[1];

                // location
                cv::Mat img_point(3, 1, CV_32FC1);
                img_point.at<float>(0) = 4.0f * (static_cast<float>(img_x) + output_bbox_preds[8 * i + 1]);
                img_point.at<float>(1) = 4.0f * (static_cast<float>(img_y) + output_bbox_preds[8 * i + 2]);
                img_point.at<float>(2) = 1.0f;
                cv::Mat cam_point = intrinsic_.inv() * img_point * z;
                float x = cam_point.at<float>(0);
                float y = cam_point.at<float>(1);

                // Dimension
                // std::cout<<"class_id:"<<class_id<<std::endl;
                // std::cout<<"w_offset:"<<bbox_preds_[8*i + 3]<<std::endl;
                float w = base_dims[class_id].x * expf(Sigmoid(output_bbox_preds[8 * i + 3]) - 0.5f);
                float l = base_dims[class_id].y * expf(Sigmoid(output_bbox_preds[8 * i + 4]) - 0.5f);
                float h = base_dims[class_id].z * expf(Sigmoid(output_bbox_preds[8 * i + 5]) - 0.5f);

                // Orientation
                float ori_norm = sqrtf(powf(output_bbox_preds[8 * i + 6], 2.0f) + powf(output_bbox_preds[8 * i + 7], 2.0f));
                output_bbox_preds[8 * i + 6] /= ori_norm;  //sin(alpha)
                output_bbox_preds[8 * i + 7] /= ori_norm;  //cos(alpha)
                float ray = atan(x / (z + 1e-7f));
                float alpha = atan(output_bbox_preds[8 * i + 6] / (output_bbox_preds[8 * i + 7] + 1e-7f));
                if (output_bbox_preds[8 * i + 7] > 0.0f)
                {
                    alpha -= M_PI / 2.0f;
                }
                else
                {
                    alpha += M_PI / 2.0f;
                }
                float angle = alpha + ray;
                if (angle > M_PI)
                {
                    angle -= 2.0f * M_PI;
                }
                else if (angle < -M_PI)
                {
                    angle += 2.0f * M_PI;
                }

                // https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/structures/cam_box3d.py#L117
                //              front z
                //                   /
                //                  /
                //    (x0, y0, z1) + -----------  + (x1, y0, z1)
                //                /|            / |
                //               / |           /  |
                // (x0, y0, z0) + ----------- +   + (x1, y1, z1)
                //              |  /      .   |  /
                //              | / origin    | /
                // (x0, y1, z0) + ----------- + -------> x right
                //              |             (x1, y1, z0)
                //              |
                //              v
                //         down y
                cv::Mat cam_corners = (cv::Mat_<float>(8, 3) << 
                    -w, -l, -h,     // (x0, y0, z0)
                    -w, -l,  h,     // (x0, y0, z1)
                    -w,  l,  h,     // (x0, y1, z1)
                    -w,  l, -h,     // (x0, y1, z0)
                    w, -l, -h,     // (x1, y0, z0)
                    w, -l,  h,     // (x1, y0, z1)
                    w,  l,  h,     // (x1, y1, z1)
                    w,  l, -h);    // (x1, y1, z0)
                cam_corners = 0.5f * cam_corners;
                cv::Mat rotation_y = cv::Mat::eye(3, 3, CV_32FC1);
                rotation_y.at<float>(0, 0) = cosf(angle);
                rotation_y.at<float>(0, 2) = sinf(angle);
                rotation_y.at<float>(2, 0) = -sinf(angle);
                rotation_y.at<float>(2, 2) = cosf(angle);
                // cos, 0, sin
                //   0, 1,   0
                //-sin, 0, cos
                cam_corners = cam_corners * rotation_y.t();
                for (int i = 0; i < 8; ++i)
                {
                    cam_corners.at<float>(i, 0) += x;
                    cam_corners.at<float>(i, 1) += y;
                    cam_corners.at<float>(i, 2) += z;
                }
                cam_corners = cam_corners * intrinsic_.t();

                int index = *cubearray_host;

                float *pout_item = cubearray_host + 1 + index * model_info->m_postProcCfg.NUM_CUBE_ELEMENT;
                std::vector<cv::Point2f> img_corners(8);
                for (int j = 0; j < 8; j++)
                {
                    float point_x = cam_corners.at<float>(j, 0) / cam_corners.at<float>(j, 2);
                    float point_y = cam_corners.at<float>(j, 1) / cam_corners.at<float>(j, 2);
                    img_corners[j].x = point_x;
                    img_corners[j].y = point_y;
                    point_x    = affine_matrix_host[0] * point_x + affine_matrix_host[1] * point_y + affine_matrix_host[2];
                    point_y    = affine_matrix_host[3] * point_x + affine_matrix_host[4] * point_y + affine_matrix_host[5];
                    *pout_item++ = point_x;
                    *pout_item++ = point_y;
                }
                *pout_item++ = score;
                *cubearray_host += 1;
            }
        }

        BatchCubeArray smoke_detector::parser_box(int num_image)
        {
            BatchCubeArray arrout(num_image);
            for (int ib = 0; ib < num_image; ib++)
            {
                float* parray = output_cubearray_.cpu() + ib * (32 + model_info->m_postProcCfg.IMAGE_MAX_CUBES_ADD_ELEMENT);
                int count = std::min(model_info->m_postProcCfg.MAX_IMAGE_CUBES, (int)*parray);
                CubeArray& output = arrout[ib];
                output.reserve(count);

                for (int i = 0; i < count; i++)
                {
                    float* pbox = parray + 1 + i * model_info->m_postProcCfg.NUM_CUBE_ELEMENT;
                    CubeBox cubebox;
                    for (int c = 0; c < 8; c++)
                    {
                        cubebox.cube_point[c].x = pbox[2 * c];
                        cubebox.cube_point[c].y = pbox[2 * c + 1];
                    }
                    cubebox.score = pbox[16];
                    output.push_back(cubebox);
                }
            }
            return arrout;
        }

        CubeArray smoke_detector::forward(const Image& image)
        {
            auto output = forwards({image});
            if (output.empty())
            {
                return {};
            }
            
            return output[0];
        }

        BatchCubeArray smoke_detector::forwards(const std::vector<Image>& images)
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
            bindings.push_back(det_bbox_predicts_.gpu());
            bindings.push_back(det_scores_predicts_.gpu());
            bindings.push_back(det_indices_predicts_.gpu());

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