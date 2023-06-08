#include "yolox_detector.h"

namespace tensorrt_infer
{
    namespace yolox_infer
    {
        void yolox_detector::initParameters(
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
            //传入参数的配置
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

            model_info->m_preProcCfg.normalize_ = Norm::alpha_beta(1.0f, 0.0f, ChannelType::RGB);

            for (int i = 0; i < num_stage; i++)
            {
                auto det_cls_output_dim = this->model_->get_network_dims(i * num_stage + 0 + 1);
                auto det_bbox_output_dim = this->model_->get_network_dims(i * num_stage + 1 + 1);
                auto det_obj_output_dim = this->model_->get_network_dims(i * num_stage + 2 + 1);

                det_output_cls_buffer_size[i] = det_cls_output_dim[1] * det_cls_output_dim[2] * det_cls_output_dim[3];
                det_output_obj_buffer_size[i] = det_obj_output_dim[1] * det_obj_output_dim[2] * det_obj_output_dim[3];
                det_output_bbox_buffer_size[i] = det_bbox_output_dim[1] * det_bbox_output_dim[2] * det_bbox_output_dim[3];
            }

            model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT = \
                model_info->m_postProcCfg.MAX_IMAGE_BOXES * model_info->m_postProcCfg.NUM_BOX_ELEMENT;

            CHECK(cudaStreamCreate(&cu_stream)); // 创建cuda流
        }

        yolox_detector::~yolox_detector()
        {
            CHECK(cudaStreamDestroy(cu_stream)); // 销毁cuda流
        }

        void yolox_detector::adjust_memory(int batch_size)
        {
            // 申请模型输入和模型输出所用到的内存
            input_buffer_.gpu(batch_size * model_info->m_preProcCfg.network_input_numel); // 申请batch个模型输入的gpu内存
            for (int i = 0; i < num_stage; i++)
            {
                det_cls_predicts_[i].gpu(batch_size * det_output_cls_buffer_size[i]);
                det_obj_predicts_[i].gpu(batch_size * det_output_obj_buffer_size[i]);
                det_bbox_predicts_[i].gpu(batch_size * det_output_bbox_buffer_size[i]);
            }

            output_boxarray_.gpu(batch_size * (32 + model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT));
            output_boxarray_.cpu(batch_size * (32 + model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT));

            if ((int)preprocess_buffers_.size() < batch_size)
            {
                for (int i = preprocess_buffers_.size(); i < batch_size; i++)
                {
                    preprocess_buffers_.push_back(std::make_shared<Memory<unsigned char>>()); // 添加batch个Memory对象
                }
            }

            // 申请batch size个仿射矩阵，由于也是动态batch指定，所以直接在这里写了
            if ((int)affine_matrixs.size() < batch_size)
            {
                for (int i = affine_matrixs.size(); i < batch_size; i++)
                {
                    affine_matrixs.push_back(AffineMatrix()); // 添加batch个AffineMatrix对象
                }
            }
        }

        void yolox_detector::preprocess_gpu(
            int ibatch, const Image& image,
            std::shared_ptr<Memory<unsigned char>> preprocess_buffer,
            AffineMatrix& affine, cudaStream_t stream_)
        {
            if (image.channels != model_info->m_preProcCfg.network_input_channels_)
            {
                INFO("Warning : Number of channels wanted differs from number of channels in the actual image \n");
                exit(-1);
            }

            affine.compute(
                std::make_tuple(image.width, image.height),
                std::make_tuple(model_info->m_preProcCfg.network_input_width_, model_info->m_preProcCfg.network_input_height_),
                DetectorType::X
            );

            float* input_device = input_buffer_.gpu() + ibatch * model_info->m_preProcCfg.network_input_numel;
            size_t size_image = image.width * image.height * image.channels;
            size_t size_matrix = upbound(sizeof(affine.d2i), 32);

            uint8_t* gpu_workspace = preprocess_buffer->gpu(size_matrix + size_image); // 这里把仿射矩阵 + image_size放在一起申请gpu内存
            float* affine_matrix_device = (float*)gpu_workspace;
            uint8_t* image_device = gpu_workspace + size_matrix; // 这里只取放射变换矩阵的gpu内存

            // 同上，只不过申请的是vpu内存
            uint8_t* cpu_workspace = preprocess_buffer->cpu(size_matrix + size_image);
            float* affine_matrix_host = (float*)cpu_workspace;
            uint8_t* image_host = cpu_workspace + size_matrix;

            // 赋值这一步并不是多余的，这个是从分页内存到固定页内存的数据传输，可以加速向gpu内存进行数据传输
            memcpy(image_host, image.bgrptr, size_image); // 给图片内存赋值
            memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i)); // 给仿射变换矩阵内存赋值

            // 从cpu-->gpu，其中image_host也可以替换为image.bgrptr然后删除上面几行，但是会慢个0.02ms左右
            checkRuntime(
                cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_)
            );

            checkRuntime(
                cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i),
                                cudaMemcpyHostToDevice, stream_)
            ); // 仿射变换矩阵 cpu内存上传到gpu内存

            // 执行resize + fill[114]
            warp_affine_bilinear_and_normalize_plane(
                image_device, image.width * image.channels, image.width,
                image.height, input_device,
                model_info->m_preProcCfg.network_input_width_, model_info->m_preProcCfg.network_input_height_,
                affine_matrix_device, const_value,
                model_info->m_preProcCfg.normalize_, stream_
            );
        }

        void yolox_detector::postprocess_gpu(int ibatch, cudaStream_t stream_)
        {
            float* boxarray_device = output_boxarray_.gpu() + ibatch * (32 + model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT);
            float* affine_matrix_device = (float*)preprocess_buffers_[ibatch]->gpu();
            checkRuntime(cudaMemsetAsync(boxarray_device, 0, sizeof(int), stream_));

            for (int stride = 0; stride < num_stage; stride++)
            {
                float* det_cls_output_device = det_cls_predicts_[stride].gpu() + ibatch * det_output_cls_buffer_size[stride];
                float* det_bbox_output_device = det_bbox_predicts_[stride].gpu() + ibatch * det_output_bbox_buffer_size[stride];
                float* det_obj_output_device = det_obj_predicts_[stride].gpu() + ibatch * det_output_obj_buffer_size[stride];

                decode_kernel_yolox_invoker(
                    det_cls_output_device, det_obj_output_device, det_bbox_output_device,
                    model_info->m_preProcCfg.infer_batch_size,
                    det_obj_len, det_bbox_len, det_cls_len,
                    model_info->m_postProcCfg.MAX_IMAGE_BOXES, model_info->m_postProcCfg.NUM_BOX_ELEMENT,
                    model_info->m_preProcCfg.network_input_height_, model_info->m_preProcCfg.network_input_width_,
                    strides[stride],
                    model_info->m_postProcCfg.confidence_threshold_, model_info->m_postProcCfg.nms_threshold_,
                    affine_matrix_device, boxarray_device, stream_
                );
            }

            nms_kernel_invoker(
                boxarray_device, model_info->m_postProcCfg.nms_threshold_, model_info->m_postProcCfg.MAX_IMAGE_BOXES,
                model_info->m_postProcCfg.NUM_BOX_ELEMENT, stream_
            );
        }

        BatchBoxArray yolox_detector::parser_box(int num_image)
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

        BoxArray yolox_detector::forward(const Image& image)
        {
            auto output = forwards({image});
            if (output.empty())
            {
                return {};
            }
            
            return output[0];
        }

        BatchBoxArray yolox_detector::forwards(const std::vector<Image>& images)
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
            for (int i = 0; i < num_stage; i++)
            {
                float* cls_output_device = det_cls_predicts_[i].gpu();
                float* bbox_output_device = det_bbox_predicts_[i].gpu();
                float* obj_output_device = det_obj_predicts_[i].gpu();
                bindings.push_back(cls_output_device);
                bindings.push_back(bbox_output_device);
                bindings.push_back(obj_output_device);
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

            checkRuntime(
                cudaMemcpyAsync(
                    output_boxarray_.cpu(), output_boxarray_.gpu(),
                    output_boxarray_.gpu_bytes(), cudaMemcpyDeviceToHost, cu_stream
                )
            );
            checkRuntime(cudaStreamSynchronize(cu_stream));

            return parser_box(num_image);
        }
    }
}