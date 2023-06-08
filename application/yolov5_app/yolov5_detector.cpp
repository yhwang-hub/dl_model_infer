#include "yolov5_detector.h"

namespace tensorrt_infer
{
    namespace yolov5_infer
    {
        void yolov5_detector::initParameters(const std::string& engine_file, float score_thr, float nms_thr)
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

            // 对输入的图片预处理进行配置，即，yolov8的预处理是除以255， 并且是RGB通道输入
            model_info->m_preProcCfg.normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::RGB);

            // 获取输出的尺寸信息
            auto output_dim = this->model_->get_network_dims(1);
            model_info->m_postProcCfg.bbox_head_dims_ = output_dim;
            model_info->m_postProcCfg.num_classes_ = output_dim[2] - 5;
            model_info->m_postProcCfg.bbox_head_dims_output_numel_ = output_dim[1] * output_dim[2];
            model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT =\
                model_info->m_postProcCfg.MAX_IMAGE_BOXES * model_info->m_postProcCfg.NUM_BOX_ELEMENT;

            CHECK(cudaStreamCreate(&cu_stream)); // 创建cuda流
        }

        yolov5_detector::~yolov5_detector()
        {
            CHECK(cudaStreamDestroy(cu_stream)); // 销毁cuda流
        }

        void yolov5_detector::adjust_memory(int batch_size)
        {
            // 申请模型输入和模型输出所用到的内存
            input_buffer_.gpu(batch_size * model_info->m_preProcCfg.network_input_numel); // 申请batch个模型输入的gpu内存
            bbox_predict_.gpu(batch_size * model_info->m_postProcCfg.bbox_head_dims_output_numel_); // 申请batch个模型输出的gpu内存

            // 申请模型解析成box时需要存储的内容， +32 是因为第一个数要设置为框的数量，防止内存溢出
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

        void yolov5_detector::preprocess_gpu(
            int ibatch, const Image& image,
            std::shared_ptr<Memory<unsigned char>> preprocess_buffer,
            AffineMatrix& affine,
            cudaStream_t stream_
        )
        {
            if (image.channels != model_info->m_preProcCfg.network_input_channels_)
            {
                INFO("Warning : Number of channels wanted differs from number of channels in the actual image \n");
                exit(-1);
            }

            affine.compute(std::make_tuple(image.width, image.height),
                        std::make_tuple(model_info->m_preProcCfg.network_input_width_, model_info->m_preProcCfg.network_input_height_),
                        DetectorType::V5);
            float* input_device = input_buffer_.gpu() + ibatch * model_info->m_preProcCfg.network_input_numel; // 获取当前batch的gpu内存指针
            size_t size_image = image.width * image.height * image.channels;
            size_t size_matrix = upbound(sizeof(affine.d2i), 32); // 向上取整
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
            ); // 图片cpu内存上传到gpu内存
            checkRuntime(
                cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i),
                                cudaMemcpyHostToDevice, stream_)
            ); // 仿射变换矩阵 cpu内存上传到gpu内存

            // 执行resize + fill[114]
            warp_affine_bilinear_and_normalize_plane(
                image_device, image.width * image.channels, image.width,
                image.height, input_device, model_info->m_preProcCfg.network_input_width_,
                model_info->m_preProcCfg.network_input_height_, affine_matrix_device, const_value,
                model_info->m_preProcCfg.normalize_, stream_
            );
        }

        void yolov5_detector::postprocess_gpu(int ibatch, cudaStream_t stream_)
        {
            // boxarray_device:对推理结果进行解析后要存储的gpu指针
            float* boxarray_device = output_boxarray_.gpu() + ibatch * (32 + model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT);
            // affine_matrix_device: 获取仿射变换矩阵+size_image的gpu指针，主要是用来归一化的框尺寸缩放相对于图片尺寸
            float* affine_matrix_device = (float*)preprocess_buffers_[ibatch]->gpu();
            // image_based_bbox_output:推理结果产生的所有预测框的gpu指针
            float* image_based_bbox_output = bbox_predict_.gpu() + ibatch * model_info->m_postProcCfg.bbox_head_dims_output_numel_;

            checkRuntime(cudaMemsetAsync(boxarray_device, 0, sizeof(int), stream_));

            decode_detect_kernel_invoker(
                image_based_bbox_output, model_info->m_postProcCfg.bbox_head_dims_[1], model_info->m_postProcCfg.num_classes_,
                model_info->m_postProcCfg.bbox_head_dims_[2], model_info->m_postProcCfg.confidence_threshold_,
                affine_matrix_device, boxarray_device, model_info->m_postProcCfg.MAX_IMAGE_BOXES,
                model_info->m_postProcCfg.NUM_BOX_ELEMENT, stream_
            );

            nms_kernel_invoker(
                boxarray_device, model_info->m_postProcCfg.nms_threshold_, model_info->m_postProcCfg.MAX_IMAGE_BOXES,
                model_info->m_postProcCfg.NUM_BOX_ELEMENT, stream_
            );
        }

        BatchBoxArray yolov5_detector::parser_box(int num_image)
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

        BoxArray yolov5_detector::forward(const Image& image)
        {
            auto output = forwards({image});
            if (output.empty())
            {
                return {};
            }
            
            return output[0];
        }

        BatchBoxArray yolov5_detector::forwards(const std::vector<Image>& images)
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

            float* bbox_output_device = bbox_predict_.gpu();
            std::vector<void*> bindings{input_buffer_.gpu(), bbox_output_device};

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