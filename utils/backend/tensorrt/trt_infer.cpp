#include "trt_infer.h"
namespace trt
{
    namespace infer
    {

        // __native_engine_context的方法实现 start
        bool __native_engine_context::construct(std::vector<unsigned char> &trtFile)
        {
            destroy();
            sample::setReportableSeverity(sample::Severity::kERROR); // 设置trt的log输出级别，这里只有报错才输出
            if (trtFile.empty())
                return false;

            runtime_ = shared_ptr<IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()), destroy_nvidia_pointer<IRuntime>);
            if (runtime_ == nullptr)
                return false;
            initLibNvInferPlugins(&(sample::gLogger.getTRTLogger()), "");
            engine_ = shared_ptr<ICudaEngine>(runtime_->deserializeCudaEngine(trtFile.data(), trtFile.size(), nullptr), destroy_nvidia_pointer<ICudaEngine>);
            if (engine_ == nullptr)
                return false;

            context_ = shared_ptr<IExecutionContext>(engine_->createExecutionContext(), destroy_nvidia_pointer<IExecutionContext>);
            return context_ != nullptr;
        }

        __native_engine_context::~__native_engine_context()
        {
            destroy();
        }

        void __native_engine_context::destroy()
        {
            context_.reset();
            engine_.reset();
            runtime_.reset();
        }
        // __native_engine_context的方法实现 end

        // InferTRT的方法实现 start

        bool InferTRT::construct_context(std::vector<unsigned char> &trtFile)
        {
            this->context_ = make_shared<__native_engine_context>();
            if (!this->context_->construct(trtFile))
            {
                return false;
            }

            setup();
            return true;
        }

        bool InferTRT::load(const string &engine_file)
        {
            std::vector<unsigned char> data = ai::utils::load_file(engine_file);
            if (data.empty())
            {
                INFO("An empty file has been loaded. Please confirm your file path: %s", engine_file.c_str());
                return false;
            }
            return this->construct_context(data);
        }

        void InferTRT::setup()
        {
            auto engine = this->context_->engine_;
            int nbBindings = engine->getNbBindings();

            binding_name_to_index_.clear();
            for (int i = 0; i < nbBindings; ++i)
            {
                const char *bindingName = engine->getBindingName(i);
                binding_name_to_index_[bindingName] = i;
            }
        }

        int InferTRT::index(const std::string &name)
        {
            auto iter = binding_name_to_index_.find(name);
            Assertf(iter != binding_name_to_index_.end(), "Can not found the binding name: %s",
                    name.c_str());
            return iter->second;
        }

        bool InferTRT::forward(const std::vector<void *> &bindings, void *stream, void *input_consum_event)
        {
            return this->context_->context_->enqueueV2((void **)bindings.data(), (cudaStream_t)stream,
                                                       (cudaEvent_t *)input_consum_event);
        }

        std::vector<int> InferTRT::get_network_dims(const std::string &name)
        {
            return get_network_dims(index(name));
        }

        std::vector<int> InferTRT::get_network_dims(int ibinding)
        {
            auto dim = this->context_->engine_->getBindingDimensions(ibinding);
            return std::vector<int>(dim.d, dim.d + dim.nbDims);
        }

        bool InferTRT::set_network_dims(const std::string &name, const std::vector<int> &dims)
        {
            return this->set_network_dims(index(name), dims);
        }

        bool InferTRT::set_network_dims(int ibinding, const std::vector<int> &dims)
        {
            Dims d;
            memcpy(d.d, dims.data(), sizeof(int) * dims.size());
            d.nbDims = dims.size();
            return this->context_->context_->setBindingDimensions(ibinding, d);
        }

        bool InferTRT::has_dynamic_dim()
        {
            // check if any input or output bindings have dynamic shapes
            int numBindings = this->context_->engine_->getNbBindings();
            for (int i = 0; i < numBindings; ++i)
            {
                nvinfer1::Dims dims = this->context_->engine_->getBindingDimensions(i);
                for (int j = 0; j < dims.nbDims; ++j)
                {
                    if (dims.d[j] == -1)
                        return true;
                }
            }
            return false;
        }

        std::string InferTRT::format_shape(const Dims &shape)
        {
            stringstream output;
            char buf[64];
            const char *fmts[] = {"%d", "x%d"};
            for (int i = 0; i < shape.nbDims; ++i)
            {
                snprintf(buf, sizeof(buf), fmts[i != 0], shape.d[i]);
                output << buf;
            }
            return output.str();
        }

        void InferTRT::print()
        {
            INFO("Infer %p [%s]", this, has_dynamic_dim() ? "DynamicShape" : "StaticShape");

            int num_input = 0;
            int num_output = 0;
            auto engine = this->context_->engine_;
            for (int i = 0; i < engine->getNbBindings(); ++i)
            {
                if (engine->bindingIsInput(i))
                    num_input++;
                else
                    num_output++;
            }

            INFO("Inputs: %d", num_input);
            for (int i = 0; i < num_input; ++i)
            {
                auto name = engine->getBindingName(i);
                auto dim = engine->getBindingDimensions(i);
                INFO("\t%d.%s : shape {%s}", i, name, format_shape(dim).c_str());
            }

            INFO("Outputs: %d", num_output);
            for (int i = 0; i < num_output; ++i)
            {
                auto name = engine->getBindingName(i + num_input);
                auto dim = engine->getBindingDimensions(i + num_input);
                INFO("\t%d.%s : shape {%s}", i, name, format_shape(dim).c_str());
            }
        }

        // InferTRT的方法实现 end

        Infer *loadraw(const std::string &file)
        {
            InferTRT *impl = new InferTRT();
            if (!impl->load(file))
            {
                delete impl;
                impl = nullptr;
            }
            return impl;
        }

        std::shared_ptr<Infer> load(const std::string &file)
        {
            return std::shared_ptr<InferTRT>((InferTRT *)loadraw(file));
        }
    }
}