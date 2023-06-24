// Copyright (c) OpenMMLab. All rights reserved

#include "trt_modulated_deform_conv_kernel.hpp"
#include "trt_modulated_deform_conv.hpp"
#include "serialize.hpp"

namespace 
{
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"MMCVModulatedDeformConv2d"};
// const char * NMS_PLUGIN_NAMESPACE{""};
}  // namespace    

namespace mmdeploy 
{
ModulatedDeformableConvPluginDynamic::ModulatedDeformableConvPluginDynamic(
    const std::string &name, const nvinfer1::Dims stride, const nvinfer1::Dims padding,
    const nvinfer1::Dims dilation, const int deformableGroup, const int group)
    : mPluginName(name),
      mStride(stride),
      mPadding(padding),
      mDilation(dilation),
      mDeformableGroup(deformableGroup),
      mGroup(group) 
{
  mWithBias = false;
}

ModulatedDeformableConvPluginDynamic::ModulatedDeformableConvPluginDynamic(const std::string name,
                                                                           const void *data,
                                                                           size_t length)
{
  mPluginName = name; 
  deserialize_value(&data, &length, &mStride);
  deserialize_value(&data, &length, &mPadding);
  deserialize_value(&data, &length, &mDilation);
  deserialize_value(&data, &length, &mDeformableGroup);
  deserialize_value(&data, &length, &mGroup);
  mWithBias = false;
}

ModulatedDeformableConvPluginDynamic::~ModulatedDeformableConvPluginDynamic() {};

const char * ModulatedDeformableConvPluginDynamic::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char * ModulatedDeformableConvPluginDynamic::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

void ModulatedDeformableConvPluginDynamic::setPluginNamespace(const char * pluginNamespace) noexcept
{
   mPluginNamespace = pluginNamespace;
}

const char * ModulatedDeformableConvPluginDynamic::getPluginNamespace() const noexcept 
{ 
    return mPluginNamespace.c_str(); 
}

int ModulatedDeformableConvPluginDynamic::initialize() noexcept
{
    return 0; //0 means success
}

void ModulatedDeformableConvPluginDynamic::terminate() noexcept
{
  return;
}

void ModulatedDeformableConvPluginDynamic::destroy() noexcept
{
    delete this;
}

int ModulatedDeformableConvPluginDynamic::getNbOutputs() const noexcept
{
    return 1;
}

size_t ModulatedDeformableConvPluginDynamic::getSerializationSize() const noexcept
{
    return serialized_size(mStride) + serialized_size(mPadding) + serialized_size(mDilation) +
         serialized_size(mDeformableGroup) + serialized_size(mGroup);
}

void ModulatedDeformableConvPluginDynamic::serialize(void *buffer) const noexcept
{
    serialize_value(&buffer, mStride);
    serialize_value(&buffer, mPadding);
    serialize_value(&buffer, mDilation);
    serialize_value(&buffer, mDeformableGroup);
    serialize_value(&buffer, mGroup);
}

nvinfer1::DataType ModulatedDeformableConvPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes, int nbInputs) const noexcept 
{
  return inputTypes[0];
}

nvinfer1::IPluginV2DynamicExt * ModulatedDeformableConvPluginDynamic::clone() const noexcept
{    
    ModulatedDeformableConvPluginDynamic* plugin = new ModulatedDeformableConvPluginDynamic(mPluginName,
      mStride,mPadding,mDilation,mDeformableGroup,mGroup);

    plugin->setPluginNamespace(mPluginNamespace.c_str());

    return plugin;
}


nvinfer1::DimsExprs ModulatedDeformableConvPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs * inputs, int nbInputs,
    nvinfer1::IExprBuilder & exprBuilder) noexcept
{
  //to be clear,input: x(b,c1,h,w) offset(c2,2,h,w) mask(c2,n,n) weight(c2,n,n)  output: y(b,c2,h,w)
  
  nvinfer1::DimsExprs ret;
  ret.nbDims = 4;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = inputs[3].d[0];

  ret.d[2] = inputs[1].d[2];
  ret.d[3] = inputs[1].d[3];

  return ret;
}

bool ModulatedDeformableConvPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc * inOut, int nbInputs,
    int nbOutputs) noexcept
{
  //pos的可能取值为0-N N为该算子的输入输出总数目-1  
  if (pos == 0) {
    return (inOut[pos].type == nvinfer1::DataType::kFLOAT &&
            inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);

  } else {
    return inOut[pos].type == inOut[0].type && inOut[pos].format == inOut[0].format;
  }
}

void ModulatedDeformableConvPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc * in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc * out, int nbOutputs) noexcept
{
  if (nbInputs == 5) {
    mWithBias = true;
  }
}

size_t ModulatedDeformableConvPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc * inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc * outputs, int nbOutputs) const noexcept
{
  int sizeof_dtype = mmdeploy::getElementSize(outputs[0].type);

  int batch_size = inputs[0].dims.d[0];
  int nInputPlane = inputs[0].dims.d[1];
  int inputHeight = inputs[0].dims.d[2];
  int inputWidth = inputs[0].dims.d[3];

  int nOutputPlane = outputs[0].dims.d[1];
  int outputHeight = outputs[0].dims.d[2];
  int outputWidth = outputs[0].dims.d[3];

  int kW = inputs[3].dims.d[2];
  int kH = inputs[3].dims.d[3];
  int im2col_step = std::min(32, batch_size);

  size_t col_size =
      mmdeploy::getAlignedSize(nInputPlane * kW * kH * outputHeight * outputWidth * sizeof_dtype);

  return col_size;

}

int ModulatedDeformableConvPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc * inputDesc, const nvinfer1::PluginTensorDesc * outputDesc,
    const void * const * inputs, void * const * outputs, void * workspace,
    cudaStream_t stream) noexcept
{
  int batch = inputDesc[0].dims.d[0];
  int channels = inputDesc[0].dims.d[1];
  int height = inputDesc[0].dims.d[2];
  int width = inputDesc[0].dims.d[3];
  int channels_out = outputDesc[0].dims.d[1];
  int kernel_h = inputDesc[3].dims.d[2];
  int kernel_w = inputDesc[3].dims.d[3];

  const void *x = inputs[0]; //input feature
  const void *offset = inputs[1]; //采样点的偏移　比如3x3的kernerl. 则shape为3x3x2(2:一个x方向偏移,一个y方向偏移)
  const void *mask = inputs[2];//
  const void *weight = inputs[3];//每个采样点的权重
  const void *bias = mWithBias ? inputs[4] : nullptr;
  void *output = outputs[0];
  int im2col_step = std::min(batch, 32);

  // TODO: add fp16 support
  auto data_type = inputDesc[0].type;
  switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
      ModulatedDeformConvForwardCUDAKernelLauncher_float(
          (float *)x, (float *)weight, (float *)bias, (float *)offset, (float *)mask,
          (float *)output, workspace, batch, channels, height, width, channels_out, kernel_w,
          kernel_h, mStride.d[0], mStride.d[1], mPadding.d[0], mPadding.d[1], mDilation.d[0],
          mDilation.d[1], mGroup, mDeformableGroup, im2col_step, m_cublas_handle, stream);
      break;
    default:
      return 1;
      break;
  }

  return 0;
}

void ModulatedDeformableConvPluginDynamic::attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext,
                       nvinfer1::IGpuAllocator *gpuAllocator) noexcept
{
  m_cublas_handle = cublasContext;
}

void ModulatedDeformableConvPluginDynamic::detachFromContext() noexcept
{

}

///////////////////////////////////////////////////////////////////////////////
nvinfer1::PluginFieldCollection ModulatedDeformableConvPluginDynamicCreator::mFC{};
std::vector<nvinfer1::PluginField> ModulatedDeformableConvPluginDynamicCreator::mPluginAttributes;

ModulatedDeformableConvPluginDynamicCreator::ModulatedDeformableConvPluginDynamicCreator()
{
  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(nvinfer1::PluginField("stride"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("padding"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("dilation"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("groups"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("deform_groups"));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char * ModulatedDeformableConvPluginDynamicCreator::getPluginName() const noexcept
{
  return PLUGIN_NAME;
}

const char * ModulatedDeformableConvPluginDynamicCreator::getPluginVersion() const noexcept
{
  return PLUGIN_VERSION;
}

void ModulatedDeformableConvPluginDynamicCreator::setPluginNamespace(const char * libNamespace) noexcept
{
  mNamespace = libNamespace;
}

const char * ModulatedDeformableConvPluginDynamicCreator::getPluginNamespace() const noexcept 
{ 
  // std::cout<<"getPluginNamespace():"<<mNamespace<<std::endl;
  return mNamespace.c_str(); 
}

const nvinfer1::PluginFieldCollection * ModulatedDeformableConvPluginDynamicCreator::getFieldNames() noexcept { return &mFC; }

nvinfer1::IPluginV2DynamicExt * ModulatedDeformableConvPluginDynamicCreator::createPlugin(
    const char * name, const nvinfer1::PluginFieldCollection * fc) noexcept
{
  //从PluginFieldCollection解析出构造plugin的参数
  nvinfer1::Dims stride{2, {1, 1}};
  nvinfer1::Dims padding{2, {0, 0}};
  nvinfer1::Dims dilation{2, {1, 1}};
  int deformableGroup = 1;
  int group = 1;

  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("deformable_group") == 0) {
      deformableGroup = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("group") == 0) {
      group = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("stride") == 0) {
      stride.nbDims = 2;
      stride.d[0] = static_cast<const int *>(fc->fields[i].data)[0];
      stride.d[1] = static_cast<const int *>(fc->fields[i].data)[1];
    }

    if (field_name.compare("padding") == 0) {
      padding.nbDims = 2;
      padding.d[0] = static_cast<const int *>(fc->fields[i].data)[0];
      padding.d[1] = static_cast<const int *>(fc->fields[i].data)[1];
    }

    if (field_name.compare("dilation") == 0) {
      dilation.nbDims = 2;
      dilation.d[0] = static_cast<const int *>(fc->fields[i].data)[0];
      dilation.d[1] = static_cast<const int *>(fc->fields[i].data)[1];
    }
  }
  ModulatedDeformableConvPluginDynamic *plugin = new ModulatedDeformableConvPluginDynamic(
      name, stride, padding, dilation, deformableGroup, group);
  plugin->setPluginNamespace(mNamespace.c_str());

  return plugin;
}

nvinfer1::IPluginV2DynamicExt * ModulatedDeformableConvPluginDynamicCreator::deserializePlugin(
  const char * name, const void * serialData, size_t serialLength) noexcept
{
  ModulatedDeformableConvPluginDynamic * plugin = new ModulatedDeformableConvPluginDynamic(name, serialData, serialLength);;
  plugin->setPluginNamespace(mNamespace.c_str());

  return plugin;
}

//向tensorrt注册plugin
// REGISTER_TENSORRT_PLUGIN(ModulatedDeformableConvPluginDynamicCreator);

}