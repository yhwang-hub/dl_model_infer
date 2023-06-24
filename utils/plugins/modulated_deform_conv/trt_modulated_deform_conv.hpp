#ifndef MODULATED_DEFORM_CONV_HPP_
#define MODULATED_DEFORM_CONV_HPP_

#include <NvInfer.h>
#include <cublas_v2.h>

#include <cassert>
#include <vector>
#include <string>
#include <stdexcept>

namespace mmdeploy 
{

inline unsigned int getElementSize(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    // case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8:
      return 1;
    default:
      throw std::runtime_error("Invalid DataType.");
  }
  throw std::runtime_error("Invalid DataType.");
  return 0;
}

inline size_t getAlignedSize(size_t origin_size, size_t aligned_number = 16) {
  return size_t((origin_size + aligned_number - 1) / aligned_number) * aligned_number;
}

class ModulatedDeformableConvPluginDynamic : public nvinfer1::IPluginV2DynamicExt
{
public:
  ModulatedDeformableConvPluginDynamic(const std::string &name, const nvinfer1::Dims stride,
                                       const nvinfer1::Dims padding, const nvinfer1::Dims dilation,
                                       const int deformableGroup, const int group);
  ModulatedDeformableConvPluginDynamic(const std::string name,const void *data,size_t length);
  ModulatedDeformableConvPluginDynamic() = delete;
  ~ModulatedDeformableConvPluginDynamic() noexcept override;
  
  //IPluginV2 Methods
  const char *getPluginType() const noexcept override;
  const char *getPluginVersion() const noexcept override;
  void setPluginNamespace(const char * libNamespace) noexcept override;
  const char * getPluginNamespace() const noexcept override;
  int initialize() noexcept override;
  void terminate() noexcept override;
  void destroy() noexcept override;
  
  int getNbOutputs() const noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;


  // IPluginV2Ext Methods
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes,
                                       int nbInputs) const noexcept override;

  // IPluginV2DynamicExt Methods
  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs * inputs, int nbInputs,
    nvinfer1::IExprBuilder & exprBuilder) noexcept override;
  bool supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc * inOut, int nbInputs,
    int nbOutputs) noexcept override;
  void configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc * in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc * out, int nbOutputs) noexcept override;
  size_t getWorkspaceSize(
    const nvinfer1::PluginTensorDesc * inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc * outputs, int nbOutputs) const noexcept override;
  int enqueue(
    const nvinfer1::PluginTensorDesc * inputDesc, const nvinfer1::PluginTensorDesc * outputDesc,
    const void * const * inputs, void * const * outputs, void * workspace,
    cudaStream_t stream) noexcept override;
  void attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext,
                       nvinfer1::IGpuAllocator *gpuAllocator) noexcept override;
  void detachFromContext() noexcept override;

private:
  nvinfer1::Dims mStride;
  nvinfer1::Dims mPadding;
  nvinfer1::Dims mDilation;
  int mDeformableGroup;
  int mGroup;
  bool mWithBias;

  std::string mPluginName;
  std::string mPluginNamespace;

  cublasHandle_t m_cublas_handle;
};

class ModulatedDeformableConvPluginDynamicCreator : public nvinfer1::IPluginCreator
{
public:
  ModulatedDeformableConvPluginDynamicCreator();
  ~ModulatedDeformableConvPluginDynamicCreator() override = default;

  const char * getPluginName() const noexcept override;
  const char * getPluginVersion() const noexcept override;
  void setPluginNamespace(const char * libNamespace) noexcept override;
  const char * getPluginNamespace() const noexcept override;

  nvinfer1::IPluginV2DynamicExt * createPlugin(
    const char * name, const nvinfer1::PluginFieldCollection * fc) noexcept override;

  nvinfer1::IPluginV2DynamicExt * deserializePlugin(
    const char * name, const void * serialData, size_t serialLength) noexcept override;
  
  const nvinfer1::PluginFieldCollection * getFieldNames() noexcept override;
private:
  std::string mNamespace; 
  static std::vector<nvinfer1::PluginField> mPluginAttributes;
  static nvinfer1::PluginFieldCollection mFC;
};

REGISTER_TENSORRT_PLUGIN(ModulatedDeformableConvPluginDynamicCreator);
}

#endif