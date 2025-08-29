#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <NvInferRuntimePlugin.h>
#include <mutex>
#include <string>
#include <iostream>
#include <memory>

#include "preprocPlugin.h"
#include "preprocKernel.h"

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::PreprocV3;
using nvinfer1::plugin::PreprocV3PluginCreator;

// This is not needed for plugin dynamic registration.
REGISTER_TENSORRT_PLUGIN(PreprocV3PluginCreator);

// In IPluginV3 interface, the plugin name, version, and name space must be
// specified for the plugin and plugin creator exactly the same.
constexpr char const *const kPREPROC_PLUGIN_NAME{"Preproc_TRT"};
constexpr char const *const kPREPROC_PLUGIN_VERSION{"1"};
constexpr char const *const kPREPROC_PLUGIN_NAMESPACE{""};

PluginFieldCollection PreprocV3PluginCreator::mFC{};
std::vector<PluginField> PreprocV3PluginCreator::mPluginAttributes;

PreprocV3PluginCreator::PreprocV3PluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> guard(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("output_batchSize", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("output_channel", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("output_height", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("output_width", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const *PreprocV3PluginCreator::getPluginName() const noexcept
{
    return kPREPROC_PLUGIN_NAME;
}

char const *PreprocV3PluginCreator::getPluginVersion() const noexcept
{
    return kPREPROC_PLUGIN_VERSION;
}

PluginFieldCollection const *PreprocV3PluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV3 *PreprocV3PluginCreator::createPlugin(
    char const *name, PluginFieldCollection const *fc, TensorRTPhase phase) noexcept
{
    try
    {
        // PLUGIN_VALIDATE(fc != nullptr);
        PluginField const *fields = fc->fields;

        // default values
        int32_t outputBatchSize = 1;
        int32_t outputChannel = 1;
        int32_t outputHeight = 1;
        int32_t outputWidth = 1;

        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            char const *attrName = fields[i].name;
            if (!strcmp(attrName, "output_batchSize"))
            {
                // PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                outputBatchSize = static_cast<int32_t>(*(static_cast<int32_t const *>(fields[i].data)));
            }
            else if (!strcmp(attrName, "output_channel"))
            {
                // PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                outputChannel = static_cast<int32_t>(*(static_cast<int32_t const *>(fields[i].data)));
            }
            else if (!strcmp(attrName, "output_height"))
            {
                // PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                outputHeight = static_cast<int32_t>(*(static_cast<int32_t const *>(fields[i].data)));
            }
            else if (!strcmp(attrName, "output_width"))
            {
                // PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                outputWidth = static_cast<int32_t>(*(static_cast<int32_t const *>(fields[i].data)));
            }
        }
        return new PreprocV3(outputBatchSize, outputChannel, outputHeight, outputWidth);
    }
    catch (std::exception const &e)
    {
        // caughtError(e);
        std::cerr << e.what() << std::endl;
    }
    return nullptr;
}

void PreprocV3PluginCreator::setPluginNamespace(char const *libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const *PreprocV3PluginCreator::getPluginNamespace() const noexcept
{
    return kPREPROC_PLUGIN_NAMESPACE;
}

PreprocV3::PreprocV3(int32_t outputBatchSize, int32_t outputChannel, int32_t outputHeight, int32_t outputWidth)
    : mOutputBatchSize(outputBatchSize), mOutputChannel(outputChannel), mOutputHeight(outputHeight), mOutputWidth(outputWidth)
{
    // PLUGIN_VALIDATE(outputHeight > 0);
    // PLUGIN_VALIDATE(outputWidth > 0);

    int32_t device;
    // PLUGIN_CUASSERT(cudaGetDevice(&device));
    cudaGetDevice(&device);
    cudaDeviceProp props;
    // PLUGIN_CUASSERT(cudaGetDeviceProperties(&props, device));
    cudaGetDeviceProperties(&props, device);

    mMaxThreadsPerBlock = props.maxThreadsPerBlock;
}

IPluginCapability *PreprocV3::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    try
    {
        if (type == PluginCapabilityType::kBUILD)
        {
            return static_cast<IPluginV3OneBuild *>(this);
        }
        if (type == PluginCapabilityType::kRUNTIME)
        {
            return static_cast<IPluginV3OneRuntime *>(this);
        }
        // PLUGIN_ASSERT(type == PluginCapabilityType::kCORE);
        return static_cast<IPluginV3OneCore *>(this);
    }
    catch (std::exception const &e)
    {
        // caughtError(e);
        std::cerr << e.what() << std::endl;
    }
    return nullptr;
}

IPluginV3 *PreprocV3::clone() noexcept
{
    try
    {
        auto plugin = std::make_unique<PreprocV3>(*this);
        return plugin.release();
    }
    catch (std::exception const &e)
    {
        // caughtError(e);
        std::cerr << e.what() << std::endl;
    }
    return nullptr;
}

char const *PreprocV3::getPluginName() const noexcept
{
    return kPREPROC_PLUGIN_NAME;
}

char const *PreprocV3::getPluginVersion() const noexcept
{
    return kPREPROC_PLUGIN_VERSION;
}

char const *PreprocV3::getPluginNamespace() const noexcept
{
    return kPREPROC_PLUGIN_NAMESPACE;
}

int32_t PreprocV3::getNbOutputs() const noexcept
{
    return 1;
}

int32_t PreprocV3::configurePlugin(
    DynamicPluginTensorDesc const *in, int32_t nbInputs, DynamicPluginTensorDesc const *out, int32_t nbOutputs) noexcept
{
    return 0;
}

bool PreprocV3::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // PLUGIN_ASSERT(inOut != nullptr);
    // PLUGIN_ASSERT(pos >= 0 && pos <= 3);
    // PLUGIN_ASSERT(nbInputs == 3);
    // PLUGIN_ASSERT(nbOutputs == 1);

    PluginTensorDesc const &desc = inOut[pos].desc;
    if (desc.format != TensorFormat::kLINEAR)
    {
        return false;
    }

    // // first input should be float16 or float32
    // if (pos == 0)
    // {
    //     return (desc.type == nvinfer1::DataType::kFLOAT || desc.type == nvinfer1::DataType::kHALF);
    // }

    // // batch_indices always has to be int32
    // if (pos == 2)
    // {
    //     return (desc.type == nvinfer1::DataType::kINT32);
    // }

    // // rois and the output should have the same type as the first input
    // return (desc.type == inOut[0].desc.type);
    // return true;
    return inOut[pos].desc.type == DataType::kFLOAT || inOut[pos].desc.type == DataType::kHALF;
}

int32_t PreprocV3::getOutputDataTypes(
    DataType *outputTypes, int32_t nbOutputs, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    // PLUGIN_ASSERT(inputTypes != nullptr);
    // PLUGIN_ASSERT(nbInputs == 3);
    // PLUGIN_ASSERT(nbOutputs == 1);
    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t PreprocV3::getOutputShapes(DimsExprs const *inputs, int32_t nbInputs, DimsExprs const *shapeInputs,
                                   int32_t nbShapeInputs, DimsExprs *outputs, int32_t nbOutputs, IExprBuilder &exprBuilder) noexcept
{
    // PLUGIN_ASSERT(inputs != nullptr);
    // PLUGIN_ASSERT(nbInputs == 3);
    // PLUGIN_ASSERT(nbOutputs == 1);

    outputs[0].nbDims = 4;

    outputs[0].d[0] = inputs[0].d[0];
    outputs[0].d[1] = inputs[0].d[3];
    outputs[0].d[2] = inputs[0].d[1];
    outputs[0].d[3] = inputs[0].d[2];

    return 0;
}

int32_t PreprocV3::enqueue(PluginTensorDesc const *inputDesc, PluginTensorDesc const *outputDesc,
                           void const *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    // PLUGIN_VALIDATE(inputDesc != nullptr && inputs != nullptr && outputs != nullptr);
    auto type = inputDesc[0].type;

    // PLUGIN_ASSERT(type == nvinfer1::DataType::kHALF || type == nvinfer1::DataType::kFLOAT);

    switch (type)
    {
    case nvinfer1::DataType::kFLOAT:
    {
        auto input = static_cast<float const *>(inputs[0]);
        auto output = static_cast<float *>(outputs[0]);
        return PreprocImpl<float>(stream, mMaxThreadsPerBlock, input, mOutputBatchSize, mOutputChannel, mOutputHeight, mOutputWidth, output);
    }
    break;
    case nvinfer1::DataType::kHALF:
    {
        auto input = static_cast<__half const *>(inputs[0]);
        auto output = static_cast<__half *>(outputs[0]);

        return PreprocImpl<__half>(stream, mMaxThreadsPerBlock, input, mOutputBatchSize, mOutputChannel, mOutputHeight, mOutputWidth, output);
    }
    break;
    default:
        return -1;
    }

    return 0;
}

int32_t PreprocV3::onShapeChange(
    PluginTensorDesc const *in, int32_t nbInputs, PluginTensorDesc const *out, int32_t nbOutputs) noexcept
{
    return 0;
}

IPluginV3 *PreprocV3::attachToContext(IPluginResourceContext *context) noexcept
{
    return clone();
}

PluginFieldCollection const *PreprocV3::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("output_batchSize", &mOutputBatchSize, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("output_channel", &mOutputChannel, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("output_height", &mOutputHeight, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("output_width", &mOutputWidth, PluginFieldType::kINT32, 1);
    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

size_t PreprocV3::getWorkspaceSize(DynamicPluginTensorDesc const *inputs, int32_t nbInputs,
                                   DynamicPluginTensorDesc const *outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

void PreprocV3::setPluginNamespace(char const *libNamespace) noexcept
{
    try
    {
        // PLUGIN_ASSERT(libNamespace != nullptr);
        mNameSpace = libNamespace;
    }
    catch (std::exception const &e)
    {
        // caughtError(e);
        std::cerr << e.what() << std::endl;
    }
}