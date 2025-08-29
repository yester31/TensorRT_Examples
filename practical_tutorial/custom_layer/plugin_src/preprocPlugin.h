#ifndef TRT_PREPROC_PLUGIN_H
#define TRT_PREPROC_PLUGIN_H

#include <cuda_runtime_api.h>
#include <stdint.h>
#include <vector>
#include <string>

#include "NvInfer.h"
#include "NvInferPlugin.h"

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>

namespace nvinfer1
{
    namespace plugin
    {

        class PreprocV3PluginCreator : public nvinfer1::IPluginCreatorV3One
        {
        public:
            PreprocV3PluginCreator();

            ~PreprocV3PluginCreator() override = default;

            char const *getPluginName() const noexcept override;

            char const *getPluginVersion() const noexcept override;

            PluginFieldCollection const *getFieldNames() noexcept override;

            IPluginV3 *createPlugin(char const *name, PluginFieldCollection const *fc, TensorRTPhase phase) noexcept override;

            void setPluginNamespace(char const *libNamespace) noexcept;

            char const *getPluginNamespace() const noexcept override;

        private:
            static PluginFieldCollection mFC;
            static std::vector<PluginField> mPluginAttributes;
            std::string mNamespace;
        };

        class PreprocV3 : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime
        {
        public:
            PreprocV3(int32_t outputBatchSize, int32_t outputChannel, int32_t outputHeight, int32_t outputWidth);
            PreprocV3(PreprocV3 const &) = default;
            ~PreprocV3() override = default;

            IPluginCapability *getCapabilityInterface(PluginCapabilityType type) noexcept override;

            IPluginV3 *clone() noexcept override;

            char const *getPluginName() const noexcept override;

            char const *getPluginVersion() const noexcept override;

            char const *getPluginNamespace() const noexcept override;

            int32_t getNbOutputs() const noexcept override;

            int32_t configurePlugin(DynamicPluginTensorDesc const *in, int32_t nbInputs, DynamicPluginTensorDesc const *out,
                                    int32_t nbOutputs) noexcept override;

            bool supportsFormatCombination(
                int32_t pos, DynamicPluginTensorDesc const *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

            int32_t getOutputDataTypes(
                DataType *outputTypes, int32_t nbOutputs, DataType const *inputTypes, int32_t nbInputs) const noexcept override;

            int32_t getOutputShapes(DimsExprs const *inputs, int32_t nbInputs, DimsExprs const *shapeInputs,
                                    int32_t nbShapeInputs, DimsExprs *outputs, int32_t nbOutputs, IExprBuilder &exprBuilder) noexcept override;

            int32_t enqueue(PluginTensorDesc const *inputDesc, PluginTensorDesc const *outputDesc, void const *const *inputs,
                            void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;

            int32_t onShapeChange(
                PluginTensorDesc const *in, int32_t nbInputs, PluginTensorDesc const *out, int32_t nbOutputs) noexcept override;

            IPluginV3 *attachToContext(IPluginResourceContext *context) noexcept override;

            PluginFieldCollection const *getFieldsToSerialize() noexcept override;

            size_t getWorkspaceSize(DynamicPluginTensorDesc const *inputs, int32_t nbInputs,
                                    DynamicPluginTensorDesc const *outputs, int32_t nbOutputs) const noexcept override;

            void setPluginNamespace(char const *libNamespace) noexcept;

        private:
            int32_t mOutputBatchSize{};
            int32_t mOutputChannel{};
            int32_t mOutputHeight{};
            int32_t mOutputWidth{};

            int32_t mMaxThreadsPerBlock{};

            std::string mNameSpace{};

            std::vector<nvinfer1::PluginField> mDataToSerialize;
            nvinfer1::PluginFieldCollection mFCToSerialize;
        };

    } // namespace plugin
} // namespace nvinfer1
#endif // TRT_PREPROC_PLUGIN_H