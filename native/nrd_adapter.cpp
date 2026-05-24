#include "nrd_adapter.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include "NRD.h"
#include "NRDDescs.h"
#include "NRDSettings.h"

struct RevolumetricNrdInstance {
    nrd::Instance* instance = nullptr;
    std::vector<NrdSamplerDesc> samplers;
    std::vector<NrdResourceRangeDesc> resourceRanges;
    std::vector<NrdPipelineDesc> pipelines;
    std::vector<NrdTextureDesc> permanentPool;
    std::vector<NrdTextureDesc> transientPool;
    std::vector<NrdResourceDesc> dispatchResources;
    std::vector<NrdDispatchDesc> dispatches;
};

static RevolumetricNrdStatus from_result(nrd::Result result) {
    return result == nrd::Result::SUCCESS ? REVOLUMETRIC_NRD_STATUS_OK
                                          : REVOLUMETRIC_NRD_STATUS_SDK_ERROR;
}

static uint32_t to_u32(nrd::DescriptorType value) {
    return static_cast<uint32_t>(value);
}

static uint32_t to_u32(nrd::ResourceType value) {
    return static_cast<uint32_t>(value);
}

static uint32_t to_u32(nrd::Format value) {
    return static_cast<uint32_t>(value);
}

static void copy_matrix(float* dst, const float* src) {
    std::memcpy(dst, src, sizeof(float) * 16);
}

static void copy_common_settings(
    nrd::CommonSettings& dst,
    const NrdCommonSettings& src) {
    copy_matrix(dst.viewToClipMatrix, src.viewToClipMatrix);
    copy_matrix(dst.viewToClipMatrixPrev, src.viewToClipMatrixPrev);
    copy_matrix(dst.worldToViewMatrix, src.worldToViewMatrix);
    copy_matrix(dst.worldToViewMatrixPrev, src.worldToViewMatrixPrev);
    std::memcpy(dst.cameraJitter, src.cameraJitter, sizeof(float) * 2);
    std::memcpy(dst.cameraJitterPrev, src.cameraJitterPrev, sizeof(float) * 2);
    std::memcpy(dst.motionVectorScale, src.motionVectorScale, sizeof(float) * 3);
    std::memcpy(dst.resourceSize, src.resourceSize, sizeof(uint16_t) * 2);
    std::memcpy(dst.resourceSizePrev, src.resourceSizePrev, sizeof(uint16_t) * 2);
    std::memcpy(dst.rectSize, src.rectSize, sizeof(uint16_t) * 2);
    std::memcpy(dst.rectSizePrev, src.rectSizePrev, sizeof(uint16_t) * 2);
    dst.denoisingRange = src.denoisingRange;
    dst.disocclusionThreshold = src.disocclusionThreshold;
    dst.disocclusionThresholdAlternate = src.disocclusionThresholdAlternate;
    dst.splitScreen = src.splitScreen;
    dst.timeDeltaBetweenFrames = src.timeDeltaBetweenFrames;
    dst.viewZScale = src.viewZScale;
    dst.frameIndex = src.frameIndex;
    dst.accumulationMode = static_cast<nrd::AccumulationMode>(src.accumulationMode);
    dst.isMotionVectorInWorldSpace = src.isMotionVectorInWorldSpace != 0;
    dst.isHistoryConfidenceAvailable = src.isHistoryConfidenceAvailable != 0;
    dst.isDisocclusionThresholdMixAvailable =
        src.isDisocclusionThresholdMixAvailable != 0;
    dst.enableValidation = src.enableValidation != 0;
}

static void copy_relax_diffuse_settings(
    nrd::RelaxSettings& dst,
    const NrdRelaxDiffuseSettings& src) {
    dst.antilagSettings.accelerationAmount = src.antilagAccelerationAmount;
    dst.antilagSettings.spatialSigmaScale = src.antilagSpatialSigmaScale;
    dst.antilagSettings.temporalSigmaScale = src.antilagTemporalSigmaScale;
    dst.antilagSettings.resetAmount = src.antilagResetAmount;
    dst.diffuseMaxAccumulatedFrameNum = src.diffuseMaxAccumulatedFrameNum;
    dst.specularMaxAccumulatedFrameNum = 0;
    dst.diffuseMaxFastAccumulatedFrameNum = src.diffuseMaxFastAccumulatedFrameNum;
    dst.specularMaxFastAccumulatedFrameNum = 0;
    dst.historyFixFrameNum = src.historyFixFrameNum;
    dst.historyFixBasePixelStride = src.historyFixBasePixelStride;
    dst.historyFixAlternatePixelStride = src.historyFixAlternatePixelStride;
    dst.historyFixEdgeStoppingNormalPower = src.historyFixEdgeStoppingNormalPower;
    dst.fastHistoryClampingSigmaScale = src.fastHistoryClampingSigmaScale;
    dst.diffusePrepassBlurRadius = src.diffusePrepassBlurRadius;
    dst.specularPrepassBlurRadius = 0.0f;
    dst.minHitDistanceWeight = src.minHitDistanceWeight;
    dst.spatialVarianceEstimationHistoryThreshold =
        src.spatialVarianceEstimationHistoryThreshold;
    dst.diffusePhiLuminance = src.diffusePhiLuminance;
    dst.specularPhiLuminance = 0.0f;
    dst.lobeAngleFraction = 0.0f;
    dst.roughnessFraction = 0.0f;
    dst.specularVarianceBoost = 0.0f;
    dst.specularLobeAngleSlack = 0.0f;
    dst.atrousIterationNum = src.atrousIterationNum;
    dst.diffuseMinLuminanceWeight = src.diffuseMinLuminanceWeight;
    dst.specularMinLuminanceWeight = 0.0f;
    dst.depthThreshold = src.depthThreshold;
    dst.confidenceDrivenRelaxationMultiplier =
        src.confidenceDrivenRelaxationMultiplier;
    dst.confidenceDrivenLuminanceEdgeStoppingRelaxation =
        src.confidenceDrivenLuminanceEdgeStoppingRelaxation;
    dst.confidenceDrivenNormalEdgeStoppingRelaxation =
        src.confidenceDrivenNormalEdgeStoppingRelaxation;
    dst.luminanceEdgeStoppingRelaxation = src.luminanceEdgeStoppingRelaxation;
    dst.normalEdgeStoppingRelaxation = src.normalEdgeStoppingRelaxation;
    dst.roughnessEdgeStoppingRelaxation = src.roughnessEdgeStoppingRelaxation;
    dst.checkerboardMode =
        static_cast<nrd::CheckerboardMode>(src.checkerboardMode);
    dst.hitDistanceReconstructionMode =
        static_cast<nrd::HitDistanceReconstructionMode>(
            src.hitDistanceReconstructionMode);
    dst.minMaterialForDiffuse = src.minMaterialForDiffuse;
    dst.minMaterialForSpecular = 0;
    dst.enableAntiFirefly = src.enableAntiFirefly != 0;
    dst.enableRoughnessEdgeStopping = src.enableRoughnessEdgeStopping != 0;
}

static void cache_instance_desc(RevolumetricNrdInstance& out) {
    out.samplers.clear();
    out.resourceRanges.clear();
    out.pipelines.clear();
    out.permanentPool.clear();
    out.transientPool.clear();

    const nrd::InstanceDesc& desc = *nrd::GetInstanceDesc(*out.instance);
    uint32_t resource_ranges_num = 0;
    for (uint32_t i = 0; i < desc.pipelinesNum; ++i) {
        resource_ranges_num += desc.pipelines[i].resourceRangesNum;
    }
    out.resourceRanges.reserve(resource_ranges_num);
    out.pipelines.reserve(desc.pipelinesNum);
    out.permanentPool.reserve(desc.permanentPoolSize);
    out.transientPool.reserve(desc.transientPoolSize);
    out.samplers.reserve(desc.samplersNum);
    for (uint32_t i = 0; i < desc.samplersNum; ++i) {
        out.samplers.push_back({static_cast<uint32_t>(desc.samplers[i])});
    }

    for (uint32_t i = 0; i < desc.permanentPoolSize; ++i) {
        const nrd::TextureDesc& texture = desc.permanentPool[i];
        out.permanentPool.push_back(
            {to_u32(texture.format), texture.downsampleFactor, 0});
    }
    for (uint32_t i = 0; i < desc.transientPoolSize; ++i) {
        const nrd::TextureDesc& texture = desc.transientPool[i];
        out.transientPool.push_back(
            {to_u32(texture.format), texture.downsampleFactor, 0});
    }

    for (uint32_t i = 0; i < desc.pipelinesNum; ++i) {
        const nrd::PipelineDesc& pipeline = desc.pipelines[i];
        const uint32_t range_offset =
            static_cast<uint32_t>(out.resourceRanges.size());
        for (uint32_t range = 0; range < pipeline.resourceRangesNum; ++range) {
            const nrd::ResourceRangeDesc& native_range =
                pipeline.resourceRanges[range];
            out.resourceRanges.push_back(
                {to_u32(native_range.descriptorType), native_range.descriptorsNum});
        }

        NrdPipelineDesc copied = {};
        copied.spirvBytecode =
            reinterpret_cast<const uint32_t*>(pipeline.computeShaderSPIRV.bytecode);
        copied.spirvBytecodeSize = pipeline.computeShaderSPIRV.size;
        copied.resourceRanges =
            out.resourceRanges.empty() ? nullptr
                                       : out.resourceRanges.data() + range_offset;
        copied.resourceRangesNum = pipeline.resourceRangesNum;
        copied.hasConstantData = pipeline.hasConstantData ? 1u : 0u;
        std::memcpy(
            copied.shaderIdentifier,
            pipeline.shaderIdentifier,
            sizeof(copied.shaderIdentifier));
        out.pipelines.push_back(copied);
    }
}

extern "C" RevolumetricNrdStatus revolumetric_nrd_create_relax_diffuse(
    uint32_t width,
    uint32_t height,
    RevolumetricNrdInstance** out_instance) {
    if (out_instance == nullptr || width == 0 || height == 0) {
        return REVOLUMETRIC_NRD_STATUS_INVALID_ARGUMENT;
    }
    *out_instance = nullptr;

    nrd::DenoiserDesc denoiser = {};
    denoiser.identifier = 0;
    denoiser.denoiser = nrd::Denoiser::RELAX_DIFFUSE;
    nrd::InstanceCreationDesc create_desc = {};
    create_desc.denoisers = &denoiser;
    create_desc.denoisersNum = 1;
    std::unique_ptr<RevolumetricNrdInstance> instance(
        new RevolumetricNrdInstance());
    const nrd::Result result = nrd::CreateInstance(create_desc, instance->instance);
    if (result != nrd::Result::SUCCESS) {
        return from_result(result);
    }
    cache_instance_desc(*instance);
    *out_instance = instance.release();
    return REVOLUMETRIC_NRD_STATUS_OK;
}

extern "C" void revolumetric_nrd_destroy(RevolumetricNrdInstance* instance) {
    if (instance == nullptr) {
        return;
    }
    if (instance->instance != nullptr) {
        nrd::DestroyInstance(*instance->instance);
    }
    delete instance;
}

extern "C" RevolumetricNrdStatus revolumetric_nrd_get_library_desc(
    NrdLibraryDesc* out_desc) {
    if (out_desc == nullptr) {
        return REVOLUMETRIC_NRD_STATUS_INVALID_ARGUMENT;
    }
    const nrd::LibraryDesc& desc = *nrd::GetLibraryDesc();
    out_desc->textureOffset = desc.spirvBindingOffsets.textureOffset;
    out_desc->samplerOffset = desc.spirvBindingOffsets.samplerOffset;
    out_desc->constantBufferOffset =
        desc.spirvBindingOffsets.constantBufferOffset;
    out_desc->storageTextureAndBufferOffset =
        desc.spirvBindingOffsets.storageTextureAndBufferOffset;
    return REVOLUMETRIC_NRD_STATUS_OK;
}

extern "C" RevolumetricNrdStatus revolumetric_nrd_get_instance_desc(
    const RevolumetricNrdInstance* instance,
    NrdInstanceDesc* out_desc) {
    if (instance == nullptr || instance->instance == nullptr || out_desc == nullptr) {
        return REVOLUMETRIC_NRD_STATUS_INVALID_ARGUMENT;
    }
    const nrd::InstanceDesc& desc = *nrd::GetInstanceDesc(*instance->instance);
    out_desc->constantBufferAndSamplersSpaceIndex =
        desc.constantBufferAndSamplersSpaceIndex;
    out_desc->resourcesSpaceIndex = desc.resourcesSpaceIndex;
    out_desc->constantBufferRegisterIndex = desc.constantBufferRegisterIndex;
    out_desc->samplersBaseRegisterIndex = desc.samplersBaseRegisterIndex;
    out_desc->resourcesBaseRegisterIndex = desc.resourcesBaseRegisterIndex;
    out_desc->constantBufferMaxDataSize = desc.constantBufferMaxDataSize;
    out_desc->samplers = instance->samplers.data();
    out_desc->samplersNum = static_cast<uint32_t>(instance->samplers.size());
    out_desc->pipelines = instance->pipelines.data();
    out_desc->pipelinesNum = static_cast<uint32_t>(instance->pipelines.size());
    out_desc->permanentPool = instance->permanentPool.data();
    out_desc->permanentPoolSize =
        static_cast<uint32_t>(instance->permanentPool.size());
    out_desc->transientPool = instance->transientPool.data();
    out_desc->transientPoolSize =
        static_cast<uint32_t>(instance->transientPool.size());
    return REVOLUMETRIC_NRD_STATUS_OK;
}

extern "C" RevolumetricNrdStatus revolumetric_nrd_set_common_settings(
    RevolumetricNrdInstance* instance,
    const NrdCommonSettings* settings) {
    if (instance == nullptr || instance->instance == nullptr || settings == nullptr) {
        return REVOLUMETRIC_NRD_STATUS_INVALID_ARGUMENT;
    }
    nrd::CommonSettings native = {};
    copy_common_settings(native, *settings);
    return from_result(nrd::SetCommonSettings(*instance->instance, native));
}

extern "C" RevolumetricNrdStatus revolumetric_nrd_set_relax_diffuse_settings(
    RevolumetricNrdInstance* instance,
    const NrdRelaxDiffuseSettings* settings) {
    if (instance == nullptr || instance->instance == nullptr || settings == nullptr) {
        return REVOLUMETRIC_NRD_STATUS_INVALID_ARGUMENT;
    }
    nrd::RelaxSettings native = {};
    copy_relax_diffuse_settings(native, *settings);
    return from_result(nrd::SetDenoiserSettings(*instance->instance, 0, &native));
}

extern "C" RevolumetricNrdStatus revolumetric_nrd_get_dispatches(
    RevolumetricNrdInstance* instance,
    const NrdDispatchDesc** out_dispatches,
    uint32_t* out_dispatches_num) {
    if (instance == nullptr || instance->instance == nullptr ||
        out_dispatches == nullptr || out_dispatches_num == nullptr) {
        return REVOLUMETRIC_NRD_STATUS_INVALID_ARGUMENT;
    }

    const nrd::DispatchDesc* native_dispatches = nullptr;
    uint32_t dispatches_num = 0;
    const nrd::Identifier identifiers[] = {0};
    const nrd::Result result = nrd::GetComputeDispatches(
        *instance->instance, identifiers, 1, native_dispatches, dispatches_num);
    if (result != nrd::Result::SUCCESS) {
        return from_result(result);
    }

    uint32_t resources_num = 0;
    for (uint32_t i = 0; i < dispatches_num; ++i) {
        resources_num += native_dispatches[i].resourcesNum;
    }
    instance->dispatchResources.clear();
    instance->dispatches.clear();
    instance->dispatchResources.reserve(resources_num);
    instance->dispatches.reserve(dispatches_num);
    for (uint32_t i = 0; i < dispatches_num; ++i) {
        const nrd::DispatchDesc& native_dispatch = native_dispatches[i];
        const uint32_t resource_offset =
            static_cast<uint32_t>(instance->dispatchResources.size());
        for (uint32_t resource = 0; resource < native_dispatch.resourcesNum;
             ++resource) {
            const nrd::ResourceDesc& native_resource =
                native_dispatch.resources[resource];
            instance->dispatchResources.push_back({
                to_u32(native_resource.descriptorType),
                to_u32(native_resource.type),
                native_resource.indexInPool,
                0,
            });
        }

        NrdDispatchDesc copied = {};
        copied.name = native_dispatch.name;
        copied.identifier = native_dispatch.identifier;
        copied.resources = instance->dispatchResources.empty()
                               ? nullptr
                               : instance->dispatchResources.data() + resource_offset;
        copied.resourcesNum = native_dispatch.resourcesNum;
        copied.constantBufferData =
            static_cast<const uint8_t*>(native_dispatch.constantBufferData);
        copied.constantBufferDataSize = native_dispatch.constantBufferDataSize;
        copied.constantBufferDataMatchesPreviousDispatch =
            native_dispatch.constantBufferDataMatchesPreviousDispatch ? 1u : 0u;
        copied.pipelineIndex = native_dispatch.pipelineIndex;
        copied.gridWidth = native_dispatch.gridWidth;
        copied.gridHeight = native_dispatch.gridHeight;
        copied.reserved0 = 0;
        instance->dispatches.push_back(copied);
    }

    *out_dispatches = instance->dispatches.data();
    *out_dispatches_num = static_cast<uint32_t>(instance->dispatches.size());
    return REVOLUMETRIC_NRD_STATUS_OK;
}
