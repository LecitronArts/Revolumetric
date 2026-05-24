#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RevolumetricNrdInstance RevolumetricNrdInstance;

typedef enum RevolumetricNrdStatus {
    REVOLUMETRIC_NRD_STATUS_OK = 0,
    REVOLUMETRIC_NRD_STATUS_INVALID_ARGUMENT = 1,
    REVOLUMETRIC_NRD_STATUS_SDK_ERROR = 2,
    REVOLUMETRIC_NRD_STATUS_INSUFFICIENT_CAPACITY = 3,
} RevolumetricNrdStatus;

typedef struct NrdLibraryDesc {
    uint32_t textureOffset;
    uint32_t samplerOffset;
    uint32_t constantBufferOffset;
    uint32_t storageTextureAndBufferOffset;
} NrdLibraryDesc;

typedef struct NrdTextureDesc {
    uint32_t format;
    uint16_t downsampleFactor;
    uint16_t reserved0;
} NrdTextureDesc;

typedef struct NrdResourceDesc {
    uint32_t descriptorType;
    uint32_t type;
    uint16_t indexInPool;
    uint16_t reserved0;
} NrdResourceDesc;

typedef struct NrdResourceRangeDesc {
    uint32_t descriptorType;
    uint32_t descriptorsNum;
} NrdResourceRangeDesc;

typedef struct NrdSamplerDesc {
    uint32_t mode;
} NrdSamplerDesc;

typedef struct NrdPipelineDesc {
    const uint32_t* spirvBytecode;
    uint64_t spirvBytecodeSize;
    const NrdResourceRangeDesc* resourceRanges;
    uint32_t resourceRangesNum;
    uint32_t hasConstantData;
    char shaderIdentifier[256];
} NrdPipelineDesc;

typedef struct NrdInstanceDesc {
    uint32_t constantBufferAndSamplersSpaceIndex;
    uint32_t resourcesSpaceIndex;
    uint32_t constantBufferRegisterIndex;
    uint32_t samplersBaseRegisterIndex;
    uint32_t resourcesBaseRegisterIndex;
    uint32_t constantBufferMaxDataSize;
    const NrdSamplerDesc* samplers;
    uint32_t samplersNum;
    const NrdPipelineDesc* pipelines;
    uint32_t pipelinesNum;
    const NrdTextureDesc* permanentPool;
    uint32_t permanentPoolSize;
    const NrdTextureDesc* transientPool;
    uint32_t transientPoolSize;
} NrdInstanceDesc;

typedef struct NrdDispatchDesc {
    const char* name;
    uint32_t identifier;
    const NrdResourceDesc* resources;
    uint32_t resourcesNum;
    const uint8_t* constantBufferData;
    uint32_t constantBufferDataSize;
    uint32_t constantBufferDataMatchesPreviousDispatch;
    uint16_t pipelineIndex;
    uint16_t gridWidth;
    uint16_t gridHeight;
    uint16_t reserved0;
} NrdDispatchDesc;

typedef struct NrdCommonSettings {
    float viewToClipMatrix[16];
    float viewToClipMatrixPrev[16];
    float worldToViewMatrix[16];
    float worldToViewMatrixPrev[16];
    float cameraJitter[2];
    float cameraJitterPrev[2];
    float motionVectorScale[3];
    uint16_t resourceSize[2];
    uint16_t resourceSizePrev[2];
    uint16_t rectSize[2];
    uint16_t rectSizePrev[2];
    float denoisingRange;
    float disocclusionThreshold;
    float disocclusionThresholdAlternate;
    float splitScreen;
    float timeDeltaBetweenFrames;
    float viewZScale;
    uint32_t frameIndex;
    uint32_t accumulationMode;
    uint32_t isMotionVectorInWorldSpace;
    uint32_t isHistoryConfidenceAvailable;
    uint32_t isDisocclusionThresholdMixAvailable;
    uint32_t enableValidation;
} NrdCommonSettings;

typedef struct NrdRelaxDiffuseSettings {
    float antilagAccelerationAmount;
    float antilagSpatialSigmaScale;
    float antilagTemporalSigmaScale;
    float antilagResetAmount;
    uint32_t diffuseMaxAccumulatedFrameNum;
    uint32_t diffuseMaxFastAccumulatedFrameNum;
    uint32_t historyFixFrameNum;
    uint32_t historyFixBasePixelStride;
    uint32_t historyFixAlternatePixelStride;
    float historyFixEdgeStoppingNormalPower;
    float fastHistoryClampingSigmaScale;
    float diffusePrepassBlurRadius;
    float minHitDistanceWeight;
    uint32_t spatialVarianceEstimationHistoryThreshold;
    float diffusePhiLuminance;
    uint32_t atrousIterationNum;
    float diffuseMinLuminanceWeight;
    float depthThreshold;
    float confidenceDrivenRelaxationMultiplier;
    float confidenceDrivenLuminanceEdgeStoppingRelaxation;
    float confidenceDrivenNormalEdgeStoppingRelaxation;
    float luminanceEdgeStoppingRelaxation;
    float normalEdgeStoppingRelaxation;
    float roughnessEdgeStoppingRelaxation;
    uint32_t checkerboardMode;
    uint32_t hitDistanceReconstructionMode;
    float minMaterialForDiffuse;
    uint32_t enableAntiFirefly;
    uint32_t enableRoughnessEdgeStopping;
} NrdRelaxDiffuseSettings;

RevolumetricNrdStatus revolumetric_nrd_create_relax_diffuse(
    uint32_t width,
    uint32_t height,
    RevolumetricNrdInstance** out_instance);
void revolumetric_nrd_destroy(RevolumetricNrdInstance* instance);
RevolumetricNrdStatus revolumetric_nrd_get_library_desc(NrdLibraryDesc* out_desc);
RevolumetricNrdStatus revolumetric_nrd_get_instance_desc(
    const RevolumetricNrdInstance* instance,
    NrdInstanceDesc* out_desc);
RevolumetricNrdStatus revolumetric_nrd_set_common_settings(
    RevolumetricNrdInstance* instance,
    const NrdCommonSettings* settings);
RevolumetricNrdStatus revolumetric_nrd_set_relax_diffuse_settings(
    RevolumetricNrdInstance* instance,
    const NrdRelaxDiffuseSettings* settings);
RevolumetricNrdStatus revolumetric_nrd_get_dispatches(
    RevolumetricNrdInstance* instance,
    const NrdDispatchDesc** out_dispatches,
    uint32_t* out_dispatches_num);

#ifdef __cplusplus
}
#endif
