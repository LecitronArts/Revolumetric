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

typedef enum RevolumetricNrdTextureFormat {
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_UNSUPPORTED = 0,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_R8_UNORM = 1,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_R8_SNORM = 2,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_R8_UINT = 3,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_R8_SINT = 4,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_RG8_UNORM = 5,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_RG8_SNORM = 6,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_RG8_UINT = 7,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_RG8_SINT = 8,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_RGBA8_UNORM = 9,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_RGBA8_SNORM = 10,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_RGBA8_UINT = 11,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_RGBA8_SINT = 12,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_RGBA8_SRGB = 13,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_R16_UNORM = 14,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_R16_SNORM = 15,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_R16_UINT = 16,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_R16_SINT = 17,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_R16_SFLOAT = 18,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_RG16_UNORM = 19,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_RG16_SNORM = 20,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_RG16_UINT = 21,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_RG16_SINT = 22,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_RG16_SFLOAT = 23,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_RGBA16_UNORM = 24,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_RGBA16_SNORM = 25,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_RGBA16_UINT = 26,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_RGBA16_SINT = 27,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_RGBA16_SFLOAT = 28,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_R32_UINT = 29,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_R32_SINT = 30,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_R32_SFLOAT = 31,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_RG32_UINT = 32,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_RG32_SINT = 33,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_RG32_SFLOAT = 34,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_RGB32_UINT = 35,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_RGB32_SINT = 36,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_RGB32_SFLOAT = 37,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_RGBA32_UINT = 38,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_RGBA32_SINT = 39,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_RGBA32_SFLOAT = 40,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_R10_G10_B10_A2_UNORM = 41,
    REVOLUMETRIC_NRD_TEXTURE_FORMAT_R11_G11_B10_UFLOAT = 42,
} RevolumetricNrdTextureFormat;

typedef enum RevolumetricNrdDescriptorType {
    REVOLUMETRIC_NRD_DESCRIPTOR_TYPE_UNSUPPORTED = 0,
    REVOLUMETRIC_NRD_DESCRIPTOR_TYPE_TEXTURE = 1,
    REVOLUMETRIC_NRD_DESCRIPTOR_TYPE_STORAGE_TEXTURE = 2,
} RevolumetricNrdDescriptorType;

typedef enum RevolumetricNrdSamplerMode {
    REVOLUMETRIC_NRD_SAMPLER_MODE_NEAREST_CLAMP = 0,
    REVOLUMETRIC_NRD_SAMPLER_MODE_LINEAR_CLAMP = 1,
    REVOLUMETRIC_NRD_SAMPLER_MODE_UNSUPPORTED = 0xFFFFFFFFu,
} RevolumetricNrdSamplerMode;

typedef enum RevolumetricNrdAccumulationMode {
    REVOLUMETRIC_NRD_ACCUMULATION_MODE_CONTINUE = 0,
    REVOLUMETRIC_NRD_ACCUMULATION_MODE_RESTART = 1,
    REVOLUMETRIC_NRD_ACCUMULATION_MODE_CLEAR_AND_RESTART = 2,
    REVOLUMETRIC_NRD_ACCUMULATION_MODE_UNSUPPORTED = 0xFFFFFFFFu,
} RevolumetricNrdAccumulationMode;

typedef enum RevolumetricNrdCheckerboardMode {
    REVOLUMETRIC_NRD_CHECKERBOARD_MODE_OFF = 0,
    REVOLUMETRIC_NRD_CHECKERBOARD_MODE_BLACK = 1,
    REVOLUMETRIC_NRD_CHECKERBOARD_MODE_WHITE = 2,
    REVOLUMETRIC_NRD_CHECKERBOARD_MODE_UNSUPPORTED = 0xFFFFFFFFu,
} RevolumetricNrdCheckerboardMode;

typedef enum RevolumetricNrdNormalEncoding {
    REVOLUMETRIC_NRD_NORMAL_ENCODING_RGBA8_UNORM = 0,
    REVOLUMETRIC_NRD_NORMAL_ENCODING_RGBA8_SNORM = 1,
    REVOLUMETRIC_NRD_NORMAL_ENCODING_R10_G10_B10_A2_UNORM = 2,
    REVOLUMETRIC_NRD_NORMAL_ENCODING_RGBA16_UNORM = 3,
    REVOLUMETRIC_NRD_NORMAL_ENCODING_RGBA16_SNORM = 4,
    REVOLUMETRIC_NRD_NORMAL_ENCODING_MAX_NUM = 5,
} RevolumetricNrdNormalEncoding;

typedef enum RevolumetricNrdRoughnessEncoding {
    REVOLUMETRIC_NRD_ROUGHNESS_ENCODING_SQ_LINEAR = 0,
    REVOLUMETRIC_NRD_ROUGHNESS_ENCODING_LINEAR = 1,
    REVOLUMETRIC_NRD_ROUGHNESS_ENCODING_SQRT_LINEAR = 2,
    REVOLUMETRIC_NRD_ROUGHNESS_ENCODING_MAX_NUM = 3,
} RevolumetricNrdRoughnessEncoding;

typedef enum RevolumetricNrdHitDistanceReconstructionMode {
    REVOLUMETRIC_NRD_HIT_DISTANCE_RECONSTRUCTION_MODE_OFF = 0,
    REVOLUMETRIC_NRD_HIT_DISTANCE_RECONSTRUCTION_MODE_AREA_3X3 = 1,
    REVOLUMETRIC_NRD_HIT_DISTANCE_RECONSTRUCTION_MODE_AREA_5X5 = 2,
    REVOLUMETRIC_NRD_HIT_DISTANCE_RECONSTRUCTION_MODE_UNSUPPORTED = 0xFFFFFFFFu,
} RevolumetricNrdHitDistanceReconstructionMode;

typedef enum RevolumetricNrdResourceType {
    REVOLUMETRIC_NRD_RESOURCE_TYPE_UNSUPPORTED = 0,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_IN_MV = 1,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_IN_NORMAL_ROUGHNESS = 2,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_IN_VIEWZ = 3,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_IN_DIFF_CONFIDENCE = 4,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_IN_SPEC_CONFIDENCE = 5,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_IN_DISOCCLUSION_THRESHOLD_MIX = 6,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_IN_DIFF_RADIANCE_HITDIST = 7,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_IN_SPEC_RADIANCE_HITDIST = 8,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_IN_DIFF_HITDIST = 9,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_IN_SPEC_HITDIST = 10,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_IN_DIFF_DIRECTION_HITDIST = 11,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_IN_DIFF_SH0 = 12,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_IN_DIFF_SH1 = 13,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_IN_SPEC_SH0 = 14,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_IN_SPEC_SH1 = 15,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_IN_PENUMBRA = 16,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_IN_TRANSLUCENCY = 17,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_IN_SIGNAL = 18,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_OUT_DIFF_RADIANCE_HITDIST = 19,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_OUT_SPEC_RADIANCE_HITDIST = 20,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_OUT_DIFF_SH0 = 21,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_OUT_DIFF_SH1 = 22,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_OUT_SPEC_SH0 = 23,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_OUT_SPEC_SH1 = 24,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_OUT_DIFF_HITDIST = 25,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_OUT_SPEC_HITDIST = 26,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_OUT_DIFF_DIRECTION_HITDIST = 27,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_OUT_SHADOW_TRANSLUCENCY = 28,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_OUT_SIGNAL = 29,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_OUT_VALIDATION = 30,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_TRANSIENT_POOL = 31,
    REVOLUMETRIC_NRD_RESOURCE_TYPE_PERMANENT_POOL = 32,
} RevolumetricNrdResourceType;

typedef struct NrdLibraryDesc {
    uint32_t textureOffset;
    uint32_t samplerOffset;
    uint32_t constantBufferOffset;
    uint32_t storageTextureAndBufferOffset;
    uint8_t normalEncoding;
    uint8_t roughnessEncoding;
    uint16_t reserved0;
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

typedef struct NrdReblurHitDistanceParameters {
    float a;
    float b;
    float c;
} NrdReblurHitDistanceParameters;

typedef struct NrdReblurDiffuseSettings {
    NrdReblurHitDistanceParameters hitDistanceParameters;
    float antilagLuminanceSigmaScale;
    float antilagLuminanceSensitivity;
    float responsiveAccumulationRoughnessThreshold;
    uint32_t responsiveAccumulationMinAccumulatedFrameNum;
    float convergenceS;
    float convergenceB;
    float convergenceP;
    uint32_t maxAccumulatedFrameNum;
    uint32_t maxFastAccumulatedFrameNum;
    uint32_t maxStabilizedFrameNum;
    uint32_t historyFixFrameNum;
    uint32_t historyFixBasePixelStride;
    uint32_t historyFixAlternatePixelStride;
    float fastHistoryClampingSigmaScale;
    float diffusePrepassBlurRadius;
    float specularPrepassBlurRadius;
    float minHitDistanceWeight;
    float minBlurRadius;
    float maxBlurRadius;
    float lobeAngleFraction;
    float roughnessFraction;
    float planeDistanceSensitivity;
    float specularProbabilityThresholdsForMvModification[2];
    float fireflySuppressorMinRelativeScale;
    float minMaterialForDiffuse;
    float minMaterialForSpecular;
    uint32_t checkerboardMode;
    uint32_t hitDistanceReconstructionMode;
    uint32_t enableAntiFirefly;
    uint32_t usePrepassOnlyForSpecularMotionEstimation;
    uint32_t returnHistoryLengthInsteadOfOcclusion;
} NrdReblurDiffuseSettings;

RevolumetricNrdStatus revolumetric_nrd_create_relax_diffuse(
    uint32_t width,
    uint32_t height,
    RevolumetricNrdInstance** out_instance);
RevolumetricNrdStatus revolumetric_nrd_create_reblur_diffuse(
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
RevolumetricNrdStatus revolumetric_nrd_set_reblur_diffuse_settings(
    RevolumetricNrdInstance* instance,
    const NrdReblurDiffuseSettings* settings);
RevolumetricNrdStatus revolumetric_nrd_get_dispatches(
    RevolumetricNrdInstance* instance,
    const NrdDispatchDesc** out_dispatches,
    uint32_t* out_dispatches_num);

#ifdef __cplusplus
}
#endif
