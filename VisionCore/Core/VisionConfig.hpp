//
//  VisionConfig.hpp
//  VisionCore
//
//  Created by Ahmad on 27/03/2026.
//

#ifndef VisionConfig_hpp
#define VisionConfig_hpp

#include <string>

namespace VisionCore {

enum class InferenceBackend {
    CPU,      // always available, slowest
    CoreML,   // iOS Neural Engine / GPU via ONNX CoreML EP
    NNAPI,    // Android GPU / DSP via ONNX NNAPI EP
    Auto      // pick best available at runtime (recommended)
};

struct VisionConfig {

    // ── Model ─────────────────────────────────────────────────
    std::string modelPath;
    int         inputWidth  = 384;
    int         inputHeight = 384;

    // ── Detection filtering ───────────────────────────────────
    float confidenceThreshold = 0.35f;
    float nmsIoUThreshold     = 0.30f;
    int   maxDetections       = 20;


    // ── ImageNet normalization ────────────────────────────────
    bool  applyNorm = false;
    float normMeanR = 0.485f;
    float normMeanG = 0.456f;
    float normMeanB = 0.406f;
    float normStdR  = 0.229f;
    float normStdG  = 0.224f;
    float normStdB  = 0.225f;

    int              numThreads   = 2;
};

} // namespace VisionCore

#endif /* VisionConfig_hpp */
