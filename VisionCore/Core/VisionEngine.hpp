//
//  VisionEngine.hpp
//  VisionCore
//
//  Created by Ahmad on 27/03/2026.
//

#ifndef VisionEngine_hpp
#define VisionEngine_hpp

#include <memory>
#include <vector>
#include <string>
#include "VisionConfig.hpp"
#include "VisionResult.hpp"
#include "Preprocessor.hpp"
#include "NMS.hpp"

namespace VisionCore {

class VisionEngine {
public:
    explicit VisionEngine(const VisionConfig& config);
    ~VisionEngine();

    // Initialise ONNX Runtime session + load model
    // Call once before running detect()
    bool initialise();

    // Main entry point — call every camera frame
    // Returns VisionResult with detections, counts, timings
    VisionResult detect(const RawFrame& frame);

    // True after initialise() succeeds
    bool isReady() const { return isReady_; }

    // Current config — read only
    const VisionConfig& config() const { return config_; }

private:
    VisionConfig              config_;
    std::unique_ptr<Preprocessor> preprocessor_;
    bool                      isReady_ = false;

    // ONNX Runtime session — forward declared to avoid
    // pulling ONNX headers into every file that includes VisionEngine.hpp
    struct OrtContext;
    std::unique_ptr<OrtContext> ort_;

    // Parse raw ONNX output tensor → vector of Detection
    std::vector<Detection> parseOutput(
        const std::vector<float>& output,
        int                       numDetections,
        int                       frameWidth,
        int                       frameHeight
    );

    // Build countMap and totalCount from final detections
    VisionResult buildResult(
        const std::vector<Detection>& detections,
        float inferenceMs,
        float totalMs
    );
};

} // namespace VisionCore

#endif /* VisionEngine_hpp */
