//
//  VisionEngine.cpp
//  VisionCore
//
//  Created by Ahmad on 27/03/2026.
//

#include "VisionEngine.hpp"
#include <stdexcept>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <cmath>

// ── ExecuTorch C++ API ────────────────────────────────────────────────────────
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>


using namespace executorch::extension;
using namespace executorch::runtime;

namespace VisionCore {

// ── COCO 91 labels ────────────────────────────────────────────────────────────
static const std::vector<std::string> kCoco91Labels = {
    "",              // 0  background
    "person",        // 1
    "bicycle",       // 2
    "car",           // 3
    "motorcycle",    // 4
    "airplane",      // 5
    "bus",           // 6
    "train",         // 7
    "truck",         // 8
    "boat",          // 9
    "traffic light", // 10
    "fire hydrant",  // 11
    "",              // 12 gap
    "stop sign",     // 13
    "parking meter", // 14
    "bench",         // 15
    "bird",          // 16
    "cat",           // 17
    "dog",           // 18
    "horse",         // 19
    "sheep",         // 20
    "cow",           // 21
    "elephant",      // 22
    "bear",          // 23
    "zebra",         // 24
    "giraffe",       // 25
    "",              // 26 gap
    "backpack",      // 27
    "umbrella",      // 28
    "",              // 29 gap
    "",              // 30 gap
    "handbag",       // 31
    "tie",           // 32
    "suitcase",      // 33
    "frisbee",       // 34
    "skis",          // 35
    "snowboard",     // 36
    "sports ball",   // 37
    "kite",          // 38
    "baseball bat",  // 39
    "baseball glove",// 40
    "skateboard",    // 41
    "surfboard",     // 42
    "tennis racket", // 43
    "bottle",        // 44
    "",              // 45 gap
    "wine glass",    // 46
    "cup",           // 47
    "fork",          // 48
    "knife",         // 49
    "spoon",         // 50
    "bowl",          // 51
    "banana",        // 52
    "apple",         // 53
    "sandwich",      // 54
    "orange",        // 55
    "broccoli",      // 56
    "carrot",        // 57
    "hot dog",       // 58
    "pizza",         // 59
    "donut",         // 60
    "cake",          // 61
    "chair",         // 62
    "couch",         // 63
    "potted plant",  // 64
    "bed",           // 65
    "",              // 66 gap
    "dining table",  // 67
    "",              // 68 gap
    "",              // 69 gap
    "toilet",        // 70
    "",              // 71 gap
    "tv",            // 72
    "laptop",        // 73
    "mouse",         // 74
    "remote",        // 75
    "keyboard",      // 76
    "cell phone",    // 77
    "microwave",     // 78
    "oven",          // 79
    "toaster",       // 80
    "sink",          // 81
    "refrigerator",  // 82
    "",              // 83 gap
    "book",          // 84
    "clock",         // 85
    "vase",          // 86
    "scissors",      // 87
    "teddy bear",    // 88
    "hair drier",    // 89
    "toothbrush"     // 90
};

// ── ExecuTorch context ────────────────────────────────────────────────────────
struct VisionEngine::OrtContext {
    std::unique_ptr<Module> module;
    bool loaded = false;
};

// ── Constructor ───────────────────────────────────────────────────────────────
VisionEngine::VisionEngine(const VisionConfig& config)
    : config_(config)
    , preprocessor_(nullptr)  // ← null, created in initialise()
    , ort_(std::make_unique<OrtContext>())
{
    printf("VisionEngine: constructed with applyNorm=%d\n", config_.applyNorm);

}

VisionEngine::~VisionEngine() = default;

// ── initialise ────────────────────────────────────────────────────────────────
bool VisionEngine::initialise() {
    if (config_.modelPath.empty()) return false;

    std::ifstream file(config_.modelPath);
    if (!file.good()) {
        printf("VisionCore: Model file not found at %s\n",
               config_.modelPath.c_str());
        return false;
    }
    file.close();

    try {
        // 1. Create module
        ort_->module = std::make_unique<Module>(
            config_.modelPath,
            Module::LoadMode::MmapUseMlock
        );

        // 2. Load model
        const auto error = ort_->module->load();
        if (error != Error::Ok) {
            printf("VisionCore: ExecuTorch load failed\n");
            return false;
        }

        // 3. Load forward method explicitly
        auto loadResult = ort_->module->load_forward();
        if (loadResult != Error::Ok) {
            printf("VisionCore: Failed to load_forward: %d\n", (int)loadResult);
            return false;
        }
        printf("VisionCore: load_forward ✅\n");
        preprocessor_ = std::make_unique<Preprocessor>(config_);
        printf("VisionCore: preprocessor created with applyNorm=%d\n", config_.applyNorm);

        // 4. Auto-detect input size from model
        auto inputMeta = ort_->module->method_meta("forward");
        if (inputMeta.ok()) {
            auto meta = inputMeta.get();
            if (meta.num_inputs() > 0) {
                auto input = meta.input_tensor_meta(0);
                if (input.ok()) {
                    auto sizes = input.get().sizes();
                    if (sizes.size() == 4) {
                        config_.inputHeight = static_cast<int>(sizes[2]);
                        config_.inputWidth  = static_cast<int>(sizes[3]);
                        printf("VisionCore: input size set to %d×%d\n",
                               config_.inputWidth, config_.inputHeight);
                    }
                }
            }
        }
        
        preprocessor_ = std::make_unique<Preprocessor>(config_);

        // 5. Warmup pass
        std::vector<executorch::aten::SizesType> warmupShape = {
            1, 3,
            static_cast<executorch::aten::SizesType>(config_.inputHeight),
            static_cast<executorch::aten::SizesType>(config_.inputWidth)
        };
        std::vector<float> dummy(
            3 * config_.inputHeight * config_.inputWidth, 0.0f);
        auto warmupTensor = executorch::extension::from_blob(
            dummy.data(),
            warmupShape,
            executorch::aten::ScalarType::Float
        );
        std::vector<EValue> warmupInputs = { EValue(warmupTensor) };
        ort_->module->execute("forward", warmupInputs);
        printf("VisionCore: warmup done ✅\n");

        ort_->loaded = true;
        isReady_     = true;
        printf("VisionCore: ExecuTorch ✅ model loaded\n");
        return true;

    } catch (const std::exception& e) {
        printf("VisionCore: ExecuTorch error: %s\n", e.what());
        isReady_ = false;
        return false;
    }
}

// ── detect ────────────────────────────────────────────────────────────────────
VisionResult VisionEngine::detect(const RawFrame& frame) {

    if (!isReady_) {
        VisionResult err;
        err.success      = false;
        err.errorMessage = "VisionEngine not initialised";
        return err;
    }

    auto pipelineStart = std::chrono::high_resolution_clock::now();

    // ── 1. Preprocess ─────────────────────────────────────────
    std::vector<float> inputData = preprocessor_->process(frame);

    // ── 2. Build ExecuTorch input tensor ─────────────────────
    std::vector<executorch::aten::SizesType> shape = {
        1, 3,
        static_cast<executorch::aten::SizesType>(config_.inputHeight),
        static_cast<executorch::aten::SizesType>(config_.inputWidth)
    };

    auto inputTensor = executorch::extension::from_blob(
        inputData.data(),
        shape,
        executorch::aten::ScalarType::Float
    );

    // ── 3. Run inference ──────────────────────────────────────
    auto inferStart = std::chrono::high_resolution_clock::now();

    auto result = ort_->module->forward(inputTensor);

    auto inferEnd = std::chrono::high_resolution_clock::now();
    float inferenceMs = std::chrono::duration<float, std::milli>(
        inferEnd - inferStart).count();

    if (!result.ok()) {
        VisionResult err;
        err.success      = false;
        err.errorMessage = "ExecuTorch inference failed";
        return err;
    }

    // ── 4. Parse outputs ──────────────────────────────────────
    auto& outputs = result.get();

    if (outputs.size() < 2) {
        VisionResult err;
        err.success      = false;
        err.errorMessage = "Unexpected output count";
        return err;
    }

    const auto& boxesTensor  = outputs[0].toTensor();
    const auto& scoresTensor = outputs[1].toTensor();
    const auto& classTensor  = outputs[2].toTensor();

    const float* boxes   = boxesTensor.const_data_ptr<float>();
    const float* scores  = scoresTensor.const_data_ptr<float>();
    const float* classes = classTensor.const_data_ptr<float>();

    const int numDetections = 300;

    std::vector<Detection> rawDetections;

    for (int i = 0; i < numDetections; ++i) {
        float score   = scores[i];
        int classId   = static_cast<int>(classes[i]);

        if (score < config_.confidenceThreshold) continue;
        if (classId < 0 ||
            classId >= static_cast<int>(kCoco91Labels.size())) continue;
        if (kCoco91Labels[classId].empty()) continue;

//        float cx = boxes[i * 4 + 0];
//        float cy = boxes[i * 4 + 1];
//        float w  = boxes[i * 4 + 2];
//        float h  = boxes[i * 4 + 3];

        // ExecuTorch .pte outputs pixel coords x1,y1,x2,y2 in 384x384 space
        float x1 = boxes[i * 4 + 0] / 384.0f;
        float y1 = boxes[i * 4 + 1] / 384.0f;
        float x2 = boxes[i * 4 + 2] / 384.0f;
        float y2 = boxes[i * 4 + 3] / 384.0f;

        Detection det;
        det.classId    = classId;
        det.label      = kCoco91Labels[classId];
        det.confidence = score;
        det.bbox       = { x1, y1, x2 - x1, y2 - y1 };  // x, y, w, h normalized
        rawDetections.push_back(det);
    }

    // ── 5. NMS ────────────────────────────────────────────────
    std::vector<Detection> cleanDetections = NMS::apply(
        rawDetections,
        config_.nmsIoUThreshold,
        config_.maxDetections
    );

    // ── 6. Build result ───────────────────────────────────────
    auto pipelineEnd = std::chrono::high_resolution_clock::now();
    float totalMs = std::chrono::duration<float, std::milli>(
        pipelineEnd - pipelineStart).count();

    return buildResult(cleanDetections, inferenceMs, totalMs);
}

// ── parseOutput — unused, kept for compatibility ──────────────────────────────
std::vector<Detection> VisionEngine::parseOutput(
    const std::vector<float>& output,
    int numDetections,
    int frameWidth,
    int frameHeight)
{
    return {};
}

// ── buildResult ───────────────────────────────────────────────────────────────
VisionResult VisionEngine::buildResult(
    const std::vector<Detection>& detections,
    float inferenceMs,
    float totalMs)
{
    VisionResult result;
    result.detections  = detections;
    result.inferenceMs = inferenceMs;
    result.totalMs     = totalMs;
    result.totalCount  = static_cast<int>(detections.size());
    result.success     = true;

    for (const auto& det : detections) {
        result.countMap[det.label]++;
    }

    return result;
}

} // namespace VisionCore
