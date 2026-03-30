//
//  Preprocessor.hpp
//  VisionCore
//
//  Created by Ahmad on 27/03/2026.
//

#ifndef Preprocessor_hpp
#define Preprocessor_hpp

#include <vector>
#include <cstdint>
#include "VisionConfig.hpp"    // ← correct, was pointing to itself before

namespace VisionCore {

struct RawFrame {
    const uint8_t* data   = nullptr;
    int            width  = 0;
    int            height = 0;
    int            stride = 0;  // bytes per row, may include padding

    enum class Format {
        BGR,   // OpenCV default
        RGB,   // standard
        BGRA,  // iOS CVPixelBuffer default
        RGBA
    } format = Format::BGRA;
};

class Preprocessor {
public:
    explicit Preprocessor(const VisionConfig& config);

    // Takes raw camera frame → returns float tensor ready for RF-DETR
    // Output shape: [1, 3, inputHeight, inputWidth] — CHW format, normalised 0–1
    std::vector<float> process(const RawFrame& frame);

private:
    int inputWidth_;
    int inputHeight_;
    VisionConfig config_;
    
    std::vector<uint8_t> resize(const RawFrame& frame);
    std::vector<uint8_t> toRGB(const std::vector<uint8_t>& resized,
                                RawFrame::Format format);
    std::vector<float>   toFloatTensor(const std::vector<uint8_t>& rgb);
};

} // namespace VisionCore

#endif /* Preprocessor_hpp */
