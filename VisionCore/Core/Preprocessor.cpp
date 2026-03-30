//
//  Preprocessor.cpp
//  VisionCore
//
//  Created by Ahmad on 27/03/2026.
//

#include "Preprocessor.hpp"
#include <stdexcept>
#include <cstring>


namespace VisionCore {

Preprocessor::Preprocessor(const VisionConfig& config)
    : inputWidth_(config.inputWidth)
    , inputHeight_(config.inputHeight)
    , config_(config)
{}

std::vector<float> Preprocessor::process(const RawFrame& frame) {
    if (!frame.data || frame.width <= 0 || frame.height <= 0) {
        throw std::invalid_argument("VisionCore::Preprocessor — invalid frame");
    }

    auto resized = resize(frame);
    auto rgb     = toRGB(resized, frame.format);
    return toFloatTensor(rgb);
}

// ── Bilinear resize ───────────────────────────────────────────────────────────
// Same concept as cv::resize — maps each output pixel back to the input
// using fractional coordinates, then blends the 4 surrounding input pixels.
std::vector<uint8_t> Preprocessor::resize(const RawFrame& frame) {
    const int channels = (frame.format == RawFrame::Format::BGRA ||
                          frame.format == RawFrame::Format::RGBA) ? 4 : 3;

    std::vector<uint8_t> output(inputWidth_ * inputHeight_ * channels);

    const float scaleX = static_cast<float>(frame.width)  / inputWidth_;
    const float scaleY = static_cast<float>(frame.height) / inputHeight_;

    for (int y = 0; y < inputHeight_; ++y) {
        for (int x = 0; x < inputWidth_; ++x) {

            // Map output pixel → input coordinate
            const float srcX = (x + 0.5f) * scaleX - 0.5f;
            const float srcY = (y + 0.5f) * scaleY - 0.5f;

            // Clamp to valid range
            const int x0 = std::max(0, static_cast<int>(srcX));
            const int y0 = std::max(0, static_cast<int>(srcY));
            const int x1 = std::min(frame.width  - 1, x0 + 1);
            const int y1 = std::min(frame.height - 1, y0 + 1);

            // Fractional parts for blending
            const float dx = srcX - x0;
            const float dy = srcY - y0;

            // Bilinear blend across 4 neighbours per channel
            for (int c = 0; c < channels; ++c) {
                const float tl = frame.data[y0 * frame.stride + x0 * channels + c];
                const float tr = frame.data[y0 * frame.stride + x1 * channels + c];
                const float bl = frame.data[y1 * frame.stride + x0 * channels + c];
                const float br = frame.data[y1 * frame.stride + x1 * channels + c];

                const float value = tl * (1 - dx) * (1 - dy)
                                  + tr *      dx   * (1 - dy)
                                  + bl * (1 - dx)  *      dy
                                  + br *      dx   *      dy;

                output[(y * inputWidth_ + x) * channels + c] =
                    static_cast<uint8_t>(value);
            }
        }
    }
    return output;
}

// ── Channel reorder → RGB ────────────────────────────────────────────────────
std::vector<uint8_t> Preprocessor::toRGB(const std::vector<uint8_t>& resized,
                                          RawFrame::Format format) {
    const int pixels = inputWidth_ * inputHeight_;
    std::vector<uint8_t> rgb(pixels * 3);

    for (int i = 0; i < pixels; ++i) {
        switch (format) {
            case RawFrame::Format::RGB:
                rgb[i*3+0] = resized[i*3+0];
                rgb[i*3+1] = resized[i*3+1];
                rgb[i*3+2] = resized[i*3+2];
                break;

            case RawFrame::Format::BGR:
                rgb[i*3+0] = resized[i*3+2]; // B→R swap
                rgb[i*3+1] = resized[i*3+1];
                rgb[i*3+2] = resized[i*3+0];
                break;

            case RawFrame::Format::RGBA:
                rgb[i*3+0] = resized[i*4+0];
                rgb[i*3+1] = resized[i*4+1];
                rgb[i*3+2] = resized[i*4+2];
                break;  // drop alpha

            case RawFrame::Format::BGRA: // CVPixelBuffer default on iOS
                rgb[i*3+0] = resized[i*4+2]; // B→R swap, drop alpha
                rgb[i*3+1] = resized[i*4+1];
                rgb[i*3+2] = resized[i*4+0];
                break;
        }
    }
    return rgb;
}

// ── HWC uint8 → CHW float ────────────────────────────────────────────────────
// HWC = [height][width][channel] — how cameras give us pixels
// CHW = [channel][height][width] — what ONNX Runtime expects
// Normalise 0–255 → 0.0–1.0 (RF-DETR does not use ImageNet mean/std)

std::vector<float> Preprocessor::toFloatTensor(const std::vector<uint8_t>& rgb) {
    const int pixels = inputWidth_ * inputHeight_;
    std::vector<float> tensor(3 * pixels);
    for (int i = 0; i < pixels; ++i) {
        float r = rgb[i*3+0] / 255.0f;
        float g = rgb[i*3+1] / 255.0f;
        float b = rgb[i*3+2] / 255.0f;

        // Apply ImageNet normalization if enabled
        if (config_.applyNorm) {
            r = (r - config_.normMeanR) / config_.normStdR;
            g = (g - config_.normMeanG) / config_.normStdG;
            b = (b - config_.normMeanB) / config_.normStdB;
        }

        tensor[0 * pixels + i] = r;
        tensor[1 * pixels + i] = g;
        tensor[2 * pixels + i] = b;
    }

    return tensor;
}

} // namespace VisionCore
