//
//  NMS.cpp
//  VisionCore
//
//  Created by Ahmad on 27/03/2026.
//

#include "NMS.hpp"
#include <algorithm>

namespace VisionCore {

std::vector<Detection> NMS::apply(
    const std::vector<Detection>& detections,
    float iouThreshold,
    int   maxDetections)
{
    if (detections.empty()) return {};

    // Step 1 — sort all detections by confidence, highest first
    // Same idea as Canny keeping strongest edges
    std::vector<Detection> sorted = detections;
    std::sort(sorted.begin(), sorted.end(),
        [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        });

    std::vector<Detection> result;
    std::vector<bool>      suppressed(sorted.size(), false);

    // Step 2 — walk from highest to lowest confidence
    for (size_t i = 0; i < sorted.size(); ++i) {

        if (suppressed[i]) continue;

        // Keep this detection — it's the best for this region
        result.push_back(sorted[i]);

        if (static_cast<int>(result.size()) >= maxDetections) break;

        // Step 3 — suppress any lower-confidence box of the same class
        // that overlaps this one above the IoU threshold
        for (size_t j = i + 1; j < sorted.size(); ++j) {
            if (suppressed[j]) continue;

            // Only suppress same class — a person and a chair
            // can overlap without suppressing each other
            if (sorted[j].classId != sorted[i].classId) continue;

            float iou = computeIoU(sorted[i].bbox, sorted[j].bbox);
            if (iou > iouThreshold) {
                suppressed[j] = true;  // duplicate — throw away
            }
        }
    }

    return result;
}

// ── IoU calculation ───────────────────────────────────────────────────────────
// Intersection area divided by union area
// All coordinates are normalised 0–1
float NMS::computeIoU(const BoundingBox& a, const BoundingBox& b) {

    // Find intersection rectangle
    const float interLeft   = std::max(a.x, b.x);
    const float interTop    = std::max(a.y, b.y);
    const float interRight  = std::min(a.x + a.width,  b.x + b.width);
    const float interBottom = std::min(a.y + a.height, b.y + b.height);

    // No overlap
    if (interRight <= interLeft || interBottom <= interTop) return 0.0f;

    const float interArea = (interRight  - interLeft)
                          * (interBottom - interTop);

    const float aArea = a.width * a.height;
    const float bArea = b.width * b.height;

    // Union = both areas minus the overlap counted twice
    const float unionArea = aArea + bArea - interArea;

    if (unionArea <= 0.0f) return 0.0f;

    return interArea / unionArea;
}

} // namespace VisionCore
