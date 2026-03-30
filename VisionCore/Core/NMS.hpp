//
//  NMS.hpp
//  VisionCore
//
//  Created by Ahmad on 27/03/2026.
//
#ifndef NMS_hpp
#define NMS_hpp

#include <vector>
#include "VisionResult.hpp"

namespace VisionCore {

class NMS {
public:
    // Takes raw detections from RF-DETR output
    // Returns cleaned detections — one box per object
    static std::vector<Detection> apply(
        const std::vector<Detection>& detections,
        float iouThreshold,
        int   maxDetections
    );

private:
    // Calculates IoU between two bounding boxes
    // 0.0 = no overlap, 1.0 = identical boxes
    static float computeIoU(const BoundingBox& a, const BoundingBox& b);
};

} // namespace VisionCore

#endif /* NMS_hpp */
