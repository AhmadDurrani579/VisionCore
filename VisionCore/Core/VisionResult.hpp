//
//  VisionResult.hpp
//  VisionCore
//
//  Created by Ahmad on 27/03/2026.
//

#ifndef VisionResult_hpp
#define VisionResult_hpp

#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace VisionCore {

struct BoundingBox {
    float x;        // left edge, normalised 0–1
    float y;        // top edge,  normalised 0–1
    float width;    // normalised 0–1
    float height;   // normalised 0–1

    float centerX() const {return x + width * 0.5f ;}
    float centerY() const {return x + height * 0.5f ;}
    float area() const {return width * height ;}
    
};

struct Detection {
    std::string label; // "person", "chair", etc.
    float       confidence;
    BoundingBox bbox;
    int         classId; // raw COCO class index
    std::vector<uint8_t> mask;
    int maskWidth  = 0;
    int maskHeight = 0;


} ;

struct VisionResult {
    std::vector<Detection>               detections;
    std::unordered_map<std::string, int> countMap;   // "person" → 2
    int   totalCount  = 0;
    float inferenceMs = 0.0f;
    float totalMs     = 0.0f;
    bool  success     = false;
    std::string errorMessage;
};

} //namespace VisionCore
#endif /* VisionResult_hpp */
