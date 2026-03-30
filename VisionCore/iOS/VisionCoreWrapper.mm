//
//  VisionCoreWrapper.mm
//  VisionCore
//
//  Created by Ahmad on 27/03/2026.
//

#import "VisionCoreWrapper.h"
#import <CoreVideo/CoreVideo.h>

// C++ engine headers
#include "VisionEngine.hpp"
#include "VisionConfig.hpp"
#include "VisionResult.hpp"

// ── VCDetection ───────────────────────────────────────────────────────────────
@implementation VCDetection
@end

// ── VCVisionResult ────────────────────────────────────────────────────────────
@implementation VCVisionResult
@end

// ── VisionCoreWrapper ─────────────────────────────────────────────────────────
@implementation VisionCoreWrapper {
    // C++ engine lives here — hidden from Swift completely
    std::unique_ptr<VisionCore::VisionEngine> _engine;
    VisionCore::VisionConfig                  _config;
}

- (instancetype)initWithModelPath:(NSString*)modelPath {
    self = [super init];
    if (self) {
        
        _config.modelPath           = std::string([modelPath UTF8String]);
        _config.inputWidth          = 384;
        _config.inputHeight         = 384;
        _config.confidenceThreshold = 0.35f;
        _config.maxDetections       = 10;
        _config.numThreads          = 2;
        
        // Apply ImageNet normalization for all RF-DETR models
        _config.applyNorm = true;
        _config.normMeanR = 0.485f;
        _config.normMeanG = 0.456f;
        _config.normMeanB = 0.406f;
        _config.normStdR  = 0.229f;
        _config.normStdG  = 0.224f;
        _config.normStdB  = 0.225f;

        _engine = std::make_unique<VisionCore::VisionEngine>(_config);
    }
    return self;
}


- (void)initialiseAsync:(void(^)(BOOL ready))completion {
    dispatch_async(dispatch_get_global_queue(
        DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^{
        bool ready = self->_engine->initialise();
        NSLog(@"VisionCore initialise: %@", ready ? @"✅ SUCCESS" : @"❌ FAILED");
        dispatch_async(dispatch_get_main_queue(), ^{
            if (completion) completion(ready ? YES : NO);
        });
    });
}


- (void)setConfidenceThreshold:(float)threshold {
    _config.confidenceThreshold = threshold;
}

- (void)setMaxDetections:(int)max {
    _config.maxDetections = max;
}

- (BOOL)isReady {
    return _engine && _engine->isReady() ? YES : NO;
}

// ── Main detection call ───────────────────────────────────────────────────────
- (VCVisionResult*)detectInPixelBuffer:(CVPixelBufferRef)pixelBuffer {

    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

    // Extract raw bytes from CVPixelBuffer
    const uint8_t* baseAddress = static_cast<const uint8_t*>(
        CVPixelBufferGetBaseAddress(pixelBuffer)
    );
    int width  = (int)CVPixelBufferGetWidth(pixelBuffer);
    int height = (int)CVPixelBufferGetHeight(pixelBuffer);
    int stride = (int)CVPixelBufferGetBytesPerRow(pixelBuffer);

    // Build RawFrame — AVCaptureSession gives us BGRA by default
    VisionCore::RawFrame frame;
    frame.data   = baseAddress;
    frame.width  = width;
    frame.height = height;
    frame.stride = stride;
    frame.format = VisionCore::RawFrame::Format::BGRA;

    // Run C++ engine
    VisionCore::VisionResult cppResult = _engine->detect(frame);

    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

    // ── Convert C++ result → Obj-C result for Swift ───────────────────────────
    VCVisionResult* result  = [[VCVisionResult alloc] init];
    result.success          = cppResult.success;
    result.totalCount       = cppResult.totalCount;
    result.inferenceMs      = cppResult.inferenceMs;
    result.totalMs          = cppResult.totalMs;
    result.errorMessage     = cppResult.errorMessage.empty()
                                ? @""
                                : @(cppResult.errorMessage.c_str());

    // Convert detections
    NSMutableArray<VCDetection*>* detections = [NSMutableArray array];
    for (const auto& d : cppResult.detections) {
        VCDetection* det  = [[VCDetection alloc] init];
        det.label         = @(d.label.c_str());
        det.confidence    = d.confidence;
        det.classId       = d.classId;
        det.boundingBox   = CGRectMake(
                                d.bbox.x,
                                d.bbox.y,
                                d.bbox.width,
                                d.bbox.height
                            );
        [detections addObject:det];
    }
    result.detections = detections;

    // Convert countMap
    NSMutableDictionary* countMap = [NSMutableDictionary dictionary];
    for (const auto& pair : cppResult.countMap) {
        countMap[@(pair.first.c_str())] = @(pair.second);
    }
    result.countMap = countMap;

    return result;
}

@end
