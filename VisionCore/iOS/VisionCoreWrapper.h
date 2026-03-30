//
//  VisionCoreWrapper.h
//  VisionCore
//
//  Created by Ahmad on 27/03/2026.
//

#import <Foundation/Foundation.h>
#import <CoreVideo/CoreVideo.h>



// Swift-friendly detection result
@interface VCDetection : NSObject
@property (nonatomic, copy)   NSString* label;
@property (nonatomic, assign) float     confidence;
@property (nonatomic, assign) CGRect    boundingBox;  // normalised 0–1
@property (nonatomic, assign) int       classId;
@end

// Swift-friendly frame result
@interface VCVisionResult : NSObject
@property (nonatomic, strong) NSArray<VCDetection*>*    detections;
@property (nonatomic, strong) NSDictionary<NSString*, NSNumber*>* countMap;
@property (nonatomic, assign) int   totalCount;
@property (nonatomic, assign) float inferenceMs;
@property (nonatomic, assign) float totalMs;
@property (nonatomic, assign) BOOL  success;
@property (nonatomic, copy)   NSString* errorMessage;
@end

// Main wrapper — Swift talks to this, never to C++ directly
@interface VisionCoreWrapper : NSObject

// Initialise with path to rfdetr_nano.onnx
- (instancetype)initWithModelPath:(NSString*)modelPath;

// Configure detection sensitivity
- (void)setConfidenceThreshold:(float)threshold;
- (void)setMaxDetections:(int)max;

// Call every camera frame — CVPixelBuffer comes directly from AVCaptureSession
- (VCVisionResult*)detectInPixelBuffer:(CVPixelBufferRef)pixelBuffer;

// Async init — calls completion on main thread when ready
- (void)initialiseAsync:(void(^)(BOOL ready))completion;

// Engine state
- (BOOL)isReady;

@end
