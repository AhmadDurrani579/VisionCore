//
//  VisionDetector.swift
//  VisionCore
//
//  Created by Ahmad on 27/03/2026.
//

import Foundation
import CoreVideo

// Swift-friendly Detection — mirrors VCDetection from Obj-C bridge
public struct Detection {
    public let label:       String
    public let confidence:  Float
    public let boundingBox: CGRect  // normalised 0–1
    public let classId:     Int

    public var confidencePercent: String {
        String(format: "%.1f%%", confidence * 100)
    }
}

// Swift-friendly VisionResult — mirrors VCVisionResult from Obj-C bridge
public struct VisionResult {
    public let detections:  [Detection]
    public let countMap:    [String: Int]   // "person" → 2
    public let totalCount:  Int
    public let inferenceMs: Float
    public let totalMs:     Float
    public let success:     Bool
    public let errorMessage: String

    // Convenience — FPS from total pipeline time
    public var fps: Float {
        totalMs > 0 ? 1000.0 / totalMs : 0
    }
}

// Main public SDK class — this is what developers use
// @Observable so SwiftUI views update automatically
@Observable
public final class VisionDetector {

    // Latest result — SwiftUI binds directly to this
    public private(set) var result: VisionResult?
    public private(set) var isReady: Bool = false

    // Configuration
    public var confidenceThreshold: Float = 0.50 {
        didSet { wrapper.setConfidenceThreshold(confidenceThreshold) }
    }

    public var maxDetections: Int = 100 {
        didSet { wrapper.setMaxDetections(Int32(maxDetections)) }
    }

    // Obj-C++ bridge — Swift never touches C++ directly
    private let wrapper: VisionCoreWrapper

    // ── Init ──────────────────────────────────────────────────
    public init(modelPath: String) {
        wrapper = VisionCoreWrapper(modelPath: modelPath)
        isReady = false

        // Initialise async — no polling, callback when done
        wrapper.initialiseAsync { [weak self] ready in
            guard let self else { return }
            self.isReady = ready
            print("VisionDetector isReady: \(ready)")
        }
    }
    
    // Convenience init — looks for model inside app bundle
    
    public convenience init(modelName: String = "rfdetr_nano") {
        guard let path = Bundle.main.path(
            forResource: modelName,
            ofType: "pte"  // ← was "onnx"
        ) else {
            fatalError("VisionCore: model \(modelName).pte not found in bundle")
        }
        self.init(modelPath: path)
    }


    // ── Main detection call ───────────────────────────────────
    // Call this every frame from AVCaptureSession
    @discardableResult
    public func detect(pixelBuffer: CVPixelBuffer) -> VisionResult {
        guard let raw = wrapper.detect(in: pixelBuffer) else {
            let failed = VisionResult(
                detections:   [],
                countMap:     [:],
                totalCount:   0,
                inferenceMs:  0,
                totalMs:      0,
                success:      false,
                errorMessage: "VisionCore: detect returned nil"
            )
            result = failed
            return failed
        }
        let mapped = mapResult(raw)
        result = mapped
        return mapped
    }

    // ── Map Obj-C result → Swift result ───────────────────────
    private func mapResult(_ raw: VCVisionResult) -> VisionResult {
        let detections = raw.detections.map { d in
            Detection(
                label:       d.label,
                confidence:  d.confidence,
                boundingBox: d.boundingBox,
                classId:     Int(d.classId)
            )
        }

        let countMap = raw.countMap.reduce(into: [String: Int]()) {
            $0[$1.key] = $1.value.intValue
        }

        return VisionResult(
            detections:   detections,
            countMap:     countMap,
            totalCount:   Int(raw.totalCount),
            inferenceMs:  raw.inferenceMs,
            totalMs:      raw.totalMs,
            success:      raw.success,
            errorMessage: raw.errorMessage
        )
    }
}
