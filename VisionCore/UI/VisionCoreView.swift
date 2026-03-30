//
//  VisionCoreView.swift
//  VisionCore
//
//  Created by Ahmad on 29/03/2026.
//

import SwiftUI

public struct VisionCoreView: View {

    // Configuration
    public var modelName:            String
    public var title:                String
    public var confidenceThreshold:  Float
    public var maxDetections:        Int
    public var showStats:            Bool
    public var showCount:            Bool

    // Callbacks
    public var onDetection: ((VisionResult) -> Void)?

    // Internal state
    @State private var detector:      VisionDetector?
    @State private var cameraManager: VisionCameraManager?

    public init(
        modelName:           String  = "rfdetr_nano",
        title:               String  = "RF-DETR Nano",
        confidenceThreshold: Float   = 0.35,
        maxDetections:       Int     = 50,
        showStats:           Bool    = true,
        showCount:           Bool    = true,
        onDetection:         ((VisionResult) -> Void)? = nil
    ) {
        self.modelName            = modelName
        self.title                = title
        self.confidenceThreshold  = confidenceThreshold
        self.maxDetections        = maxDetections
        self.showStats            = showStats
        self.showCount            = showCount
        self.onDetection          = onDetection
    }

    public var body: some View {
        ZStack {
            if let manager = cameraManager {
                // ── 1. Camera feed ────────────────────────────
                VisionPreviewView(session: manager.captureSession)
                    .ignoresSafeArea()

                // ── 2. Detection overlay ──────────────────────
                VisionOverlayView(result: manager.latestResult)
                    .ignoresSafeArea()

                // ── 3. HUD ────────────────────────────────────
                VStack {
                    if showStats {
                        statsBar(manager.latestResult)
                    }
                    Spacer()
                    if showCount {
                        countBar(manager.latestResult)
                    }
                }

            } else {
                // ── Loading state ─────────────────────────────
                ZStack {
                    Color.black.ignoresSafeArea()
                    VStack(spacing: 16) {
                        ProgressView()
                            .tint(.white)
                            .scaleEffect(1.5)
                        Text("Loading VisionCore...")
                            .foregroundStyle(.white)
                            .font(.system(size: 16, weight: .medium))
                        Text("First launch may take a moment")
                            .foregroundStyle(.white.opacity(0.6))
                            .font(.system(size: 13))
                    }
                }
            }
        }
        .background(Color.black)
        .onAppear { setup() }
        .onDisappear { cameraManager?.stop() }
        .onChange(of: cameraManager?.latestResult?.totalCount) { _, _ in
            if let result = cameraManager?.latestResult {
                onDetection?(result)
            }
        }
    }

    // ── Setup ─────────────────────────────────────────────────
    private func setup() {
        let det = VisionDetector(modelName: modelName)
        det.confidenceThreshold = confidenceThreshold
        det.maxDetections       = maxDetections
        detector                = det

        let manager = VisionCameraManager(detector: det)
        cameraManager           = manager
        manager.start()
    }

    // ── Stats bar ─────────────────────────────────────────────
    @ViewBuilder
    private func statsBar(_ result: VisionResult?) -> some View {
        HStack(spacing: 16) {
            Text(title)
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(.white)
            Spacer()
            if let result {
                statPill(value: String(format: "%.0f ms", result.inferenceMs))
                statPill(value: String(format: "%.0f FPS", result.fps))
                statPill(value: "\(result.totalCount) objects")
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial)
    }

    // ── Count bar ─────────────────────────────────────────────
    @ViewBuilder
    private func countBar(_ result: VisionResult?) -> some View {
        if let result, !result.countMap.isEmpty {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(
                        result.countMap.sorted { $0.value > $1.value },
                        id: \.key
                    ) { label, count in
                        HStack(spacing: 4) {
                            Text(label)
                                .font(.system(size: 12, weight: .medium))
                            Text("×\(count)")
                                .font(.system(size: 12, weight: .bold))
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(.ultraThinMaterial)
                        .clipShape(Capsule())
                        .foregroundStyle(.white)
                    }
                }
                .padding(.horizontal, 16)
            }
            .padding(.bottom, 32)
        }
    }

    // ── Stat pill ─────────────────────────────────────────────
    private func statPill(value: String) -> some View {
        Text(value)
            .font(.system(size: 13, weight: .bold))
            .foregroundStyle(.white)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Color.white.opacity(0.15))
            .clipShape(Capsule())
    }
}
