//
//  VisionCameraManager.swift
//  VisionCore
//
//  Created by Ahmad on 29/03/2026.
//

import AVFoundation
import CoreVideo
import Observation
import UIKit

@Observable
public final class VisionCameraManager: NSObject {

    // Latest detection result
    public private(set) var latestResult: VisionResult?
    public private(set) var isRunning = false
    public private(set) var frameSize: CGSize = .zero
    public var error: String?

    // Internal
    private let session         = AVCaptureSession()
    private let videoOutput     = AVCaptureVideoDataOutput()
    private let sessionQueue    = DispatchQueue(label: "com.visioncore.camera")
    private let processingQueue = DispatchQueue(label: "com.visioncore.processing")
    private let detector: VisionDetector

    public var captureSession: AVCaptureSession { session }

    // ── Init ──────────────────────────────────────────────────
    public init(detector: VisionDetector) {
        self.detector = detector
        super.init()
        setupNotifications()

    }

    // ── Start ─────────────────────────────────────────────────
    public func start() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.configureSession()
            self.session.startRunning()
            DispatchQueue.main.async { self.isRunning = true }
        }
    }
    
    private func setupNotifications() {
        // Camera session interruptions
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(sessionWasInterrupted),
            name: .AVCaptureSessionWasInterrupted,
            object: session
        )
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(sessionInterruptionEnded),
            name: .AVCaptureSessionInterruptionEnded,
            object: session
        )

        // Use didEnterBackground instead of willResignActive
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(appDidEnterBackground),
            name: UIApplication.didEnterBackgroundNotification,
            object: nil
        )
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(appDidBecomeActive),
            name: UIApplication.didBecomeActiveNotification,
            object: nil
        )
    }



    // ── Stop ──────────────────────────────────────────────────
    public func stop() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.session.stopRunning()
            DispatchQueue.main.async { self.isRunning = false }
        }
    }

    // ── Configure session ─────────────────────────────────────
    private func configureSession() {
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720

        guard let device = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: .video,
            position: .back
        ),
        let input = try? AVCaptureDeviceInput(device: device),
        session.canAddInput(input) else {
            DispatchQueue.main.async { self.error = "Camera not available" }
            session.commitConfiguration()
            return
        }
        session.addInput(input)

        videoOutput.setSampleBufferDelegate(self, queue: processingQueue)
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]

        guard session.canAddOutput(videoOutput) else {
            session.commitConfiguration()
            return
        }
        session.addOutput(videoOutput)

        if let connection = videoOutput.connection(with: .video) {
            if connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90
            }
        }

        // ── Limit frame rate to match inference speed ─────────────
        // Saves battery and reduces thermal load
        try? device.lockForConfiguration()
        device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 3)
        device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 3)
        device.unlockForConfiguration()

        session.commitConfiguration()
    }
    
    @objc private func appDidEnterBackground() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.session.stopRunning()
            DispatchQueue.main.async { self.isRunning = false }
        }
    }

    @objc private func appDidBecomeActive() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            if !self.session.isRunning {
                self.session.startRunning()
                DispatchQueue.main.async { self.isRunning = true }
            }
        }
    }

    @objc private func sessionWasInterrupted(notification: NSNotification) {
        DispatchQueue.main.async {
            self.isRunning = false
        }
    }

    @objc private func sessionInterruptionEnded(notification: NSNotification) {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.session.startRunning()
            DispatchQueue.main.async { self.isRunning = true }
        }
    }

}

// ── AVCaptureVideoDataOutputSampleBufferDelegate ──────────────────────────────
extension VisionCameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {

    public func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection)
    {
        guard detector.isReady else { return }

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }

        if frameSize == .zero {
            let w = CVPixelBufferGetWidth(pixelBuffer)
            let h = CVPixelBufferGetHeight(pixelBuffer)
            DispatchQueue.main.async {
                self.frameSize = CGSize(width: w, height: h)
            }
        }

        let result = detector.detect(pixelBuffer: pixelBuffer)

        DispatchQueue.main.async { [weak self] in
            self?.latestResult = result
        }
    }
}
