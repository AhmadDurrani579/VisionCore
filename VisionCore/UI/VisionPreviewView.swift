//
//  VisionPreviewView.swift
//  VisionCore
//
//  Created by Ahmad on 29/03/2026.
//

import SwiftUI
import AVFoundation

public struct VisionPreviewView: UIViewRepresentable {

    let session: AVCaptureSession

    public init(session: AVCaptureSession) {
        self.session = session
    }

    public func makeUIView(context: Context) -> VisionPreviewUIView {
        let view = VisionPreviewUIView()
        view.session = session
        return view
    }

    public func updateUIView(_ uiView: VisionPreviewUIView, context: Context) {}
}

// ── VisionPreviewUIView ───────────────────────────────────────────────────────
public final class VisionPreviewUIView: UIView {

    public override class var layerClass: AnyClass {
        AVCaptureVideoPreviewLayer.self
    }

    var previewLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }

    var session: AVCaptureSession? {
        didSet {
            previewLayer.session      = session
            previewLayer.videoGravity = .resizeAspectFill
        }
    }

    public override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer.frame = bounds
    }
}
