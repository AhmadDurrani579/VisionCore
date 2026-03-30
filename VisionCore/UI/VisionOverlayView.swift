//
//  VisionOverlayView.swift
//  VisionCore
//
//  Created by Ahmad on 29/03/2026.
//

import SwiftUI

public struct VisionOverlayView: View {

    public let result: VisionResult?
    
    // Animate box transitions smoothly
    @State private var animatedDetections: [Detection] = []

    public init(result: VisionResult?) {
        self.result = result
    }

    public var body: some View {
        GeometryReader { geo in
            Canvas { context, size in
                for detection in animatedDetections {
                    draw(detection: detection, in: &context, size: size)
                }
            }
        }
        .onChange(of: result?.totalCount) { _, _ in
            withAnimation(.easeInOut(duration: 0.15)) {
                animatedDetections = result?.detections ?? []
            }
        }
    }


    // ── Draw single detection ─────────────────────────────────
    private func draw(
        detection: Detection,
        in context: inout GraphicsContext,
        size: CGSize)
    {
        let rect = CGRect(
            x:      detection.boundingBox.minX * size.width,
            y:      detection.boundingBox.minY * size.height,
            width:  detection.boundingBox.width  * size.width,
            height: detection.boundingBox.height * size.height
        )

        let color = colorForLabel(detection.label)

        // Bounding box
        let boxPath = Path(roundedRect: rect, cornerRadius: 4)
        context.stroke(boxPath, with: .color(color), lineWidth: 2)

        // Label
        let label    = "\(detection.label)  \(detection.confidencePercent)"
        let fontSize: CGFloat = 13
        let padding:  CGFloat = 6
        let textSize = labelSize(label, fontSize: fontSize)

        let labelRect = CGRect(
            x:      rect.minX,
            y:      rect.minY - textSize.height - padding * 2,
            width:  textSize.width  + padding * 2,
            height: textSize.height + padding * 2
        )

        let bgPath = Path(roundedRect: labelRect, cornerRadius: 4)
        context.fill(bgPath, with: .color(color))

        context.draw(
            Text(label)
                .font(.system(size: fontSize, weight: .semibold))
                .foregroundColor(.white),
            at: CGPoint(x: labelRect.minX + padding, y: labelRect.minY + padding),
            anchor: .topLeading
        )
    }

    // ── Colour per class ──────────────────────────────────────
    private func colorForLabel(_ label: String) -> Color {
        switch label {
        case "person":
            return .green
        case "car", "truck", "bus", "motorcycle",
             "bicycle", "airplane", "train", "boat":
            return .blue
        case "laptop", "tv", "cell phone", "keyboard",
             "mouse", "remote", "microwave", "oven",
             "toaster", "refrigerator":
            return .purple
        case "bottle", "wine glass", "cup", "fork",
             "knife", "spoon", "bowl", "banana", "apple",
             "sandwich", "orange", "broccoli", "carrot",
             "hot dog", "pizza", "donut", "cake":
            return .orange
        case "chair", "couch", "bed", "dining table", "toilet":
            return .cyan
        case "bird", "cat", "dog", "horse", "sheep",
             "cow", "elephant", "bear", "zebra", "giraffe":
            return Color(red: 1.0, green: 0.8, blue: 0.0)
        default:
            return .red
        }
    }

    // ── Label size ────────────────────────────────────────────
    private func labelSize(_ text: String, fontSize: CGFloat) -> CGSize {
        let attrs: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: fontSize, weight: .semibold)
        ]
        return (text as NSString).size(withAttributes: attrs)
    }
}
