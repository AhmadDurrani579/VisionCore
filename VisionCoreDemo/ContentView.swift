//
//  ContentView.swift
//  VisionCoreDemo
//
//  Created by Ahmad on 27/03/2026.
//

import SwiftUI
import VisionCore

struct ContentView: View {
    var body: some View {
        VisionCoreView(
            modelName: "rfdetr_nano",
            title: "VisionCore Demo",
            confidenceThreshold: 0.35,
            showStats: true,
            showCount: true,
            onDetection: { result in
                print("Detected \(result.totalCount) objects")
                for det in result.detections {
                    print("  → \(det.label) \(det.confidencePercent)")
                }
            }
        )
    }
}
#Preview {
    ContentView()
}
    
