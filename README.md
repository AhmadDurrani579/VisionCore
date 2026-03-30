# VisionCore

> **Open-source cross-platform object detection SDK for iOS — powered by RF-DETR Nano and ExecuTorch XNNPACK**

VisionCore is a production-grade, native iOS SDK for real-time on-device object detection. Built with a **pure C++ inference engine**, it runs RF-DETR Nano using Meta's ExecuTorch framework with no cloud dependency, no React Native, and no compromise on architecture.

Add live object detection to any iOS app in **3 lines of Swift**.

```swift
VisionCoreView(
    modelName: "rfdetr_nano",
    confidenceThreshold: 0.35,
    showStats: true
)
```

---

## Demo

| Street Scene | Apple Store | Food Detection |
|:---:|:---:|:---:|
| ![Street](docs/screenshots/street.png) | ![Apple Store](docs/screenshots/apple_store.png) | ![Pizza](docs/screenshots/pizza.png) |
| bus 92.4%, motorcycle 82.7% | person ×5, laptop ×4 | pizza, cell phone, dining table |

---

## Architecture

VisionCore uses a pure C++ core that is fully reusable across iOS and Android.

```
┌─────────────────────────────────────┐
│         SwiftUI Demo App            │
│      VisionCoreView (3 lines)       │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│         VisionDetector.swift        │
│        @Observable public API       │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│      VisionCoreWrapper.mm           │
│      Obj-C++ iOS Bridge             │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐  ← Shared C++ Core
│         VisionEngine.cpp            │    (iOS + Android)
│   ExecuTorch XNNPACK Inference      │
├─────────────────────────────────────┤
│         Preprocessor.cpp            │
│  BGRA→RGB + ImageNet Normalization  │
├─────────────────────────────────────┤
│            NMS.cpp                  │
│   Non-Maximum Suppression           │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│         rfdetr_nano.pte             │
│   RF-DETR Nano — COCO 80 classes    │
│   ExecuTorch XNNPACK format         │
└─────────────────────────────────────┘
```

---

## Features

- **Pure C++ engine** — `VisionEngine.cpp`, `Preprocessor.cpp`, `NMS.cpp` are platform-agnostic
- **ExecuTorch XNNPACK backend** — 190MB memory vs 720MB with ONNX Runtime + CoreML EP
- **RF-DETR Nano model** — transformer-based, 80 COCO classes, higher mAP than YOLO Nano
- **Auto input size detection** — reads model metadata at runtime, works with any ExecuTorch model
- **ImageNet normalization** — applied automatically in the C++ preprocessor
- **Warmup inference** — first frame is fast, no cold-start lag
- **SwiftUI overlay** — colour-coded bounding boxes with class-specific colours
- **Detection callback** — `onDetection` closure fires on every frame
- **Custom title** — pass any title string to the stats bar
- **Count bar** — live object counts grouped by class
- **Camera interruption handling** — auto-resumes after screenshot, background, lock screen

---

## Performance

Benchmarked on **iPhone 15 (A16 Bionic)**:

| Stage | Time |
|-------|------|
| Preprocessing (BGRA→RGB + resize + norm) | ~27ms |
| Inference (ExecuTorch XNNPACK forward) | ~350ms |
| Postprocessing (parse + NMS) | ~0ms |
| **Total** | **~377ms → 3 FPS** |

Expected on newer devices:

| Device | Chip | Inference | FPS |
|--------|------|-----------|-----|
| iPhone 15 | A16 Bionic | ~350ms | 3 FPS |
| iPhone 16 | A18 | ~130ms | 7 FPS |
| iPhone 17 Pro | A19 Pro | ~101ms | 9 FPS |

> Memory usage: **190MB** (vs ~720MB with ONNX Runtime + CoreML EP)

---

## Requirements

- iOS 17.0+
- Xcode 16.0+
- iPhone with arm64 (no simulator support for ExecuTorch XNNPACK)

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/AhmadDurrani579/VisionCore.git
cd VisionCore
```

### 2. Download the model

The `.pte` model file is too large for GitHub. Download it from Google Drive:

| Model | Size | Backend | Download |
|-------|------|---------|----------|
| rfdetr_nano.pte | 106MB | ExecuTorch XNNPACK | [Google Drive](https://drive.google.com/drive/folders/1SHzv8wbI2_R9zq5vR0S_qFyanOnwOYTx?usp=share_link) |

Place the downloaded file at:
```
VisionCoreDemo/Model/rfdetr_nano.pte
```

### 3. Open in Xcode

```bash
open VisionCore.xcodeproj
```

### 4. Run on device

Select your iPhone as the build target and press `Cmd+R`.

> **Note:** ExecuTorch XNNPACK requires a physical device. Simulator is not supported.

---

## Usage

### Basic — one line

```swift
import VisionCore

struct ContentView: View {
    var body: some View {
        VisionCoreView()
    }
}
```

### With configuration

```swift
VisionCoreView(
    modelName: "rfdetr_nano",
    title: "My App",
    confidenceThreshold: 0.35,
    maxDetections: 20,
    showStats: true,
    showCount: true
)
```

### With detection callback

```swift
VisionCoreView(
    modelName: "rfdetr_nano",
    onDetection: { result in
        print("Detected \(result.totalCount) objects")
        
        for detection in result.detections {
            print("\(detection.label) — \(detection.confidencePercent)")
            print("Box: \(detection.boundingBox)")
        }
        
        print("Inference: \(result.inferenceMs)ms")
        print("FPS: \(result.fps)")
    }
)
```

### VisionResult structure

```swift
public struct VisionResult {
    public let detections:   [Detection]     // all detections this frame
    public let countMap:     [String: Int]   // label → count
    public let totalCount:   Int             // total objects detected
    public let inferenceMs:  Float           // inference time in ms
    public let totalMs:      Float           // full pipeline time in ms
    public var fps:          Float           // frames per second
    public let success:      Bool
}

public struct Detection {
    public let label:        String          // e.g. "person"
    public let confidence:   Float           // 0.0 – 1.0
    public let boundingBox:  CGRect          // normalised 0–1
    public let classId:      Int             // COCO class index
}
```

---

## Project Structure

```
VisionCore/
├── VisionCore/
│   ├── Core/                    ← Pure C++ (shared iOS + Android)
│   │   ├── VisionEngine.cpp     ← ExecuTorch inference engine
│   │   ├── VisionEngine.hpp
│   │   ├── Preprocessor.cpp     ← BGRA→RGB resize + ImageNet norm
│   │   ├── Preprocessor.hpp
│   │   ├── NMS.cpp              ← Non-maximum suppression
│   │   ├── NMS.hpp
│   │   ├── VisionConfig.hpp     ← Config struct
│   │   └── VisionResult.hpp     ← Result + Detection structs
│   │
│   ├── iOS/                     ← iOS bridge
│   │   ├── VisionCoreWrapper.h
│   │   └── VisionCoreWrapper.mm ← Obj-C++ bridge
│   │
│   ├── Camera/
│   │   └── VisionCameraManager.swift
│   │
│   ├── UI/
│   │   ├── VisionCoreView.swift
│   │   ├── VisionOverlayView.swift
│   │   └── VisionPreviewView.swift
│   │
│   └── VisionDetector.swift     ← Public Swift API
│
├── VisionCoreDemo/
│   ├── ContentView.swift        ← 3 lines of code
│   └── Model/                   ← Place .pte file here
│
└── ExecutorchLib.xcframework    ← Pre-built ExecuTorch binary
```

---

## Model

VisionCore uses **RF-DETR Nano** — a real-time transformer-based object detector from Roboflow.

| Property | Value |
|----------|-------|
| Architecture | RF-DETR (DINOv2 backbone) |
| Format | ExecuTorch XNNPACK (.pte) |
| Input | 384×384 |
| Classes | 80 COCO |
| mAP50-95 | 48.4 |
| Parameters | ~29M |

### Why RF-DETR over YOLO?

| | RF-DETR Nano | YOLO Nano |
|--|--|--|
| Architecture | Transformer | CNN |
| mAP50-95 | **48.4** | ~37 |
| Accuracy | Higher | Lower |
| Speed | 3 FPS (iPhone 15) | ~10 FPS |
| Small objects | Better | Worse |

RF-DETR trades speed for accuracy. If you need higher FPS, swap the model file for YOLO26N — no code changes needed.

---

## Roadmap

- [ ] **Android JNI bridge** — same C++ core, Kotlin API
- [ ] **CoreML backend** — Neural Engine acceleration (~20 FPS on iPhone 15)
- [ ] **YOLO26N support** — 10+ FPS option
- [ ] **Instance segmentation** — C++ mask decoder
- [ ] **Custom model support** — any ExecuTorch `.pte` model

---

## Acknowledgements

- [Roboflow](https://roboflow.com) — RF-DETR model architecture
- [Software Mansion](https://github.com/software-mansion/react-native-executorch) — ExecutorchLib.xcframework and pre-exported .pte model
- [Meta ExecuTorch](https://github.com/pytorch/executorch) — on-device inference framework

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Ahmad Durrani**  
Senior iOS Engineer · London  
[GitHub](https://github.com/AhmadDurrani579) · [LinkedIn](https://linkedin.com/in/ahmaddurrani)
