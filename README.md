# Real-Time Depth Anything V2 Point Cloud

![openFrameworks](https://img.shields.io/badge/openFrameworks-0.12.1-black?style=flat-square)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue?style=flat-square&logo=c%2B%2B)
![OS](https://img.shields.io/badge/OS-Linux-orange?style=flat-square&logo=linux&logoColor=black)
![ONNXRuntime](https://img.shields.io/badge/ONNXRuntime-1.18+-blue?style=flat-square&logo=onnx)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-red?style=flat-square)

A real-time application built with openFrameworks and ONNXRuntime, using the Depth Anything V2 model to perform monocular depth estimation on a standard webcam, generating an interactive 3D point cloud and a depth map simultaneously.

### Project Structure
```text
├── src/
│   ├── main.cpp
│   ├── ofApp.h
│   ├── ofApp.cpp
│   └── depth_anything.hpp
├── bin/
│   └── data/
│       ├── vitb_metric_indoor.onnx
│       └── vitb_metric_indoor.onnx.data
└── libs/
    └── onnxruntime/
```

### Downloads & Model

You can download the pre-exported Metric Vit-B model as a release ([SyntheticPointCloud Release v1.0](https://github.com/JHermosillaD/SyntheticPointCloud/releases/tag/v1.0)). Both files (`.onnx` and `.onnx.data`) must be placed in the `bin/data/` folder.

### Dependencies

- **openFrameworks** (v0.12.1+)
- **ONNXRuntime C++ API** (v1.18+) 
- **OpenCV** (v4.x)

### Credits

* **Base Model:** [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) by the @DepthAnything team.
* **ONNX Export Script:** [Depth-Anything-ONNX (v2.0.0)](https://github.com/fabio-sim/Depth-Anything-ONNX/releases/tag/v2.0.0) by @fabio-sim.
* **C++ Inference Header:** [Depths-CPP](https://github.com/Geekgineer/Depths-CPP) by @Geekgineer.
