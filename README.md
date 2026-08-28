# Panoptic Perception Engine for Autonomous Navigation

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![CUDA](https://img.shields.io/badge/Hardware-Edge_GPU-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-AGPL-A31F34?style=for-the-badge)

A real-time, Multi-Task Learning (MTL) vision system designed for autonomous vehicle edge-deployment. This engine utilizes a shared ResNet backbone and a Feature Pyramid Network (FPN) to simultaneously power object detection (YOLO) and semantic segmentation (U-Net). 

Instead of running decoupled, latency-heavy networks, this architecture processes a single forward pass to understand both discrete obstacles (vehicles, pedestrians) and continuous environments (drivable areas). A deterministic spatial routing algorithm maps these outputs together to calculate real-time collision risks and trigger zero-latency brake alerts.

## Table of Contents
- [Architecture](#architecture)
- [Core Features](#core-features)
- [Target Performance Benchmarks](#target-performance-benchmarks)
- [Installation & Usage](#installation--usage)

---

## Architecture

The system mimics the "HydraNet" paradigm, decoupling feature extraction from task-specific heads to optimize VRAM and maximize FPS on edge hardware.

```mermaid
graph TD
    A["Input Video Frame<br>[1, 3, 384, 640]"] --> B["Shared Backbone<br>(ResNet-18/34)"]
    
    subgraph Multi-Scale Feature Fusion
        B --> C["FPN Neck<br>Channel Equalization & Upsampling"]
        C --> D["P4 (Deep/Semantic)"]
        C --> E["P3 (Intermediate)"]
        C --> F["P2 (Shallow/Spatial)"]
    end

    D --> G["YOLO Head (Large Objects)"]
    E --> G["YOLO Head (Medium Objects)"]
    F --> G["YOLO Head (Small Objects)"]
    
    F --> H["U-Net Decoder<br>(Transposed Convs)"]

    G --> I["Bounding Boxes<br>(x, y, w, h, class)"]
    H --> J["Binary Segmentation Mask<br>(Drivable Area)"]

    I --> K{"Dynamic Spatial Routing<br>Do bbox (x,y) coords intersect<br>the segmentation mask?"}
    J --> K

    K -- Yes --> L["⚠️ ASYNC BRAKE ALERT"]
    K -- No --> M["Standard Visualization"]

```

---

## Core Features

* **Multi-Task Feature Extractor:** A unified PyTorch `nn.Module` replacing decoupled models. The ResNet backbone branches into an FPN neck, balancing gradient flows for simultaneous YOLO bounding box regression and U-Net pixel-dense segmentation.
* **Dynamic Spatial Routing (Collision Logic):** A highly optimized $O(N)$ algorithmic bridge that extracts the bottom-center coordinates of bounding boxes and calculates matrix intersections against the semantic mask to determine physical drivable-area intrusions.
* **Asynchronous Processing Pipeline:** A thread-safe Python multiprocessing architecture that decouples video I/O from GPU tensor inference, ensuring no dropped frames and maximizing VRAM utilization.
* **Custom Joint Loss Function:** Stabilizes backpropagation by mathematically balancing the severe scale variance between massive continuous pixel blocks (the road) and sparse, tiny pixel clusters (distant traffic lights).

---

## Target Performance Benchmarks

*Designed to hit industry baselines (BDD100K equivalents).*

| Metric | Target | Notes |
| --- | --- | --- |
| **Inference Latency** | 30 - 41 FPS | Evaluated on edge-equivalent GPU hardware |
| **Detection (mAP50)** | > 78.0% | Prioritizing high recall for pedestrian classes |
| **Segmentation (mIoU)** | > 91.0% | Drivable area classification via P2 FPN layer |

---

## Installation & Usage

**1. Clone the repository**

```bash
git clone [https://github.com/Asmit159/Panoptic-Perception-Engine.git](https://github.com/Asmit159/Panoptic-Perception-Engine.git)
cd Panoptic-Perception-Engine

```

**2. Install dependencies**

```bash
pip install -r requirements.txt

```

**3. Run the live inference engine**

```bash
python engine.py --source data/dashcam_sample.mp4 --conf_thres 0.45

```

---

**Author:** Asmit Mandal
