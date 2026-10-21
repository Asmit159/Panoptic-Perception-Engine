<div align="center">
    
# Panoptic Perception
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python 3.10+](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Albumentations](https://img.shields.io/badge/Albumentations-8A2BE2?style=for-the-badge)
![License: AGPL 3](https://img.shields.io/badge/License-AGPL_3-blue.svg?style=for-the-badge)
![Computer Vision](https://img.shields.io/badge/Computer_Vision-FF9E0F?style=for-the-badge)
![Autonomous Driving](https://img.shields.io/badge/Autonomous_Driving-00A98F?style=for-the-badge)
![Edge AI](https://img.shields.io/badge/Edge_AI-111111?style=for-the-badge)
![Torchvision](https://img.shields.io/badge/Torchvision-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Dataset: BDD100K](https://img.shields.io/badge/Dataset-BDD100K-8A2BE2?style=for-the-badge)
![TensorRT](https://img.shields.io/badge/TensorRT-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![NVIDIA Jetson](https://img.shields.io/badge/NVIDIA_Jetson-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

</div>

A high-performance, real-time perception Engine built from scratch in PyTorch for autonomous driving applications. By utilizing a shared CSPDarknet backbone, SPPF block, and decoupled task heads, the architecture simultaneously performs object detection and drivable area segmentation on the BDD100K dataset. The system is engineered strictly for edge deployment, prioritizing low latency and minimal VRAM consumption without relying on heavy, pre-packaged frameworks.


<div align="center">
  <video src="
https://github.com/user-attachments/assets/ce63f7f4-e550-4422-990a-41fca6893cf9" autoplay loop muted playsinline width="100%"></video>
</div>



---

## System Overview
Standard multi-task models often deploy parallel networks that duplicate feature extraction, leading to massive memory overhead and latency spikes. This engine resolves that bottleneck by processing the spatial environment through a single optimized backbone before branching into specialized heads. 

To guarantee real-time throughput and memory efficiency on edge hardware (e.g., NVIDIA Jetson architectures), the system implements several low-level optimizations:

* **Mixed Precision Operations:** Native `bfloat16` precision across forward passes and loss calculations prevents NaN gradient overflow and accelerates training.
* **Vectorized Post-Processing:** Hand-written, vectorized implementations for YOLO bounding box decoding, NMS handling, and IoU matching eliminate Python loop bottlenecks.
* **In-Memory Caching:** Custom dataloader pipeline caches 10,000+ BDD100K labels directly into system RAM to eliminate storage I/O latency during epochs.

---

## Neural Architecture
The network utilizes a shared feature extraction pipeline that feeds into specialized, decoupled heads for distinct panoptic tasks.

```mermaid
graph TD
    A["Input Tensor<br>640x384 RGB"] --> B("CSPDarknet Backbone<br>Spatial & Semantic Extraction")
    B --> C("SPPF Block<br>Spatial Pyramid Pooling - Fast")
    C --> D{"Feature Pyramid Network<br>Multi-Scale Aggregation"}

    subgraph Decoupled_Heads [Decoupled Heads]
        D --> E["YOLO-Style Detection Head<br>Strides: 8, 16, 32"]
        D --> F["ASPP Segmentation Head<br>Atrous Spatial Pyramid Pooling"]
    end

    E --> G["Bounding Boxes & Classes"]
    F --> H["Drivable Area Mask"]

    classDef coralHighlight fill:#F45B69,stroke:#0B2027,stroke-width:2px,color:#FFFFFF;
    classDef darkBlock fill:#0B2027,stroke:#70A9A1,stroke-width:2px,color:#F6F1D1;
    classDef tealBlock fill:#40798C,stroke:#0B2027,stroke-width:1.5px,color:#F6F1D1;
    classDef container fill:none,stroke:#70A9A1,stroke-width:2px,stroke-dasharray: 4 4;

    class A,G,H coralHighlight;
    class B,C,D darkBlock;
    class E,F tealBlock;
    class Decoupled_Heads container;

```

### Core Components

1. **CSPDarknet Backbone:** Cross Stage Partial network that effectively extracts spatial gradients and semantic features while cutting computational overhead in half compared to standard residual networks.
2. **SPPF (Spatial Pyramid Pooling - Fast):** Maximizes the receptive field by pooling features at multiple scales, ensuring the network can detect large objects in the immediate foreground and small objects in the distance.
3. **Feature Pyramid Network (FPN):** Aggregates the multi-scale contextual features from the backbone to feed the task-specific heads.
4. **Decoupled Heads:**
* **Detection:** A YOLO-style head utilizing Custom Focal Loss and Non-Maximum Suppression (NMS) to output bounding boxes across three stride levels.
* **Segmentation:** An ASPP block that processes the aggregated features to decode and project a binary drivable area mask.

---

## Regularization & Training Dynamics
To prevent structural overfitting, the pipeline introduces advanced regularization protocols:

* **Albumentations Pipeline:** Integrated spatial and color transformations including **Mosaic**, **MixUp**, and **Color Jittering**. This forces the model to learn physical object geometry rather than memorizing absolute screen coordinates, completely eliminating spatial hallucinations (e.g., detecting pedestrians in the sky).
* **Early Stopping Mechanics:** Implemented dynamic validation loss tracking with a patience threshold. This automatically halts the training loop upon plateau, securing the optimal checkpoint and preventing the overfitting.

---

## Performance Benchmarks & Validation
Evaluated directly on the BDD100K validation split using an NVIDIA A100 GPU at `bfloat16` precision.

<div align="center">
    
| Metric | Current Score |
| --- | --- |
| **Segmentation mIoU** | **81.88%** |
| **Segmentation Pixel Accuracy** | **97.11%** |
| **Detection F1/mAP** | **0.4283** |
| **Inference Latency** | **5.05 ms** | 
| **Throughput** | **198.21 FPS** | 
| **Peak VRAM Usage** | **2.37 GB** |

<img width="1370" height="1600" alt="WhatsApp Image 2026-10-21 at 10 48 48 AM (2)" src="https://github.com/user-attachments/assets/adfe975a-e9b4-4c00-a639-4af3f63a8380" />
<img width="814" height="286" alt="WhatsApp Image 2026-10-21 at 10 48 48 AM (3)" src="https://github.com/user-attachments/assets/8e2b5313-d6ba-4b91-89f8-b2ba356e14d3" />
<img width="850" height="470" alt="WhatsApp Image 2026-10-21 at 10 48 48 AM" src="https://github.com/user-attachments/assets/d612d502-3845-44b9-aa99-e7f01349e651" />
<img width="841" height="547" alt="WhatsApp Image 2026-10-21 at 10 48 48 AM (1)" src="https://github.com/user-attachments/assets/a550187e-1bb2-4d22-80ae-28667dc6005b" />

</div>
---

### Industry SOTA vs. Panoptic Engine Comparison

State-of-the-art (SOTA) object detection and segmentation frameworks often prioritize absolute mAP on leaderboards at the cost of operational deployment viability.

| Metric / Feature | Panoptic Engine (This Project) | Industry SOTA (e.g., YOLOv8x, Mask R-CNN) |
| --- | --- | --- |
| **Architecture** | Custom CSPDarknet + SPPF + FPN | Deep pre-trained transformers / Heavy CNNs |
| **Task Execution** | Simultaneous (Shared Backbone) | Often sequential or require multiple isolated models |
| **Peak Memory Overhead** | **~2.37 GB VRAM** | 8.0 GB to 16.0 GB VRAM |
| **Inference Latency** | **~5.05 ms** | 15.0 ms to 45.0+ ms |
| **Throughput** | **~200 FPS** | < 60 FPS (without TensorRT compilation) |
---
