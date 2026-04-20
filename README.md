# VehicleNet Vehicle Detection

## Overview
Brief description of the project and problem.

## Setup & Running the App
1. Install dependencies: `pip install gradio ultralytics huggingface_hub opencv-python`
2. Run: `python website.py`
3. Or open in Google Colab: [link]

## Theoretical Design

VehicleNet uses **VehicleNet-Y26s** by Perception365, a fine-tuned variant of YOLO26s from Ultralytics. The model belongs to the YOLO (You Only Look Once) family of real-time object detectors — an architecture lineage that reframes object detection as a **single regression problem** rather than a multi-stage pipeline.

### The YOLO Detection Paradigm

Traditional object detectors (e.g. R-CNN, HOG+SVM) used sliding-window or region-proposal pipelines that evaluated thousands of candidate regions per image. YOLOv1 (Redmon et al., 2015) replaced this with a single convolutional forward pass over the full image:

1. The image is divided into an **S×S grid** (7×7 in the original paper).
2. Each grid cell predicts **B bounding boxes** — each described by `(x, y, w, h, confidence)` — and **C conditional class probabilities**.
3. The output is a single tensor of shape `S × S × (B·5 + C)`.
4. **Confidence** is defined as `Pr(Object) × IoU(pred, truth)` — encoding both object presence and localization quality.

The model is trained with a five-term composite loss function:
- **Center coordinate loss** — weighted by λ_coord = 5
- **Width/height loss** — uses square roots of w and h for scale invariance
- **Objectness confidence loss** (cells containing objects)
- **No-object confidence loss** — weighted by λ_noobj = 0.5 to prevent empty-cell dominance
- **Classification loss** — cross-entropy over class probabilities

All layers use **Leaky ReLU** activation: `φ(x) = x if x > 0, else 0.1x`.

### YOLO26s Architecture (Modern Improvements)

YOLO26s is the latest generation in the Ultralytics YOLO family, incorporating decades of improvements over the v1/v2 baseline:

| Feature | YOLOv1 | YOLO26s |
|---|---|---|
| Anchor prediction | Direct coordinates via FC layers | **Anchor-free** detection head |
| Normalization | None (uses dropout) | **Batch normalization** on all conv layers |
| Feature scale | Single scale, 7×7 output | **Multi-scale FPN neck** for large & small objects |
| Input resolution | Fixed 448×448 | Flexible multi-scale |
| Backbone | GoogLeNet-inspired (24 conv) | Optimized lightweight backbone |

The `s` (small) suffix denotes a reduced-capacity variant optimized for **speed and low-latency inference**, making it deployable on edge devices such as dashcams and traffic cameras.

### VehicleNet-Y26s Model Specifications

The base YOLO26s weights were fine-tuned on the **UVH-26-MV Dataset** released by the Indian Institute of Science (IISc), Bangalore — a dataset designed for dense, heterogeneous Indian traffic conditions across **14 vehicle categories**: Hatchback, Sedan, SUV, MUV, Bus, Truck, Three-wheeler, Two-wheeler, LCV, Mini-bus, Tempo-traveller, Bicycle, Van, and Others.

| Property | Value |
|---|---|
| Parameters | 9.47 million |
| Layers | 122 |
| Compute | 20.6 GFLOPs |
| mAP@50 | 0.727 |
| mAP@50:95 | 0.643 |
| Precision | 0.681 |
| Recall | 0.690 |
| Training epochs | 60 (best checkpoint at epoch 40, early stopping) |
| Training hardware | 2× NVIDIA Tesla T4 GPUs, batch size 80 |

The model performs strongest on structurally distinct classes (Bus, Truck, Two-wheeler) and shows expected confusion between visually similar car subtypes (Sedan, Hatchback, SUV). No additional fine-tuning was performed — the pretrained VehicleNet-Y26s checkpoint from HuggingFace was used directly, as it was already purpose-built for this vehicle classification task.

## Software Architecture

The application is built with **Gradio** for the user interface and **Ultralytics YOLO** for inference, designed to run on Google Colab with a T4 GPU.

### High-Level Architecture

```
User (browser) → Gradio Interface → predict_image() / predict_video() → YOLO26s inference → Annotated output → Gradio display
```

### Tensor Encoding & Decoding

All tensor encoding and decoding is handled internally by the Ultralytics `model.predict()` API:

**Input encoding:**
1. User uploads an image (returned as a PIL Image) or video (returned as a file path).
2. Ultralytics resizes the input to the model's native resolution (640×640), normalizes pixel values to [0, 1], and casts to a `float32` tensor of shape `[1, 3, H, W]` (batch × channels × height × width).
3. A single forward pass produces output tensors encoding predicted bounding box coordinates, objectness scores, and class probability distributions across the detection grid.

**Output decoding:**
1. **Non-Maximum Suppression (NMS)** is applied using the user-supplied IoU threshold to remove duplicate detections.
2. Detections below the confidence threshold are discarded.
3. `results[0].plot()` renders bounding boxes and labels onto the frame as a BGR NumPy array.
4. For images, the array is converted BGR→RGB and returned to `gr.Image`. For video, frames are written directly to a `.webm` file via `cv2.VideoWriter` (VP9 codec) and returned to `gr.Video`.

### Image Detection Pipeline

```python
def predict_image(img, conf_threshold, iou_threshold, filter_text):
    # 1. Build predict args with user thresholds
    # 2. Map filter text → class IDs via model.names
    # 3. model.predict() handles tensor encoding + forward pass + NMS
    # 4. results[0].plot() → BGR array → convert to RGB → return to Gradio
```

### Video Detection Pipeline

```python
def predict_video(video_path, conf_threshold, iou_threshold, filter_text):
    # 1. OpenCV VideoCapture reads frames (BGR format)
    # 2. Each frame passed to model.predict() — same tensor pipeline as image
    # 3. Annotated BGR frames written to .webm via cv2.VideoWriter (VP9)
    # 4. Output file path returned to gr.Video component
```

### Class Filtering

The text filter maps user-provided label strings to integer class IDs using `model.names` (a dictionary of `{class_id: class_name}` stored in the model). Matched IDs are passed as the `classes` parameter to `model.predict()`, which restricts detections before NMS runs — cleaner and faster than post-filtering.

```python
target_ids = [
    cls_id for cls_id, cls_name in model.names.items()
    if any(term in cls_name.lower() for term in search_terms)
]
predict_args["classes"] = target_ids
```

### Gradio Component Mapping

| Gradio Component | Data Type | Role |
|---|---|---|
| `gr.Image` | PIL Image | Image upload input |
| `gr.Video` | File path | Video upload input / annotated output |
| `gr.Slider` (×2) | float | Confidence and IoU thresholds |
| `gr.Textbox` | str | Class name filter |
| `gr.Tabs` | — | Separates image and video workflows |

## Screenshots
[Add a few screenshots of the Gradio app with detections]

## References
- VehicleNet-Y26s: https://huggingface.co/Perception365/VehicleNet-Y26s
- UVH-26-MV Dataset: IISc Bangalore
- Ultralytics YOLO: https://github.com/ultralytics/ultralytics

## Rubric
- Signup on time ✔️
- Problem formulation ✔️
- Neural network models ✔️
- AI integration
- Github repo quality
- Presentation slide quality
- Video presentation quality
