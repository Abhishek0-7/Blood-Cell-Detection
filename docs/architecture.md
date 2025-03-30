# Blood Cell Detection System Architecture

## Overview
The blood cell detection system uses a YOLOv8 model trained to detect and classify three types of blood cells:
- Red Blood Cells (RBCs)
- White Blood Cells (WBCs)
- Platelets

## Components

### 1. Data Pipeline
- **Input**: Microscopic blood cell images (JPEG/PNG)
- **Preprocessing**:
  - Image resizing (640x640)
  - Data augmentation (flips, rotations, brightness adjustments)
  - Annotation conversion (XML to YOLO format)

### 2. Model Architecture
- **Base Model**: YOLOv8n (nano version)
- **Modifications**:
  - Output layer adjusted for 3 classes
  - Input size: 640x640 pixels
  - Anchor boxes optimized for blood cells

### 3. Training Process
- **Epochs**: 50
- **Batch Size**: 16
- **Optimizer**: Adam
- **Learning Rate**: 0.001 with cosine annealing
- **Data Split**:
  - 70% Training
  - 15% Validation
  - 15% Testing

### 4. Inference Pipeline
1. Image preprocessing
2. Model inference
3. Non-max suppression
4. Results visualization
5. Statistical analysis

### 5. Web Interface
- Built with Gradio
- Features:
  - Image upload
  - Confidence threshold adjustment
  - Detection visualization
  - Statistical charts
  - Performance metrics display

## Performance Metrics
| Class     | Precision | Recall | mAP50 | mAP50-95 |
|-----------|-----------|--------|-------|----------|
| RBC       | 0.73      | 0.91   | 0.91  | 0.68     |
| WBC       | 0.37      | 1.00   | 0.42  | 0.35     |
| Platelets | 0.41      | 0.70   | 0.47  | 0.28     |
| All       | 0.50      | 0.87   | 0.60  | 0.43     |
