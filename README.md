# 🩸 Blood Cell Detection using YOLOv8

A deep learning-powered system that automatically detects and classifies blood cells (RBCs, WBCs, and Platelets) in microscopic images with bounding boxes and statistical analysis.

## 🌟 Key Features

- **Accurate Detection**: Identifies 3 types of blood cells with 85%+ accuracy for RBCs
- **Real-time Analysis**: Processes images in under 100ms on GPU
- **Interactive Interface**: User-friendly Gradio web app
- **Detailed Analytics**: Provides counts, confidence scores, and distribution charts
- **Adjustable Sensitivity**: Customizable confidence threshold (0.1-0.9)

## 🗃️ Dataset Information

### Source Dataset
We use the **BCCD (Blood Cell Count Dataset)** available from:
- Primary Source: [GitHub Repository](https://github.com/Shenggan/BCCD_Dataset)
- Alternative Source: [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/blood-cells)

### Dataset Specifications
| Characteristic | Details |
|---------------|---------|
| Total Images | 364 |
| Annotation Format | XML (converted to YOLO) |
| Classes | RBC, WBC, Platelets |
| Image Resolution | 640x480px |
| License | CC-BY 4.0 |

### Preprocessing Steps
1. **Conversion**: XML → YOLO format
2. **Splitting**:
   - 70% Training (255 images)
   - 15% Validation (55 images)
   - 15% Testing (54 images)
3. **Augmentation**:
   - Horizontal flips
   - Random rotations
   - Brightness adjustments


## 🧠 How It Works

This system uses a fine-tuned YOLOv8 (You Only Look Once) object detection model trained on 500+ annotated blood cell images. The pipeline:

1. **Input**: Microscopic blood cell image (JPEG/PNG)
2. **Processing**:
   - Image normalization
   - YOLOv8 inference
   - Non-max suppression
3. **Output**:
   - Annotated image with bounding boxes
   - Cell counts and confidence metrics
   - Interactive pie chart of cell distribution

## 🛠️ Tech Stack

**Core Components**:
- **Deep Learning**: YOLOv8 (PyTorch backend)
- **Web Interface**: Gradio
- **Image Processing**: OpenCV, PIL

**Supporting Libraries**:
- Data Augmentation: Albumentations
- Visualization: Matplotlib
- Data Handling: Pandas, NumPy

**System Requirements**:
- Python 3.8+
- GPU recommended (CUDA compatible)
- 4GB+ RAM
- 2GB+ disk space

## 📊 Performance Metrics

| Class       | Precision | Recall | mAP50 | mAP50-95 |
|-------------|-----------|--------|-------|----------|
| RBC         | 0.73      | 0.91   | 0.91  | 0.68     |
| WBC         | 0.37      | 1.00   | 0.42  | 0.35     |
| Platelets   | 0.41      | 0.70   | 0.47  | 0.28     |
| **Overall** | 0.50      | 0.87   | 0.60  | 0.43     |

## Additional Notes

1. **Dataset Preparation**: 
   - Primary dataset: [BCCD Dataset](https://github.com/Shenggan/BCCD_Dataset)
   - Alternative: [Kaggle Blood Cells](https://www.kaggle.com/datasets/paultimothymooney/blood-cells)
   - Preprocessing script: `python scripts/data_preprocessing.py`

2. **Deployment Options**:
   - Temporary: ngrok (`python app/main.py` with ngrok tunnel)
   - Permanent:
     1. Heroku (recommended for beginners)
     2. AWS EC2 (for GPU acceleration)
     3. Docker containerization

3. **Environment Variables**: Create `.env` file with:
           # MODEL_PATH=models/best.pt
           # CONFIDENCE_THRESHOLD=0.25

4. **Testing**: 
- Test cases available in `/tests`
- Run with: `python -m pytest tests/`

## 🤝 Contributing
Contributions are welcome! Please follow these steps:
1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**Contribution Guidelines**:
- Document new functions with docstrings
- Include unit tests for new features
- Maintain consistent code style (PEP 8)

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

## 📬 Contact
Email - abhishek261kumar@gmail.com  
Project Link: [https://github.com/Abhishek0-7/Blood-Cell-Detection](https://github.com/Abhishek0-7/Blood-Cell-Detection)

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Abhishek0-7/Blood-Cell-Detection.git
cd blood-cell-detection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r app/requirements.txt

# Convert and prepare dataset
python scripts/data_preprocessing.py --input_path ./raw_data --output_path ./processed_data
