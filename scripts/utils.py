import cv2
import numpy as np
from PIL import Image

def preprocess_image(image):
    """Preprocess image for model inference"""
    if isinstance(image, Image.Image):
        img = np.array(image)
    elif isinstance(image, str):
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image
    
    if img.shape[-1] == 4:  # Remove alpha channel if exists
        img = img[:, :, :3]
    
    return img

def visualize_detections(image, boxes, class_names, class_colors):
    """Visualize detections on image"""
    img = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cls = int(box[5])
        conf = box[4]
        
        color = class_colors[class_names[cls]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[cls]}: {conf:.2f}"
        cv2.putText(img, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img

def calculate_metrics(detections, ground_truth):
    """Calculate precision, recall and other metrics"""
    # Implementation of metrics calculation
    pass
