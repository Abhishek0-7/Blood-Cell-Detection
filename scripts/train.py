from ultralytics import YOLO
import torch
import argparse

def train_model(data_path, epochs=50, imgsz=640, batch=16, model_name='yolov8n.pt'):
    """Train YOLO model on blood cell dataset"""
    # Load model
    model = YOLO(model_name)
    
    # Train the model
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device='0' if torch.cuda.is_available() else 'cpu',
        patience=10,
        project='blood_cell_detection',
        name='train',
        exist_ok=True
    )
    
    # Export to ONNX
    model.export(format='onnx')
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Pretrained model')
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model_name=args.model
    )
