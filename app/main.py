import gradio as gr
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import os

# Initialize model
model = YOLO('../models/best.pt')
class_names = ["RBC", "WBC", "Platelets"]
class_colors = {"RBC": (0, 255, 0), "WBC": (255, 0, 0), "Platelets": (0, 0, 255)}

def detect_blood_cells(input_image, conf_threshold=0.25):
    """Detect blood cells in an image and return results"""
    # Convert input to numpy array
    img = np.array(input_image) if isinstance(input_image, Image.Image) else input_image
    
    # Run inference
    results = model(img, conf=conf_threshold)
    result = results[0]
    
    # Create annotated image
    annotated_img = result.plot()
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    # Process detections
    detections = []
    counts = {cls: 0 for cls in class_names}
    confidences = {cls: [] for cls in class_names}
    
    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls = int(box.cls[0])
        class_name = class_names[cls]
        conf = float(box.conf[0])
        
        counts[class_name] += 1
        confidences[class_name].append(conf)
        detections.append([i+1, class_name, f"{conf:.2f}", f"[{x1}, {y1}, {x2}, {y2}]"])
    
    # Calculate averages and create DataFrames
    avg_conf = {cls: np.mean(confs) if confs else 0 for cls, confs in confidences.items()}
    
    detections_df = pd.DataFrame(detections, 
                               columns=["#", "Class", "Confidence", "Bounding Box"])
    
    stats_df = pd.DataFrame({
        "Class": class_names,
        "Count": [counts[cls] for cls in class_names],
        "Avg Confidence": [f"{avg_conf[cls]:.2f}" for cls in class_names]
    })
    
    # Generate pie chart
    chart_img = None
    if sum(counts.values()) > 0:
        plt.figure(figsize=(5,5))
        plt.pie([counts[cls] for cls in class_names], 
                labels=class_names, 
                colors=[tuple(np.array(class_colors[cls])/255) for cls in class_names],
                autopct='%1.1f%%')
        plt.title("Cell Distribution")
        chart_path = "cell_distribution.png"
        plt.savefig(chart_path, bbox_inches='tight')
        plt.close()
        chart_img = Image.open(chart_path)
        os.remove(chart_path)
    
    # Create summary
    summary = f"Detected {sum(counts.values())} cells:\n"
    for cls in class_names:
        summary += f"- {counts[cls]} {cls} (avg confidence: {avg_conf[cls]:.2f})\n"
    
    return annotated_img, detections_df, stats_df, summary, chart_img

# Create Gradio interface
with gr.Blocks(title="Blood Cell Detection") as app:
    gr.Markdown("# Blood Cell Detection Dashboard")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Blood Cell Image", type="pil")
            conf_slider = gr.Slider(0.1, 0.9, value=0.25, label="Confidence Threshold")
            submit_btn = gr.Button("Analyze", variant="primary")
            
            # Add example images
            example_dir = os.path.join(os.path.dirname(__file__), "static/examples")
            if os.path.exists(example_dir):
                example_images = [os.path.join(example_dir, f) 
                                for f in os.listdir(example_dir) 
                                if f.endswith(('.jpg', '.jpeg', '.png'))]
                if example_images:
                    gr.Examples(examples=example_images, inputs=input_image)
        
        with gr.Column():
            output_image = gr.Image(label="Detection Results")
            output_chart = gr.Image(label="Cell Distribution")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Detection Details")
            output_detections = gr.Dataframe(
                headers=["#", "Class", "Confidence", "Bounding Box"],
                datatype=["number", "str", "str", "str"]
            )
        
        with gr.Column():
            gr.Markdown("### Detection Statistics")
            output_stats = gr.Dataframe(
                headers=["Class", "Count", "Avg Confidence"]
            )
    
    output_summary = gr.Textbox(label="Analysis Summary")
    
    submit_btn.click(
        fn=detect_blood_cells,
        inputs=[input_image, conf_slider],
        outputs=[output_image, output_detections, output_stats, output_summary, output_chart]
    )

if __name__ == "__main__":
    app.launch()
