import os
import cv2
import glob
import random
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm

def convert_xml_to_yolo(xml_file, img_width, img_height):
    """Convert XML annotation to YOLO format"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    annotations = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name == 'RBC':
            class_id = 0
        elif class_name == 'WBC':
            class_id = 1
        elif class_name == 'Platelets':
            class_id = 2
        else:
            continue
            
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return annotations

def prepare_dataset(dataset_path, output_dir):
    """Prepare YOLO formatted dataset"""
    os.makedirs(os.path.join(output_dir, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images/val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images/test'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels/val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels/test'), exist_ok=True)

    # Find all XML files
    xml_files = glob.glob(os.path.join(dataset_path, '**/Annotations/*.xml'), recursive=True)
    random.shuffle(xml_files)
    
    # Split dataset (70% train, 15% val, 15% test)
    train_files = xml_files[:int(0.7*len(xml_files))]
    val_files = xml_files[int(0.7*len(xml_files)):int(0.85*len(xml_files))]
    test_files = xml_files[int(0.85*len(xml_files)):]
    
    for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        for xml_file in tqdm(files, desc=f"Processing {split} data"):
            img_file = xml_file.replace('Annotations', 'JPEGImages').replace('.xml', '.jpg')
            if not os.path.exists(img_file):
                continue
                
            img = cv2.imread(img_file)
            h, w = img.shape[:2]
            
            # Convert annotations
            annotations = convert_xml_to_yolo(xml_file, w, h)
            
            # Save files
            base_name = os.path.basename(img_file).split('.')[0]
            dest_img = os.path.join(output_dir, f'images/{split}/{base_name}.jpg')
            dest_label = os.path.join(output_dir, f'labels/{split}/{base_name}.txt')
            
            shutil.copy(img_file, dest_img)
            with open(dest_label, 'w') as f:
                f.write('\n'.join(annotations))
    
    # Create data.yaml
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(f"""train: {os.path.join(output_dir, 'images/train')}
val: {os.path.join(output_dir, 'images/val')}
test: {os.path.join(output_dir, 'images/test')}

nc: 3
names: ['RBC', 'WBC', 'Platelets']""")

if __name__ == "__main__":
    prepare_dataset('path/to/raw/dataset', 'bccd_yolo')
