# Blood Cell Dataset

This directory should contain the blood cell dataset. Due to size limitations, the actual dataset files are not included in this repository.

## Recommended Datasets

1. **BCCD Dataset (Primary)**
   - Source: [GitHub](https://github.com/Shenggan/BCCD_Dataset)
   - Download: `git clone https://github.com/Shenggan/BCCD_Dataset.git`
   - Contains 364 annotated blood cell images

2. **Kaggle Alternative**
   - Source: [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/blood-cells)
   - Download instructions:
     ```bash
     kaggle datasets download -d paultimothymooney/blood-cells
     unzip blood-cells.zip
     ```


Run the preprocessing script to convert to YOLO format:
```bash
python ../scripts/data_preprocessing.py --input_path ./BCCD_Dataset/BCCD --output_path ./processed
