# USV-Obstacle-Detection
Real-time Obstacle detection system for Unmanned Surface Vessels(USVs)

Dataset Link - https://www.kaggle.com/datasets/k0vidsharma/water-land-edge-segmentation-mods

Main Dataset - https://vision.fe.uni-lj.si/public/mods/

## Description

#### Goals
This is my ongoing B.Tech Project. My goals for the project are:
1. To explore and experiment existing obstacle detection approaches in Indian scenarios.
2. To create an annotated dataset for Indian waters to boost the research in this field.
3. To develop a robust obstacle detection system that can work on an Embedded GPU.

## Results
#### Segmentation
![Segmentation Results](Results/Results_Segmentation.png)
#### Detection
![Detection Results](Results/Result_Detection.png)
#### Detection Example
![YOLOv8 Dectection Example](Results/YOLOv8_Example.png)
## Repository Structure
```
.
├── BTP Half Yearly PPT.pptx
├── BTP Half Yearly Thesis.pdf
├── LICENSE
├── README.md
├── Results
│   ├── Result_Detection.png
│   ├── Results_Segmentation.png
│   └── YOLOv8_Example.png
├── Segmentation Code
│   ├── Other Models
│   │   ├── dataset.py
│   │   ├── train.py
│   │   └── utils.py
│   └── U-Net
│       ├── model.py
│       └── training_unet.ipynb
├── create_masks.py
└── metadata.py
```
