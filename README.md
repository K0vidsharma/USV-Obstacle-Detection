# USV-Obstacle-Detection
Real-time Obstacle detection system for Unmanned Surface Vessels(USVs)

Dataset Link - https://www.kaggle.com/datasets/k0vidsharma/water-land-edge-segmentation-mods

Main Dataset - https://vision.fe.uni-lj.si/public/mods/

Kaggle Notebook Link - https://www.kaggle.com/code/k0vidsharma/water-land-segment/notebook
### Description
This is my ongoing B.Tech Project. My goals for the project are:
* Assess the current research and opportunities for Unmanned Surface Vessels(USV).
* Create an IMU Synchronization system to calibrate the camera and other sensors.
* Create a real-time obstacle detection system that can work onboard using an embedded GPU.
### Repo Structure

* `create_masks.py`: Creates masks from the water-edge coordinates given in the original dataset.
* `Water-edge-segmentation.ipynb`: Ipython notebook for experimentation.
* `model.py`: Code for U-Net with Batch Normalization.
* Btech Project Quarterly PPT
* Btech Project Quarterly Thesis

Model code inspired from: https://youtu.be/IHq1t7NxS8k?si=m5u-aZETj-SCd5t3
