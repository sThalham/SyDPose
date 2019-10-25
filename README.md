# SyDPose

SyDPose: Object Detection and Pose Estimation in Cluttered Real-World Depth Images Trained using only Synthetic Data
by Stefan Thalhammer, Timothy Patten, and Markus Vincze.

## Installation

1) Clone this repository.
2) No installation necessary.
3) Dependencies have to be installed manually.

# Training
RetNetPose/bin/train.py "dataset" /path/to/dataset-train

# Testing
RetNetPose/bin/evaluate.py "dataset" /path/to/dataset-val /path/to/training/model.h5 --convert-model

Utility scripts are provided for datasets LineMOD, Occlusion and Tless. No trained models are provided.

### Notes
* This repository requires Keras 2.2.4 or higher.

