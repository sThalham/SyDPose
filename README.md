# SyDPose

SyDPose: Object Detection and Pose Estimation in Cluttered Real-World Depth Images Trained using only Synthetic Data
by Stefan Thalhammer, Timothy Patten, and Markus Vincze.

## Library version requirements

* Keras >= 2.2.4
* blender <= 2.79b
* Tensorflow >= 1.9.0
* OpenCV <= 3.4.0.19

# Training
1) render synthetic training data using the scripts "render_<dataset>.blend" in repo data\_creation.
2) annotate data to create training dataset using "annotate\_syn\_<dataset>.py" in repo data\_creation.
3) train using "RetNetPose/bin/train.py <dataset> </path/to/training\_data>"

# Testing
RetNetPose/bin/evaluate.py "dataset" /path/to/dataset-val /path/to/training/model.h5 --convert-model

Utility scripts are provided for datasets LineMOD, Occlusion and Tless. No trained models are provided.

### Notes
- branch "master" uses one sub network for control point regression
- branch "separate" uses separate regression layers. Fine-tuning of the loss weights in dependance of the amount of epochs to train is adamant.

