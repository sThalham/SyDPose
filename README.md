# SyDPose

SyDPose: Object Detection and Pose Estimation in Cluttered Real-World Depth Images Trained using only Synthetic Data
by Stefan Thalhammer, Timothy Patten, and Markus Vincze.

@INPROCEEDINGS{SyDPose, author={S. {Thalhammer} and T. {Patten} and M. {Vincze}}, booktitle={2019 International Conference on 3D Vision (3DV)}, title={SyDPose: Object Detection and Pose Estimation in Cluttered Real-World Depth Images Trained using Only Synthetic Data}, year={2019}, volume={}, number={}, pages={106-115},} 

## Version requirements

* Keras >= 2.2.4
* blender <= 2.79b
* Tensorflow >= 1.9.0
* OpenCV <= 3.4.0.19
* OpenEXR
* pyfastnoisesimd
* prgressbar2
* h5py
* open3d
* ...

# Training
1) render synthetic training data using the scripts "render_<dataset>.blend" in repo data\_creation.
   - that requires changing paths to meshes
2) annotate data to create training dataset using "annotate\_syn\_<dataset>.py" in repo data\_creation.
   - the 3D bounding boxes used to establish 2D-3D correspondences are hard coded. "annotate\_syn\_tless.py gives an example on how to load object dimensions from the files provided by [BOP](https://bop.felk.cvut.cz/home/). A good source for object meshes, test and val data.
3) train using "SyDPose/bin/train.py <dataset> </path/to/training\_data>".

# Testing
SyDPose/bin/evaluate.py <dataset> </path/to/dataset\_val> </path/to/training/model.h5> --convert-model

Utility scripts are provided for datasets LineMOD, Occlusion and Tless. No trained models are provided.

### Notes
* branch "master" uses a shared network for control point regression. If "separate" regression layers shall be used things have to be uncommented... contact me for guidelines, since the network's training becomes fragile.
* augmentation for training moved to online augmentation
