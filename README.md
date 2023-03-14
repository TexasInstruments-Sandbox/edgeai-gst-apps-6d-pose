# Edge AI GStreamer Apps
> Repository to host GStreamer based Edge AI applications for TI devices

## 6D Pose Estimation
Object pose estimation aims to estimate the 3D orientation and 3D translation of objects in a given environment. It is useful in a wide range of applications like robotic manipulation for bin-picking, motion planning, and human-robot interaction task such as learning from demonstration.

## YOLO-6D-Pose Based Multi-Object 6D Pose Estimation Models

* YOLO-6D-Pose a multi-object 6D pose estimation framework which is an enhancement of the popular YOLOX object detector. The network is end-to-end trainable and detects each object along with its pose from a single RGB image without any additional post-processing . It uses a certain parameterization of the 6D pose which is decoded to get the final translation and rotation. These models achieve competitive accuracy without further refinement or any intermediate representations. For further details refer to this [paper]().

* YOLO-6D-Pose based models are supported as part of TI Deep Learning Library(TIDL) with full hardware acceleration. These models can be trained and exported following the instruction in this [repository](https://github.com/TexasInstruments/edgeai-yolox). 

* The exported models can be further compiled in edgeai-benchmark [repository](https://github.com/TexasInstruments/edgeai-benchmark) with the corresponding [configs](https://github.com/TexasInstruments/edgeai-benchmark/blob/master/configs/object_6d_pose_estimation.py)
