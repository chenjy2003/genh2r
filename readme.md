# GenH2R

GenH2R is the official code for the following CVPR 2024 paper:

**GenH2R: Learning Generalizable Human-to-Robot Handover via Scalable Simulation, Demonstration, and Imitation**

Zifan Wang*, Junyu Chen*, Ziqing Chen, Pengwei Xie, Rui Chen, Li Yi

IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024

[ [website](https://genh2r.github.io/) ] [ [arXiv](https://arxiv.org/abs/2401.00929) ] [ [video](https://www.youtube.com/watch?v=BbphK5QlS1Y) ] 

## Introduction

GenH2R is a framework for learning generalizable vision-based human-to-robot (H2R) handover skills. The goal is to equip robots with the ability to reliably receive objects with unseen geometry handed over by humans in various complex trajectories.

We acquire such generalizability by learning H2R handover at scale with a comprehensive solution including procedural simulation assets creation, automated demonstration generation, and effective imitation learning. We leverage large-scale 3D model repositories, dexterous grasp generation methods, and curve-based 3D animation to create an H2R handover simulation environment named GenH2R-Sim, surpassing the number of scenes in existing simulators by three orders of magnitude. We further introduce a distillation-friendly demonstration generation method that automatically generates a million high-quality demonstrations suitable for learning. Finally, we present a 4D imitation learning method augmented by a future forecasting objective to distill demonstrations into a visuo-motor handover policy.

## Release News and Updates

Building upon [handover-sim](https://github.com/NVlabs/handover-sim), [GA-DDPG](https://github.com/liruiw/GA-DDPG) and [OMG-Planner](https://github.com/liruiw/OMG-Planner), our original codebase is a bit bulky. For better readability and extensibility, we decide to refactor our codebase to provide a simplified version. 

- `2024.06.18` We have released the evaluation scripts and the pre-trained models.

We are actively cleaning the code for simulation scene construction, demonstration generation and policy training, and will release as soon as possible.

## Usage

### Clone Repository
``` bash
git clone --recursive git@github.com:chenjy2003/genh2r.git
```

### Create Python Environment
``` bash
conda create -n genh2r python=3.10
conda activate genh2r
pip install -r requirements.txt
```

### Install Third Party Packages
#### PyKDL
``` bash
cd env/third_party/orocos_kinematics_dynamics
sudo apt-get update
sudo apt-get install libeigen3-dev libcppunit-dev
cd orocos_kdl
mkdir build
cd build
cmake .. -DENABLE_TESTS:BOOL=ON
make
sudo make install
make check
cd ../../python_orocos_kdl
mkdir build
cd build
ROS_PYTHON_VERSION=3.10 cmake ..
make
sudo make install
cp PyKDL.so $CONDA_PREFIX/lib/python3.10/site-packages/
## test
python3 ../tests/PyKDLtest.py
```
#### PointNet++
``` bash
cd third_party/Pointnet2_PyTorch
pip install pointnet2_ops_lib/.
```
### Prepare Data
#### DexYCB (TODO: add download command)
Download [`dex-ycb-cache-20220323.tar.gz`](https://drive.google.com/uc?export=download&id=1Jqe2iqI7inoEdE3BL4vEs25eT5M7aUHd) and `ycb_grasps.tar.gz` to `env/data/tmp`, then run
``` bash
cd env/data/tmp
tar -xvf dex-ycb-cache-20220323.tar.gz
mkdir ycb_grasps
tar -xvf ycb_grasps.tar.gz -C ycb_grasps
cd ../../..
python -m env.tools.process_dexycb
```

The processed 1000 scenes will be in `data/scene/00/00`, from `data/scene/00/00/00/00000000.npz` to `data/scene/00/00/09/00000999.npz`.

### Evaluate Pre-trained Models
We use [ray](https://github.com/ray-project/ray) for parallel evaluation in order to support larger test set. One can feel free to adjust `CUDA_VISIBLE_DEVICES` (the GPUs to use) and `num_runners` (the total number of runners) according to the local machine, without changing the evaluation results.

We observed that the evaluation results can be a bit different on different devices, which unfortunately originates in some third-party packages. Our evaluation is done on `NVIDIA GeForce RTX 3090`.
#### on DexYCB Test Set (Simultaneous)
``` bash
CUDA_VISIBLE_DEVICES=0 python -m evaluate \
    setup=s0 split=test num_runners=16 \
    policy=pointnet2 pointnet2.processor.flow_frame_num=3 pointnet2.model.obj_pose_pred_frame_num=3 \
    pointnet2.model.pretrained_dir=${model_dir} \
    pointnet2.model.pretrained_source=handoversim
```
#### on DexYCB Test Set (Sequential)
``` bash
CUDA_VISIBLE_DEVICES=0 python -m evaluate \
    setup=s0 split=test num_runners=16 \
    policy=pointnet2 pointnet2.processor.flow_frame_num=3 pointnet2.model.obj_pose_pred_frame_num=3 \
    pointnet2.model.pretrained_dir=${model_dir} \
    pointnet2.model.pretrained_source=handoversim \
    pointnet2.wait_time=3
```
### Visualization
To visualize the handover process, one can have two options:
#### Use `pybullet` GUI
``` bash
CUDA_VISIBLE_DEVICES=0 python -m evaluate \
    scene_ids=[209,214,219] use_ray=False \
    env.visualize=True \
    policy=pointnet2 pointnet2.processor.flow_frame_num=3 pointnet2.model.obj_pose_pred_frame_num=3 \
    pointnet2.model.pretrained_dir=/share1/haoran/HRI/generalizable_handover/output/t450_bc_omg_replan5_landmark_smooth0.08_use_hand_flow3_pred3_wd0.0001_pred0.5_300w_no_accum_3 \
    pointnet2.model.pretrained_source=handoversim
```
#### Record Egocentric Video from the Wrist-mounted Camera or Record Video from Third Person View
``` bash

```
## Acknowledgements
This repo is built based on [handover-sim](https://github.com/NVlabs/handover-sim), [GA-DDPG](https://github.com/liruiw/GA-DDPG), [OMG-Planner](https://github.com/liruiw/OMG-Planner), [acornym](https://github.com/NVlabs/acronym), [orocos_kinematics_dynamics](https://github.com/orocos/orocos_kinematics_dynamics), and [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch). We sincerely appreciate their contributions to open source.

## Citation
If GenH2R is useful or relevant to your research, please kindly recognize our contributions by citing our paper:
```
@article{wang2024genh2r,
  title={GenH2R: Learning Generalizable Human-to-Robot Handover via Scalable Simulation, Demonstration, and Imitation},
  author={Wang, Zifan and Chen, Junyu and Chen, Ziqing and Xie, Pengwei and Chen, Rui and Yi, Li},
  journal={arXiv preprint arXiv:2401.00929},
  year={2024}
}
```