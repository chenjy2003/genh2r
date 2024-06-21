# GenH2R

GenH2R is the official code for the following CVPR 2024 paper:

**GenH2R: Learning Generalizable Human-to-Robot Handover via Scalable Simulation, Demonstration, and Imitation**

Zifan Wang*, Junyu Chen*, Ziqing Chen, Pengwei Xie, Rui Chen, Li Yi

IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024

[ [website](https://genh2r.github.io/) ] [ [arXiv](https://arxiv.org/abs/2401.00929) ] [ [video](https://www.youtube.com/watch?v=BbphK5QlS1Y) ] 

![logo](assets/1_logo.gif)

## Introduction

GenH2R is a framework for learning generalizable vision-based human-to-robot (H2R) handover skills. The goal is to equip robots with the ability to reliably receive objects with unseen geometry handed over by humans in various complex trajectories.

We acquire such generalizability by learning H2R handover at scale with a comprehensive solution including procedural simulation assets creation, automated demonstration generation, and effective imitation learning. We leverage large-scale 3D model repositories, dexterous grasp generation methods, and curve-based 3D animation to create an H2R handover simulation environment named GenH2R-Sim, surpassing the number of scenes in existing simulators by three orders of magnitude. We further introduce a distillation-friendly demonstration generation method that automatically generates a million high-quality demonstrations suitable for learning. Finally, we present a 4D imitation learning method augmented by a future forecasting objective to distill demonstrations into a visuo-motor handover policy.

## Release News and Updates

Building upon [handover-sim](https://github.com/NVlabs/handover-sim), [GA-DDPG](https://github.com/liruiw/GA-DDPG) and [OMG-Planner](https://github.com/liruiw/OMG-Planner), our original codebase is a bit bulky. For better readability and extensibility, we decide to refactor our codebase to provide a simplified version. 

- `2024.06.20` We have released the evaluation scripts and the pre-trained models.

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
# install pytorch according to your cuda version (https://pytorch.org/get-started/previous-versions/)
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

### Install Third Party Packages
#### PyKDL
We highly recommand users to install this third-party package for robot kinematics. But inference and evaluation of pre-trained models can be done without this package, with `env.panda.IK_solver=pybullet` added to the following commands, and with slightly different results.
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
cd ../..
```
### Prepare Data
#### DexYCB
Download [`dex-ycb-cache-20220323.tar.gz`](https://drive.google.com/uc?export=download&id=1Jqe2iqI7inoEdE3BL4vEs25eT5M7aUHd) (from [handover-sim](https://github.com/NVlabs/handover-sim)) to `env/data/tmp`.

Download [`assets.tar.gz`](https://drive.google.com/file/d/1jLi23goHESWHMIud2wpNiQNdZ45dMQ49/view?usp=drive_link) (the object and hand models are from [handover-sim](https://github.com/NVlabs/handover-sim)) to `env/data`.

Then run
``` bash
cd env/data/tmp
tar -xvf dex-ycb-cache-20220323.tar.gz
cd ..
tar -xvf assets.tar.gz
cd ../..
python -m env.tools.process_dexycb
```

The processed 1000 scenes will be in `data/scene/00/00`, from `data/scene/00/00/00/00000000.npz` to `data/scene/00/00/09/00000999.npz`.

### Evaluate Pre-trained Models
Our pre-trained models can be downloaded [here](https://drive.google.com/drive/folders/1kbQR3xXJJp4rUZ-pytH7vTQVBNaJgM4O?usp=drive_link).

We use [ray](https://github.com/ray-project/ray) for parallel evaluation in order to support larger test set. One can feel free to adjust `CUDA_VISIBLE_DEVICES` (the GPUs to use) and `num_runners` (the total number of runners) according to the local machine, without changing the evaluation results.

We observed that the evaluation results can be slightly different on different devices, which unfortunately originates in some third-party packages. Our evaluation is done on `NVIDIA GeForce RTX 3090`.
#### on DexYCB Test Set (Simultaneous)
``` bash
CUDA_VISIBLE_DEVICES=0 python -m evaluate \
    setup=s0 split=test num_runners=16 \
    policy=pointnet2 pointnet2.processor.flow_frame_num=3 pointnet2.model.obj_pose_pred_frame_num=3 \
    pointnet2.model.pretrained_dir=${model_dir} \
    pointnet2.model.pretrained_source=handoversim
```
Here `model_dir` should be the path of the folder containing model parameters.
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
To use GUI, one can only start a single process, therefore parallel evaluation should be disabled by setting `use_ray=False`. Then GUI can be enabled by setting `env.visualize=True`. Note that one can have a finer control of which scenes to evaluate by setting `scene_ids` instead of `setup` and `split`. 
``` bash
CUDA_VISIBLE_DEVICES=0 python -m evaluate \
    scene_ids=[214,219] use_ray=False \
    env.visualize=True \
    policy=pointnet2 pointnet2.processor.flow_frame_num=3 pointnet2.model.obj_pose_pred_frame_num=3 \
    pointnet2.model.pretrained_dir=${model_dir} \
    pointnet2.model.pretrained_source=handoversim
```
#### Record Egocentric Video from the Wrist-mounted Camera or Record Video from Third Person View
To record videos, we need to set `demo_dir` (where to store the videos), `record_ego_video=True` and `record_third_person_video=True`.
``` bash
CUDA_VISIBLE_DEVICES=0 python -m evaluate \
    setup=s0 split=test num_runners=16 \
    policy=pointnet2 pointnet2.processor.flow_frame_num=3 pointnet2.model.obj_pose_pred_frame_num=3 \
    pointnet2.model.pretrained_dir=${model_dir} \
    pointnet2.model.pretrained_source=handoversim \
    demo_dir=data/tmp record_ego_video=True record_third_person_video=True
```
There is an argument `demo_structure` controlling how the demonstration data are arranged. If set to `hierarchical` by default, then the data will be stored in a hierarchical way, like `data/tmp/00/00/02/00000209_ego_rgb.mp4`. If set to `flat`, then all the data will be stored in the same folder `data/tmp`.

One can also adjust the position and orientation of the third person camera by setting `env.third_person_camera.pos` (default `[1.5,-0.1,1.8]`), `env.third_person_camera.target` (default `[0.6,-0.1,1.3]`), and `env.third_person_camera.up_vector` (default `[0.,0.,1.]`).

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
