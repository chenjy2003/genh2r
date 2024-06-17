We have restructured our codebase to provide a simplified version that supports more extensions and improves readability. Our previous model was trained on a modified version of HandoverSim, where we transformed the RL scheme to the IL scheme. The model trained on HandoverSim can be easily loaded into our codebase.



We now provide a skeleton of our codebase and the evaluation step. More about the demonstration generation and training will be released as soon as possible.

```
git clone https://github.com/chenjy2003/genh2r --recursive
```

1.Create Python Environment

```
conda create -n genh2r python=3.10
conda activate genh2r
```

2.Install Required Packages

```
pip install pybullet
pip install numpy
# pip install yacs
pip install omegaconf
pip install scipy
pip install ipdb
pip install psutil
pip install lxml
pip install ray
pip install transforms3d
pip install easydict
pip install opencv-python
pip install imageio[ffmpeg]
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install tqdm
pip install mayavi
pip install PyQt5
pip install open3d
pip install h5py
pip install bezier
pip install wandb
```

or 

```
pip install -r requirements.txt
```





3.Initialize Third-Party Libraries

```
git submodule init
git submodule update
```



4.Install PyKDL

```
cd env/third_party/orocos_kinematics_dynamics
git submodule update --init
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



5.Install PointNet++

```
cd third_party/Pointnet2_PyTorch
pip install pointnet2_ops_lib/.
```



6.Download Data

1).dexycb

````
Download `dex-ycb-cache-20220323.tar.gz` and `ycb_grasps.tar.gz` to `env/data/tmp`, then run
``` bash
cd env/data/tmp
tar -xvf dex-ycb-cache-20220323.tar.gz
mkdir ycb_grasps
tar -xvf ycb_grasps.tar.gz -C ycb_grasps
cd ../../..
python -m env.tools.process_dexycb
```
The processed 1000 scenes will be in `data/scene/00/00`, from `data/scene/00/00/00/00000000.npz` to `data/scene/00/00/09/00000999.npz`.
````



2).assets,Download(in a separate environment with numpy=1.23.0)

 mano_v1_2 from [MANO website](http://mano.is.tue.mpg.de/) to env/data/tmp, then run

```
cd env/data/tmp
unzip mano_v1_2.zip
cd ../../..
```

