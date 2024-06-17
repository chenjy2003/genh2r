We have restructured our codebase to provide a simplified version that supports more extensions and improves readability. Our previous model was trained on a modified version of HandoverSim, where we transformed the RL scheme to the IL scheme. The model trained on HandoverSim can be easily loaded into our codebase.



We now provide a skeleton of our codebase and the evaluation step. More about the demonstration generation and training will be released as soon as possible.

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

### Evaluate Models
