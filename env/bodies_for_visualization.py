import os
import numpy as np
from numpy.typing import NDArray
from pybullet_utils.bullet_client import BulletClient
from dataclasses import dataclass
from typing import Tuple, Optional, Any
from omegaconf import MISSING
from scipy.spatial.transform import Rotation as Rt

from .body import Body, BodyConfig
from .utils.transform import mat_to_pos_ros_quat

@dataclass
class GraspConfig(BodyConfig):
    name: str = "grasp"
    urdf_file: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "assets", "grasp", "grasp_simplified.urdf")
    # collision
    collision_mask: int = 0
    # base
    use_fixed_base: bool = True
    base_position: Tuple[float] = MISSING
    base_orientation: Tuple[float] = MISSING
    # visual
    link_color: Tuple[float] = MISSING

def get_grasp_config(pose_mat: NDArray, color=[1., 0., 0., 1.]): # NDArray is not supported by OmegaConf
    pos, orn = mat_to_pos_ros_quat(pose_mat)
    grasp_config = GraspConfig(base_position=tuple(pos.tolist()), base_orientation=tuple(orn.tolist()), link_color=color)
    return grasp_config

class Grasp(Body):
    def __init__(self, bullet_client: BulletClient, cfg: GraspConfig):
        super().__init__(bullet_client, cfg)
        self.cfg: GraspConfig
        self.load()

@dataclass
class SphereConfig(BodyConfig):
    name: str = "sphere"
    urdf_file: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "assets", "sphere", "sphere.urdf")
    scale: float = MISSING
    # collision
    collision_mask: int = 0
    # base
    use_fixed_base: bool = True
    base_position: Tuple[float] = MISSING
    base_orientation: Tuple[float] = (0., 0., 0., 1.)
    # visual
    link_color: Tuple[float] = MISSING

class Sphere(Body):
    def __init__(self, bullet_client: BulletClient, cfg: SphereConfig):
        super().__init__(bullet_client, cfg)
        self.cfg: SphereConfig
        self.load()

def debug():
    from omegaconf import OmegaConf
    import pybullet
    import code
    grasp_cfg = OmegaConf.to_object(OmegaConf.structured(get_grasp_config(pose_mat=np.eye(4))))
    sphere_cfg = OmegaConf.to_object(OmegaConf.structured(SphereConfig(scale=1., base_position=(0., 0., 0.1), link_color=(1., 0., 0., 1.))))
    bullet_client = BulletClient(connection_mode=pybullet.GUI) # or pybullet.DIRECT
    grasp = Grasp(bullet_client, grasp_cfg)
    sphere = Sphere(bullet_client, sphere_cfg)
    code.interact(local=dict(globals(), **locals()))

if __name__ == "__main__":
    debug()

"""
python -m env.bodies_for_visualization

CUDA_VISIBLE_DEVICES=2 python -m evaluate evaluate.setup s0 evaluate.split train evaluate.use_ray False policy.name offline policy.offline.demo_dir data/demo/s0/train/pointnet2_simultaneous_dart/0 env.visualize True env.show_trajectory True
scene_id 5, step 68, status 1, reward 1.0, reached frame 6760, done frame 9184
"""