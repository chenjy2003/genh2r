import os
import numpy as np
from numpy.typing import NDArray
from pybullet_utils.bullet_client import BulletClient
from dataclasses import dataclass
from omegaconf import MISSING
from xml.etree.ElementTree import parse, ElementTree
import copy
from typing import Tuple, List, Optional
import open3d as o3d
import code

from .body import Body, BodyConfig
from .utils.transform import pos_ros_quat_to_mat, se3_transform_pc

@dataclass
class ObjectConfig(BodyConfig):
    name: str = ""
    urdf_file: str = ""
    # collision
    collision_mask: Optional[int] = None
    # base
    use_fixed_base: bool = True
    base_position: Tuple[float] = (0., 0., 0.)
    base_orientation: Tuple[float] = (0., 0., 0., 1.)
    # links
    num_dofs: int = 6
    dof_position: Optional[Tuple[float]] = None
    dof_velocity: Tuple[float] = (0.0,)*9

    translation_max_force: Tuple[float] = (50.0,)*3
    translation_position_gain: Tuple[float] = (0.2,)*3
    translation_velocity_gain: Tuple[float] = (1.0,)*3
    rotation_max_force: Tuple[float] = (5.0,)*3
    rotation_position_gain: Tuple[float] = (0.2,)*3
    rotation_velocity_gain: Tuple[float] = (1.0,)*3

    dof_max_force: Tuple[float] = "${concat_tuples:${.translation_max_force},${.rotation_max_force}}"
    dof_position_gain: Tuple[float] = "${concat_tuples:${.translation_position_gain},${.rotation_position_gain}}"
    dof_velocity_gain: Tuple[float] = "${concat_tuples:${.translation_velocity_gain},${.rotation_velocity_gain}}"
    # concat_tuples is defined in body.py
    # the "." is necessary in nested configs, see https://github.com/omry/omegaconf/issues/1099#issuecomment-1624496194

    compute_target_real_displacement: bool = False

max_target_real_displacement = 0.

class Object(Body):
    def __init__(self, bullet_client: BulletClient, cfg: ObjectConfig):
        super().__init__(bullet_client, cfg)
        self.cfg: ObjectConfig

    def reset(self, name: str, path: str, pose: NDArray[np.float32], collision_mask: int):
        self.base_reset()
        # config
        self.cfg.name = name
        self.cfg.urdf_file = path
        self.cfg.collision_mask = collision_mask

        # process pose data
        self.pose = pose # (num_frames, 6)
        self.num_frames = self.pose.shape[0]
        self.frame = 0

        self.cfg.dof_position = self.pose[self.frame].tolist() # can not use tuple(self.pose[self.frame]), which keeps the type np.float32 for scalars
        self.load()
        self.set_dof_target(self.pose[self.frame])

        self.object_pc: Optional[NDArray[np.float64]] = None

    def pre_step(self):
        self.frame += 1
        self.frame = min(self.frame, self.num_frames-1)
        self.set_dof_target(self.pose[self.frame])
    
    def post_step(self):
        if self.cfg.compute_target_real_displacement:
            global max_target_real_displacement
            max_target_real_displacement = max(max_target_real_displacement, np.abs(self.pose[self.frame][:3]-self.get_joint_positions()[:3]).max())
            print(f"max_target_real_displacement={max_target_real_displacement}")

    def get_world_to_obj(self) -> NDArray[np.float64]:
        world_to_obj = self.get_link_pose(5)
        return world_to_obj
    
    def get_world_to_object_pc(self) -> NDArray[np.float64]:
        if self.object_pc is None:
            tree: ElementTree = parse(self.cfg.urdf_file)
            root = tree.getroot()
            collision_file_name: str = root.findall("link")[-1].find("collision").find("geometry").find("mesh").get("filename")
            collision_file_path = os.path.join(os.path.dirname(self.cfg.urdf_file), collision_file_name)
            object_mesh = o3d.io.read_triangle_mesh(collision_file_path)
            self.object_pc = np.array(object_mesh.vertices)
        world_to_object = self.get_world_to_obj()
        world_to_object_pc = se3_transform_pc(world_to_object, self.object_pc)
        return world_to_object_pc

@dataclass
class ObjectsConfig:
    collision_masks: Tuple[int] = (2**2, 2**3, 2**4, 2**5, 2**6, 2**7) # allow up to 6 objects
    collision_mask_release: int = -1-2**1

class Objects:
    def __init__(self, bullet_client: BulletClient, cfg_objects: ObjectsConfig, cfg_object: ObjectConfig):
        self.bullet_client = bullet_client
        self.cfg_objects = cfg_objects
        self.cfg_object = cfg_object
        self.objects: List[Object] = []
    
    def reset(self, names: List[str], paths: List[str], grasp_id: int, pose: NDArray[np.float32]):
        for obj in self.objects:
            del obj
        self.objects: List[Object] = []
        self.grasp_id = grasp_id
        for idx, (name, path) in enumerate(zip(names, paths)):
            new_object = Object(self.bullet_client, copy.deepcopy(self.cfg_object))
            new_object.reset(name, path, pose[idx], self.cfg_objects.collision_masks[idx])
            self.objects.append(new_object)
            if idx == grasp_id:
                self.target_object = new_object
        self.released = False

    def pre_step(self):
        for obj in self.objects:
            obj.pre_step()

    def post_step(self):
        for obj in self.objects:
            obj.post_step()
        
    def release(self):
        self.target_object.set_collision_mask(self.cfg_objects.collision_mask_release)
        self.target_object.cfg.dof_max_force = (0.,)*6
        self.target_object.set_dof_target(self.target_object.pose[self.target_object.frame]) # refresh the controller setting
        self.released = True

def debug():
    from omegaconf import OmegaConf
    import pybullet
    import time
    from env.utils import load_scene_data
    objects_cfg = OmegaConf.structured(ObjectsConfig)
    object_cfg = OmegaConf.structured(ObjectConfig)
    bullet_client = BulletClient(connection_mode=pybullet.GUI) # or pybullet.DIRECT
    objects = Objects(bullet_client, objects_cfg, object_cfg)
    scene_data = load_scene_data(0)
    objects.reset(scene_data["object_names"], scene_data["object_paths"], scene_data["object_grasp_id"], scene_data["object_poses"])
    code.interact(local=dict(globals(), **locals()))
    while True:
        print(f"frame {objects.target_object.frame}")
        objects.pre_step()
        bullet_client.stepSimulation()
        objects.post_step()
        # time.sleep(0.01)

if __name__ == "__main__":
    debug()

"""
DISPLAY="localhost:11.0" python -m env.objects

CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=32 env.panda.IK_solver=PyKDL setup=s0 split=train policy=chomp chomp.wait_time=13 env.object.compute_target_real_displacement=True
"""