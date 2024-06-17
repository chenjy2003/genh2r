import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple
import os
import open3d as o3d

from env.handover_env import Observation
from env.utils.scene import scene_id_to_dir
from .base_policy import BasePolicy
from .offline_policy_config import OfflinePolicyConfig

def get_o3d_pcd(points: NDArray[np.float64], color: Optional[Tuple[float]]=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if color is not None:
        pcd.paint_uniform_color(color)
    return pcd

def scene_id_to_demo_path(scene_id: int, demo_structure: str) -> str:
    scene_dir = scene_id_to_dir(scene_id, demo_structure)
    scene_path = os.path.join(scene_dir, f"{scene_id:08d}.npz")
    return scene_path

class OfflinePolicy(BasePolicy):
    def __init__(self, cfg: OfflinePolicyConfig):
        super().__init__(cfg)
        self.cfg: OfflinePolicyConfig
        self.data_dir = self.cfg.demo_dir
        assert self.data_dir is not None
        self.data_structure = self.cfg.demo_structure
        assert self.data_structure in ["flat", "hierarchical"]
        self.data_source = self.cfg.demo_source

    def reset(self, scene_id: int):
        self.base_reset()
        if self.data_source == "genh2r":
            self.data_path = os.path.join(self.data_dir, scene_id_to_demo_path(scene_id, self.data_structure))
        elif self.data_source == "handoversim":
            self.data_path = os.path.join(self.data_dir, f"{scene_id}.npz")
        else:
            raise NotImplementedError
        self.traj_data = np.load(self.data_path)
        self.num_steps = self.traj_data["num_steps"]
        if self.data_source == "genh2r":
            self.actions = self.traj_data["action"]
            self.world_to_target_grasps = self.traj_data["world_to_target_grasp"]
        elif self.data_source == "handoversim":
            self.actions = self.traj_data["expert_action"]
        else:
            raise NotImplementedError
        self.step = 0

    def plan(self, observation: Observation):
        if self.step == self.num_steps:
            action, action_type, reached, info = None, None, True, {}
        else:
            action = np.append(self.actions[self.step], 0.04)
            action_type, reached, info = "ego_cartesian", False, {"world_to_target_grasp": self.world_to_target_grasps[self.step]}
            if self.cfg.check_input_pcd:
                if f"object_points_{self.step}" in self.traj_data:
                    object_points = self.traj_data[f"object_points_{self.step}"]
                else:
                    object_points = np.zeros((0, 3))
                if f"hand_points_{self.step}" in self.traj_data:
                    hand_points = self.traj_data[f"hand_points_{self.step}"]
                else:
                    hand_points = np.zeros((0, 3))
                object_pcd = get_o3d_pcd(object_points, color=(0., 0., 1.))
                hand_pcd = get_o3d_pcd(hand_points, (0., 1., 0.))
                info["input_pcd"] = object_pcd+hand_pcd
        self.step += 1
        return action, action_type, reached, info

"""
CUDA_VISIBLE_DEVICES=0 python -m evaluate use_ray=False env.panda.IK_solver=PyKDL setup=s0 split=train policy=offline offline.demo_dir=/data2/haoran/HRI/expert_demo/OMG/s0/know_dest_smooth_0.08 offline.demo_source=handoversim env.visualize=True env.verbose=True

CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_DEDUP_LOGS=0 python -m evaluate env.panda.IK_solver=PyKDL setup=s0 split=train policy=offline offline.demo_dir=/data2/haoran/HRI/expert_demo/OMG/s0/know_dest_smooth_0.08 offline.demo_source=handoversim
"""