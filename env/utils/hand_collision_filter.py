import os
import numpy as np
from numpy.typing import NDArray
import torch
from typing import Dict, List, Optional
from omegaconf import OmegaConf
import open3d as o3d
from dataclasses import dataclass
import code

from .robot_kinematics import RobotKinematics, RobotKinematicsConfig
from .transform import se3_inverse
from .sdf_loss import SDFData, SDFDataTensor, se3_transform_pc_tensor, compute_distance

class HandKinematics:
    def __init__(self, hand_urdf_path: str):
        self.hand_dir = os.path.dirname(hand_urdf_path)
        kinematics_cfg: RobotKinematicsConfig = OmegaConf.to_object(OmegaConf.structured(RobotKinematicsConfig(urdf_file=hand_urdf_path, chain_tip=None)))
        self.kinematics = RobotKinematics(kinematics_cfg)

        self.link_points_list: List[Optional[NDArray[np.float64]]] = []
        for link in self.kinematics.links:
            if link.collision_file_path is None:
                self.link_points_list.append(None)
                continue
            link_collision_file_path = os.path.join(self.hand_dir, link.collision_file_path)
            link_mesh = o3d.io.read_triangle_mesh(link_collision_file_path)
            self.link_points_list.append(np.array(link_mesh.vertices))

    def get_hand_points(self, hand_joint_values: NDArray[np.float64], device: torch.device) -> torch.DoubleTensor:
        hand_joint_values: torch.DoubleTensor = torch.tensor(hand_joint_values, device=device)
        base_to_links = self.kinematics.joint_to_cartesian_for_all_links(hand_joint_values)
        base_to_link_points_list: List[torch.DoubleTensor] = []
        for base_to_link, link_points in zip(base_to_links, self.link_points_list):
            if link_points is None:
                continue
            link_points: torch.DoubleTensor = torch.tensor(link_points, device=device)
            base_to_link_points = se3_transform_pc_tensor(base_to_link, link_points)
            base_to_link_points_list.append(base_to_link_points)
        hand_points = torch.cat(base_to_link_points_list, dim=0)
        return hand_points

@dataclass
class HandCollisionFilterConfig:
    device: str = "cpu"
    threshold: float = 0.
    use_bbox: bool = True

class HandCollisionFilter:
    def __init__(self, cfg: HandCollisionFilterConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.hand_kinematics_cache: Dict[str, HandKinematics] = {}

        env_dir = os.path.dirname(os.path.dirname(__file__))
        ee_dir = os.path.join(env_dir, "data", "assets", "franka_panda", "ee")
        if cfg.use_bbox:
            ee_mesh = o3d.io.read_triangle_mesh(os.path.join(ee_dir, "model.obj"))
            self.ee_min_coords = np.array(ee_mesh.vertices).min(axis=0)-cfg.threshold
            self.ee_max_coords = np.array(ee_mesh.vertices).max(axis=0)+cfg.threshold
        else:
            self.ee_sdf_data = SDFData(os.path.join(ee_dir, "sdf.npz"))
    
    def filter_hand_collision(self, hand_urdf_path: str, hand_joint_values: NDArray[np.float64], world_to_ees: NDArray[np.float64]) -> NDArray[np.bool_]:
        if hand_urdf_path not in self.hand_kinematics_cache:
            self.hand_kinematics_cache[hand_urdf_path] = HandKinematics(hand_urdf_path)
        hand_kinematics = self.hand_kinematics_cache[hand_urdf_path]
        hand_points = hand_kinematics.get_hand_points(hand_joint_values, device=self.device) # hand_points is in the world coordinate system, since its base is the origin in the world

        filter_results: List[bool] = []
        if not self.cfg.use_bbox:
            ee_sdf_data_tensor = SDFDataTensor(self.ee_sdf_data, self.device)
        for world_to_ee in world_to_ees:
            ee_to_world = torch.tensor(se3_inverse(world_to_ee), device=self.device)
            if not self.cfg.use_bbox:
                inside_bbox_mask, inside_bbox_value = compute_distance(hand_points, ee_sdf_data_tensor, ee_to_world) # this bbox is the bbox of SDF
                filter_results.append((inside_bbox_value>=self.cfg.threshold).all().item())
            else:
                ee_to_hand_points = se3_transform_pc_tensor(ee_to_world, hand_points)
                inside_bbox_mask: torch.BoolTensor = (
                    (ee_to_hand_points[:, 0] >= self.ee_min_coords[0])
                    & (ee_to_hand_points[:, 0] <= self.ee_max_coords[0])
                    & (ee_to_hand_points[:, 1] >= self.ee_min_coords[1])
                    & (ee_to_hand_points[:, 1] <= self.ee_max_coords[1])
                    & (ee_to_hand_points[:, 2] >= self.ee_min_coords[2])
                    & (ee_to_hand_points[:, 2] <= self.ee_max_coords[2])
                ) # this bbox is the bbox of ee model
                filter_results.append(not inside_bbox_mask.any())

        filter_results = np.array(filter_results, dtype=bool)
        return filter_results

def debug():
    hand_kinematics = HandKinematics("env/data/assets/hand/20200709-subject-01_left/mano.urdf")
    # for i in range(51):
    #     hand_joint_values = np.zeros(51)
    #     hand_joint_values[i] = np.pi/6
    #     hand_points = hand_kinematics.get_hand_points(hand_joint_values, device=torch.device("cpu"))
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(hand_points.numpy())
    #     o3d.io.write_point_cloud(f"tmp/debug_hand_collision_filter/{i}.xyz", pcd)
    #     # o3d.visualization.draw_geometries([pcd])

    hand_collision_filter = HandCollisionFilter("env/data/assets/hand/20200709-subject-01_left/mano.urdf")

if __name__ == "__main__":
    debug()

"""
python -m policies.hand_collision_filter
"""