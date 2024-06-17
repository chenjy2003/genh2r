import numpy as np
from numpy.typing import NDArray
from typing import Optional
from dataclasses import dataclass
from omegaconf import OmegaConf

from env.utils.transform import se3_inverse, se3_transform_pc
import open3d as o3d

def regularize_pc_point_count(pc: NDArray, num_points: int, np_random: np.random.RandomState) -> NDArray:
    """
    pc: (N, D)
    If point cloud pc has less points than num_points, it oversamples.
    Otherwise, it downsample the input pc to have num_points points.
    """
    if pc.shape[0] > num_points:
        selected_indices = np_random.choice(range(pc.shape[0]), size=num_points, replace=False)
        regularized_pc = pc[selected_indices]
    elif pc.shape[0] == num_points:
        regularized_pc = pc
    else:
        required = num_points-pc.shape[0]
        selected_indices = np_random.choice(range(pc.shape[0]), size=required)
        regularized_pc = np.concatenate((pc, pc[selected_indices]), axis=0)
    return regularized_pc

OmegaConf.register_new_resolver("eval", eval)

@dataclass
class PointCloudProcessorConfig:
    num_points: int = 1024
    use_hand: bool = True
    flow_frame_num: int = 0
    in_channel: int = "${eval:'3+2*int(${.use_hand})+3*${.flow_frame_num}'}"

class PointCloudProcessor:
    def __init__(self, cfg: PointCloudProcessorConfig):
        self.cfg = cfg
        self.np_random = np.random.RandomState(0)
        self.world_to_object_points = np.zeros((0, 3))
        self.world_to_hand_points = np.zeros((0, 3))
        self.world_to_previous_object_points = np.zeros((0, 3))
        self.world_to_previous_hand_points = np.zeros((0, 3))
        self.pre_hand_flow_matrix_list = [np.eye(4) for i in range(self.cfg.flow_frame_num)]
        self.pre_object_flow_matrix_list = [np.eye(4) for i in range(self.cfg.flow_frame_num)]
    
    def _get_flow_matrix(self, pre_point, cur_point):
        " pre_point: (..., 3)    cur_point: (..., 3) "
        if pre_point.shape[0] == 0 or cur_point.shape[0] == 0:
            return np.eye(4)
        pre_pcd = o3d.geometry.PointCloud()
        pre_pcd.points = o3d.utility.Vector3dVector(pre_point)
        cur_pcd = o3d.geometry.PointCloud()
        cur_pcd.points = o3d.utility.Vector3dVector(cur_point)

        threshold = 10  # Correspondence distance threshold
        reg_p2p = o3d.pipelines.registration.registration_icp(
        pre_pcd, cur_pcd, threshold, np.identity(4), o3d.pipelines.registration.TransformationEstimationPointToPoint())

        return reg_p2p.transformation
    
    def _compute_pre_points(self, matrix, points):                
        return se3_transform_pc(se3_inverse(matrix), points)
    
    def reset(self):
        self.np_random.seed(0)
        self.world_to_object_points = np.zeros((0, 3))
        self.world_to_hand_points = np.zeros((0, 3))
        self.world_to_previous_object_points = np.zeros((0, 3))
        self.world_to_previous_hand_points = np.zeros((0, 3))
        self.pre_hand_flow_matrix_list = [np.eye(4) for i in range(self.cfg.flow_frame_num)]
        self.pre_object_flow_matrix_list = [np.eye(4) for i in range(self.cfg.flow_frame_num)]
        
    def process(self, object_points, hand_points, world_to_ee) -> Optional[NDArray[np.float32]]:
        self.world_to_previous_object_points = self.world_to_object_points
        self.world_to_previous_hand_points = self.world_to_hand_points

        # update world to points
        if object_points.shape[0] > 0 or hand_points.shape[0] > 0:
            world_to_new_object_points = se3_transform_pc(world_to_ee, object_points)
            world_to_new_hand_points = se3_transform_pc(world_to_ee, hand_points)
            self.world_to_object_points = world_to_new_object_points
            self.world_to_hand_points = world_to_new_hand_points

        # convert the stored world_to_points to egocentric
        ee_to_world = se3_inverse(world_to_ee)
        object_points = se3_transform_pc(ee_to_world, self.world_to_object_points)
        hand_points = se3_transform_pc(ee_to_world, self.world_to_hand_points)

        if object_points.shape[0]+hand_points.shape[0] == 0:
            return None

        if self.cfg.flow_frame_num > 0:
            object_flow_matrix = self._get_flow_matrix(self.world_to_previous_object_points, self.world_to_object_points)
            hand_flow_matrix = self._get_flow_matrix(self.world_to_previous_hand_points, self.world_to_hand_points)
            self.pre_object_flow_matrix_list.append(np.eye(4))
            self.pre_hand_flow_matrix_list.append(np.eye(4))

            for i in range(-self.cfg.flow_frame_num, 0):
                self.pre_object_flow_matrix_list[i] = object_flow_matrix @ self.pre_object_flow_matrix_list[i]
                self.pre_hand_flow_matrix_list[i] = hand_flow_matrix @ self.pre_hand_flow_matrix_list[i]

        input_points = np.zeros((object_points.shape[0]+hand_points.shape[0], 5), dtype=np.float32)
        input_points[:object_points.shape[0], :3] = object_points
        input_points[:object_points.shape[0], 3] = 1
        input_points[object_points.shape[0]:, :3] = hand_points
        input_points[object_points.shape[0]:, 4] = 1
        input_points = regularize_pc_point_count(input_points, self.cfg.num_points, self.np_random)

        
        if self.cfg.flow_frame_num > 0:
            point_flow_feature = np.zeros((input_points.shape[0], 3 * self.cfg.flow_frame_num))
            object_mask = input_points[:,3] == True
            hand_mask = input_points[:,4] == True
            world_to_new_object_points = se3_transform_pc(world_to_ee, input_points[object_mask, :3])
            world_to_new_hand_points = se3_transform_pc(world_to_ee, input_points[hand_mask, :3])

            for i in range(self.cfg.flow_frame_num):
                point_flow_feature[object_mask, i*3 : (i+1)*3] = self._compute_pre_points(self.pre_object_flow_matrix_list[-(i+1)], world_to_new_object_points)
                point_flow_feature[hand_mask, i*3 : (i+1)*3] = self._compute_pre_points(self.pre_hand_flow_matrix_list[-(i+1)],   world_to_new_hand_points)
                point_flow_feature[:, i*3 : (i+1)*3] = se3_transform_pc(ee_to_world, point_flow_feature[:, i*3 : (i+1)*3])
        
            input_points = np.concatenate([input_points[:,:3], point_flow_feature, input_points[:,3:]], axis = 1)

        return input_points
