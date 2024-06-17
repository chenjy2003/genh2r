import numpy as np
from numpy.typing import NDArray
import torch
import code

from env.handover_env import Observation
from env.utils.transform import se3_transform_pc, se3_inverse, pos_ros_quat_to_mat
from models.policy_network import PolicyNetwork
from .base_policy import BasePolicy
from .pointnet2_policy_config import PointNet2PolicyConfig
from .utils.point_cloud import regularize_pc_point_count, PointCloudProcessor

class PointNet2Policy(BasePolicy):
    def __init__(self, cfg: PointNet2PolicyConfig, device=torch.device("cuda")):
        super().__init__(cfg)
        self.cfg: PointNet2PolicyConfig
        torch.backends.cudnn.deterministic = True
        self.device = device
        self.policy_network = PolicyNetwork(cfg.model).to(device)
        self.policy_network.eval()
        self.point_cloud_processor = PointCloudProcessor(cfg.processor)
    
    def reset(self):
        self.base_reset()
        self.point_cloud_processor.reset()

    def termination_heuristics(self, world_to_ee: NDArray) -> bool:
        ee_to_world = se3_inverse(world_to_ee)
        object_points = se3_transform_pc(ee_to_world, self.point_cloud_processor.world_to_object_points)
        if object_points.shape[0] == 0:
            return False
        cage_points_mask = (
            (object_points[:, 2] > +0.06)
            & (object_points[:, 2] < +0.11)
            & (object_points[:, 1] > -0.05)
            & (object_points[:, 1] < +0.05)
            & (object_points[:, 0] > -0.02)
            & (object_points[:, 0] < +0.02)
        )
        cage_points_mask_reg = regularize_pc_point_count(cage_points_mask[:, None], 1024, self.np_random)
        return np.sum(cage_points_mask_reg) > 50

    @torch.no_grad()
    def plan(self, observation: Observation):
        info = {}
        object_points, hand_points = observation.get_visual_observation()[3]
        world_to_ee = observation.world_to_ee
        input_points = self.point_cloud_processor.process(object_points, hand_points, world_to_ee)
        if input_points is None:
            action = np.zeros(6)
        else:
            input_points_tensor = torch.tensor(input_points, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, goal_pred, obj_pose_pred = self.policy_network(input_points_tensor)
            action, goal_pred, obj_pose_pred = action[0].cpu().numpy(), goal_pred[0].cpu().numpy(), obj_pose_pred[0].cpu().numpy()
            if self.policy_network.cfg.goal_pred:
                ee_to_target_grasp = pos_ros_quat_to_mat(goal_pred[:3], goal_pred[3:])
                world_to_target_grasp = world_to_ee@ee_to_target_grasp
                info["world_to_target_grasp"] = world_to_target_grasp
        action = np.append(action, 0.04)
        reached = self.termination_heuristics(world_to_ee)
        return action, "ego_cartesian", reached, info

"""
CUDA_VISIBLE_DEVICES=1 python -m evaluate setup=s0 split=train use_ray=False policy=pointnet2 pointnet2.model.pretrained_dir=/share1/haoran/HRI/generalizable_handover/output/t450_bc_omg_replan5_landmark_smooth0.08_use_hand_flow0_pred0_wd0.0001_pred0.5_300w_13s_no_accum_3 pointnet2.model.pretrained_source=handoversim env.visualize=True
scene_id 5, step 62, status 1, reward 1.0, reached frame 5720, done frame 8380

CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_DEDUP_LOGS=0 python -m evaluate setup=s0 split=test num_runners=32 policy=pointnet2 pointnet2.model.pretrained_dir=/share1/haoran/HRI/generalizable_handover/output/t450_bc_omg_replan5_landmark_smooth0.08_use_hand_flow0_pred0_wd0.0001_pred0.5_300w_13s_no_accum_3 pointnet2.model.pretrained_source=handoversim
success rate: 123/144=0.8541666666666666
contact rate: 5/144=0.034722222222222224
   drop rate: 16/144=0.1111111111111111
timeout rate: 0/144=0.0
average done frame        : 6378.659722222223
average success done frame: 6421.861788617886
average success num steps : 47.0650406504065
average success           : 0.4322831196581196
evaluting uses 112.1101815700531 seconds

CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_DEDUP_LOGS=0 python -m evaluate setup=s0 split=train num_runners=32 policy=pointnet2 pointnet2.model.pretrained_dir=/share1/haoran/HRI/generalizable_handover/output/t450_bc_omg_replan5_landmark_smooth0.08_use_hand_flow0_pred0_wd0.0001_pred0.5_300w_13s_no_accum_3 pointnet2.model.pretrained_source=handoversim
success rate: 613/720=0.8513888888888889
contact rate: 20/720=0.027777777777777776
   drop rate: 68/720=0.09444444444444444
timeout rate: 19/720=0.02638888888888889
average done frame        : 6652.265277777778
average success done frame: 6543.5057096247965
average success num steps : 47.99347471451876
average success           : 0.42291068376068375
evaluting uses 370.2707829475403 seconds
"""