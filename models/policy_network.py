import os
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from omegaconf import OmegaConf, MISSING
from typing import Optional, Dict, Tuple
import code

from env.handover_env import CartesianActionSpace
from .encoders.pointnet2 import PointNet2Encoder, PointNet2EncoderConfig
from .utils import get_submodule_weights
from .loss import compute_bc_loss, compute_goal_pred_loss

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

@dataclass
class PolicyNetworkConfig:
    num_points: int = 1024
    pretrained_dir: Optional[str] = None
    pretrained_suffix: str = "latest"
    pretrained_source: str = "genh2r" # "handoversim"
    in_channel: int = MISSING
    state_dim: int = 512
    hidden_dim: int = 256
    goal_pred: bool = True
    obj_pose_pred_frame_num: int = 0
    obj_pose_pred_coff: float = 0.0

    encoder: PointNet2EncoderConfig = field(default_factory=lambda: PointNet2EncoderConfig(
        in_channel="${..in_channel}", 
    ))

class PolicyNetwork(nn.Module):
    def __init__(self, cfg: PolicyNetworkConfig):
        super().__init__()
        self.cfg = cfg
        num_actions = 6
        self.goal_pred_dim = 7 if cfg.goal_pred else 0
        self.obj_pose_pred_dim = 6*cfg.obj_pose_pred_frame_num

        self.encoder = PointNet2Encoder(cfg.encoder)

        self.linear1 = nn.Linear(cfg.state_dim, cfg.hidden_dim)
        self.linear2 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

        self.action_head = nn.Linear(cfg.hidden_dim, num_actions)

        self.goal_pred_head = nn.Linear(cfg.hidden_dim, self.goal_pred_dim)
        self.obj_pose_pred_head = nn.Linear(cfg.hidden_dim, self.obj_pose_pred_dim)

        action_space = CartesianActionSpace()
        self.action_scale = nn.Parameter(torch.tensor((action_space.high-action_space.low)/2.0, dtype=torch.float32), requires_grad=False)
        self.action_bias = nn.Parameter(torch.tensor((action_space.high+action_space.low)/2.0, dtype=torch.float32), requires_grad=False)

        self.init_model()

    def init_model(self):
        if self.cfg.pretrained_dir is None:
            self.apply(weights_init_)
            return
    
        if self.cfg.pretrained_source == "genh2r":
            checkpoint = torch.load(os.path.join(self.cfg.pretrained_dir, f"{self.cfg.pretrained_suffix}.pth"))
            self.load_state_dict(checkpoint["state_dict"])
        elif self.cfg.pretrained_source == "handoversim":
            actor_weight = torch.load(os.path.join(self.cfg.pretrained_dir, f"BC_actor_PandaYCBEnv_{self.cfg.pretrained_suffix}"))["net"]
            encoder_weight = torch.load(os.path.join(self.cfg.pretrained_dir, f"BC_state_feat_PandaYCBEnv_{self.cfg.pretrained_suffix}"))["net"]
            self.encoder.SA_modules[0].load_state_dict(get_submodule_weights(encoder_weight, "module.encoder.0.0."))
            self.encoder.SA_modules[1].load_state_dict(get_submodule_weights(encoder_weight, "module.encoder.0.1."))
            self.encoder.SA_modules[2].load_state_dict(get_submodule_weights(encoder_weight, "module.encoder.0.2."))
            self.encoder.fc_layer.load_state_dict(get_submodule_weights(encoder_weight, "module.encoder.1."))
            self.linear1.load_state_dict(get_submodule_weights(actor_weight, "linear1."))
            self.linear2.load_state_dict(get_submodule_weights(actor_weight, "linear2."))
            self.action_head.load_state_dict(get_submodule_weights(actor_weight, "mean."))
            self.goal_pred_head.load_state_dict(get_submodule_weights(actor_weight, "extra_pred."))
            assert (self.obj_pose_pred_dim > 0) == any(key.startswith("extra_pred_obj") for key in actor_weight.keys())
            if self.obj_pose_pred_dim > 0:
                self.obj_pose_pred_head.load_state_dict(get_submodule_weights(actor_weight, "extra_pred_obj."))
        else:
            raise NotImplementedError

    def forward(self, pc):
        state = self.encoder(pc)
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        action = self.action_head(x)
        action = torch.tanh(action)*self.action_scale+self.action_bias
        goal_pred = self.goal_pred_head(x)
        if self.cfg.goal_pred: # pos+quat
            goal_pred = torch.cat([goal_pred[:, :3], F.normalize(goal_pred[:, 3:], p=2, dim=-1)], dim=-1)
        obj_pose_pred = self.obj_pose_pred_head(x)
        return action, goal_pred, obj_pose_pred

    def compute_loss(self, batch_data: dict) -> Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        point_cloud, expert_action = batch_data["point_clouds"], batch_data["expert_actions"]
        gt_grasp_pose, gt_object_pred_pose = batch_data["grasp_poses"], batch_data["object_pred_poses"]
        action, goal_pred, obj_pose_pred = self.forward(point_cloud)

        # print("point_cloud", point_cloud.shape)         #(B, 1024, 5 + flow_frame * 3)
        # print("expert_action", expert_action.shape, "action", action.shape)        #(B, 6)
        # print("gt_grasp_pose", gt_grasp_pose.shape, "goal_pred", goal_pred.shape)   # (B, 7)
        # print("gt_object_pred_pose", gt_object_pred_pose.shape, "obj_pose_pred", obj_pose_pred.shape)  # (B, 6 * pred_frame)

        loss = 0
        loss_dict: Dict[str, torch.FloatTensor] = {}
        bc_loss = compute_bc_loss(action, expert_action)
        loss = loss+bc_loss
        loss_dict["bc_loss"] = bc_loss.item()

        if self.cfg.goal_pred:
            goal_pred_loss = compute_goal_pred_loss(goal_pred, gt_grasp_pose)
            loss = loss+goal_pred_loss
            loss_dict["goal_pred_loss"] = goal_pred_loss.item()

        if self.cfg.obj_pose_pred_frame_num > 0:
            obj_pose_pred_loss = self.cfg.obj_pose_pred_coff*(obj_pose_pred.view(-1)-gt_object_pred_pose.view(-1)).abs().mean()
            loss = loss+obj_pose_pred_loss
            loss_dict["obj_pose_pred_loss"] = obj_pose_pred_loss.item()

        return loss, loss_dict

def debug():
    from omegaconf import OmegaConf
    default_cfg = OmegaConf.structured(PolicyNetworkConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg: PolicyNetworkConfig = OmegaConf.to_object(OmegaConf.merge(default_cfg, cli_cfg))
    policy_network = PolicyNetwork(cfg).cuda()
    state = torch.load("/share1/haoran/HRI/generalizable_handover/state.pkl")
    x = F.relu(policy_network.linear1(state))
    x = F.relu(policy_network.linear2(x))
    action = policy_network.action_head(x)
    action = torch.tanh(action)*policy_network.action_scale+policy_network.action_bias
    # pc = torch.randn(2, 1024, 17).cuda()
    code.interact(local=dict(globals(), **locals()))

if __name__ == "__main__":
    debug()

"""
conda activate gaddpg
cd /share1/haoran/HRI/generalizable_handover

OMG_PLANNER_DIR=/share/haoran/HRI/OMG-Planner PYTHONUNBUFFERED=True CUDA_VISIBLE_DEVICES=3 python -m core_multi_frame.train_online_handover TRAIN_DDPG.test True BENCHMARK.SETUP s0 TRAIN_DDPG.num_remotes 1 TRAIN_DDPG.pretrained output/t450_bc_omg_replan5_landmark_smooth0.08_use_hand_flow0_pred0_wd0.0001_pred0.5_300w_13s_no_accum_3 SIM.RENDER False TEST.split test TRAIN_DDPG.occupy False TRAIN_DDPG.debug True

CUDA_VISIBLE_DEVICES=7 python -m models.policy_network pretrained_dir=/share1/haoran/HRI/generalizable_handover/output/t450_bc_omg_replan5_landmark_smooth0.08_use_hand_flow0_pred0_wd0.0001_pred0.5_300w_13s_no_accum_3 pretrained_source=handoversim in_channel=5
>>> action
tensor([[ 0.0194, -0.0011,  0.0022, -0.0022, -0.0011, -0.0042]],
       device='cuda:0', grad_fn=<AddBackward0>)
"""