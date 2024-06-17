import numpy as np
import code

from env.handover_env import Observation
from .base_policy_config import BasePolicyConfig

class BasePolicy:
    def __init__(self, cfg: BasePolicyConfig):
        self.cfg = cfg
        self.init_joint = np.array(cfg.init_joint)
        self.wait_steps = int(cfg.wait_time/cfg.step_time)
        self.action_repeat_steps = int(cfg.action_repeat_time/cfg.step_time)
        self.close_gripper_steps = int(cfg.close_gripper_time/cfg.step_time)
        self.np_random = np.random.RandomState()

    def base_reset(self):
        self.np_random.seed(self.cfg.seed)
        self.init = False
        self.reached = False
        self.reached_frame = -1
        self.grasped = False
        self.retreat_step = 0

    def plan(self, observation: Observation):
        pass
        # return action, action_type, reached, info
    
    def run_policy(self, observation: Observation):
        if not self.init:
            self.init = True
            if self.wait_steps > 0:
                # Wait
                action, action_type, repeat, stage, info = self.init_joint, "joint", self.wait_steps, "wait", {}
                return action, action_type, repeat, stage, info
        
        if not self.reached:
            action, action_type, reached, info = self.plan(observation)
            if not reached:
                # Approach object until reaching grasp pose.
                repeat = self.action_repeat_steps
                stage = "reach"
                return action, action_type, repeat, stage, info
            else:
                self.reached = True
                self.reached_frame = observation.frame

        if not self.grasped:
            # Close gripper
            self.grasped = True
            action, action_type, repeat, stage, info = np.zeros(7), "ego_cartesian", self.close_gripper_steps, "grasp", {}
            return action, action_type, repeat, stage, info

        # Retreat
        if self.retreat_step < self.cfg.retreat_steps:
            self.retreat_step += 1
            action = np.array([0., 0., -self.cfg.retreat_step_size, 0., 0., 0., 0.])
            action_type, repeat, stage, info = "ego_cartesian", self.action_repeat_steps, "retreat", {}
            return action, action_type, repeat, stage, info

        # Retrieve
        pos = observation.world_to_ee[:3, 3]
        pos_displacement = self.cfg.goal_center-pos
        if np.linalg.norm(pos_displacement) <= self.cfg.retrive_step_size:
            action_pos = pos_displacement
        else:
            action_pos = pos_displacement/np.linalg.norm(pos_displacement)*self.cfg.retrive_step_size
        action, action_type, repeat, stage, info = np.append(action_pos, 0), "world_pos", self.action_repeat_steps, "retrieve", {}
        return action, action_type, repeat, stage, info

class DebugPolicy(BasePolicy):
    def __init__(self, cfg: BasePolicyConfig):
        super().__init__(cfg)

    def reset(self, traj_data_path):
        self.base_reset()
        self.traj_data = np.load(traj_data_path)
        self.num_steps = self.traj_data["num_steps"]
        self.step = 0

    def plan(self, observation: Observation):
        if self.step == self.num_steps:
            action, action_type, reached, info = None, None, True, {}
        else:
            action = np.append(self.traj_data["expert_action"][self.step], 0.04)
            action_type, reached, info = "ego_cartesian", False, {}
        self.step += 1
        return action, action_type, reached, info

def debug():
    from env.handover_env import GenH2RSim, GenH2RSimConfig
    from omegaconf import OmegaConf
    from dataclasses import dataclass, field
    import ipdb

    @dataclass
    class EvalConfig:
        env: GenH2RSimConfig = field(default_factory=GenH2RSimConfig)
        policy: BasePolicyConfig = field(default_factory=lambda: BasePolicyConfig(name="debug", init_joint="${..env.panda.dof_default_position}", step_time="${..env.step_time}", goal_center="${..env.status_checker.goal_center}"))
    
    eval_base_cfg = OmegaConf.structured(EvalConfig)
    cli_cfg = OmegaConf.from_cli()
    eval_cfg = OmegaConf.to_object(OmegaConf.merge(eval_base_cfg, cli_cfg))
    env = GenH2RSim(eval_cfg.env)
    policy = DebugPolicy(eval_cfg.policy)

    while True:
        scene_id = int(input("scene_id:"))
        env.reset(scene_id)
        policy.reset(f"/data2/haoran/HRI/expert_demo/OMG/s0/know_dest_smooth_0.08/{scene_id}.npz")
        while True:
            observation = env.get_observation()
            action, action_type, repeat, stage, info = policy.run_policy(observation)
            if action_type == "joint":
                reward, done, info = env.joint_step(action, repeat)
            elif action_type == "ego_cartesian":
                reward, done, info = env.ego_cartesian_step(action, repeat)
            elif action_type == "world_pos":
                reward, done, info = env.world_pos_step(action, repeat)
            print(f"reward {reward}, done {done}, info {info}")
            if done:
                break

if __name__ == "__main__":
    debug()

"""
python -m policies.base_policy env.verbose=True env.panda.IK_solver=PyKDL env.visualize=True
scene_id: 10
frame 7052, status 1, reward 1.0, done True
scene_id: 12
frame 7018, status 1, reward 1.0, done True
"""