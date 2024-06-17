from dataclasses import dataclass
from typing import Tuple
from omegaconf import MISSING

@dataclass
class BasePolicyConfig:
    name: str = MISSING
    seed: int = 0
    init_joint: Tuple[float] = MISSING
    step_time: float = MISSING
    wait_time: float = 0.
    action_repeat_time: float = 0.13
    close_gripper_time: float = 0.5
    retreat_step_size: float = 0.03
    retrive_step_size: float = 0.03
    goal_center: Tuple[float] = MISSING
    retreat_steps: int = 0
    verbose: bool = False

"""
It is nice to have all configs in the same file with objects. But in some cases, only the configs are wanted to be imported.
"""
