import numpy as np
from numpy.typing import NDArray
import os
from typing import Tuple, Optional, ContextManager
from dataclasses import dataclass
from omegaconf import MISSING
from pybullet_utils.bullet_client import BulletClient
import code

from .body import Body, BodyConfig

@dataclass
class HandConfig(BodyConfig):
    name: str = ""
    urdf_file: str = ""
    # collision
    collision_mask: int = 2**1
    # base
    use_fixed_base: bool = True
    base_position: Tuple[float] = (0., 0., 0.)
    base_orientation: Tuple[float] = (0., 0., 0., 1.)
    # links
    num_dofs: int = 51
    dof_position: Optional[Tuple[float]] = None
    dof_velocity: Tuple[float] = (0.0,)*51
    link_lateral_friction: Tuple[float] = (5.0,)*53
    link_spinning_friction: Tuple[float] = (5.0,)*53
    link_restitution: Tuple[float] = (0.5,)*53
    link_linear_damping: float = 10.0
    link_angular_damping: float = 10.0

    translation_max_force: Tuple[float] = (50.0,)*3
    translation_position_gain: Tuple[float] = (0.2,)*3
    translation_velocity_gain: Tuple[float] = (1.0,)*3
    rotation_max_force: Tuple[float] = (5.0,)*3
    rotation_position_gain: Tuple[float] = (0.2,)*3
    rotation_velocity_gain: Tuple[float] = (1.0,)*3
    joint_max_force: Tuple[float] = (0.5,)*45
    joint_position_gain: Tuple[float] = (0.1,)*45
    joint_velocity_gain: Tuple[float] = (1.0,)*45

    dof_max_force: Tuple[float] = "${concat_tuples:${.translation_max_force},${.rotation_max_force},${.joint_max_force}}"
    dof_position_gain: Tuple[float] = "${concat_tuples:${.translation_position_gain},${.rotation_position_gain},${.joint_position_gain}}"
    dof_velocity_gain: Tuple[float] = "${concat_tuples:${.translation_velocity_gain},${.rotation_velocity_gain},${.joint_velocity_gain}}"
    # concat_tuples is defined in body.py
    # the "." is necessary in nested configs, see https://github.com/omry/omegaconf/issues/1099#issuecomment-1624496194
    # visual
    link_color: Tuple[float] = (0., 1., 0., 1.)

class Hand(Body):
    def __init__(self, bullet_client: BulletClient, cfg: HandConfig):
        super().__init__(bullet_client, cfg)
        self.cfg: HandConfig

    def reset(self, name: str, side: str, path: str, pose: NDArray[np.float32]):
        self.base_reset()
        # config
        self.cfg.name = "{}_{}".format(name, side)
        self.cfg.urdf_file = path

        # process pose data
        self.pose = pose # (num_frames, 51)
        nonzero_mask = np.any(self.pose!=0, axis=1)
        if np.any(nonzero_mask):
            nonzeros = np.where(nonzero_mask)[0]
            self.start_frame = nonzeros[0]
            self.end_frame = nonzeros[-1]+1
        else:
            self.start_frame, self.end_frame = -1, -1
        self.num_frames = self.pose.shape[0]
        self.frame = 0

        if self.frame == self.start_frame:
            self.cfg.dof_position = self.pose[self.frame].tolist() # can not use tuple(self.pose[self.frame]), which keeps the type np.float32 for scalars
            self.load()
            self.set_dof_target(self.pose[self.frame])

    def pre_step(self, disable_rendering: ContextManager):
        self.frame += 1
        self.frame = min(self.frame, self.num_frames-1)

        if self.frame == self.start_frame:
            self.cfg.dof_position = self.pose[self.frame].tolist() # can not use tuple(self.pose[self.frame]), which keeps the type np.float32 for scalars
            with disable_rendering():
                self.load()
        if self.start_frame <= self.frame < self.end_frame:
            self.set_dof_target(self.pose[self.frame])
        if self.frame == self.end_frame:
            self.clear()
    
    def post_step(self):
        pass

def debug():
    from omegaconf import OmegaConf
    import pybullet
    import time
    from env.utils import load_scene_data
    hand_cfg = OmegaConf.structured(HandConfig)
    bullet_client = BulletClient(connection_mode=pybullet.GUI) # or pybullet.DIRECT
    hand = Hand(bullet_client, hand_cfg)
    scene_data = load_scene_data(0)
    hand.reset(scene_data["hand_name"], scene_data["hand_side"], scene_data["hand_path"], scene_data["hand_pose"])
    while True:
        print(f"frame {hand.frame}")
        hand.pre_step()
        bullet_client.stepSimulation()
        hand.post_step()
        time.sleep(0.01)
    code.interact(local=dict(globals(), **locals()))

if __name__ == "__main__":
    debug()

"""
DISPLAY="localhost:11.0" python -m env.hand
"""