import os
from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray
import torch
from dataclasses import dataclass
from omegaconf import MISSING, OmegaConf
import pybullet
from pybullet_utils.bullet_client import BulletClient
import code

from .utils.transform import pos_ros_quat_to_mat
from .utils.sdf_loss import SDFData

OmegaConf.register_new_resolver("concat_tuples", lambda *tuples: sum(tuples, start=tuple()))

@dataclass
class BodyConfig:
    name: str = MISSING
    urdf_file: str = MISSING
    scale: float = 1.
    verbose: bool = False
    # collision
    use_self_collision: bool = False
    collision_mask: Optional[int] = None
    # base
    use_fixed_base: bool = False
    base_position: Optional[Tuple[float]] = None # (3, )
    base_orientation: Optional[Tuple[float]] = None # (4, ), xyzw, ros quaternion
    base_linear_velocity: Optional[Tuple[float]] = None
    base_angular_velocity: Optional[Tuple[float]] = None
    # links
    num_dofs: int = 0
    dof_position: Optional[Tuple[float]] = None
    dof_velocity: Optional[Tuple[float]] = None
    link_lateral_friction: Optional[Tuple[float]] = None
    link_spinning_friction: Optional[Tuple[float]] = None
    link_rolling_friction: Optional[Tuple[float]] = None
    link_restitution: Optional[Tuple[float]] = None
    link_linear_damping: Optional[float] = None
    link_angular_damping: Optional[float] = None
    dof_max_force: Optional[Tuple[float]] = None
    dof_position_gain: Optional[Tuple[float]] = None
    dof_velocity_gain: Optional[Tuple[float]] = None
    dof_max_velocity: Optional[Tuple[float]] = None
    # visual
    link_color: Optional[Tuple[float]] = None

class Body:
    def __init__(self, bullet_client: BulletClient, cfg: BodyConfig):
        self.bullet_client = bullet_client
        self.cfg = cfg
        self.body_id = None
        self.sdf_data: Optional[SDFData] = None
    
    def base_reset(self):
        self.body_id = None
        # self.clear() # bodies are already removed by bullet_client.resetSimulation()
    
    def set_collision_mask(self, collision_mask):
        for i in range(-1, self.num_links-1):
            self.bullet_client.setCollisionFilterGroupMask(self.body_id, i, collision_mask, collision_mask)

    def load(self):
        # assert self.body_id is None

        kwargs = {}
        if self.cfg.use_self_collision:
            kwargs["flags"] = pybullet.URDF_USE_SELF_COLLISION
        kwargs["useFixedBase"] = self.cfg.use_fixed_base
        if self.cfg.scale != 1.:
            kwargs["globalScaling"] = self.cfg.scale
        self.body_id = self.bullet_client.loadURDF(self.cfg.urdf_file, **kwargs)

        dof_indices = []
        for j in range(self.bullet_client.getNumJoints(self.body_id)):
            joint_info = self.bullet_client.getJointInfo(self.body_id, j)
            if joint_info[2] != pybullet.JOINT_FIXED:
                dof_indices.append(j)
        self.dof_indices = np.array(dof_indices, dtype=np.int64)
        self.num_links = self.bullet_client.getNumJoints(self.body_id)+1

        # Reset base state.
        if self.cfg.base_position is not None and self.cfg.base_orientation is not None:
            self.bullet_client.resetBasePositionAndOrientation(self.body_id, self.cfg.base_position, self.cfg.base_orientation)
        self.bullet_client.resetBaseVelocity(self.body_id, linearVelocity=self.cfg.base_linear_velocity, angularVelocity=self.cfg.base_angular_velocity)

        # Reset DoF state.
        if self.cfg.dof_position is not None:
            for i, j in enumerate(self.dof_indices):
                kwargs = {}
                if self.cfg.dof_velocity is not None:
                    kwargs["targetVelocity"] = self.cfg.dof_velocity[i]
                self.bullet_client.resetJointState(self.body_id, j, self.cfg.dof_position[i], **kwargs)
        
        # Set collision mask.
        if self.cfg.collision_mask is not None:
            self.set_collision_mask(self.cfg.collision_mask)

        # Set link dynamics.
        kwargs = {}
        if self.cfg.link_lateral_friction is not None:
            kwargs["lateralFriction"] = self.cfg.link_lateral_friction
        if self.cfg.link_spinning_friction is not None:
            kwargs["spinningFriction"] = self.cfg.link_spinning_friction
        if self.cfg.link_rolling_friction is not None:
            kwargs["rollingFriction"] = self.cfg.link_rolling_friction
        if self.cfg.link_restitution is not None:
            kwargs["restitution"] = self.cfg.link_restitution
        if len(kwargs) > 0:
            for i in range(-1, self.num_links-1):
                self.bullet_client.changeDynamics(self.body_id, i, **{k: v[i+1] for k, v in kwargs.items()})
        # Bullet only sets `linearDamping` and `angularDamping` for link index -1. See:
        #     https://github.com/bulletphysics/bullet3/blob/740d2b978352b16943b24594572586d95d476466/examples/SharedMemory/PhysicsClientC_API.cpp#L3419
        #     https://github.com/bulletphysics/bullet3/blob/740d2b978352b16943b24594572586d95d476466/examples/SharedMemory/PhysicsClientC_API.cpp#L3430
        kwargs = {}
        if self.cfg.link_linear_damping is not None:
            kwargs["linearDamping"] = self.cfg.link_linear_damping
        if self.cfg.link_angular_damping is not None:
            kwargs["angularDamping"] = self.cfg.link_angular_damping
        if len(kwargs) > 0:
            self.bullet_client.changeDynamics(self.body_id, -1, **kwargs)

        # Set link color.
        if self.cfg.link_color is not None:
            for i in range(-1, self.num_links-1):
                self.bullet_client.changeVisualShape(self.body_id, i, rgbaColor=self.cfg.link_color)
    
    def clear(self):
        if self.body_id is not None:
            self.bullet_client.removeBody(self.body_id)
            self.body_id = None

    def set_dof_target(self, dof_target_position=None, dof_target_velocity=None):
        kwargs = {}
        if dof_target_position is not None:
            kwargs["targetPositions"] = dof_target_position
        if dof_target_velocity is not None:
            kwargs["targetVelocities"] = dof_target_velocity
        if self.cfg.dof_position_gain is not None:
            kwargs["positionGains"] = self.cfg.dof_position_gain
        if self.cfg.dof_velocity_gain is not None:
            kwargs["velocityGains"] = self.cfg.dof_velocity_gain
        if self.cfg.dof_max_force is not None:
            kwargs["forces"] = self.cfg.dof_max_force
        # The redundant if-else block below is an artifact due to `setJointMotorControlArray()`
        # not supporting `maxVelocity`. `setJointMotorControlArray()` is still preferred when
        # `maxVelocity` is not needed due to better speed performance.
        if self.cfg.dof_max_velocity is None:
            self.bullet_client.setJointMotorControlArray(self.body_id, self.dof_indices, pybullet.POSITION_CONTROL, **kwargs)
        else: # For Bullet, 'dof_max_velocity' has no effect when not in the POSITION_CONROL mode.
            kwargs["maxVelocity"] = self.cfg.dof_max_velocity
            for i, j in enumerate(self.dof_indices):
                self.bullet_client.setJointMotorControl2(self.body_id, j, pybullet.POSITION_CONTROL, **{k: v[i] for k, v in kwargs.items()})

    def get_link_pos(self, link_id) -> NDArray[np.float64]:
        if self.body_id is None:
            pos = np.nan*np.ones(3, dtype=np.float64)
        else:
            link_state = self.bullet_client.getLinkState(self.body_id, link_id, computeForwardKinematics=1)
            pos = np.array(link_state[4], dtype=np.float64)
        return pos
    
    def get_link_orn(self, link_id) -> NDArray[np.float64]:
        if self.body_id is None:
            orn = np.nan*np.ones(4, dtype=np.float64)
        else:
            link_state = self.bullet_client.getLinkState(self.body_id, link_id, computeForwardKinematics=1)
            orn = np.array(link_state[5], dtype=np.float64)
        return orn
    
    def get_link_pos_orn(self, link_id: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        if self.body_id is None:
            pos, orn = np.nan*np.ones(3, dtype=np.float64), np.nan*np.ones(4, dtype=np.float64)
        else:
            link_state = self.bullet_client.getLinkState(self.body_id, link_id, computeForwardKinematics=1)
            pos, orn = np.array(link_state[4], dtype=np.float64), np.array(link_state[5], dtype=np.float64)
        return pos, orn

    def get_link_pose(self, link_id: int) -> NDArray[np.float64]:
        pos, orn = self.get_link_pos_orn(link_id)
        pose = pos_ros_quat_to_mat(pos, orn)
        return pose
    
    def get_link_states(self):
        link_indices = list(range(0, self.num_links-1))
        return self.bullet_client.getLinkStates(self.body_id, link_indices, computeLinkVelocity=1, computeForwardKinematics=1)

    def get_joint_positions(self) -> NDArray[np.float64]:
        if self.body_id is None:
            joint_positions = np.nan*np.ones(self.cfg.num_dofs, dtype=np.float64)
        else:
            joint_states = self.bullet_client.getJointStates(self.body_id, self.dof_indices)
            joint_positions = np.array([joint_state[0] for joint_state in joint_states], dtype=np.float64)
        return joint_positions
    
    def get_sdf(self) -> SDFData:
        if self.sdf_data is None:
            sdf_path = os.path.join(os.path.dirname(self.cfg.urdf_file), "sdf.npz")
            assert os.path.exists(sdf_path), f"{self.cfg.name} doesn't have SDF in {sdf_path}"
            self.sdf_data = SDFData(sdf_path)
        return self.sdf_data

    def pre_step(self):
        pass

    def post_step(self):
        pass

def debug():
    from omegaconf import OmegaConf
    class BodyTest(Body):
        pass
    cli_cfg = OmegaConf.from_cli()
    body_base_cfg = OmegaConf.structured(BodyConfig)
    body_cfg = OmegaConf.merge(body_base_cfg, cli_cfg)
    bullet_client = BulletClient(connection_mode=pybullet.GUI) # or pybullet.DIRECT
    body_test = BodyTest(bullet_client, body_cfg)
    body_test.load()
    code.interact(local=dict(globals(), **locals()))

if __name__ == "__main__":
    debug()

"""
DISPLAY="localhost:11.0" python -m env.body name=table urdf_file=data/assets/table/table.urdf
"""