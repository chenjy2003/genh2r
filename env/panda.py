import os
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, List
from dataclasses import dataclass, field
from omegaconf import MISSING
from pybullet_utils.bullet_client import BulletClient
from scipy.spatial.transform import Rotation as Rt
import ipdb
import code

from .body import Body, BodyConfig
from .camera import Camera, CameraConfig
from .utils.transform import pos_ros_quat_to_mat, pos_euler_to_mat, mat_to_pos_ros_quat, se3_inverse, se3_transform_pc
from .utils.robot_kinematics import RobotKinematics, RobotKinematicsConfig

@dataclass
class PandaConfig(BodyConfig):
    name: str = "panda"
    urdf_file: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "assets", "franka_panda", "panda_gripper_hand_camera.urdf")
    # collision
    use_self_collision: bool = True
    collision_mask: int = -1
    # base
    use_fixed_base: bool = True
    base_position: Tuple[float] = (0.61, -0.50, 0.875)
    base_orientation: Tuple[float] = (0.0, 0.0, 0.7071068, 0.7071068)
    # links
    num_dofs: int = 9
    dof_default_position: Tuple[float] = (0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04)
    dof_position: Optional[Tuple[float]] = None
    dof_velocity: Tuple[float] = (0.0,)*9
    dof_max_force: Tuple[float] = (250.0,)*9
    dof_position_gain: Tuple[float] = (0.01,)*9
    dof_velocity_gain: Tuple[float] = (1.0,)*9

    # IK
    IK_solver: str = "PyKDL" # "pybullet"
    IK_solver_max_iter: int = 100
    IK_solver_eps: float = 1e-6
    # camera
    step_time: float = MISSING
    camera: CameraConfig = field(default_factory=lambda: CameraConfig(width=224, height=224, fov=90., near=0.035, far=2.0, step_time="${..step_time}")) # the format of nested structured configs must be like this https://omegaconf.readthedocs.io/en/2.3_branch/structured_config.html#nesting-structured-configs
    robot_kinematics: RobotKinematicsConfig = field(default_factory=lambda: RobotKinematicsConfig(
        urdf_file="${..urdf_file}",
        IK_solver_max_iter="${..IK_solver_max_iter}", 
        IK_solver_eps="${..IK_solver_eps}",
        chain_tip="panda_hand",
    ))

class Panda(Body):
    def __init__(self, bullet_client: BulletClient, cfg: PandaConfig):
        super().__init__(bullet_client, cfg)
        self.cfg: PandaConfig
        self.ee_link_id = 7
        self.fingers_link_id = (8, 9)
        self.camera_link_id = 10
        self.world_to_base = pos_ros_quat_to_mat(cfg.base_position, cfg.base_orientation)
        self.base_to_world = se3_inverse(self.world_to_base)

        self.camera = Camera(bullet_client, cfg.camera)
        self.hand_to_camera = np.eye(4)
        self.hand_to_camera[:3, 3] = (0.036, 0.0, 0.036)
        self.hand_to_camera[:3, :3] = Rt.from_euler("XYZ", (0.0, 0.0, np.pi/2)).as_matrix()

        if cfg.IK_solver == "PyKDL":
            self.robot_kinematics = RobotKinematics(cfg.robot_kinematics)

    def reset(self, joint_state: Optional[Tuple[float]]=None):
        self.base_reset()
        if joint_state is not None:
            self.cfg.dof_position = joint_state
        else:
            self.cfg.dof_position = self.cfg.dof_default_position
        self.load()
        self.set_dof_target(self.cfg.dof_position)
        camera_link_state = self.bullet_client.getLinkState(self.body_id, self.camera_link_id, computeForwardKinematics=1)
        self.camera.update_pose(camera_link_state[4], camera_link_state[5])

    def pre_step(self, dof_target_position):
        self.set_dof_target(dof_target_position)
        self.camera.pre_step()
    
    def post_step(self):
        camera_link_state = self.bullet_client.getLinkState(self.body_id, self.camera_link_id, computeForwardKinematics=1)
        self.camera.update_pose(camera_link_state[4], camera_link_state[5])
        self.camera.post_step()
    
    def get_world_to_ee(self):
        world_to_ee = self.get_link_pose(self.ee_link_id)
        return world_to_ee

    def get_tip_pos(self):
        world_to_ee = self.get_world_to_ee()
        tip_pos = world_to_ee[:3, 3]+0.108*world_to_ee[:3, 2]
        return tip_pos

    def ego_cartesian_action_to_dof_target_position(self, pos: NDArray[np.float64], orn: NDArray[np.float64], width: float, orn_type="euler") -> NDArray[np.float64]:
        world_to_ee = self.get_world_to_ee()
        if orn_type == "euler":
            ee_to_new_ee = pos_euler_to_mat(pos, orn)
        else:
            raise NotImplementedError
        world_to_new_ee = world_to_ee@ee_to_new_ee
        
        if self.cfg.IK_solver == "PyKDL":
            base_to_new_ee = self.base_to_world@world_to_new_ee
            joint_positions = self.get_joint_positions()
            dof_target_position, info = self.robot_kinematics.cartesian_to_joint(base_to_new_ee, seed=joint_positions[:7])
            if self.cfg.verbose and info<0: print(f"PyKDL IK error: {info}")
            dof_target_position = np.append(dof_target_position, [width, width])
        elif self.cfg.IK_solver == "pybullet":
            world_to_new_ee_pos, world_to_new_ee_ros_quat = mat_to_pos_ros_quat(world_to_new_ee)
            dof_target_position = self.bullet_client.calculateInverseKinematics(self.body_id, self.ee_link_id, world_to_new_ee_pos, world_to_new_ee_ros_quat, maxNumIterations=self.cfg.IK_solver_max_iter, residualThreshold=self.cfg.IK_solver_eps)
            dof_target_position = np.array(dof_target_position)
            dof_target_position[7:9] = width
        else:
            raise NotImplementedError
        return dof_target_position

    def world_pos_action_to_dof_target_position(self, pos, width):
        world_to_ee = self.get_world_to_ee()
        world_to_new_ee_pos = world_to_ee[:3, 3]+pos
        # if self.cfg.IK_solver == "PyKDL":
        #     base_to_new_ee_pos = se3_transform_pc(self.base_to_world, world_to_new_ee_pos)
        #     joint_positions = self.get_joint_positions()
        #     dof_target_position, info = self.robot_kinematics.inverse_kinematics(position=base_to_new_ee_pos, seed=joint_positions[:7])
        #     if self.root_cfg.env.verbose and info<0: print(f"PyKDL IK error: {info}")
        #     dof_target_position = np.append(dof_target_position, [width, width])
        # elif self.cfg.IK_solver == "pybullet":
        dof_target_position = self.bullet_client.calculateInverseKinematics(self.body_id, self.ee_link_id, world_to_new_ee_pos)
        dof_target_position = np.array(dof_target_position)
        dof_target_position[7:9] = width
        return dof_target_position

    def get_visual_observation(self, segmentation_ids: List[int]=[]):
        camera_link_state = self.bullet_client.getLinkState(self.body_id, self.camera_link_id, computeForwardKinematics=1)
        self.camera.update_pose(camera_link_state[4], camera_link_state[5])
        color, depth, segmentation, points = self.camera.render(segmentation_ids)
        for i in range(len(points)):
            points[i] = se3_transform_pc(self.hand_to_camera, points[i])
        return color, depth, segmentation, points

def debug():
    from omegaconf import OmegaConf
    import pybullet
    import copy
    import time
    default_cfg = OmegaConf.structured(PandaConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg: PandaConfig = OmegaConf.to_object(OmegaConf.merge(default_cfg, cli_cfg))
    bullet_client = BulletClient(connection_mode=pybullet.GUI) # or pybullet.DIRECT
    bullet_client.resetDebugVisualizerCamera(cameraDistance=2.4, cameraPitch=-58, cameraYaw=102, cameraTargetPosition=[0, 0, 0])
    panda = Panda(bullet_client, cfg)
    panda.reset()
    dof_default_position = np.array(cfg.dof_default_position)

    def move_to(dof_target_position: NDArray[np.float64], steps=1) -> None:
        dof_current_position = panda.get_joint_positions()
        for i in range(steps):
            dof_target_position_i = (dof_target_position-dof_current_position)/steps*(i+1)+dof_current_position
            for _ in range(130):
                panda.pre_step(dof_target_position_i)
                bullet_client.stepSimulation()
                panda.post_step()
                # time.sleep(0.003)
            panda.get_visual_observation()
    
    # for i in range(len(dof_default_position)-2): # arm links
    #     print(f"moving link {i}")
    #     dof_target_position = dof_default_position.copy()
    #     dof_target_position[i] += np.pi/2
    #     move_to(dof_target_position, 10)
    #     move_to(dof_default_position, 10)
    
    # dof_target_position = dof_default_position.copy()
    # dof_target_position[7:] = 0.
    # move_to(dof_target_position) # gripper
    # move_to(dof_default_position)

    for i in range(6): # cartesian action
        pos, orn = np.array([0., 0., 0.]), np.array([0., 0., 0.])
        if i<3:
            pos[i] = 0.1
        else:
            orn[i-3] = np.pi/2
        dof_target_position = panda.ego_cartesian_action_to_dof_target_position(pos, orn, 0.04)
        move_to(dof_target_position, 10) # forward
        move_to(dof_default_position, 10) # backward
    code.interact(local=dict(globals(), **locals()))

if __name__ == "__main__":
    debug()

"""
DISPLAY="localhost:11.0" python -m env.panda
DISPLAY="localhost:11.0" python -m env.panda IK_solver=pybullet
"""