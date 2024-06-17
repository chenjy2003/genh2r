import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from omegaconf import OmegaConf
import pybullet
from pybullet_utils.bullet_client import BulletClient
import pybullet_data
from functools import partial
from typing import TypedDict, Callable, List, Tuple, Optional
from scipy.spatial.transform import Rotation as Rt
from contextlib import contextmanager
import time
import code

from .table import Table, TableConfig
from .panda import Panda, PandaConfig
from .camera import Camera, CameraConfig
from .hand import Hand, HandConfig
from .objects import Objects, ObjectConfig, ObjectsConfig
from .status_checker import StatusChecker, StatusCheckerConfig, EpisodeStatus
from .bodies_for_visualization import Grasp, GraspConfig, get_grasp_config, Sphere, SphereConfig
from .utils.scene import load_scene_data

OmegaConf.register_new_resolver("divide_to_int", lambda x, y: int(x/y) if x is not None else None)

@dataclass
class GenH2RSimConfig:
    gravity: Tuple[float] = (0.0, 0.0, -9.8)
    substeps: int = 1
    table_height: float = 0.92

    step_time: float = 0.001
    max_time: float = 13.0
    stop_moving_time: Optional[float] = None

    max_frames: int = "${divide_to_int:${.max_time},${.step_time}}"
    stop_moving_frame: Optional[int] = "${divide_to_int:${.stop_moving_time},${.step_time}}"
    frame_interval: int = "${divide_to_int:${.step_time},0.001}"
    # the "." is necessary in nested configs, see https://github.com/omry/omegaconf/issues/1099#issuecomment-1624496194

    stop_moving_dist: Optional[float] = None

    # GUI
    visualize: bool = False
    viewer_camera_distance: float = 2.4
    viewer_camera_yaw: float = 102.
    viewer_camera_pitch: float = -58.
    viewer_camera_target: Tuple[float] = (0., 0., 0.)
    # DRAW_VIEWER_AXES = True
    show_trajectory: bool = False
    show_camera: bool = False

    verbose: bool = False

    table: TableConfig = field(default_factory=TableConfig) # the format of nested structured configs must be like this https://omegaconf.readthedocs.io/en/2.3_branch/structured_config.html#nesting-structured-configs
    hand: HandConfig = field(default_factory=HandConfig)
    object: ObjectConfig = field(default_factory=ObjectConfig)
    objects: ObjectsConfig = field(default_factory=ObjectsConfig)
    panda: PandaConfig = field(default_factory=lambda: PandaConfig(step_time="${..step_time}"))
    third_person_camera: CameraConfig = field(default_factory=lambda: CameraConfig(width=1280, height=720, fov=60., near=0.1, far=10.0, pos=(1.5, -0.1, 1.8), target=(0.6, -0.1, 1.3), up_vector=(0., 0., 1.), step_time="${..step_time}"))
    status_checker: StatusCheckerConfig = field(default_factory=lambda: StatusCheckerConfig(table_height="${..table_height}", max_frames="${..max_frames}"))

class CartesianActionSpace: # not a hard constraint. only for policy learning
    def __init__(self):
        self.high = np.array([ 0.06,  0.06,  0.06,  np.pi/6,  np.pi/6,  np.pi/6]) #, np.pi/10
        self.low  = np.array([-0.06, -0.06, -0.06, -np.pi/6, -np.pi/6, -np.pi/6]) # , -np.pi/3
        self.shape = [6]
        self.bounds = np.vstack([self.low, self.high])

@dataclass
class Observation:
    frame: int
    world_to_ee: NDArray[np.float64]
    joint_positions: NDArray[np.float64]
    get_visual_observation: "GenH2RSim.get_visual_observation"
    env: "GenH2RSim"

class GenH2RSim:
    def __init__(self, cfg: GenH2RSimConfig):
        self.cfg = cfg

        if self.cfg.visualize:
            self.bullet_client = BulletClient(connection_mode=pybullet.GUI)
            self.bullet_client.resetDebugVisualizerCamera(cameraDistance=self.cfg.viewer_camera_distance, cameraYaw=self.cfg.viewer_camera_yaw, cameraPitch=self.cfg.viewer_camera_pitch, cameraTargetPosition=self.cfg.viewer_camera_target)
            self.bullet_client.rendering_enabled = True
        else:
            self.bullet_client = BulletClient(connection_mode=pybullet.DIRECT)
            self.bullet_client.rendering_enabled = False
        self.bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.table = Table(self.bullet_client, cfg.table)
        self.panda = Panda(self.bullet_client, cfg.panda)
        self.hand = Hand(self.bullet_client, cfg.hand)
        self.objects = Objects(self.bullet_client, cfg.objects, cfg.object)
        self.status_checker = StatusChecker(self.bullet_client, cfg.status_checker)
        self.third_person_camera = Camera(self.bullet_client, cfg.third_person_camera)
    
    @contextmanager
    def disable_rendering(self): # speedup setting up scene
        rendering_enabled = self.bullet_client.rendering_enabled
        if rendering_enabled:
            self.bullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
            self.bullet_client.rendering_enabled = False
        yield
        if rendering_enabled:
            self.bullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
            self.bullet_client.rendering_enabled = True

    def reset(self, scene_id):
        self.bullet_client.resetSimulation() # remove all objects from the scene
        self.bullet_client.setGravity(*self.cfg.gravity)
        self.bullet_client.setPhysicsEngineParameter(fixedTimeStep=self.cfg.step_time, numSubSteps=self.cfg.substeps, deterministicOverlappingPairs=1)

        self.scene_id = scene_id
        self.frame = 0
        self.scene_data = load_scene_data(scene_id, table_height=self.cfg.table_height, stop_moving_frame=self.cfg.stop_moving_frame, frame_interval=self.cfg.frame_interval)
        self.target_object_stopped_because_of_dist = False

        with self.disable_rendering():
            self.ground_id = self.bullet_client.loadURDF("plane_implicit.urdf", basePosition=(0., 0., 0.))
            self.table.reset()
            self.panda.reset()
            self.hand.reset(self.scene_data["hand_name"], self.scene_data["hand_side"], self.scene_data["hand_path"], self.scene_data["hand_pose"])
            self.objects.reset(self.scene_data["object_names"], self.scene_data["object_paths"], self.scene_data["object_grasp_id"], self.scene_data["object_poses"])
            self.status_checker.reset()

        # bodies for visualization
        self.grasps: List[Grasp] = []
        self.spheres: List[Sphere] = []

    def get_visual_observation(self):
        return self.panda.get_visual_observation([self.objects.target_object.body_id, self.hand.body_id])

    def get_observation(self) -> Observation:
        # color, depth, segmentation, points = self.panda.get_visual_observation([self.objects.target_object.body_id, self.hand.body_id])
        # "color": color,
        # "depth": depth,
        # "segmentation": segmentation,
        # "object_points": points[0],
        # "hand_points": points[1]
        if self.cfg.show_camera:
            self.get_visual_observation() # to refresh the camera for visualization
        observation = Observation(
            frame=self.frame,
            world_to_ee=self.panda.get_world_to_ee(),
            joint_positions=self.panda.get_joint_positions(),
            get_visual_observation=self.get_visual_observation,
            env=self,
        )
        return observation

    # no disable_rendering because there are always multiple loadings together, so disable_rendering is placed there
    def load_grasp(self, pose_mat: NDArray[np.float64], color: Tuple[float]) -> Grasp:
        grasp_cfg: GraspConfig = OmegaConf.to_object(OmegaConf.structured(get_grasp_config(pose_mat=pose_mat, color=color)))
        grasp = Grasp(self.bullet_client, grasp_cfg)
        self.grasps.append(grasp)
        return grasp

    def clear_grasps(self):
        with self.disable_rendering():
            for grasp in self.grasps:
                grasp.clear()
        self.grasps = []

    # no disable_rendering because there are always multiple loadings together, so disable_rendering is placed there
    def load_sphere(self, pos: NDArray[np.float64], color: Tuple[float], scale: float) -> Sphere:
        sphere_cfg: SphereConfig = OmegaConf.to_object(OmegaConf.structured(SphereConfig(base_position=tuple(pos.tolist()), link_color=color, scale=scale)))
        sphere = Sphere(self.bullet_client, sphere_cfg)
        self.spheres.append(sphere)
        return sphere

    def clear_spheres(self):
        with self.disable_rendering():
            for sphere in self.spheres:
                sphere.clear()
        self.spheres = []
    
    def get_panda_object_dist(self) -> float:
        world_to_tip_pos = self.panda.get_tip_pos()
        world_to_object_pc = self.objects.target_object.get_world_to_object_pc()
        dists = np.square(world_to_object_pc-world_to_tip_pos).sum(axis=1)
        min_dist, min_dist_idx = np.sqrt(dists.min()), dists.argmin()
        return min_dist

    def sim_step(self, panda_dof_target_position, increase_frame=True):
        info = {}
        # pre step
        if increase_frame:
            self.frame += 1
        # self.table.pre_step()
        self.panda.pre_step(panda_dof_target_position)
        increase_hand_object_frame = increase_frame
        if self.cfg.stop_moving_dist is not None:
            panda_object_dist = self.get_panda_object_dist()
            if panda_object_dist < self.cfg.stop_moving_dist:
                increase_hand_object_frame = False
            self.target_object_stopped_because_of_dist = panda_object_dist < self.cfg.stop_moving_dist
            if self.cfg.verbose:
                print(f"frame: {self.frame}, panda object dist {panda_object_dist}")
        if increase_hand_object_frame:
            self.hand.pre_step(self.disable_rendering)
            self.objects.pre_step()
        # self.status_checker.pre_step()
        # self.third_person_camera.pre_step()
        self.bullet_client.stepSimulation()
        # post step
        # self.table.post_step()
        self.panda.post_step()
        # self.hand.post_step()
        self.objects.post_step()
        status, release = self.status_checker.post_step(self.table, self.panda, self.hand, self.objects, self.frame)
        self.third_person_camera.post_step()
        if release:
            self.objects.release()
        reward = float(status == EpisodeStatus.SUCCESS)
        done = status != 0
        if self.cfg.verbose and status != 0: print(f"frame {self.frame}, status {status}, reward {reward}, done {done}")
        info["status"] = status
        return reward, done, info
    
    def joint_step(self, panda_dof_target_position, repeat, increase_frame=True):
        " panda_dof_target_position: (9, ) "
        if self.cfg.verbose:
            print(f"in joint_step, frame={self.frame}", end=" ")
            for position in panda_dof_target_position:
                print(position, end=" ")
            print("")
        if self.cfg.show_trajectory:
            with self.disable_rendering():
                self.load_sphere(self.objects.target_object.get_world_to_obj()[:3, 3], color=(1., 0.75, 0., 1.), scale=0.1)
                self.load_sphere(self.panda.get_tip_pos(), color=(1., 0., 0., 1.), scale=0.1)
                self.load_grasp(self.panda.get_world_to_ee(), color=(1., 0., 0., 1.))
        for _ in range(repeat):
            reward, done, info = self.sim_step(panda_dof_target_position, increase_frame)
            if done: break
        return reward, done, info
    
    def ego_cartesian_step(self, action, repeat, increase_frame=True):
        " action: (7,) pos+euler+width "
        if self.cfg.verbose:
            print(f"in ego_cartesian_step, frame={self.frame}", end=" ")
            for action_i in action:
                print(action_i, end=" ")
            print("")
        panda_dof_target_position = self.panda.ego_cartesian_action_to_dof_target_position(pos=action[:3], orn=action[3:6], width=action[6], orn_type="euler")
        reward, done, info = self.joint_step(panda_dof_target_position, repeat, increase_frame)
        return reward, done, info

    def world_pos_step(self, action, repeat):
        " action: (4,) pos+width "
        panda_dof_target_position = self.panda.world_pos_action_to_dof_target_position(pos=action[:3], width=action[3])
        reward, done, info = self.joint_step(panda_dof_target_position, repeat)
        return reward, done, info
