import numpy as np
from typing import Tuple, Optional, Union, List
from dataclasses import dataclass
from omegaconf import OmegaConf, MISSING
import pybullet
from pybullet_utils.bullet_client import BulletClient
from scipy.spatial.transform import Rotation as Rt
import imageio
import code

# def get_view_matrix(pos: Optional[Tuple[float]], orn: Optional[Tuple[float]], target: Optional[Tuple[float]], up_vector: Optional[Tuple[float]]) -> Tuple[float]:
#     if pos is not None and orn is not None:
#         assert target is None and up_vector is None
#         urdf_to_opengl = Rt.from_euler("XYZ", (-np.pi/2, 0.0, -np.pi)).as_matrix()
#         R = Rt.from_quat(orn).as_matrix()@urdf_to_opengl
#         t = -np.array(pos).dot(R)
#         view_matrix = np.eye(4, dtype=float)
#         view_matrix[:3, :3] = R
#         view_matrix[3, :3] = t
#         view_matrix = tuple(view_matrix.flatten())
#         return view_matrix
#     elif pos is not None and target is not None and up_vector is not None:
#         assert orn is None
#         view_matrix = pybullet.computeViewMatrix(cameraEyePosition=pos, cameraTargetPosition=target, cameraUpVector=up_vector)
#         return view_matrix
#     else:
#         raise NotImplementedError
# OmegaConf.register_new_resolver("get_view_matrix", get_view_matrix)

@dataclass
class CameraConfig:
    width: int = MISSING
    height: int = MISSING
    fov: float = MISSING
    near: float = MISSING
    far: float = MISSING
    pos: Optional[Tuple[float]] = None
    orn: Optional[Tuple[float]] = None
    target: Optional[Tuple[float]] = None
    up_vector: Optional[Tuple[float]] = None
    step_time: float = MISSING
    @property
    def view_matrix(self) -> Tuple[float]:
        if self.pos is not None and self.orn is not None:
            assert self.target is None and self.up_vector is None
            urdf_to_opengl = Rt.from_euler("XYZ", (-np.pi/2, 0.0, -np.pi)).as_matrix()
            R = Rt.from_quat(self.orn).as_matrix()@urdf_to_opengl
            t = -np.array(self.pos).dot(R)
            view_matrix = np.eye(4, dtype=float)
            view_matrix[:3, :3] = R
            view_matrix[3, :3] = t
            view_matrix = tuple(view_matrix.flatten())
            return view_matrix
        elif self.pos is not None and self.target is not None and self.up_vector is not None:
            assert self.orn is None
            view_matrix = pybullet.computeViewMatrix(cameraEyePosition=self.pos, cameraTargetPosition=self.target, cameraUpVector=self.up_vector)
            return view_matrix
        else:
            raise NotImplementedError
    # view_matrix: Tuple[float] = "${get_view_matrix:${.pos},${.orn},${.target},${.up_vector}}" # the "." is necessary in nested configs, see https://github.com/omry/omegaconf/issues/1099#issuecomment-1624496194

class Camera:
    def __init__(self, bullet_client: BulletClient, cfg: CameraConfig):
        self.bullet_client = bullet_client
        self.cfg = cfg

        self.projection_matrix = self.bullet_client.computeProjectionMatrixFOV(fov=self.cfg.fov, aspect=self.cfg.width/self.cfg.height, nearVal=self.cfg.near, farVal=self.cfg.far)

        self.urdf_to_opengl = Rt.from_euler("XYZ", (-np.pi/2, 0.0, -np.pi)).as_matrix()

        K = np.eye(3, dtype=float)
        K[0, 0] = self.cfg.width/2/np.tan(np.deg2rad(self.cfg.fov)*self.cfg.width/self.cfg.height/2)
        K[1, 1] = self.cfg.height/2/np.tan(np.deg2rad(self.cfg.fov)/2)
        K[0, 2] = self.cfg.width/2
        K[1, 2] = self.cfg.height/2
        K_inv = np.linalg.inv(K)
        x, y = np.meshgrid(np.arange(self.cfg.width), np.arange(self.cfg.height))
        x, y = x.astype(float), y.astype(float)
        ones = np.ones((self.cfg.height, self.cfg.width), dtype=float)
        xy1s = np.stack((x, y, ones), axis=2).reshape(self.cfg.width*self.cfg.height, 3).T
        self.deproject = np.matmul(K_inv, xy1s).T

    def update_pose(self, pos, orn):
        self.cfg.pos, self.cfg.orn = pos, orn

    def process_depth(self, depth):
        # depth_1_mask = depth==1.0
        depth = self.cfg.far*self.cfg.near/(self.cfg.far-(self.cfg.far-self.cfg.near)*depth)
        # depth[depth_1_mask] = 0.0
        return depth

    def get_points(self, depth, mask):
        points = np.tile(depth[mask].reshape(-1, 1), (1, 3))*self.deproject[mask.reshape(-1), :]
        return points

    def render(self, segmentation_ids=[]):
        # code.interact(local=dict(globals(), **locals()))
        _, _, color, depth, segmentation = self.bullet_client.getCameraImage(width=self.cfg.width, height=self.cfg.height, viewMatrix=self.cfg.view_matrix, projectionMatrix=self.projection_matrix, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL) # (224, 224, 4), (224, 224), (224, 224)
        depth = self.process_depth(depth)
        points = []
        for segmentation_id in segmentation_ids:
            points.append(self.get_points(depth, segmentation==segmentation_id))
        return color, depth, segmentation, points

    def setup_video_writer(self, video_filename, fps=20):
        print(f"setting up video writer to {video_filename}...")
        self.video_writer = imageio.get_writer(video_filename, mode='I', fps=fps)
        self.save_video_interval = int(1/fps/self.cfg.step_time)
        self.frame = 0
    
    def close_video_writer(self):
        print("closing video writer...")
        if self.video_writer is not None:
            self.video_writer.close()
        delattr(self, "video_writer")

    def pre_step(self):
        pass

    def post_step(self):
        if not hasattr(self, "video_writer"):
            return
        self.frame += 1
        if self.frame % self.save_video_interval == 0:
            color, depth, segmentation, points = self.render([])
            self.video_writer.append_data(color)

def debug():
    camera_cfg = OmegaConf.structured(CameraConfig(pos=(0., 0., 0.), orn=(0., 0., 0., 1.)))
    code.interact(local=dict(globals(), **locals()))

if __name__ == "__main__":
    debug()

"""
DISPLAY="localhost:11.0" python -m env.camera
"""