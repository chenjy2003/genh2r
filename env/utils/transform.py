import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Union, List
from scipy.spatial.transform import Rotation as Rt

def se3_transform_pc(T: NDArray[np.float64], points: NDArray[np.float64]) -> NDArray[np.float64]:
    " T: (4, 4) points: (..., 3) "
    return points@T[:3, :3].transpose()+T[:3, 3]

def pos_ros_quat_to_mat(pos: Union[NDArray[np.float64], List[float]], ros_quat: Union[NDArray[np.float64], List[float]]) -> NDArray[np.float64]:
    " pos: (..., 3) ros_quat: (..., 4) xyzw "
    if not isinstance(pos, np.ndarray):
        pos = np.array(pos, dtype=np.float64)
    if not isinstance(ros_quat, np.ndarray):
        ros_quat = np.array(ros_quat, dtype=np.float64)
    mat = np.tile(np.eye(4), pos.shape[:-1]+(1, 1))
    mat[..., :3, :3] = Rt.from_quat(ros_quat.reshape(-1, 4)).as_matrix().reshape(ros_quat.shape[:-1]+(3, 3))
    mat[..., :3, 3] = pos
    return mat

def mat_to_pos_ros_quat(mat):
    pos = mat[:3, 3].copy()
    quat = Rt.from_matrix(mat[:3, :3]).as_quat()
    return pos, quat

def pos_euler_to_mat(pos, euler):
    mat = np.eye(4)
    mat[:3, 3] = pos
    mat[:3, :3] = Rt.from_euler("xyz", euler).as_matrix() # extrinsic euler. equivalent to transforms3d.euler.euler2mat
    return mat

def mat_to_pos_euler(mat: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    pos = mat[:3, 3].copy()
    euler = Rt.from_matrix(mat[:3, :3]).as_euler("xyz")
    return pos, euler

def se3_inverse(mat: NDArray[np.float64]) -> NDArray[np.float64]:
    R = mat[:3, :3]
    mat_inv = np.eye(4, dtype=mat.dtype)
    mat_inv[:3, :3] = R.transpose()
    mat_inv[:3, 3] = -R.transpose()@mat[:3, 3]
    return mat_inv

def to_ros_quat(tf_quat):  # wxyz -> xyzw
    quat = np.zeros(4)
    quat[-1] = tf_quat[0]
    quat[:-1] = tf_quat[1:]
    return quat

def to_tf_quat(ros_quat):  # xyzw -> wxyz
    quat = np.zeros(4)
    quat[0] = ros_quat[-1]
    quat[1:] = ros_quat[:-1]
    return quat
