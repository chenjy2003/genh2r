import os
import numpy as np
from numpy.typing import NDArray
from typing import TypedDict, List, Optional
from scipy.spatial.transform import Rotation as Rt

def scene_id_to_hierarchical_dir(scene_id: int) -> str:
    scene_id_str = f"{scene_id:08d}" # "12345678"
    hierarchical_dir = os.path.join(*(scene_id_str[i:i+2] for i in range(0, len(scene_id_str)-2, 2))) # "12/34/56/"
    return hierarchical_dir

def scene_id_to_dir(scene_id: int, demo_structure: str) -> str:
    if demo_structure == "hierarchical":
        scene_dir = scene_id_to_hierarchical_dir(scene_id)
    elif demo_structure == "flat":
        scene_dir = ""
    else:
        raise NotImplementedError
    return scene_dir

env_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scene_root_dir = os.path.join(env_dir, "data", "scene")
def scene_id_to_scene_dir(scene_id: int) -> str:
    scene_path = os.path.join(scene_root_dir, scene_id_to_hierarchical_dir(scene_id))
    return scene_path

class SceneData(TypedDict):
    hand_name: str
    hand_side: str
    hand_path: str
    hand_pose: NDArray[np.float32]
    object_names: List[str]
    object_paths: List[str]
    object_grasp_id: int
    object_poses: NDArray[np.float32]
    endpoints: Optional[NDArray[np.int64]]

objects_dir = os.path.join(env_dir, "data", "assets", "objects")
hand_dir = os.path.join(env_dir, "data", "assets", "hand")
def load_scene_data(scene_id: int, table_height: float=0., stop_moving_frame: Optional[int]=None, frame_interval: int=1) -> SceneData:
    scene_dir = scene_id_to_scene_dir(scene_id)
    scene_data_path = os.path.join(scene_dir, f"{scene_id:08d}.npz")
    scene_data = dict(np.load(scene_data_path)) # "hand_name", "hand_side", "hand_path", "hand_pose", "object_names", "object_paths", "object_grasp_id", "object_poses", "endpoints", "source"
    source = scene_data["source"]
    scene_data["hand_name"] = scene_data["hand_name"].item()
    scene_data["hand_side"] = scene_data["hand_side"].item()
    scene_data["hand_path"] = os.path.join(hand_dir, scene_data["hand_path"].item())
    if source == "dexycb":
        scene_data["hand_pose"] = scene_data["hand_pose"][::frame_interval] # (T, 51)
    elif source == "genh2r":
        hand_pose = scene_data["hand_pose"][::frame_interval]
        scene_data["hand_pose"] = np.concatenate([hand_pose, np.tile(scene_data["hand_theta"], (hand_pose.shape[0], 1))], axis=1)
    else:
        raise NotImplementedError
    scene_data["object_names"] = scene_data["object_names"].tolist()
    scene_data["object_paths"] = [os.path.join(objects_dir, object_path) for object_path in scene_data["object_paths"].tolist()]
    scene_data["object_poses"] = scene_data["object_poses"][:, ::frame_interval] # (#objects, T, 6)

    if stop_moving_frame is not None:
        scene_data["hand_pose"] = scene_data["hand_pose"][:stop_moving_frame]
        scene_data["object_poses"] = scene_data["object_poses"][:, :stop_moving_frame]
    if table_height != 0.:
        hand_nonzero_mask = np.any(scene_data["hand_pose"]!=0, axis=1)
        hand_nonzeros = np.where(hand_nonzero_mask)[0]
        hand_start_frame = hand_nonzeros[0]
        hand_end_frame = hand_nonzeros[-1]+1
        scene_data["hand_pose"][hand_start_frame:hand_end_frame, 2] += table_height
        scene_data["object_poses"][:, :, 2] += table_height
    
    if "endpoints" in scene_data:
        scene_data["endpoints"] //= frame_interval
    else:
        scene_data["endpoints"] = None

    return scene_data

def six_d_to_mat(six_d: NDArray[np.float64]) -> NDArray[np.float64]:
    " (..., 6) "
    shape_prefix = six_d.shape[:-1]
    mat = np.zeros(shape_prefix+(4, 4))
    mat[..., :3, 3] = six_d[..., :3]
    mat[..., :3, :3] = Rt.from_euler("XYZ", six_d[..., 3:].reshape(-1, 3)).as_matrix().reshape(shape_prefix+(3, 3))
    mat[..., 3, 3] = 1
    return mat

def mat_to_six_d(mat: NDArray[np.float64]) -> NDArray[np.float64]:
    " (..., 4, 4) "
    shape_prefix = mat.shape[:-2]
    return np.concatenate([mat[..., :3, 3], Rt.from_matrix(mat[..., :3, :3].reshape(-1, 3, 3)).as_euler("XYZ").reshape(shape_prefix + (3, ))], axis=-1)