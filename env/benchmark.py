import numpy as np
from typing import List, Optional
from .utils.scene import load_scene_data

def get_dexycb_scene_ids(setup: str, split: str) -> List[int]:
    _EVAL_SKIP_OBJECT = [0, 15]
    if setup == "s0": # Seen subjects, camera views, grasped objects.
        if split == "train":
            subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            sequence_ind = [i for i in range(100) if i % 5 != 4]
        if split == "val":
            subject_ind = [0, 1]
            sequence_ind = [i for i in range(100) if i % 5 == 4]
        if split == "test":
            subject_ind = [2, 3, 4, 5, 6, 7, 8, 9]
            sequence_ind = [i for i in range(100) if i % 5 == 4]
        mano_side = ["right", "left"]
    elif setup == "s1": # Unseen subjects.
        if split == "train":
            subject_ind = [0, 1, 2, 3, 4, 5, 9]
        if split == "val":
            subject_ind = [6]
        if split == "test":
            subject_ind = [7, 8]
        sequence_ind = [*range(100)]
        mano_side = ["right", "left"]
    elif setup == "s2": # Unseen handedness.
        if split == "train":
            subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            mano_side = ["right"]
        if split == "val":
            subject_ind = [0, 1]
            mano_side = ["left"]
        if split == "test":
            subject_ind = [2, 3, 4, 5, 6, 7, 8, 9]
            mano_side = ["left"]
        sequence_ind = [*range(100)]
    elif setup == "s3": # Unseen grasped objects.
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        if split == "train":
            sequence_ind = [i for i in range(100) if i // 5 not in (3, 7, 11, 15, 19)]
        if split == "val":
            sequence_ind = [i for i in range(100) if i // 5 in (3, 19)]
        if split == "test":
            sequence_ind = [i for i in range(100) if i // 5 in (7, 11, 15)]
        mano_side = ["right", "left"]
    
    scene_ids = []
    for i in range(1000):
        if i // 5 % 20 in _EVAL_SKIP_OBJECT:
            continue
        if i // 100 in subject_ind and i % 100 in sequence_ind:
            if mano_side == ["right", "left"]:
                scene_ids.append(i)
            else:
                if i % 5 != 4:
                    if (i % 5 in (0, 1) and mano_side == ["right"] or i % 5 in (2, 3) and mano_side == ["left"]):
                        scene_ids.append(i)
                elif mano_side[0] == load_scene_data(i)["hand_side"]:
                    scene_ids.append(i)
    return scene_ids

def get_genh2rsim_scene_ids(setup: str, split: str, start_object_idx: Optional[int]=None, end_object_idx: Optional[int]=None, start_traj_idx: Optional[int]=None, end_traj_idx: Optional[int]=None) -> List[int]:
    if setup == "t0":
        start_id = 1000000
        num_objects = 8836 # number of acronym objects
        num_trajs_per_object = 128
        all_scene_ids = np.arange(start_id, start_id+num_objects*num_trajs_per_object)
        if split == "train":
            object_mask = np.arange(num_objects)%10<8
        elif split == "val":
            object_mask = np.arange(num_objects)%10==8
        elif split == "test":
            object_mask = np.arange(num_objects)%10==9
        else:
            raise ValueError(split)
        if start_object_idx is not None and end_object_idx is not None:
            object_idxs = np.arange(num_objects)
            object_mask &= (start_object_idx<=object_idxs)&(object_idxs<end_object_idx)
        traj_mask = np.ones(num_trajs_per_object, dtype=bool)
        if start_traj_idx is not None and end_traj_idx is not None:
            traj_idxs = np.arange(num_trajs_per_object)
            traj_mask &= (start_traj_idx<=traj_idxs)&(traj_idxs<end_traj_idx)
        scene_ids = all_scene_ids.reshape((num_trajs_per_object, num_objects))[traj_mask][:, object_mask].reshape(-1)
        return scene_ids.tolist()

def get_scene_ids(setup: str, split: str, start_object_idx: Optional[int]=None, end_object_idx: Optional[int]=None, start_traj_idx: Optional[int]=None, end_traj_idx: Optional[int]=None) -> List[int]:
    if setup in ["s0", "s1", "s2", "s3"]:
        return get_dexycb_scene_ids(setup, split)
    elif setup in ["t0"]:
        return get_genh2rsim_scene_ids(setup, split, start_object_idx, end_object_idx, start_traj_idx, end_traj_idx)
    else:
        raise ValueError(setup)
