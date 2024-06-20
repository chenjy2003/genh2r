import numpy as np
from transforms3d.euler import mat2euler
import json
import os
from tqdm import tqdm
import code

from ..utils.scene import scene_id_to_scene_dir
from .sdf import gen_sdf

OBJ_ID_TO_NAME = {
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    20: "052_extra_large_clamp",
    21: "061_foam_brick",
}

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def rotZ(rotz):
    RotZ = np.array(
        [
            [np.cos(rotz), -np.sin(rotz), 0, 0],
            [np.sin(rotz), np.cos(rotz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    return RotZ

def ycb_special_case(pose_grasp, name):
    if name == '037_scissors': # only accept top down for edge cases
        z_constraint = np.where((np.abs(pose_grasp[:, 2, 3]) > 0.09) * \
                 (np.abs(pose_grasp[:, 1, 3]) > 0.02) * (np.abs(pose_grasp[:, 0, 3]) < 0.05)) 
        pose_grasp = pose_grasp[z_constraint[0]]
        top_down = []
        
        for pose in pose_grasp:
            top_down.append(mat2euler(pose[:3, :3]))
        
        top_down = np.array(top_down)[:,1]
        rot_constraint = np.where(np.abs(top_down) > 0.06) 
        pose_grasp = pose_grasp[rot_constraint[0]]
    
    elif name == '024_bowl' or name == '025_mug':
        if name == '024_bowl':
            angle = 30
        else:
            angle = 15
        top_down = []
        for pose in pose_grasp:
            top_down.append(mat2euler(pose[:3, :3]))
        top_down = np.array(top_down)[:,1]
        rot_constraint = np.where(np.abs(top_down) > angle * np.pi / 180)
        pose_grasp = pose_grasp[rot_constraint[0]]
    return pose_grasp

ycb_grasps_dir = os.path.join(data_dir, "tmp", "ycb_grasps")
def process_object(object_name: str):
    object_dir = os.path.join(data_dir, "assets", "objects", "ycb", object_name)
    # grasps
    object_grasp_path_original = os.path.join(ycb_grasps_dir, object_name+".npy")
    object_grasp_path = os.path.join(object_dir, "grasps.npy")
    if os.path.exists(object_grasp_path):
        pass
    elif not os.path.exists(object_grasp_path_original):
        print(f"grasps not found for {object_name}")
    else:
        simulator_grasp = np.load(
            object_grasp_path_original,
            allow_pickle=True,
            fix_imports=True,
            encoding="bytes",
        )
        pose_grasp = simulator_grasp.item()[b"transforms"]
        offset_pose = np.array(rotZ(np.pi / 2))  # and
        pose_grasp = np.matmul(pose_grasp, offset_pose)  # flip x, y
        # print(f"load {pose_grasp.shape[0]} grasps")
        pose_grasp = ycb_special_case(pose_grasp, object_name)
        np.save(object_grasp_path, pose_grasp)
    # sdf
    object_sdf_path = os.path.join(object_dir, "sdf.npz")
    if not os.path.exists(object_sdf_path):
        gen_sdf(os.path.join(object_dir, "model_normalized_convex.obj"))

dexycb_data_dir = os.path.join(data_dir, "tmp", "dex-ycb-cache")
def process(scene_id: int):
    meta_path = os.path.join(dexycb_data_dir, f"meta_{scene_id:03d}.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    pose_path = os.path.join(dexycb_data_dir, f"pose_{scene_id:03d}.npz")
    pose = np.load(pose_path)

    scene_dir = scene_id_to_scene_dir(scene_id)
    os.makedirs(scene_dir, exist_ok=True)
    scene_data_path = os.path.join(scene_dir, f"{scene_id:08d}.npz")

    hand_name = meta["name"].split("/")[0]
    hand_side = meta["mano_sides"][0]
    hand_path = os.path.join(f"{hand_name}_{hand_side}", "mano.urdf")
    object_names = []
    object_paths = []
    for object_id in meta["ycb_ids"]:
        object_name = OBJ_ID_TO_NAME[object_id]
        object_path = os.path.join("ycb", object_name, "model_normalized.urdf")
        object_names.append(object_name)
        object_paths.append(object_path)
    object_names = np.array(object_names)
    scene_data = {
        "hand_name": hand_name,
        "hand_side": hand_side,
        "hand_path": hand_path,
        "hand_pose": pose["pose_m"][:, 0],
        "object_names": object_names,
        "object_paths": object_paths,
        "object_grasp_id": meta["ycb_grasp_ind"],
        "object_poses": pose["pose_y"].transpose(1, 0, 2),
        "source": "dexycb",
    }
    np.savez(scene_data_path, **scene_data)

def main():
    for key, object_name in OBJ_ID_TO_NAME.items():
        process_object(object_name)

    for i in tqdm(range(1000)):
        process(i)

if __name__ == "__main__":
    main()
