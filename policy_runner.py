import os
import ray
import numpy as np
from numpy.typing import NDArray
from typing import TypedDict, List, Optional
import open3d as o3d
import code

from env.utils.scene import scene_id_to_dir
from env.status_checker import EpisodeStatus
from env.handover_env import GenH2RSim, Observation
from evaluate_config import EvaluateConfig

result_dtype = np.dtype({
    "names": ["scene_id", "status", "reward", "reached_frame", "done_frame", "num_steps"],
    "formats": [np.int32, np.int32, np.float32, np.int32, np.int32, np.int32],
    "offsets": [0, 4, 8, 12, 16, 20],
    "itemsize": 24,
})

def scene_id_to_demo_path(scene_id: int, demo_structure: str) -> str:
    scene_dir = scene_id_to_dir(scene_id, demo_structure)
    scene_path = os.path.join(scene_dir, f"{scene_id:08d}.npz")
    return scene_path

@ray.remote(num_cpus=1)
class Distributer:
    def __init__(self, scene_ids: List[int]):
        self.idx = 0
        self.scene_ids = scene_ids

    def get_next_task(self):
        if self.idx == len(self.scene_ids):
            return None
        print(f"idx={self.idx}, scene_id={self.scene_ids[self.idx]}")
        scene_id = self.scene_ids[self.idx]
        self.idx += 1
        return scene_id

class DemoData(TypedDict):
    frame: NDArray[np.int32]
    world_to_ee: NDArray[np.float32]
    world_to_object: NDArray[np.float32]
    world_to_hand: NDArray[np.float32]
    joint_positions: NDArray[np.float32]
    action: NDArray[np.float32]
    world_to_target_grasp: NDArray[np.float32]
    num_steps: int
    object_points_0: NDArray[np.float32] # and other steps
    hand_points_0: NDArray[np.float32]
    status: int
    reward: float
    reached_frame: int
    done_frame: int

class PolicyRunner:
    def __init__(self, cfg: EvaluateConfig, distributer: Distributer=None):
        self.cfg = cfg
        self.env = GenH2RSim(cfg.env)
        if cfg.policy == "offline":
            from policies.offline_policy import OfflinePolicy
            self.policy = OfflinePolicy(cfg.offline)
        elif cfg.policy == "pointnet2":
            from policies.pointnet2_policy import PointNet2Policy
            self.policy = PointNet2Policy(cfg.pointnet2)
        else:
            raise NotImplementedError
        self.distributer = distributer

        self.demo_array_keys = ["frame", "world_to_ee", "world_to_object", "world_to_hand", "joint_positions", "action", "world_to_target_grasp"]
        self.np_random = np.random.RandomState()
    
    def get_dart_action(self, step: int) -> Optional[NDArray[np.float64]]:
        if self.cfg.policy == "offline":
            if f"dart_action_{step}" in self.policy.traj_data:
                dart_action: NDArray[np.float64] = self.policy.traj_data[f"dart_action_{step}"]
            else:
                dart_action = None
        else:
            if self.cfg.dart and not self.policy.reached and step >= self.cfg.dart_min_step and step <= self.cfg.dart_max_step and self.np_random.uniform() <= self.cfg.dart_ratio:
                trans = self.np_random.uniform(-0.04, 0.04, size=(3,))
                rot = self.np_random.uniform(-0.2, 0.2, size=(3,))
                dart_action = np.concatenate([trans, rot, np.array([0.04])])
            else:
                dart_action = None
        return dart_action

    def init_demo_data(self, scene_id: int) -> str:
        self.demo_data = {key: [] for key in self.demo_array_keys}
        self.demo_data["num_steps"] = 0
        demo_path = os.path.join(self.cfg.demo_dir, scene_id_to_demo_path(scene_id, self.cfg.demo_structure))
        os.makedirs(os.path.dirname(demo_path), exist_ok=True)
        return demo_path

    def clip_object_hand_points(self, object_points, hand_points):
        num_points = object_points.shape[0]+hand_points.shape[0]
        num_object_points = object_points.shape[0]
        if num_points <= 1024:
            return object_points, hand_points
        selected_idxs = np.random.choice(range(num_points), size=1024, replace=False)
        object_points = object_points[selected_idxs[selected_idxs<num_object_points]]
        hand_points = hand_points[selected_idxs[selected_idxs>=num_object_points]-num_object_points]
        return object_points, hand_points

    def add_demo_data(self, stage: str, observation: Observation, action: NDArray, world_to_target_grasp: NDArray, dart_action: NDArray=None, demo_path: str=None): # depend on clip_object_hand_points
        step = self.demo_data["num_steps"]
        if dart_action is not None:
            self.demo_data[f"dart_action_{step}"] = dart_action
        if stage == "reach":
            self.demo_data["num_steps"] += 1
            self.demo_data[f"object_points_{step}"], self.demo_data[f"hand_points_{step}"] = self.clip_object_hand_points(*observation.get_visual_observation()[3])
            self.demo_data["frame"].append(observation.frame)
            self.demo_data["world_to_ee"].append(self.env.panda.get_world_to_ee())
            self.demo_data["world_to_object"].append(self.env.objects.target_object.get_world_to_obj())
            self.demo_data["world_to_hand"].append(self.env.hand.get_joint_positions())
            self.demo_data["joint_positions"].append(self.env.panda.get_joint_positions())
            self.demo_data["action"].append(action)
            self.demo_data["world_to_target_grasp"].append(world_to_target_grasp)
        if demo_path is not None and self.cfg.save_state:
            scene_id = int(demo_path.split("/")[-1][:-4])
            self.env.bullet_client.saveBullet(os.path.join(os.path.dirname(demo_path), f"{scene_id}_step_{step}.bullet"))
    
    def save_demo_data(self, demo_path: str, status: int, reward: float, reached_frame: int, done_frame: int):
        for key in self.demo_array_keys:
            if len(self.demo_data[key]) == 0:
                print(f"no data added for {demo_path}")
                self.demo_data[key] = None
            else:
                self.demo_data[key] = np.stack(self.demo_data[key], axis=0)
        self.demo_data.update({"status": status, "reward": reward, "reached_frame": reached_frame, "done_frame": done_frame})
        if status != EpisodeStatus.SUCCESS:
            for step in range(self.demo_data["num_steps"]):
                del self.demo_data[f"object_points_{step}"], self.demo_data[f"hand_points_{step}"]
        np.savez(demo_path, **self.demo_data)
        self.demo_data = {} # free the space

    def run(self, scene_id): # depend on init_demo_data, add_demo_data, save_demo_data
        self.env.reset(scene_id)
        self.np_random.seed(self.cfg.seed+scene_id)
        if self.cfg.policy in ["offline", "chomp"]:
            self.policy.reset(scene_id)
        else:
            self.policy.reset()
        step = 0
        if self.cfg.demo_dir is not None:
            demo_path = self.init_demo_data(scene_id)
            if self.cfg.record_ego_video:
                self.env.panda.camera.setup_video_writer(os.path.join(os.path.dirname(demo_path), f"{scene_id:08d}_ego_rgb.mp4"))
            if self.cfg.record_third_person_video:
                self.env.third_person_camera.setup_video_writer(os.path.join(os.path.dirname(demo_path), f"{scene_id:08d}_third_person_rgb.mp4"))
        while True:
            dart_action = self.get_dart_action(step)
            if dart_action is not None:
                reward, done, info = self.env.ego_cartesian_step(dart_action, self.policy.action_repeat_steps, increase_frame=False)
                if self.cfg.policy in ["OMG_original", "chomp"]:
                    self.policy.reset()
            self.env.clear_grasps()
            observation = self.env.get_observation()
            action, action_type, repeat, stage, info = self.policy.run_policy(observation)
            if "world_to_target_grasp" in info:
                world_to_target_grasp = info["world_to_target_grasp"]
                if self.cfg.show_target_grasp:
                    self.env.load_grasp(world_to_target_grasp, [0., 1., 0., 1.])
            else:
                world_to_target_grasp = np.nan*np.ones((4, 4), dtype=np.float32)
            if "input_pcd" in info: # offline.check_input_point_cloud
                assert self.cfg.demo_dir is not None
                if np.array(info["input_pcd"].points).shape[0] > 0:
                    o3d.io.write_point_cloud(os.path.join(os.path.dirname(demo_path), f"{scene_id:08d}_{step}_input_pcd.pcd"), info["input_pcd"])
            if self.cfg.demo_dir is not None:
                self.add_demo_data(stage, observation, action, world_to_target_grasp, dart_action, demo_path)
            if self.cfg.policy == "offline":
                # print(f"actual joint positions {self.env.panda.get_joint_positions()}")
                # print(f"loaded joint positions {self.policy.traj_data['joint_positions'][step]}")
                # print(f"actual world to obj {self.env.objects.target_object.get_world_to_obj()}")
                # print(f"loaded world to obj {self.policy.traj_data['world_to_object'][step]}")
                saved_state_path = os.path.join(os.path.dirname(self.policy.data_path), f"{scene_id}_step_{step}.bullet")
                if os.path.exists(saved_state_path):
                    self.env.bullet_client.saveBullet("tmp.bullet")
                    print(f"stage {stage} step {step} diff", os.system(f"diff {saved_state_path} tmp.bullet"))
            step += 1
            if self.cfg.verbose:
                print(f"step {step}, frame {observation.frame}, action {action}, repeat {repeat}")
            if action_type == "joint":
                reward, done, info = self.env.joint_step(action, repeat)
            elif action_type == "ego_cartesian":
                reward, done, info = self.env.ego_cartesian_step(action, repeat)
            elif action_type == "world_pos":
                reward, done, info = self.env.world_pos_step(action, repeat)
            else:
                raise NotImplementedError
            if done:
                break
        reached_frame, done_frame, status = self.policy.reached_frame, self.env.frame, info["status"]
        print(f"scene_id {scene_id}, step {step}, status {status}, reward {reward}, reached frame {reached_frame}, done frame {done_frame}")
        if self.cfg.policy == "offline":
            if status != self.policy.traj_data["status"]:
                print(f"scene_id {scene_id}, status {status}, loaded status {self.policy.traj_data['status']}")
        if self.cfg.demo_dir is not None:
            self.save_demo_data(demo_path, status, reward, reached_frame, done_frame)
            if self.cfg.record_ego_video:
                self.env.panda.camera.close_video_writer()
            if self.cfg.record_third_person_video:
                self.env.third_person_camera.close_video_writer()
        return status, reward, reached_frame, done_frame, step

    def work(self):
        results = []
        while True:
            scene_id = ray.get(self.distributer.get_next_task.remote())
            if scene_id is None: break
            result: NDArray[result_dtype] = np.empty((1, ), dtype=result_dtype)
            result["scene_id"] = scene_id
            demo_data_existed = False
            if self.cfg.demo_dir is not None:
                demo_path = os.path.join(self.cfg.demo_dir, scene_id_to_demo_path(scene_id, self.cfg.demo_structure))
                if os.path.exists(demo_path):
                    demo_data_existed = True
            if demo_data_existed and not self.cfg.overwrite_demo:
                demo_data = np.load(demo_path)
                result["status"], result["reward"], result["reached_frame"], result["done_frame"], result["num_steps"] = demo_data["status"], demo_data["reward"], demo_data["reached_frame"], demo_data["done_frame"], demo_data["num_steps"]
            else:
                result["status"], result["reward"], result["reached_frame"], result["done_frame"], result["num_steps"] = self.run(scene_id)
            results.append(result)
        if len(results) > 0:
            results: NDArray[result_dtype] = np.stack(results)
        else:
            results: NDArray[result_dtype] = np.empty((0, 1), dtype=result_dtype)
        return results

def get_policy_runner_remote(num_gpus):
    @ray.remote(num_cpus=1, num_gpus=num_gpus)
    class PolicyRunnerRemote(PolicyRunner):
        pass
    return PolicyRunnerRemote
