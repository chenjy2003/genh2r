import numpy as np
from pybullet_utils.bullet_client import BulletClient
from dataclasses import dataclass
from omegaconf import MISSING
from typing import Tuple
import code

from .table import Table
from .panda import Panda
from .hand import Hand
from .objects import Object, Objects
from .utils.transform import se3_transform_pc, pos_ros_quat_to_mat, se3_inverse

@dataclass
class StatusCheckerConfig:
    table_height: float = MISSING # "${..table_height}"
    max_frames: int = MISSING # "${..max_frames}"
    # release object check
    release_force_thresh: float = 0.0
    release_time_thresh: float = 0.1
    release_step_thresh: int = "${divide_to_int:${.release_time_thresh},${..step_time}}" # divide_to_int is defined in handover_env.py
    release_contact_region_range_x: Tuple[float] = (-0.0110, +0.0110)
    release_contact_region_range_y: Tuple[float] = (-0.0090, +0.0090)
    release_contact_region_range_z: Tuple[float] = (+0.0000, +0.0550)
    release_displacement_thresh: float = 0.03
    # draw_release_contact = False
    # release_contact_region_color = (0.85, 0.19, 0.21, 0.5)
    # release_contact_vertex_radius = 0.001
    # release_contact_vertex_color = (0.85, 0.19, 0.21, 1.0)

    # collision check for two failure cases: human hand collision and drop
    contact_force_thresh: float = 0.0

    # success check
    goal_center: Tuple[float] = (0.61, -0.20, 1.25)
    goal_radius: float = 0.15
    # draw_goal = False
    goal_color: Tuple[float] = (0.85, 0.19, 0.21, 0.5)
    success_time_thresh: float = 0.1
    success_step_thresh: int = "${divide_to_int:${.success_time_thresh},${..step_time}}" # divide_to_int is defined in handover_env.py

    verbose: bool = False

class EpisodeStatus:
    SUCCESS = 1
    FAILURE_HUMAN_CONTACT = 2
    FAILURE_OBJECT_DROP = 4
    FAILURE_TIMEOUT = 8

class StatusChecker:
    def __init__(self, bullet_client: BulletClient, cfg: StatusCheckerConfig):
        self.bullet_client = bullet_client
        self.cfg = cfg

        self.DTYPE = np.dtype(
            {
                "names": [
                    "body_id_a",
                    "body_id_b",
                    "link_id_a",
                    "link_id_b",
                    "position_a_world",
                    "position_b_world",
                    "position_a_link",
                    "position_b_link",
                    "normal",
                    "force",
                ],
                "formats": [
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                    [("x", np.float32), ("y", np.float32), ("z", np.float32)],
                    [("x", np.float32), ("y", np.float32), ("z", np.float32)],
                    [("x", np.float32), ("y", np.float32), ("z", np.float32)],
                    [("x", np.float32), ("y", np.float32), ("z", np.float32)],
                    [("x", np.float32), ("y", np.float32), ("z", np.float32)],
                    np.float32,
                ],
                "offsets": [0, 4, 8, 12, 16, 28, 40, 52, 64, 76],
                "itemsize": 80,
            }
        )
    
    def reset(self):
        self.release_step_counter_passive = 0
        self.release_step_counter_active = 0
        self.dropped = False
        self.success_step_counter = 0
    
    def get_contact(self):
        contact_points = self.bullet_client.getContactPoints()
        self.contact = np.empty(len(contact_points), dtype=self.DTYPE)
        if len(contact_points) == 0:
            return
        body_id_a, body_id_b, link_id_a, link_id_b, position_a_world, position_b_world, position_a_link, position_b_link, normal, force = [], [], [], [], [], [], [], [], [], []
        for contactFlag, bodyUniqueIdA, bodyUniqueIdB, linkIndexA, linkIndexB, positionOnA, positionOnB, contactNormalOnB, contactDistance, normalForce, lateralFriction1, lateralFrictionDir1, lateralFriction2, lateralFrictionDir2 in contact_points:
            body_id_a.append(bodyUniqueIdA) # if bodyUniqueIdA != self.ground_id else -1)
            body_id_b.append(bodyUniqueIdB) # if bodyUniqueIdB != self.ground_id else -1)
            link_id_a.append(linkIndexA)
            link_id_b.append(linkIndexB)
            position_a_world.append(positionOnA)
            position_b_world.append(positionOnB)
            position_a_link.append(np.nan)
            position_b_link.append(np.nan)
            normal.append(contactNormalOnB)
            force.append(normalForce)
        # code.interact(local=dict(globals(), **locals()))
        self.contact["body_id_a"] = body_id_a
        self.contact["body_id_b"] = body_id_b
        self.contact["link_id_a"] = link_id_a
        self.contact["link_id_b"] = link_id_b
        self.contact["position_a_world"] = position_a_world
        self.contact["position_b_world"] = position_b_world
        self.contact["position_a_link"] = position_a_link
        self.contact["position_b_link"] = position_b_link
        self.contact["normal"] = normal
        self.contact["force"] = force
    
    def check_object_release(self, target_object: Object, panda: Panda, frame: int):
        object_id, panda_id = target_object.body_id, panda.body_id # if hand is not loaded, hand.body_id = None
        contact = self.contact[self.contact["force"] > self.cfg.release_force_thresh].copy()

        if len(contact) == 0:
            contact_panda_release_region = [False] * len(panda.fingers_link_id)
            contact_panda_body = False
        else:
            contact_1 = contact[(contact["body_id_a"] == object_id) & (contact["body_id_b"] == panda_id)]
            contact_2 = contact[(contact["body_id_a"] == panda_id) & (contact["body_id_b"] == object_id)]
            contact_2[["body_id_a", "body_id_b"]] = contact_2[["body_id_b", "body_id_a"]]
            contact_2[["link_id_a", "link_id_b"]] = contact_2[["link_id_b", "link_id_a"]]
            contact_2[["position_a_world", "position_b_world"]] = contact_2[["position_b_world", "position_a_world"]]
            contact_2[["position_a_link", "position_b_link"]] = contact_2[["position_b_link", "position_a_link"]]
            contact_2["normal"]["x"] *= -1
            contact_2["normal"]["y"] *= -1
            contact_2["normal"]["z"] *= -1
            contact = np.concatenate((contact_1, contact_2))

            contact_panda_body = len(contact) > 0
            contact_panda_release_region = []

            for link_id in panda.fingers_link_id:
                contact_link = contact[contact["link_id_b"] == link_id]
                if len(contact_link) == 0:
                    contact_panda_release_region.append(False)
                    continue
                if np.any(np.isnan(contact_link["position_b_link"]["x"])):
                    pos, orn = panda.get_link_pos_orn(link_id)
                    world_to_link = pos_ros_quat_to_mat(pos, orn)
                    link_to_world = se3_inverse(world_to_link)
                    position = np.ascontiguousarray(contact_link["position_b_world"]).view(np.float32).reshape(-1, 3)
                    position = se3_transform_pc(link_to_world, position)
                else:
                    position = (
                        np.ascontiguousarray(contact_link["position_b_link"])
                        .view(np.float32)
                        .reshape(-1, 3)
                    )

                is_in_release_region = (
                    (position[:, 0] > self.cfg.release_contact_region_range_x[0])
                    & (position[:, 0] < self.cfg.release_contact_region_range_x[1])
                    & (position[:, 1] > self.cfg.release_contact_region_range_y[0])
                    & (position[:, 1] < self.cfg.release_contact_region_range_y[1])
                    & (position[:, 2] > self.cfg.release_contact_region_range_z[0])
                    & (position[:, 2] < self.cfg.release_contact_region_range_z[1])
                )
                contact_panda_release_region.append(np.any(is_in_release_region))

        if not any(contact_panda_release_region) and contact_panda_body:
            self.release_step_counter_passive += 1
        else:
            self.release_step_counter_passive = 0

        if all(contact_panda_release_region):
            self.release_step_counter_active += 1
        else:
            self.release_step_counter_active = 0
        if self.cfg.verbose and self.release_step_counter_passive+self.release_step_counter_active > 0:
            print(f"frame: {frame}, release step counter passive: {self.release_step_counter_passive}, release step counter active: {self.release_step_counter_active}")
        
        target_real_displacement = np.abs(target_object.pose[target_object.frame, :3]-target_object.get_joint_positions()[:3]).max()

        return self.release_step_counter_passive >= self.cfg.release_step_thresh or self.release_step_counter_active >= self.cfg.release_step_thresh or target_real_displacement >= self.cfg.release_displacement_thresh

    def check_panda_hand_collision(self, panda: Panda, hand: Hand):
        panda_id, hand_id = panda.body_id, hand.body_id # if hand is not loaded, hand.body_id = None
        contact_1 = self.contact[(self.contact["body_id_a"] == hand_id) & (self.contact["body_id_b"] == panda_id)]
        contact_2 = self.contact[(self.contact["body_id_a"] == panda_id) & (self.contact["body_id_b"] == hand_id)]
        contact = np.concatenate((contact_1, contact_2))
        if contact.shape[0] > 0 and contact["force"].max() > self.cfg.contact_force_thresh:
            if self.cfg.verbose: print("detect pand hand collision")
            return True
        else:
            return False

    def check_object_dropped(self, objects: Objects, panda: Panda, table: Table) -> Tuple[bool, bool]:
        if self.dropped:
            return True, False
        target_object_id = objects.target_object.body_id
        contact_1 = self.contact[self.contact["body_id_a"] == target_object_id].copy()
        contact_2 = self.contact[self.contact["body_id_b"] == target_object_id].copy()
        contact_2[["body_id_a", "body_id_b"]] = contact_2[["body_id_b", "body_id_a"]]
        contact_2[["link_id_a", "link_id_b"]] = contact_2[["link_id_b", "link_id_a"]]
        contact_2[["position_a_world", "position_b_world"]] = contact_2[["position_b_world", "position_a_world"]]
        contact_2[["position_a_link", "position_b_link"]] = contact_2[["position_b_link", "position_a_link"]]
        contact_2["normal"]["x"] *= -1
        contact_2["normal"]["y"] *= -1
        contact_2["normal"]["z"] *= -1
        contact = np.concatenate((contact_1, contact_2))

        contact_panda = contact[contact["body_id_b"] == panda.body_id]
        contact_table = contact[contact["body_id_b"] == table.body_id]
        contact_other_objects = []
        for i in range(len(objects.objects)):
            if i != objects.grasp_id:
                contact_other_objects.append(contact[contact["body_id_b"] == objects.objects[i].body_id])

        panda_link_ind = contact_panda["link_id_b"][contact_panda["force"] > self.cfg.contact_force_thresh]
        contact_panda_fingers = set(panda.fingers_link_id).issubset(panda_link_ind)
        contact_table = np.any(contact_table["force"] > self.cfg.contact_force_thresh)
        if len(contact_other_objects) > 0:
            contact_other_objects = np.concatenate(contact_other_objects)
            contact_other_objects = np.any(contact_other_objects["force"] > self.cfg.contact_force_thresh)
        else:
            contact_other_objects = False

        target_object_pos = objects.target_object.get_link_pos(5)
        is_below_table = target_object_pos[2] < self.cfg.table_height

        if not contact_panda_fingers and (contact_table or contact_other_objects or is_below_table):
            if self.cfg.verbose:
                print("detect contact panda fingers")
                if contact_table: print("object drop because contact table")
                if contact_other_objects: print("object drop because contact other objects")
                if is_below_table: print("object drop because is below table")
            self.dropped = True
        return self.dropped, contact_panda_fingers

    def check_success(self, panda: Panda, contact_panda_fingers: bool):
        if not contact_panda_fingers:
            self.success_step_counter = 0
            return 0
        pos = panda.get_link_pos(panda.ee_link_id)
        dist = np.linalg.norm(np.array(self.cfg.goal_center)-np.array(pos))
        is_within_goal = dist < self.cfg.goal_radius
        if not is_within_goal:
            self.success_step_counter = 0
            return 0
        self.success_step_counter += 1
        if self.success_step_counter >= self.cfg.success_step_thresh:
            if self.cfg.verbose: print("detect success")
            return EpisodeStatus.SUCCESS
        else:
            return 0

    def pre_step(self):
        pass

    def post_step(self, table: Table, panda: Panda, hand: Hand, objects: Objects, frame: int) -> Tuple[int, bool]:
        self.get_contact()
        release = not objects.released and self.check_object_release(objects.target_object, panda, frame)
        status = 0
        if self.check_panda_hand_collision(panda, hand):
            status |= EpisodeStatus.FAILURE_HUMAN_CONTACT
        dropped, contact_panda_fingers = self.check_object_dropped(objects, panda, table)
        if dropped:
            status |= EpisodeStatus.FAILURE_OBJECT_DROP
        if status != 0:
            return status, release
        if self.check_success(panda, contact_panda_fingers):
            status |= EpisodeStatus.SUCCESS
        if frame >= self.cfg.max_frames and status != EpisodeStatus.SUCCESS:
            status |= EpisodeStatus.FAILURE_TIMEOUT
        return status, release
