import numpy as np
from numpy.typing import NDArray
import torch
from xml.etree.ElementTree import parse, ElementTree
from dataclasses import dataclass
from omegaconf import MISSING
from typing import Tuple, Optional, List, Dict, Union
from scipy.spatial.transform import Rotation as Rt
try:
    import PyKDL
except:
    print("warning: failed to import PyKDL, the functionality of robot_kinematics will be limited")
import code

from .transform import mat_to_pos_ros_quat, pos_ros_quat_to_mat

def str_to_float_list(s):
    return list(map(float, s.split(" ")))

def list_to_jnt_array(q: List[float]):
    q_kdl = PyKDL.JntArray(len(q))
    for i, q_i in enumerate(q):
        q_kdl[i] = q_i
    return q_kdl

def urdf_origin_to_pose(origin: Optional[ElementTree]) -> NDArray[np.float64]:
    pose = np.eye(4)
    if origin is not None:
        xyz, rpy = str_to_float_list(origin.get("xyz")), str_to_float_list(origin.get("rpy"))
        pose[:3, 3] = xyz
        pose[:3, :3] = Rt.from_euler("xyz", rpy).as_matrix()
    return pose

def axis_angle_to_rotation_matrix_torch(axis: torch.DoubleTensor, angle: torch.DoubleTensor) -> torch.DoubleTensor:
    """
    Axis angle to rotation matrix for pytorch, numpy can use Rt.from_rotvec(...).as_matrix()
    Input:
        axis: (..., 3), angle: (...)
    Return:
        rotation_matrix:  (..., 3, 3)
    """
    # Components of the axis vector
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    # Using Rodrigues' rotation formula
    cos, sin = angle.cos(), angle.sin()
    one_minus_cos = 1-cos
    R = torch.stack([
        torch.stack([cos + x*x*one_minus_cos, x*y*one_minus_cos - z*sin, x*z*one_minus_cos + y*sin], dim=-1), # (..., 3)
        torch.stack([y*x*one_minus_cos + z*sin, cos + y*y*one_minus_cos, y*z*one_minus_cos - x*sin], dim=-1), # (..., 3)
        torch.stack([z*x*one_minus_cos - y*sin, z*y*one_minus_cos + x*sin, cos + z*z*one_minus_cos], dim=-1), # (..., 3)
    ], dim=-2) # (..., 3, 3)
    return R

class Link:
    name: str
    parent: Optional["Link"]
    parent_joint: Optional["Joint"]
    children: List["Link"]
    inertial: ElementTree
    idx: int
    collision_file_path: Optional[str]

    def __init__(self, cfg: ElementTree, idx: int):
        self.name = cfg.get("name")
        self.inertial = cfg.find("inertial")
        self.parent = None
        self.parent_joint = None
        self.children = []
        self.idx = idx
        try:
            self.collision_file_path = cfg.find("collision").find("geometry").find("mesh").get("filename").lstrip("package://")
        except:
            self.collision_file_path = None
    
    def __repr__(self) -> str:
        return f"Link("\
            f"name={self.name}, "\
            f"parent={self.parent.name if self.parent is not None else None}, "\
            f"parent_joint={self.parent_joint.name if self.parent_joint is not None else None}, "\
            f"children={[child.name for child in self.children]}, "\
        f")"

class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: ElementTree
    parent_to_child_default: NDArray[np.float64]
    axis: Optional[List[float]]
    limit: Optional[ElementTree]
    safety_controller: Optional[ElementTree]
    dynamics: Optional[ElementTree]
    def __init__(self, cfg: ElementTree):
        self.name = cfg.get("name")
        self.type = cfg.get("type")
        self.parent = cfg.find("parent").get("link")
        self.child = cfg.find("child").get("link")

        self.origin = cfg.find("origin")
        assert self.origin is not None
        self.parent_to_child_default = urdf_origin_to_pose(self.origin)
        if cfg.find("axis") is None:
            self.axis = None
        else:
            self.axis = str_to_float_list(cfg.find("axis").get("xyz"))
        
        self.limit = cfg.find("limit")
        self.safety_controller = cfg.find("safety_controller")
        self.dynamics = cfg.find("dynamics")
    
    def get_parent_to_child(self, theta: Union[float, NDArray[np.float64], torch.DoubleTensor]) -> Union[NDArray[np.float64], torch.DoubleTensor]:
        """
        Get the pose of the child in parent's frame
        Input:
            theta: (...)
        Return:
            parent_to_child: (..., 4, 4)
        """
        if self.type in ["revolute", "continuous"]:
            if isinstance(theta, torch.DoubleTensor) or isinstance(theta, torch.cuda.DoubleTensor):
                device = theta.device
                axis = torch.tensor(self.axis, device=device, dtype=torch.float64)
                child_default_to_child_rot = axis_angle_to_rotation_matrix_torch(axis, theta) # (..., 3, 3)
                parent_to_child = torch.tensor(self.parent_to_child_default, device=device).repeat(theta.shape+(1, 1)) # (..., 4, 4)
                parent_to_child[..., :3, :3] = parent_to_child[..., :3, :3].clone()@child_default_to_child_rot  # (..., 4, 4)
            elif isinstance(theta, float) or (isinstance(theta, np.ndarray) and theta.dtype == np.float64):
                if isinstance(theta, float):
                    theta = np.array(theta)
                rotvec = np.array(self.axis)*theta[..., None] # (..., 3)
                child_default_to_child_rot = Rt.from_rotvec(rotvec.reshape(-1, 3)).as_matrix().reshape(rotvec.shape[:-1]+(3, 3)) # (..., 3, 3)
                parent_to_child = np.tile(self.parent_to_child_default, theta.shape+(1, 1)) # (..., 4, 4)
                parent_to_child[..., :3, :3] = parent_to_child[..., :3, :3]@child_default_to_child_rot  # (..., 4, 4)
            else:
                raise TypeError(type(theta))
        elif self.type == "prismatic":
            if isinstance(theta, torch.DoubleTensor) or isinstance(theta, torch.cuda.DoubleTensor):
                device = theta.device
                child_default_to_child_trans = torch.tensor(self.axis, device=device, dtype=torch.float64)*theta[..., None] # (..., 3)
                parent_to_child = torch.tensor(self.parent_to_child_default, device=device).repeat(theta.shape+(1, 1)) # (..., 4, 4)
                parent_to_child[..., :3, 3] = parent_to_child[..., :3, 3].clone()+child_default_to_child_trans  # (..., 4, 4)
            elif isinstance(theta, float) or (isinstance(theta, np.ndarray) and theta.dtype == np.float64):
                if isinstance(theta, float):
                    theta = np.array(theta)
                child_default_to_child_trans = np.array(self.axis)*theta[..., None] # (..., 3)
                parent_to_child = np.tile(self.parent_to_child_default, theta.shape+(1, 1)) # (..., 4, 4)
                parent_to_child[..., :3, 3] = parent_to_child[..., :3, 3]+child_default_to_child_trans  # (..., 4, 4)
            else:
                raise TypeError(type(theta))
        else:
            raise NotImplementedError
        return parent_to_child

def parse_urdf(urdf_file) -> Tuple[List[Link], Dict[str, Link], List[Joint], Dict[str, Joint], Link]:
    tree: ElementTree = parse(urdf_file)
    root = tree.getroot()
    link_cfgs = root.findall("link")
    joint_cfgs = root.findall("joint")
    links = [Link(link_cfg, idx) for idx, link_cfg in enumerate(link_cfgs)]
    link_map = {link.name: link for link in links}
    joints = [Joint(joint_cfg) for joint_cfg in joint_cfgs]
    joint_map = {joint.name: joint for joint in joints}
    for joint in joints:
        parent_link = link_map[joint.parent]
        child_link = link_map[joint.child]
        child_link.parent = parent_link
        child_link.parent_joint = joint
        parent_link.children.append(child_link)
    root_link = None
    for link in links:
        if link.parent is None:
            assert root_link is None, "multiple roots detected"
            root_link = link
    assert root_link is not None, "no roots detected"
    return links, link_map, joints, joint_map, root_link

def urdf_origin_to_kdl_frame(origin: Optional[ElementTree]):
    if origin is not None:
        xyz = str_to_float_list(origin.get("xyz"))
        rpy = str_to_float_list(origin.get("rpy"))
    else:
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]
    return PyKDL.Frame(PyKDL.Rotation.Quaternion(*Rt.from_euler("xyz", rpy).as_quat()), PyKDL.Vector(*xyz))

def urdf_joint_to_kdl_joint(joint: Joint):
    origin_frame = urdf_origin_to_kdl_frame(joint.origin)
    if joint.type == "fixed":
        return PyKDL.Joint(joint.name, PyKDL.Joint.Fixed)
    axis = PyKDL.Vector(*joint.axis)
    if joint.type in ["revolute", "continuous"]:
        return PyKDL.Joint(joint.name, origin_frame.p, origin_frame.M*axis, PyKDL.Joint.RotAxis)
    if joint.type == "prismatic":
        return PyKDL.Joint(joint.name, origin_frame.p, origin_frame.M*axis, PyKDL.Joint.TransAxis)
    raise ValueError(f"Unknown joint type: {joint.type}.")

def urdf_inertial_to_kdl_rbi(inertial: ElementTree):
    origin_frame = urdf_origin_to_kdl_frame(inertial.find("origin"))
    mass = float(inertial.find("mass").get("value"))
    inertia = inertial.find("inertia")
    rbi = PyKDL.RigidBodyInertia(
        mass,
        origin_frame.p,
        PyKDL.RotationalInertia(
            float(inertia.get("ixx")),
            float(inertia.get("iyy")),
            float(inertia.get("izz")),
            float(inertia.get("ixy")),
            float(inertia.get("ixz")),
            float(inertia.get("iyz")),
        ),
    )
    return origin_frame.M*rbi

def build_kdl_tree(link_map: Dict[str, Link], joints: List[Joint], root_link: Link):
    tree = PyKDL.Tree(root_link.name)
    for joint in joints:
        child_link = link_map[joint.child]
        kdl_joint = urdf_joint_to_kdl_joint(joint)
        kdl_frame = urdf_origin_to_kdl_frame(joint.origin)
        kdl_rbi = urdf_inertial_to_kdl_rbi(child_link.inertial)
        kdl_segment = PyKDL.Segment(joint.child, kdl_joint, kdl_frame, kdl_rbi)
        tree.addSegment(kdl_segment, joint.parent)
    return tree

def get_chain_joint_limit(tip_link: Link) -> Tuple[List[float], List[float]]:
    q_min, q_max = [], []
    cur_link = tip_link
    while cur_link.parent_joint is not None:
        limit = cur_link.parent_joint.limit
        if limit is not None:
            q_min.append(float(limit.get("lower")))
            q_max.append(float(limit.get("upper")))
        cur_link = cur_link.parent
    q_min.reverse()
    q_max.reverse()
    return q_min, q_max

def get_home_pose_and_screw_axes(tip_link: Link) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    current_link = tip_link
    current_link_to_tip_link = np.eye(4)
    chain_links: List[Link] = [current_link]
    while current_link.parent is not None:
        current_link_to_tip_link = current_link.parent_joint.parent_to_child_default@current_link_to_tip_link
        current_link = current_link.parent
        chain_links.append(current_link)
    home_pose = current_link_to_tip_link
    # screw axes
    chain_links.reverse()
    base_link_to_current_link = np.eye(4)
    screw_axes: List[NDArray[np.float64]] = []
    for link in chain_links[1:]:
        joint = link.parent_joint
        base_link_to_current_link = base_link_to_current_link@joint.parent_to_child_default
        if joint.type in ["revolute", "continuous"]:
            rot_axis = base_link_to_current_link[:3, :3]@np.array(joint.axis) # (3,)
            linear_velocity = -np.cross(rot_axis, base_link_to_current_link[:3, 3]) # (3,)
            # Reference: MODERN ROBOTICS: MECHANICS, PLANNING, AND CONTROL. 4.1.1 First Formulation: Screw Axes in the Base Frame
            screw_axis = np.concatenate([rot_axis, linear_velocity])
            screw_axes.append(screw_axis)
        elif joint.type == "fixed":
            pass
        else:
            raise NotImplementedError
    screw_axes: NDArray[np.float64] = np.stack(screw_axes, axis=0)
    return home_pose, screw_axes

def vector_to_so3(vec: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Input:
        vec:        (..., 3)
    Return:
        matrices:   (..., 3, 3)
    Reference:
        MODERN ROBOTICS: MECHANICS, PLANNING, AND CONTROL. 3.30
        [[  0, -x2,  x1],
         [ x2,   0, -x0],
         [-x1,  x0,   0]]
    """
    assert vec.shape[-1] == 3
    x0, x1, x2 = vec[..., 0], vec[..., 1], vec[..., 2]
    zeros = np.zeros_like(x0)
    return np.stack([
        np.stack([zeros, -x2, x1], axis=-1),
        np.stack([x2, -zeros, -x0], axis=-1),
        np.stack([-x1, x0, zeros], axis=-1),
    ], axis=-2)

def screw_axes_and_thetas_to_matrix_exponentials(screw_axes: NDArray[np.float64], thetas: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Input:
        screw_axes:          (..., 6)
        thetas:              (...)
    Return:
        matrix_exponentials: (..., 4, 4)
    Reference:
        MODERN ROBOTICS: MECHANICS, PLANNING, AND CONTROL. 3.51 and 3.88
        [[I+sin(theta)[w]+(1-cos(theta))[w]^2, (theta*I+(1-cos(theta))[w]+(theta-sin(theta))[w]^2)v]
         [0, 1]]
    """
    omegas = vector_to_so3(screw_axes[..., :3]) # (..., 3, 3)
    omegas_square = omegas@omegas # (..., 3, 3)
    sin_thetas = np.sin(thetas) # (...)
    one_minus_cos_thetas = 1-np.cos(thetas) # (...)
    rotations = np.eye(3)+sin_thetas[..., None, None]*omegas+one_minus_cos_thetas[..., None, None]*omegas_square # (..., 3, 3)
    translations = (thetas[..., None, None]*np.eye(3)+one_minus_cos_thetas[..., None, None]*omegas+(thetas-sin_thetas)[..., None, None]*omegas_square)@screw_axes[..., 3:, None] # (..., 3, 1)
    last_row = np.tile(np.array([0., 0., 0., 1.]), rotations.shape[:-2]+(1, 1)) # (..., 1, 4)
    matrix_exponentials = np.concatenate([
        np.concatenate([rotations, translations], axis=-1), # (..., 3, 4)
        last_row
    ], axis=-2) # (..., 4, 4)
    return matrix_exponentials

@dataclass
class RobotKinematicsConfig:
    urdf_file: str = "env/data/assets/franka_panda/panda_gripper_hand_camera.urdf"
    IK_solver_max_iter: int = 100
    IK_solver_eps: float = 1e-6
    chain_tip: Optional[str] = "panda_hand"

class RobotKinematics:
    def __init__(self, cfg: RobotKinematicsConfig):
        self.cfg = cfg

        self.links, self.link_map, self.joints, self.joint_map, self.root_link = parse_urdf(cfg.urdf_file)
        
        if cfg.chain_tip is not None:
            self.kdl_tree = build_kdl_tree(self.link_map, self.joints, self.root_link)
            self.arm_chain = self.kdl_tree.getChain(self.root_link.name, cfg.chain_tip)
            self.num_joints: int = self.arm_chain.getNrOfJoints()
            self.tip_link = self.link_map[cfg.chain_tip]
            self.q_min, self.q_max = get_chain_joint_limit(self.tip_link)
            q_min_kdl, q_max_kdl = list_to_jnt_array(self.q_min), list_to_jnt_array(self.q_max)
            self.fk_p_kdl = PyKDL.ChainFkSolverPos_recursive(self.arm_chain)
            self.ik_v_kdl = PyKDL.ChainIkSolverVel_pinv(self.arm_chain)
            self.ik_p_kdl = PyKDL.ChainIkSolverPos_NR_JL(self.arm_chain, q_min_kdl, q_max_kdl, self.fk_p_kdl, self.ik_v_kdl, maxiter=cfg.IK_solver_max_iter, eps=cfg.IK_solver_eps)
            self.home_pose, self.screw_axes = get_home_pose_and_screw_axes(self.tip_link)
    
    def cartesian_to_joint(self, base_to_ee: NDArray[np.float64], seed: Optional[NDArray[np.float64]]=None) -> Tuple[NDArray[np.float64], int]:
        """
        Inverse kinematics
        Input:
            base_to_ee: (4, 4)
            seed:       (self.num_joints,)
        Return:
            result:     (self.num_joints,)
            info:       int
        """
        pos, ros_quat = mat_to_pos_ros_quat(base_to_ee)
        if seed is None:
            seed = np.zeros(self.num_joints)
        else:
            assert seed.shape[0] == self.num_joints
        seed_kdl = list_to_jnt_array(seed)
        # Make IK Call
        goal_pose = PyKDL.Frame(
            PyKDL.Rotation.Quaternion(ros_quat[0], ros_quat[1], ros_quat[2], ros_quat[3]),
            PyKDL.Vector(pos[0], pos[1], pos[2])
        )
        result_angles = PyKDL.JntArray(self.num_joints)
        info = self.ik_p_kdl.CartToJnt(seed_kdl, goal_pose, result_angles) # E_NOERROR = 0, E_NOT_UP_TO_DATE = -3, E_SIZE_MISMATCH = -4, E_MAX_ITERATIONS_EXCEEDED = -5, E_IKSOLVERVEL_FAILED = -100, E_FKSOLVERPOS_FAILED = -101 
        result = np.array(list(result_angles))
        return result, info

    def joint_to_cartesian(self, joint_values: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Forward kinematics
        Input:
            joint_values: (..., self.num_joints)
        Return:
            base_to_ees:  (..., 4, 4)
        """
        matrix_exponentials = screw_axes_and_thetas_to_matrix_exponentials(self.screw_axes, joint_values) # (..., self.num_joints, 4, 4)
        base_to_ees = self.home_pose
        for i in reversed(range(self.num_joints)):
            base_to_ees = matrix_exponentials[..., i, :, :]@base_to_ees
        return base_to_ees

    def joint_to_cartesian_for_all_links(self, joint_values: torch.DoubleTensor) -> torch.DoubleTensor:
        """
        Forward kinematics for all links
        Input:
            joint_values: (..., number of joints in self.joints that are not fixed)
        Return:
            base_to_links:  (..., len(self.links), 4, 4)
        """
        device = joint_values.device
        base_to_base = torch.eye(4, dtype=torch.float64, device=device).repeat(joint_values.shape[:-1]+(1, 1)) # (..., 4, 4)
        base_to_links: List[torch.DoubleTensor] = [base_to_base]
        joint_idx = 0
        for link in self.links[1:]: # assume links are in a topological order
            parent_link_idx = link.parent.idx
            parent_joint = link.parent_joint
            base_to_parent_link = base_to_links[parent_link_idx] # (..., 4, 4)
            if parent_joint.type == "fixed":
                base_to_link = base_to_parent_link@torch.tensor(parent_joint.parent_to_child_default, device=device) # (..., 4, 4)
            elif parent_joint.type in ["revolute", "prismatic", "continuous"]:
                base_to_link = base_to_parent_link@parent_joint.get_parent_to_child(joint_values[..., joint_idx]) # (..., 4, 4)
                joint_idx += 1
            else:
                raise NotImplementedError(f"forward kinematics not implemented for type {parent_joint.type}")
            base_to_links.append(base_to_link)
        base_to_links = torch.stack(base_to_links, dim=-3) # (..., len(self.links), 4, 4)
        return base_to_links

def debug_cartesian_to_joint(robot: RobotKinematics):
    pos = np.array([ 0.1439894 , -0.00910749,  0.71072687])
    ros_quat = np.array([ 0.96438653,  0.03465594,  0.2612568 , -0.02241564])
    base_to_ee = pos_ros_quat_to_mat(pos, ros_quat)
    seed = np.array([ 0.   , -1.285,  0.   , -2.356,  0.   ,  1.571,  0.785])
    ik_joint_values, info = robot.cartesian_to_joint(base_to_ee, seed) # array([ 0.03839981, -1.29161734, -0.04006584, -2.35181459,  0.01433843, 1.58850796,  0.73411765])
    print(f"ik_joint_values {ik_joint_values}")

def debug_joint_to_cartesian(robot: RobotKinematics):
    print(f"robot.home_pose {robot.home_pose}") 
    # array([[ 0.70710678,  0.70710678,  0.        ,  0.088     ],
    #        [ 0.70710678, -0.70710678, -0.        , -0.        ],
    #        [-0.        ,  0.        , -1.        ,  0.926     ],
    #        [ 0.        ,  0.        ,  0.        ,  1.        ]])
    joint_values = np.array([
        [0., 0., 0., 0., 0., 0., 0.],
        [.1, 0., 0., 0., 0., 0., 0.],
        [0., .1, 0., 0., 0., 0., 0.],
        [0., 0., .1, 0., 0., 0., 0.],
        [0., 0., 0., .1, 0., 0., 0.],
        [0., 0., 0., 0., .1, 0., 0.],
        [0., 0., 0., 0., 0., .1, 0.],
        [0., 0., 0., 0., 0., 0., .1],
        [.1, .2, .3, .4, .5, .6, .7],
    ])
    for joint_value in joint_values:
        print(f"joint_value {joint_value} base_to_ee\n{robot.joint_to_cartesian(joint_value)}")
    print(f"batch base_to_ee\n{robot.joint_to_cartesian(joint_values)}")
    # joint_value [0. 0. 0. 0. 0. 0. 0.], base_to_ee
    # [[ 0.70710678  0.70710678  0.          0.088     ]
    # [ 0.70710678 -0.70710678 -0.         -0.        ]
    # [-0.          0.         -1.          0.926     ]
    # [ 0.          0.          0.          1.        ]]
    # joint_value [0.1 0.  0.  0.  0.  0.  0. ], base_to_ee
    # [[ 0.63298131  0.77416708  0.          0.08756037]
    # [ 0.77416708 -0.63298131 -0.          0.00878534]
    # [-0.          0.         -1.          0.926     ]
    # [ 0.          0.          0.          1.        ]]
    # joint_value [0.  0.1 0.  0.  0.  0.  0. ], base_to_ee
    # [[ 0.70357419  0.70357419 -0.09983342  0.14676158]
    # [ 0.70710678 -0.70710678 -0.         -0.        ]
    # [-0.07059289 -0.07059289 -0.99500417  0.91425213]
    # [ 0.          0.          0.          1.        ]]
    # joint_value [0.  0.  0.1 0.  0.  0.  0. ], base_to_ee
    # [[ 0.63298131  0.77416708  0.          0.08756037]
    # [ 0.77416708 -0.63298131 -0.          0.00878534]
    # [-0.          0.         -1.          0.926     ]
    # [ 0.          0.          0.          1.        ]]
    # joint_value [0.  0.  0.  0.1 0.  0.  0. ], base_to_ee
    # [[ 0.70357419  0.70357419  0.09983342  0.06031867]
    # [ 0.70710678 -0.70710678 -0.         -0.        ]
    # [ 0.07059289  0.07059289 -0.99500417  0.92516524]
    # [ 0.          0.          0.          1.        ]]
    # joint_value [0.  0.  0.  0.  0.1 0.  0. ], base_to_ee
    # [[ 0.63298131  0.77416708  0.          0.08756037]
    # [ 0.77416708 -0.63298131 -0.          0.00878534]
    # [-0.          0.         -1.          0.926     ]
    # [ 0.          0.          0.          1.        ]]
    # joint_value [0.  0.  0.  0.  0.  0.1 0. ], base_to_ee
    # [[ 0.70357419  0.70357419  0.09983342  0.09824254]
    # [ 0.70710678 -0.70710678 -0.         -0.        ]
    # [ 0.07059289  0.07059289 -0.99500417  0.93531989]
    # [ 0.          0.          0.          1.        ]]
    # joint_value [0.  0.  0.  0.  0.  0.  0.1], base_to_ee
    # [[ 0.77416708  0.63298131 -0.          0.088     ]
    # [ 0.63298131 -0.77416708 -0.         -0.        ]
    # [-0.          0.         -1.          0.926     ]
    # [ 0.          0.          0.          1.        ]]
    # joint_value [0.1 0.2 0.3 0.4 0.5 0.6 0.7], base_to_ee
    # [[ 0.3429257   0.80404361  0.48571168  0.08508066]
    # [ 0.60596605 -0.58444466  0.53965691  0.06370813]
    # [ 0.7177793   0.10926257 -0.68764422  0.97517365]
    # [ 0.          0.          0.          1.        ]]

    # batch base_to_ee
    # [[[ 0.70710678  0.70710678  0.          0.088     ]
    # [ 0.70710678 -0.70710678 -0.         -0.        ]
    # [-0.          0.         -1.          0.926     ]
    # [ 0.          0.          0.          1.        ]]

    # [[ 0.63298131  0.77416708  0.          0.08756037]
    # [ 0.77416708 -0.63298131 -0.          0.00878534]
    # [-0.          0.         -1.          0.926     ]
    # [ 0.          0.          0.          1.        ]]

    # [[ 0.70357419  0.70357419 -0.09983342  0.14676158]
    # [ 0.70710678 -0.70710678 -0.         -0.        ]
    # [-0.07059289 -0.07059289 -0.99500417  0.91425213]
    # [ 0.          0.          0.          1.        ]]

    # [[ 0.63298131  0.77416708  0.          0.08756037]
    # [ 0.77416708 -0.63298131 -0.          0.00878534]
    # [-0.          0.         -1.          0.926     ]
    # [ 0.          0.          0.          1.        ]]

    # [[ 0.70357419  0.70357419  0.09983342  0.06031867]
    # [ 0.70710678 -0.70710678 -0.         -0.        ]
    # [ 0.07059289  0.07059289 -0.99500417  0.92516524]
    # [ 0.          0.          0.          1.        ]]

    # [[ 0.63298131  0.77416708  0.          0.08756037]
    # [ 0.77416708 -0.63298131 -0.          0.00878534]
    # [-0.          0.         -1.          0.926     ]
    # [ 0.          0.          0.          1.        ]]

    # [[ 0.70357419  0.70357419  0.09983342  0.09824254]
    # [ 0.70710678 -0.70710678 -0.         -0.        ]
    # [ 0.07059289  0.07059289 -0.99500417  0.93531989]
    # [ 0.          0.          0.          1.        ]]

    # [[ 0.77416708  0.63298131  0.          0.088     ]
    # [ 0.63298131 -0.77416708 -0.         -0.        ]
    # [-0.          0.         -1.          0.926     ]
    # [ 0.          0.          0.          1.        ]]

    # [[ 0.3429257   0.80404361  0.48571168  0.08508066]
    # [ 0.60596605 -0.58444466  0.53965691  0.06370813]
    # [ 0.7177793   0.10926257 -0.68764422  0.97517365]
    # [ 0.          0.          0.          1.        ]]]

def debug_joint_to_cartesian_for_all_links(robot: RobotKinematics):
    # joint_values = torch.zeros(9, dtype=torch.float64)
    joint_values = torch.tensor([.1, .2, .3, .4, .5, .6, .7, .04, .04], dtype=torch.float64)
    world_to_links = robot.joint_to_cartesian_for_all_links(joint_values)
    # [[[ 1.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
    # [ 0.0000e+00,  1.0000e+00,  0.0000e+00,  0.0000e+00],
    # [ 0.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00],
    # [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

    # [[ 9.9500e-01, -9.9833e-02,  0.0000e+00,  0.0000e+00],
    # [ 9.9833e-02,  9.9500e-01,  0.0000e+00,  0.0000e+00],
    # [ 0.0000e+00,  0.0000e+00,  1.0000e+00,  3.3300e-01],
    # [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

    # [[ 9.7517e-01, -1.9768e-01, -9.9833e-02,  0.0000e+00],
    # [ 9.7843e-02, -1.9834e-02,  9.9500e-01,  0.0000e+00],
    # [-1.9867e-01, -9.8007e-01,  4.8966e-12,  3.3300e-01],
    # [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

    # [[ 9.0211e-01, -3.8356e-01,  1.9768e-01,  6.2466e-02],
    # [ 3.8752e-01,  9.2165e-01,  1.9834e-02,  6.2675e-03],
    # [-1.8980e-01,  5.8711e-02,  9.8007e-01,  6.4270e-01],
    # [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

    # [[ 9.0788e-01, -1.6923e-01,  3.8356e-01,  1.3689e-01],
    # [ 3.6465e-01, -1.3264e-01, -9.2165e-01,  3.8238e-02],
    # [ 2.0684e-01,  9.7661e-01, -5.8711e-02,  6.2704e-01],
    # [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

    # [[ 6.1285e-01, -7.7186e-01, -1.6923e-01, -2.9931e-03],
    # [ 7.6187e-01,  6.3400e-01, -1.3264e-01, -4.2779e-02],
    # [ 2.0967e-01, -4.7642e-02,  9.7661e-01,  9.8500e-01],
    # [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

    # [[ 4.1026e-01, -4.8571e-01,  7.7186e-01, -2.9931e-03],
    # [ 5.5391e-01, -5.3966e-01, -6.3400e-01, -4.2779e-02],
    # [ 7.2448e-01,  6.8764e-01,  4.7642e-02,  9.8500e-01],
    # [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

    # [[ 8.1103e-01,  3.2606e-01,  4.8571e-01,  3.3110e-02],
    # [ 1.5218e-02, -8.4175e-01,  5.3966e-01,  5.9648e-03],
    # [ 5.8481e-01, -4.3029e-01, -6.8764e-01,  1.0488e+00],
    # [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

    # [[ 3.4293e-01,  8.0404e-01,  4.8571e-01,  8.5081e-02],
    # [ 6.0597e-01, -5.8444e-01,  5.3966e-01,  6.3708e-02],
    # [ 7.1778e-01,  1.0926e-01, -6.8764e-01,  9.7517e-01],
    # [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

    # [[ 3.4293e-01,  8.0404e-01,  4.8571e-01,  1.4561e-01],
    # [ 6.0597e-01, -5.8444e-01,  5.3966e-01,  7.1846e-02],
    # [ 7.1778e-01,  1.0926e-01, -6.8764e-01,  9.3939e-01],
    # [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

    # [[ 3.4293e-01,  8.0404e-01,  4.8571e-01,  8.1284e-02],
    # [ 6.0597e-01, -5.8444e-01,  5.3966e-01,  1.1860e-01],
    # [ 7.1778e-01,  1.0926e-01, -6.8764e-01,  9.3064e-01],
    # [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

    # [[-8.0404e-01, -4.8571e-01,  3.4293e-01,  1.1491e-01],
    # [ 5.8444e-01, -5.3966e-01,  6.0597e-01,  1.0495e-01],
    # [-1.0926e-01,  6.8764e-01,  7.1778e-01,  9.7626e-01],
    # [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]]
    code.interact(local=dict(globals(), **locals()))

def debug():
    np.set_printoptions(suppress=True) # no scientific notation
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_object(OmegaConf.structured(RobotKinematicsConfig))
    robot = RobotKinematics(cfg)
    # debug_cartesian_to_joint(robot)
    # debug_joint_to_cartesian(robot)
    debug_joint_to_cartesian_for_all_links(robot)
    code.interact(local=dict(globals(), **locals()))

if __name__ == "__main__":
    debug()

"""
python -m env.utils.robot_kinematics
"""