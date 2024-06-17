import numpy as np
import torch

def euler_to_rotation_matrix(az, el, th, batched=False):
    if batched:
        cx = torch.cos(torch.reshape(az, [-1, 1]))
        cy = torch.cos(torch.reshape(el, [-1, 1]))
        cz = torch.cos(torch.reshape(th, [-1, 1]))
        sx = torch.sin(torch.reshape(az, [-1, 1]))
        sy = torch.sin(torch.reshape(el, [-1, 1]))
        sz = torch.sin(torch.reshape(th, [-1, 1]))

        ones = torch.ones_like(cx)
        zeros = torch.zeros_like(cx)

        rx = torch.cat([ones, zeros, zeros, zeros, cx, -sx, zeros, sx, cx], dim=-1)
        ry = torch.cat([cy, zeros, sy, zeros, ones, zeros, -sy, zeros, cy], dim=-1)
        rz = torch.cat([cz, -sz, zeros, sz, cz, zeros, zeros, zeros, ones], dim=-1)

        rx = torch.reshape(rx, [-1, 3, 3])
        ry = torch.reshape(ry, [-1, 3, 3])
        rz = torch.reshape(rz, [-1, 3, 3])

        return torch.matmul(rz, torch.matmul(ry, rx))
    else:
        cx = torch.cos(az)
        cy = torch.cos(el)
        cz = torch.cos(th)
        sx = torch.sin(az)
        sy = torch.sin(el)
        sz = torch.sin(th)

        rx = torch.stack([[1.0, 0.0, 0.0], [0, cx, -sx], [0, sx, cx]], dim=0)
        ry = torch.stack([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dim=0)
        rz = torch.stack([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dim=0)

        return torch.matmul(rz, torch.matmul(ry, rx))

def get_control_points(batch_size: int, device: torch.device, rotz=False) -> torch.FloatTensor:
    " Outputs a tensor of shape (batch_size x 6 x 3) "
    control_points = np.array([
        [ 0.   ,  0.   ,  0.   ],
        [ 0.   ,  0.   ,  0.   ],
        [ 0.053, -0.   ,  0.075],
        [-0.053,  0.   ,  0.075],
        [ 0.053, -0.   ,  0.105],
        [-0.053,  0.   ,  0.105]
    ], dtype=np.float32)
    control_points = np.tile(np.expand_dims(control_points, 0), [batch_size, 1, 1])

    if rotz:
        RotZ = np.array([
            [0., -1., 0.],
            [1., 0., 0.],
            [0., 0., 1.],
        ])
        control_points = np.matmul(control_points, RotZ)

    return torch.tensor(control_points, dtype=torch.float32, device=device)

def rotate_vector_by_quaternion(q, v):
    """
    Rotate vector(s) v about the rotation described by ros quaternion(s) q.
    Input:
        q: (..., 4)
        v: (..., 3)
    Return:
        (..., 3)
    """
    qvec = q[..., :3]
    uv = torch.cross(qvec, v, dim=-1) # (..., 3)
    uuv = torch.cross(qvec, uv, dim=-1) # (..., 3)
    return v+2*(q[..., 3:]*uv+uuv)

def pos_ros_quat_transform_points(pos_ros_quat: torch.FloatTensor, control_points: torch.FloatTensor):
    """
    Transforms canonical points using pos and ros_quat.
    Input:
        pos_ros_quat: (B, 7)
        control_points: (B, N, 3)
    Return:
        transformed_control_points: (B, N, 3)
    """
    pos = pos_ros_quat[:, :3] # (B, 3)
    ros_quat = pos_ros_quat[:, 3:] # (B, 4)
    rotated_control_points = rotate_vector_by_quaternion(ros_quat[:, None], control_points) # (B, N, 3)
    transformed_control_points = rotated_control_points+pos[:, None] # (B, N, 3)
    return transformed_control_points

def control_points_from_cartesian_action(action: torch.FloatTensor):
    " action: (B, 6) "
    rot = euler_to_rotation_matrix(action[:, 3], action[:, 4], action[:, 5], batched=True)
    grasp_pc = get_control_points(action.shape[0], device=action.device)
    grasp_pc = torch.matmul(grasp_pc, rot.permute(0, 2, 1))
    grasp_pc += action[:, :3].unsqueeze(1)
    return grasp_pc

def compute_bc_loss(action: torch.FloatTensor, expert_action: torch.FloatTensor) -> torch.FloatTensor:
    """
    PM loss for behavior clone
    action: (B, 6)
    expert_action: (B, 6)
    """
    action_control_points = control_points_from_cartesian_action(action)
    expert_action_control_points = control_points_from_cartesian_action(expert_action)
    return torch.mean(torch.abs(action_control_points-expert_action_control_points).sum(-1))

def compute_goal_pred_loss(pred: torch.FloatTensor, target: torch.FloatTensor) -> torch.FloatTensor:
    """
    PM loss for grasp pose detection
    Input:
        pred: (B, 7)
        target: (B, 7)
    """
    control_points = get_control_points(pred.shape[0], device=pred.device, rotz=True) # (N, 3)
    pred_control_points = pos_ros_quat_transform_points(pred, control_points) # (B, N, 3)
    target_control_points = pos_ros_quat_transform_points(target, control_points) # (B, N, 3)
    return torch.mean(torch.abs(pred_control_points-target_control_points).sum(-1))

def debug():
    import code
    torch.manual_seed(0)
    action = torch.randn(8, 6).cuda()
    pos = torch.randn(8, 3).cuda()
    rot = torch.randn(8, 4).cuda()
    rot = rot/rot.norm(dim=1, keepdim=True)
    print(f"action: {action.sum()}, pos: {pos.sum()}")

    bc_loss = compute_bc_loss(action[:4], action[4:])
    goal_loss = compute_goal_pred_loss(torch.cat([pos[:4], rot[:4, 1:], rot[:4, :1]], dim=1), torch.cat([pos[4:], rot[4:, 1:], rot[4:, :1]], dim=1))
    print(f"bc loss {bc_loss}, goal loss {goal_loss}")
    code.interact(local=dict(globals(), **locals()))

if __name__ == "__main__":
    debug()

"""
conda activate genh2r
cd /share1/haoran/HRI/GenH2R
python -m models.loss
"""