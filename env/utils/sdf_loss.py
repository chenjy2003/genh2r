import numpy as np
from numpy.typing import NDArray
import torch
from typing import List, Tuple
import code

class SDFData:
    def __init__(self, sdf_path: str):
        sdf_data = np.load(sdf_path)
        self.min_coords: NDArray[np.float64] = sdf_data["min_coords"]
        self.delta: NDArray[np.float64] = sdf_data["delta"]
        self.sdf: NDArray[np.float64] = sdf_data["sdf"]

class SDFDataTensor:
    def __init__(self, sdf_data: SDFData, device: torch.device=torch.device("cpu")):
        self.min_coords: torch.DoubleTensor = torch.tensor(sdf_data.min_coords, device=device)
        self.delta: torch.DoubleTensor = torch.tensor(sdf_data.delta, device=device)
        self.sdf: torch.DoubleTensor = torch.tensor(sdf_data.sdf, device=device)

def se3_transform_pc_tensor(T: torch.DoubleTensor, points: torch.DoubleTensor) -> torch.DoubleTensor:
    " T: (4, 4) points: (..., 3) "
    return points@T[:3, :3].transpose(0, 1)+T[:3, 3]

def interpolate(data: torch.DoubleTensor, coords_0: torch.LongTensor, coords_frac: torch.DoubleTensor) -> torch.DoubleTensor:
    """
    Input:
        data: (nx, ny, nz)
        coords: (N, 3) within data range
        coords_frac: (N, 3)
    Return:
        value: (N, )
    """
    x0, y0, z0 = coords_0.unbind(dim=1) # (N,), (N,), (N,)
    x1, y1, z1 = x0+1, y0+1, z0+1 # (N,), (N,), (N,)
    xc, yc, zc = coords_frac.unbind(dim=1) # (N,), (N,), (N,)
    value_00 = data[x0, y0, z0]*(1-zc)+data[x0, y0, z1]*zc # (N,)
    value_01 = data[x0, y1, z0]*(1-zc)+data[x0, y1, z1]*zc # (N,)
    value_10 = data[x1, y0, z0]*(1-zc)+data[x1, y0, z1]*zc # (N,)
    value_11 = data[x1, y1, z0]*(1-zc)+data[x1, y1, z1]*zc # (N,)
    value_0 = value_00*(1-yc)+value_01*yc # (N,)
    value_1 = value_10*(1-yc)+value_11*yc # (N,)
    value = value_0*(1-xc)+value_1*xc # (N,)
    return value

def compute_distance(points: torch.DoubleTensor, sdf_data: SDFDataTensor, body_to_base: torch.DoubleTensor) -> Tuple[torch.BoolTensor, torch.DoubleTensor]:
    """
    Input:
        point: (N, 3)
        sdf_data
        body_to_base: (4, 4)
    Return:
        inside_bbox_mask: (N,)
        inside_bbox_value: (M,)
    """
    body_to_points = se3_transform_pc_tensor(body_to_base, points) # (N, 3)
    point_coords: torch.DoubleTensor = (body_to_points-sdf_data.min_coords)/sdf_data.delta # (N, 3)
    point_coords_0: torch.LongTensor = point_coords.floor().long() # (N, 3)
    point_coords_fraction: torch.DoubleTensor = point_coords-point_coords_0 # (N, 3)
    nx, ny, nz = sdf_data.sdf.shape
    x0, y0, z0 = point_coords_0.unbind(dim=1) # (N,)
    
    inside_bbox_mask = (x0>=0)&(y0>=0)&(z0>=0)&(x0<nx-1)&(y0<ny-1)&(z0<nz-1) # (N,)
    inside_bbox_value = interpolate(sdf_data.sdf, point_coords_0[inside_bbox_mask], point_coords_fraction[inside_bbox_mask]) # (M,)
    return inside_bbox_mask, inside_bbox_value

def compute_sdf_loss(points: torch.DoubleTensor, sdf_data_list: List[SDFDataTensor], body_to_base_list: List[torch.DoubleTensor], epsilon: float=0.2) -> torch.DoubleTensor:
    " points: (N, 3) "
    points = points.detach()
    points.requires_grad = True
    device = points.device
    loss: torch.DoubleTensor = torch.zeros(points.shape[0], dtype=torch.float64, device=device)
    grad: torch.DoubleTensor = torch.zeros_like(points)
    for sdf_data, body_to_base in zip(sdf_data_list, body_to_base_list):
        inside_bbox_mask, inside_bbox_value = compute_distance(points, sdf_data, body_to_base) # (N,), (M,)
        inside_object_loss = torch.zeros(inside_bbox_value.shape[0], dtype=torch.float64, device=device) # (M,)
        inside_object_mask = inside_bbox_value<=0 # (M,)
        inside_object_loss[inside_object_mask] = -inside_bbox_value[inside_object_mask]+0.5*epsilon
        within_sdf_epsilon_mask = (inside_bbox_value<=epsilon)&(~inside_object_mask) # (M,)
        inside_object_loss[within_sdf_epsilon_mask] = 1/(2*epsilon)*(inside_bbox_value[within_sdf_epsilon_mask]-epsilon)**2
        loss[inside_bbox_mask] = loss[inside_bbox_mask]+inside_object_loss
        # loss = loss+(-inside_bbox_value[inside_object_mask]+0.5*epsilon).sum()
        # loss = loss+(1/(2*epsilon)*(inside_bbox_value[within_sdf_epsilon_mask]-epsilon)**2).sum()
    grad = torch.autograd.grad(loss.sum(), points)[0]
    return loss.detach(), grad

def visualize(points, vectors):
    import matplotlib.pyplot as plt

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot 3D points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')

    # Plot vectors starting from the points
    for i, (point, vector) in enumerate(zip(points, vectors)):
        ax.quiver(point[0], point[1], point[2], vector[0], vector[1], vector[2], color='b')

    # Set labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    # ax.set_zlim(-1, 1)

    plt.show()

def debug():
    import time
    # torch.manual_seed(0)
    # points = torch.randn(100, 3, dtype=torch.float64)*0.5
    # points.requires_grad = True
    # points = torch.tensor([
    #     [0.1540,  0.0932, -0.2383],
    #     # [-0.1441, -0.0051, -0.0321],
    #     # [-0.0338, -0.0653, -0.1720],
    #     # [-0.0500,  0.0788, -0.2203],
    #     # [ 0.0766, -0.1635,  0.0624],
    #     # [ 0.1436, -0.0409,  0.0021],
    #     # [-0.1285, -0.1161,  0.0298],
    #     # [-0.0441,  0.0190, -0.2006]
    # ], dtype=torch.float64, requires_grad=True)
    # sdf_data = SDFData("tmp/debug_sdf/box/sdf.npz")
    device = torch.device("cuda")
    sdf_data = SDFData("tmp/debug_sdf/003_cracker_box/sdf.npz")
    sdf_data_tensor = SDFDataTensor(sdf_data, device=device)
    min_coords = sdf_data_tensor.min_coords
    max_coords = sdf_data_tensor.min_coords+sdf_data_tensor.delta*torch.tensor(sdf_data_tensor.sdf.shape, device=device)
    side_length = 10
    points_x, points_y, points_z = torch.meshgrid(
        torch.linspace(min_coords[0], max_coords[0], side_length, dtype=torch.float64, device=device), 
        torch.linspace(min_coords[1], max_coords[1], side_length, dtype=torch.float64, device=device), 
        torch.linspace(min_coords[2], max_coords[2], side_length, dtype=torch.float64, device=device)
    )
    points = torch.stack([points_x, points_y, points_z], dim=3).reshape(-1, 3)
    points.requires_grad = True

    body_to_base = torch.eye(4, dtype=torch.float64, device=device)

    torch.cuda.synchronize()
    start_time = time.time()
    loss, grad = compute_sdf_loss(points, [sdf_data_tensor], [body_to_base])
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"compute sdf loss takes {end_time-start_time} seconds")
    code.interact(local=dict(globals(), **locals()))
    visualize(points.cpu().data, grad.cpu().data/10)

if __name__ == "__main__":
    debug()

"""
DISPLAY="localhost:11.0" CUDA_VISIBLE_DEVICES=3 python -m env.utils.sdf_loss
"""