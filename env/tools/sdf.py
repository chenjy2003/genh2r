import os
import numpy as np
from numpy.typing import NDArray
import open3d as o3d
import code

env_path = os.path.dirname(os.path.dirname(__file__))
sdfgen_path = os.path.join(env_path, "third_party", "SDFGen", "build", "bin", "SDFGen")

def load_sdf(sdf_path: str):
    with open(sdf_path, "r") as f:
        nx, ny, nz = map(int, f.readline().split(" "))
        x0, y0, z0 = map(float, f.readline().split(" "))
        delta = float(f.readline().strip())
        sdf = np.loadtxt(f).reshape((nz, ny, nx)).transpose((2, 1, 0))
    return np.array([x0, y0, z0]), delta, sdf

def gen_sdf(object_path: str, reg_size: float=0.2, dim: int=32, padding: int=20):
    " object should be convex decomposed before generating sdf "
    object_mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(object_path)
    object_vertices: NDArray[np.float64] = np.array(object_mesh.vertices)
    min_coords, max_coords = object_vertices.min(axis=0), object_vertices.max(axis=0)
    extent: NDArray[np.float64] = max_coords-min_coords
    max_extent = extent.max()
    scale = max_extent/reg_size
    dim = np.clip(dim*scale, 32, 100)
    delta = max_extent/dim
    padding = min(int(padding*scale), 30)

    print(f"generating sdf for {object_path} with min_coords={min_coords}, max_coords={max_coords}, extent={extent}, delta={delta}, padding={padding}")
    sdfgen_cmd = f"{sdfgen_path} \"{object_path}\" {delta} {padding}"
    os.system(sdfgen_cmd) # bound box size: min=min_coords-padding*delta, max=max_coords+padding*delta

    sdf_path = object_path[:-4]+".sdf"
    sdf_min_coords, sdf_delta, sdf = load_sdf(sdf_path)
    print(f"sdf generation complete, sdf_min_coords={sdf_min_coords}, sdf_delta={sdf_delta}")
    np.savez(os.path.join(os.path.dirname(object_path), "sdf.npz"), min_coords=sdf_min_coords, delta=sdf_delta, sdf=sdf)

def vis_sdf(sdf_path: str):
    from mayavi import mlab
    min_coords, delta, sdf = load_sdf(sdf_path)

    src = mlab.pipeline.scalar_field(sdf)
    mlab.pipeline.iso_surface(src, contours=[0, ], opacity=0.3)
    mlab.show()

if __name__ == "__main__":
    # gen_sdf("tmp/debug_sdf/box/box.obj")
    # sdf_data = np.load("tmp/debug_sdf/box/sdf.npz")
    gen_sdf(os.path.join("env", "data", "assets", "table", "table.obj"))
    code.interact(local=dict(globals(), **locals()))
    # gen_sdf("tmp/debug_sdf/003_cracker_box/model_normalized_convex.obj")

"""
python -m env.tools.sdf
"""