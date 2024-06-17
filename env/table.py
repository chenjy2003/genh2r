import os
from typing import Tuple
from dataclasses import dataclass
import pybullet
from pybullet_utils.bullet_client import BulletClient
import code

from env.body import Body, BodyConfig
from env.utils.transform import pos_ros_quat_to_mat

@dataclass
class TableConfig(BodyConfig):
    name: str = "table"
    urdf_file: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "assets", "table", "table.urdf")
    # collision
    collision_mask: int = 2**0
    # base
    use_fixed_base: bool = True
    base_position: Tuple[float] = (0.61, 0.28, 0.0)
    base_orientation: Tuple[float] = (0., 0., 0., 1.)
    # visual
    link_color: Tuple[float] = (1., 1., 1., 1.)

class Table(Body):
    def __init__(self, bullet_client: BulletClient, cfg: TableConfig):
        super().__init__(bullet_client, cfg)
        self.cfg: TableConfig
        self.world_to_table = pos_ros_quat_to_mat(cfg.base_position, cfg.base_orientation)
    
    def reset(self):
        self.base_reset()
        self.load()

    def pre_step(self):
        pass

    def post_step(self):
        pass

def debug():
    from omegaconf import OmegaConf
    table_cfg = OmegaConf.structured(TableConfig)
    bullet_client = BulletClient(connection_mode=pybullet.GUI) # or pybullet.DIRECT
    table = Table(bullet_client, table_cfg)
    table.reset()
    code.interact(local=dict(globals(), **locals()))

if __name__ == "__main__":
    debug()

"""
DISPLAY="localhost:11.0" python -m env.table
"""