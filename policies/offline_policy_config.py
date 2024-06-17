from dataclasses import dataclass
from omegaconf import MISSING
from typing import Optional

from .base_policy_config import BasePolicyConfig

@dataclass
class OfflinePolicyConfig(BasePolicyConfig):
    name: str = "offline"
    demo_dir: Optional[str] = None
    demo_structure: str = "hierarchical" # "flat"
    demo_source: str = "genh2r" # "handoversim"
    check_input_pcd: bool = False
