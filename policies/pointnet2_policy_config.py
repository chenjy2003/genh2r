from dataclasses import dataclass, field

from .base_policy_config import BasePolicyConfig
from .utils.point_cloud import PointCloudProcessorConfig
from models.policy_network import PolicyNetworkConfig

@dataclass
class PointNet2PolicyConfig(BasePolicyConfig):
    name: str = "pointnet2"

    processor: PointCloudProcessorConfig = field(default_factory=PointCloudProcessorConfig)
    model: PolicyNetworkConfig = field(default_factory=lambda: PolicyNetworkConfig(
        in_channel="${..processor.in_channel}", 
    ))
