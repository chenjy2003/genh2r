import torch
from torch import nn
from pointnet2_ops.pointnet2_modules import PointnetSAModule
from dataclasses import dataclass
from omegaconf import MISSING
import code

@dataclass
class PointNet2EncoderConfig:
    in_channel: int = MISSING

class PointNet2Encoder(nn.Module):
    def __init__(self, cfg: PointNet2EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.in_channel = cfg.in_channel

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=32,
                radius=0.02,
                nsample=64,
                mlp=[self.in_channel, 64, 64, 128],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=32,
                radius=0.04,
                nsample=128,
                mlp=[128, 128, 128, 256],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 256, 512]
            )
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
        )

    def forward(self, pc):
        " pc: (B, N, 3+C) "
        xyz = pc[:, :, :3].contiguous()
        features = pc.transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.fc_layer(features.squeeze(-1))

def debug():
    from omegaconf import OmegaConf
    default_cfg = OmegaConf.structured(PointNet2EncoderConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg: PointNet2EncoderConfig = OmegaConf.to_object(OmegaConf.merge(default_cfg, cli_cfg))
    encoder = PointNet2Encoder(cfg).cuda()
    pc = torch.randn(2, 1024, cfg.in_channel).cuda()
    feature = encoder(pc)
    code.interact(local=dict(globals(), **locals()))

if __name__ == "__main__":
    debug()

"""
CUDA_VISIBLE_DEVICES=7 python -m models.encoders.pointnet2 in_channel=10
"""