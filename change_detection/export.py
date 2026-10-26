from __future__ import annotations

import numpy as np
import torch

from utils.sh_utils import SH2RGB


def gaussian_rgb_from_dc(gaussians):
    dc = gaussians._features_dc.detach()[:, 0, :]
    rgb = SH2RGB(dc)
    return torch.clamp(rgb, 0.0, 1.0)


def gather_gaussian_arrays(gaussians):
    with torch.no_grad():
        return {
            "gaussian_id": np.arange(gaussians.get_xyz.shape[0], dtype=np.int64),
            "xyz": gaussians.get_xyz.detach().cpu().numpy(),
            "scale": gaussians.get_scaling.detach().cpu().numpy(),
            "opacity_pre": gaussians.get_opacity.detach().cpu().numpy(),
            "rgb": gaussian_rgb_from_dc(gaussians).detach().cpu().numpy(),
        }
