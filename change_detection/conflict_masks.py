from __future__ import annotations

import torch


def robust_color_correct(pre_rgb, post_rgb, valid_mask, eps=1e-6):
    mask = valid_mask.squeeze(0)
    if mask.sum() < 32:
        return pre_rgb, torch.ones(3, device=pre_rgb.device), torch.zeros(3, device=pre_rgb.device)

    scales = []
    offsets = []
    corrected = []
    for channel in range(3):
        pre_values = pre_rgb[channel][mask]
        post_values = post_rgb[channel][mask]
        pre_std = torch.std(pre_values, unbiased=False)
        post_std = torch.std(post_values, unbiased=False)
        scale = torch.clamp(post_std / (pre_std + eps), 0.5, 2.0)
        offset = torch.median(post_values) - scale * torch.median(pre_values)
        offset = torch.clamp(offset, -0.25, 0.25)
        scales.append(scale)
        offsets.append(offset)
        corrected.append(torch.clamp(scale * pre_rgb[channel] + offset, 0.0, 1.0))
    return torch.stack(corrected, dim=0), torch.stack(scales), torch.stack(offsets)


@torch.no_grad()
def build_conflict_masks(
    pre_pack,
    post_rgb,
    post_depth=None,
    valid_post_depth=None,
    alpha_min=0.2,
    depth_margin=0.03,
    rgb_thresh=0.12,
    rgb_quantile=0.90,
    use_rgb_conflict=True,
    use_depth_conflict=True,
):
    pre_rgb = torch.clamp(pre_pack["render"], 0.0, 1.0)
    pre_alpha = torch.clamp(pre_pack["rend_alpha"], 0.0, 1.0)
    pre_depth = pre_pack["surf_depth"]
    post_rgb = torch.clamp(post_rgb, 0.0, 1.0)

    valid_pre = (pre_alpha > alpha_min) & (pre_depth > 0.0)
    valid_mask = pre_alpha > alpha_min

    if use_depth_conflict and post_depth is not None and valid_post_depth is not None:
        m_free = valid_post_depth & valid_pre & (pre_depth < post_depth - depth_margin)
        m_new_occlusion = valid_post_depth & valid_pre & (post_depth < pre_depth - depth_margin)
    else:
        m_free = torch.zeros_like(pre_alpha, dtype=torch.bool)
        m_new_occlusion = torch.zeros_like(pre_alpha, dtype=torch.bool)

    pre_corr, scale, offset = robust_color_correct(pre_rgb, post_rgb, valid_mask)
    rgb_residual = torch.mean(torch.abs(pre_corr - post_rgb), dim=0, keepdim=True)

    if use_rgb_conflict:
        rgb_valid = valid_mask & (~m_new_occlusion)
        if rgb_thresh < 0.0 and rgb_valid.any():
            threshold = torch.quantile(rgb_residual[rgb_valid], rgb_quantile)
        else:
            threshold = torch.tensor(rgb_thresh, dtype=rgb_residual.dtype, device=rgb_residual.device)
        m_rgb = (rgb_residual > threshold) & rgb_valid
    else:
        threshold = torch.tensor(rgb_thresh, dtype=rgb_residual.dtype, device=rgb_residual.device)
        m_rgb = torch.zeros_like(pre_alpha, dtype=torch.bool)

    return {
        "pre_rgb": pre_rgb,
        "pre_alpha": pre_alpha,
        "pre_depth": pre_depth,
        "post_rgb": post_rgb,
        "post_depth": post_depth,
        "valid_post_depth": valid_post_depth,
        "pre_corrected": pre_corr,
        "rgb_residual": rgb_residual,
        "rgb_threshold": threshold,
        "color_scale": scale,
        "color_offset": offset,
        "M_rgb": m_rgb,
        "M_free": m_free,
        "M_new_occlusion": m_new_occlusion,
        "M_weak": m_rgb,
        "M_strong": m_free,
        "valid_mask": valid_mask,
    }
