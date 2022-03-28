from urllib.parse import non_hierarchical
from numpy import nonzero
import torch
import torch.nn.functional as F
import einops


def make_coord_grid(shape, range, device=None):
    """
        Args:
            shape: tuple
            range: [minv, maxv] or [[minv_1, maxv_1], ..., [minv_d, maxv_d]] for each dim
        Returns:
            grid: shape (*shape, )
    """
    l_lst = []
    for i, s in enumerate(shape):
        l = (0.5 + torch.arange(s, device=device)) / s
        if isinstance(range[0], int) or isinstance(range[0], float):
            minv, maxv = range
        else:
            minv, maxv = range[i]
        l = minv + (maxv - minv) * l
        l_lst.append(l)
    grid = torch.meshgrid(*l_lst, indexing='ij')
    grid = torch.stack(grid, dim=-1)
    return grid


def poses_to_rays(poses, H, W, focal):
    """
        Args:
            poses: (b n 3 4)
        Returns:
            rays_o, rays_d: shape (b n h w 3)
    """
    if isinstance(focal, torch.Tensor):
        if focal.dim() == 0 or len(focal) == 1:
            focal = focal.item()
    if isinstance(focal, float) or isinstance(focal, int):
        focal = [focal, focal]
    B = poses.shape[0]
    device = poses.device
    x, y = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
    x, y = x + 0.5, y + 0.5 # modified to + 0.5
    dirs = torch.stack([(x - W * .5) / focal[0], -(y - H * .5) / focal[1], -torch.ones_like(x)], dim=-1) # (H, W, 3)

    poses = poses.unsqueeze(2).unsqueeze(2) # b n 3 4 -> b n 1 1 3 4
    dirs = dirs.unsqueeze(0).unsqueeze(0).unsqueeze(-2) # h w 3 -> 1 1 h w 1 3
    rays_o = poses[..., -1].repeat(1, 1, H, W, 1) # b n h w 3
    rays_d = (dirs * poses[..., :3]).sum(dim=-1) # b n h w 3
    return rays_o, rays_d


def volume_rendering(nerf, rays_o, rays_d, near, far, points_per_ray, use_viewdirs, rand):
    """
        Args:
            rays_o, rays_d: shape (b ... 3)
        Returns:
            pred: (b ... 3)
    """
    B = rays_o.shape[0]
    rays_shape = rays_o.shape[1: -1]
    rays_o = rays_o.view(B, -1, 3)
    rays_d = rays_d.view(B, -1, 3)
    n_rays = rays_o.shape[1]
    device = rays_o.device

    # Compute 3D query points
    z_vals = torch.linspace(near, far, points_per_ray, device=device)
    z_vals = einops.repeat(z_vals, 'p -> n p', n=n_rays)
    if rand:
        d = (far - near) / (points_per_ray - 1) # modified as points_per_ray - 1
        z_vals = z_vals + torch.rand(n_rays, points_per_ray, device=device) * d

    pts = rays_o.view(B, n_rays, 1, 3) + rays_d.view(B, n_rays, 1, 3) * z_vals.view(1, n_rays, points_per_ray, 1)

    # Run network
    pts_flat = einops.rearrange(pts, 'b n p d -> b (n p) d')
    if not use_viewdirs:
        raw = nerf(pts_flat)
    else:
        viewdirs = einops.repeat(rays_d, 'b n d -> b n p d', p=points_per_ray)
        raw = nerf(pts_flat, viewdirs=viewdirs)
    raw = einops.rearrange(raw, 'b (n p) c -> b n p c', n=n_rays)

    # Compute opacities and colors
    rgb, sigma_a = raw[..., :3], raw[..., 3]
    rgb = torch.sigmoid(rgb) # b n p 3
    sigma_a = F.relu(sigma_a) # b n p

    # Do volume rendering
    dists = torch.cat([z_vals[:, 1:] - z_vals[:, :-1], torch.ones_like(z_vals[:, -1:]) * 1e-3], dim=-1) # n p
    alpha = 1. - torch.exp(-sigma_a * dists) # b n p
    trans = torch.clamp(1. - alpha + 1e-10, max=1.) # b n p
    trans = torch.cat([torch.ones_like(trans[..., :1]), trans[..., :-1]], dim=-1)
    weights = alpha * torch.cumprod(trans, dim=-1) # b n p

    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)
    acc_map = torch.sum(weights, dim=-1)
    rgb_map = rgb_map + (1. - acc_map).unsqueeze(-1) # white background
    # depth_map = torch.sum(weights * z_vals, dim=-1)

    rgb_map = rgb_map.view(B, *rays_shape, 3)
    return rgb_map

def batched_volume_rendering(nerf, rays_o, rays_d, *args, batch_size=1, **kwargs):
    """
        Args:
            rays_o, rays_d: (b ... 3)
        Returns:
            pred: (b ... 3)
    """
    B = rays_o.shape[0]
    rays_shape = rays_o.shape[1: -1]
    rays_o = rays_o.view(B, -1, 3)
    rays_d = rays_d.view(B, -1, 3)
    lq = 0
    ret = []

    while lq < rays_o.shape[1]:
        rq = min(lq + batch_size, rays_o.shape[1])
        _rays_o = rays_o[:, lq: rq, :]
        _rays_d = rays_d[:, lq: rq, :]
        t = volume_rendering(nerf, _rays_o, _rays_d, *args, **kwargs)
        ret.append(t)
        lq = rq

    ret = torch.cat(ret, dim=1)
    ret = ret.view(B, *rays_shape, 3)
    return ret


def projection_to_views(x, poses, H, W, focal):
    """
        Args:
            x: (b p 3)
            poses: (b n 3 4)
        Returns:
            pi_x: (b n p 2)
    """
    if isinstance(focal, torch.Tensor):
        if focal.dim() == 0 or len(focal) == 1:
            focal = focal.item()
    if isinstance(focal, float) or isinstance(focal, int):
        focal = [focal, focal]
    H, W = H / focal[1], W / focal[0] ##
    x = x.unsqueeze(1) # b 1 p 3
    poses = poses.unsqueeze(2) # b n 1 3 4
    x = x - poses[..., -1] # b n p 3
    x = (x.unsqueeze(-1) * poses[..., :-1]).sum(dim=-2) # b n p 3
    pi_x = torch.ones_like(x[..., :2]) * (max(H, W) * 999) # set inf as outside view
    nonzero_z = (torch.abs(x[..., -1]) > 1e-9)
    pi_x[nonzero_z] = -x[nonzero_z][:, :2] / x[nonzero_z][:, -1:]
    pi_x[..., 0] /= (W / 2)
    pi_x[..., 1] /= (H / 2)
    return pi_x
