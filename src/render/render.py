import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.model import SEM_NGP
from src.rays import sample_pdf

from src.utils import load_config, arg_parse 


def render_rays(
    nerf_model,
    rays_o,
    rays_d,
    near,
    far,
    N_samples=64,
    N_importance=128,
    semantic_ch=2,
):
    device = rays_o.device
    N_rays = rays_o.shape[0]

    t_lin = torch.linspace(0.0, 1.0, N_samples, device=device)[None, :]
    t_vals = near[:, None] * (1.0 - t_lin) + far[:, None] * t_lin

    mids = 0.5 * (t_vals[:, :-1] + t_vals[:, 1:])
    lower = torch.cat([t_vals[:, :1], mids], dim=-1)
    upper = torch.cat([mids, t_vals[:, -1:]], dim=-1)
    t_coarse = lower + (upper - lower) * torch.rand_like(lower)

    pts = rays_o[:, None, :] + rays_d[:, None, :] * t_coarse[:, :, None]
    pts_flat = pts.reshape(-1, 3)
    dirs_flat = rays_d.repeat_interleave(N_samples, dim=0)

    colors, sigma, semantics = nerf_model(pts_flat, dirs_flat)

    colors = colors.view(N_rays, N_samples, 3)
    sigma = sigma.view(N_rays, N_samples)
    semantics = semantics.view(N_rays, N_samples, semantic_ch)

    deltas = torch.cat(
        [t_coarse[:, 1:] - t_coarse[:, :-1], 1e10 * torch.ones(N_rays, 1, device=device)],
        dim=-1,
    )

    alpha = 1 - torch.exp(-sigma * deltas)
    trans = torch.cumprod(
        torch.cat([torch.ones(N_rays, 1, device=device), 1 - alpha + 1e-10], dim=-1),
        dim=-1,
    )[:, :-1]
    weights = alpha * trans

    rgb = torch.sum(weights[..., None] * colors, dim=1)
    sem = torch.sum(weights[..., None] * semantics, dim=1)
    depth = torch.sum(weights * t_coarse, dim=1)

    return rgb, sem, depth


# ORTHOGRAPHISCHE AXIS-RENDERFUNKTION
def render_axis(
    checkpoint_path,
    transforms_path,
    axis="z",
    H=512,
    W=512,
    margin=0.05,
    device="cuda",
    out_dir="axis_renders",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- NeRF laden --------
    model = SEM_NGP().to(device)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()

    # -------- AABB laden --------
    with open(transforms_path, "r") as f:
        meta = json.load(f)

    aabb_min = np.array(meta["point_bbox"]["min"], dtype=np.float32)
    aabb_max = np.array(meta["point_bbox"]["max"], dtype=np.float32)

    xs = np.linspace(aabb_min[0], aabb_max[0], W, dtype=np.float32)
    ys = np.linspace(aabb_max[1], aabb_min[1], H, dtype=np.float32)

    # -------- Strahlen erzeugen --------
    if axis == "z":  # Top-Down
        gx, gy = np.meshgrid(xs, ys)
        z0 = aabb_max[2] + margin
        origins = np.stack([gx, gy, np.full_like(gx, z0)], axis=-1)
        directions = np.array([0, 0, -1], dtype=np.float32)
        near = np.zeros(H * W)
        far = z0 - aabb_min[2]

    elif axis == "y":
        gx, gz = np.meshgrid(xs, ys)
        y0 = aabb_max[1] + margin
        origins = np.stack([gx, np.full_like(gx, y0), gz], axis=-1)
        directions = np.array([0, -1, 0], dtype=np.float32)
        near = np.zeros(H * W)
        far = y0 - aabb_min[1]

    elif axis == "x":
        gy, gz = np.meshgrid(xs, ys)
        x0 = aabb_max[0] + margin
        origins = np.stack([np.full_like(gy, x0), gy, gz], axis=-1)
        directions = np.array([-1, 0, 0], dtype=np.float32)
        near = np.zeros(H * W)
        far = x0 - aabb_min[0]

    else:
        raise ValueError("axis must be x, y or z")

    rays_o = torch.from_numpy(origins.reshape(-1, 3)).to(device)
    rays_d = torch.from_numpy(np.tile(directions, (H * W, 1))).to(device)
    near = torch.from_numpy(near).float().to(device)
    far = torch.full_like(near, far).to(device)

    # -------- Rendern --------
    with torch.no_grad():
        chunk_size = 8192

        rgbs = []
        sems = []
        depths = []

        with torch.no_grad():
            for start in range(0, rays_o.shape[0], chunk_size):
                end = min(start + chunk_size, rays_o.shape[0])

                rgb_c, sem_c, depth_c = render_rays(
                    model,
                    rays_o[start:end],
                    rays_d[start:end],
                    near[start:end],
                    far[start:end],
                )

                rgbs.append(rgb_c)
                sems.append(sem_c)
                depths.append(depth_c)

        rgb = torch.cat(rgbs, dim=0)
    rgb = rgb.reshape(H, W, 3).cpu().numpy()

    # -------- Speichern --------
    plt.imsave(out_dir / f"{axis}_rgb.png", np.clip(rgb, 0, 1))
    print(f"Saved {axis}-axis render to {out_dir}")



if __name__ == "__main__":

    args = arg_parse()
    cfg = load_config(args.config_path)

    ckpt_path = Path(cfg.paths.checkpoints)/  "sem_nerf_model.pth"
    transform_path = Path(cfg.files.transforms_json)
    output_dir = Path(cfg.paths.root) / "axis_renders"
    render_axis(
        checkpoint_path=ckpt_path,
        transforms_path=transform_path,
        axis="z",
        out_dir=output_dir,
    )
