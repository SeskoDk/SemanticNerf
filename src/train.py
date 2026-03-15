from pathlib import Path
import shutil
from typing import Dict

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from src.dataset import RayDataset
from src.utils import load_config, arg_parse, prepare_output_dirs
from src.model import SEM_NGP
from src.rays import sample_pdf

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def create_eval_dir(output_dir):
    out_dir = Path(output_dir, "novel_views").expanduser().resolve()

    if out_dir.exists():
        shutil.rmtree(out_dir)

    (out_dir / "color").mkdir(parents=True, exist_ok=True)
    (out_dir / "depth").mkdir(parents=True, exist_ok=True)
    (out_dir / "sematic").mkdir(parents=True, exist_ok=True)


def render_rays(
    nerf_model,
    rays_o,
    rays_d,
    near,
    far,
    N_samples: int = 64,
    N_importance: int = 128,
    semantic_ch: int = 2,
    fixd_interval: bool = False,
):
    device = rays_o.device
    N_rays = rays_o.shape[0]

    if fixd_interval:
        hn = 0
        hf = 1
        t_vals = torch.linspace(hn, hf, N_samples, device=device)
        t_vals = t_vals.expand(N_rays, N_samples)
    else:
        t_lin = torch.linspace(0.0, 1.0, N_samples, device=device)
        t_lin = t_lin[None, :]  #
        near = near[:, None]  # (B,1)
        far = far[:, None]  # (B,1)
        t_vals = near * (1.0 - t_lin) + far * t_lin

    mids = 0.5 * (t_vals[:, :-1] + t_vals[:, 1:])
    lower = torch.cat([t_vals[:, :1], mids], dim=-1)
    upper = torch.cat([mids, t_vals[:, -1:]], dim=-1)
    t_coarse = lower + (upper - lower) * torch.rand_like(lower)

    # Convert samples to 3D points
    pts_coarse = rays_o[:, None, :] + rays_d[:, None, :] * t_coarse[:, :, None]

    dirs_c = rays_d.repeat_interleave(N_samples, dim=0)  # (B * N_samples) x 3
    pts_c = pts_coarse.reshape(-1, 3)  # (B * N_samples) x 3
    colors_c, sigma_c, sematic_c = nerf_model(pts_c, dirs_c)

    colors_c = colors_c.reshape(N_rays, N_samples, 3)
    sigma_c = sigma_c.reshape(N_rays, N_samples)
    sematic_c = sematic_c.reshape(N_rays, N_samples, semantic_ch)

    # Distance deltas
    deltas = torch.cat(
        [
            t_coarse[:, 1:] - t_coarse[:, :-1],
            1e10 * torch.ones(N_rays, 1, device=device),
        ],
        dim=-1,
    )

    # Volume rendering
    alpha = 1 - torch.exp(-sigma_c * deltas)
    trans = torch.cumprod(
        torch.cat([torch.ones(N_rays, 1, device=device), 1 - alpha + 1e-10], dim=-1),
        dim=-1,
    )[:, :-1]
    weights = alpha * trans  # (N_rays, N_samples)

    # Coarse RGB
    rgb_coarse = torch.sum(weights[..., None] * colors_c, dim=1)
    # Coarse Semantic
    sem_coarse = torch.sum(weights[..., None] * sematic_c, dim=1)  # B x semantic_chs

    # 2. IMPORTANCE SAMPLING (sample_pdf)
    if N_importance > 0:
        # Bin midpoints for sampling
        mids = 0.5 * (t_coarse[:, :-1] + t_coarse[:, 1:])

        t_fine = sample_pdf(
            bins=mids,
            weights=weights[:, 1:-1],  # drop first/last per NeRF implementation
            N_importance=N_importance,
            det=False,
        ).detach()

        t_all = torch.sort(torch.cat([t_coarse, t_fine], dim=-1), dim=-1).values

        # Compute 3D points for fine samples
        pts_fine = rays_o[:, None, :] + rays_d[:, None, :] * t_all[:, :, None]

        N_all = t_all.shape[1]
        dirs_f = rays_d.repeat_interleave(N_all, dim=0)
        pts_f = pts_fine.reshape(-1, 3)

        colors_f, sigma_f, sematic_f = nerf_model(pts_f, dirs_f)

        colors_f = colors_f.reshape(N_rays, N_all, 3)
        sematic_f = sematic_f.reshape(N_rays, N_all, semantic_ch)
        sigma_f = sigma_f.reshape(N_rays, N_all)

        # Recompute deltas
        deltas_f = torch.cat(
            [t_all[:, 1:] - t_all[:, :-1], 1e10 * torch.ones(N_rays, 1, device=device)],
            dim=-1,
        )

        alpha_f = 1 - torch.exp(-sigma_f * deltas_f)
        trans_f = torch.cumprod(
            torch.cat(
                [torch.ones(N_rays, 1, device=device), 1 - alpha_f + 1e-10], dim=-1
            ),
            dim=-1,
        )[:, :-1]
        weights_f = alpha_f * trans_f

        # Final rendered RGB
        rgb_fine = torch.sum(weights_f[..., None] * colors_f, dim=1)

        sematic_fine = torch.sum(weights_f[..., None] * sematic_f, dim=1)

        # Depth estimation
        depth = torch.sum(weights_f * t_all, dim=-1)  # (N_rays,)

        # return rgb_fine  # use fine output
        return rgb_fine, sematic_fine, depth

    depth = torch.sum(weights * t_coarse, dim=-1)
    return rgb_coarse, sem_coarse, depth


def eval_model(cfg, model, data, device, chunk_size=1024 * 6):
    model.eval()
    semantic_ch = model.semantic_channels

    rays_o = torch.from_numpy(data["rays_o"]).to(device)  # (N,H*W,3)
    rays_d = torch.from_numpy(data["rays_d"]).to(device)
    near = torch.from_numpy(data["near"]).to(device)
    far = torch.from_numpy(data["far"]).to(device)

    H = int(data["H"])
    W = int(data["W"])
    N_imgs = rays_o.shape[0]

    with torch.no_grad():
        for img_idx in range(N_imgs):
            colors = []
            depths = []
            semantics = []

            ro = rays_o[img_idx]  # (H*W,3)
            rd = rays_d[img_idx]
            n = near[img_idx]
            f = far[img_idx]

            for start in range(0, H * W, chunk_size):
                end = min(start + chunk_size, H * W)

                rgb, sem_logits, depth = render_rays(
                    model,
                    ro[start:end],
                    rd[start:end],
                    n[start:end],
                    f[start:end],
                )

                colors.append(rgb)
                semantics.append(sem_logits)
                depths.append(depth)

            # Stack and reshape
            img = torch.cat(colors).reshape(H, W, 3).cpu().numpy()
            img = np.clip(img, 0.0, 1.0)

            depth_img = torch.cat(depths).reshape(H, W).cpu().numpy()
            sem_logits_img = (
                torch.cat(semantics).reshape(H, W, semantic_ch).cpu().numpy()
            )
            novel_path = cfg.paths.novel_views
            # Save RGB
            plt.imsave(f"{novel_path}/color/img_{img_idx:04d}.png", img)

            # Save semantics
            sem_ids = sem_logits_img.argmax(axis=-1)
            plt.imsave(
                f"{novel_path}/sematic/img_{img_idx:04d}.png",
                sem_ids,
            )

            # Save depth
            depth_norm = depth_img / max(depth_img.max(), 1e-6)
            plt.imsave(
                f"{novel_path}/depth/depth_{img_idx:04d}.png",
                depth_norm,
                cmap="binary",
            )
            np.save(
                f"{novel_path}/depth/depth_{img_idx:04d}.npy",
                depth_img,
            )


def train(
    cfg,
    nerf_model,
    optimizer,
    scheduler,
    data_loader,
    lambda_sem: float,
    num_epochs: int,
    device: torch.device,
    H: int,
    W: int,
    N_samples: int = 64,
    N_importance: int = 128,
    eval_dir: Path = None,
    eval_data_set=None,
):

    training_loss = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        pbar = tqdm(
            data_loader,
            total=len(data_loader),
            desc=f"Epoch [{epoch + 1} /{num_epochs}]",
        )

        for idx, batch in enumerate(pbar):

            rays_o = batch["rays_o"].to(device)
            rays_d = batch["rays_d"].to(device)
            rgb_gt = batch["rgb"].to(device)

            mask_gt = batch["mask"]
            mask_gt = mask_gt.long().squeeze(-1).to(device)

            near = batch["near"].to(device)
            far = batch["far"].to(device)

            rgb_pred, mask_pred, _ = render_rays(
                nerf_model, rays_o, rays_d, near, far, N_samples, N_importance
            )

            rgb_loss = F.mse_loss(rgb_pred, rgb_gt)
            sem_loss = F.cross_entropy(mask_pred, mask_gt)
            loss = rgb_loss + lambda_sem * sem_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss.append(loss.item())
            epoch_loss += loss.item()

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.6f}",
                    "loss_rgb": f"{rgb_loss.item():.6f}",
                    "loss_sem": f"{sem_loss.item():.6f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

        print(f"Epoch {epoch + 1}: avg loss = {epoch_loss  / len(data_loader):.6f}")
        scheduler.step()

        if eval_dir is not None and eval_data_set is not None:
            print("Render test images")
            create_eval_dir(eval_dir)
            eval_model(cfg, nerf_model, eval_data_set, device)


def main():
    args = arg_parse()
    cfg = load_config(args.config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    ray_path = Path(cfg.paths.root) / "rays.npz"
    dataset = RayDataset(ray_path)

    model = SEM_NGP().to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        model_optimizer, milestones=[2, 4, 8], gamma=0.5
    )

    data_loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)
    H, W = dataset.H, dataset.W

    eval_data_path = Path(cfg.paths.root) / "rays_eval.npz"
    eval_data_set = np.load(eval_data_path, allow_pickle=True)
    train(
        cfg,
        model,
        model_optimizer,
        scheduler,
        data_loader,
        lambda_sem=cfg.train.lambda_sem,
        num_epochs=cfg.train.num_epochs,
        H=H,
        W=W,
        N_samples=cfg.train.N_samples,
        N_importance=cfg.train.N_importance,
        device=device,
        eval_dir=cfg.paths.root,
        eval_data_set=eval_data_set,
    )

    print("Training completed.")
    ckpts_dir = Path(cfg.paths.checkpoints) 
    ckpts_dir.mkdir(parents=True, exist_ok=True)

    model_path = ckpts_dir / "sem_nerf_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved under: {model_path}")


if __name__ == "__main__":
    main()
