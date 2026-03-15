import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from vispy import scene, app

from src.model import SEM_NGP
from src.utils import load_config, arg_parse

import torch
import torch.nn.functional as F


def create_grid_from_bbox(bbox_min, bbox_max, res):
    """
    bbox_min: (3,) tensor or list [xmin, ymin, zmin]
    bbox_max: (3,) tensor or list [xmax, ymax, zmax]
    res: int or tuple (nx, ny, nz)
    """
    if isinstance(res, int):
        res = (res, res, res)

    x = torch.linspace(bbox_min[0], bbox_max[0], res[0])
    y = torch.linspace(bbox_min[1], bbox_max[1], res[1])
    z = torch.linspace(bbox_min[2], bbox_max[2] + 0.1, res[2])

    grid = torch.stack(
        torch.meshgrid(x, y, z, indexing="ij"), dim=-1
    )  # (nx, ny, nz, 3)

    pts = grid.reshape(-1, 3)  # (N, 3)

    return pts, grid.shape[:3]


def sample_density_color_semantic(model, pts, device, batch=50000):
    all_sigma = []
    all_rgb = []
    all_sem = []

    viewdir = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32).to(device)

    for i in tqdm(range(0, len(pts), batch), desc="Sampling NeRF"):
        p = pts[i : i + batch].to(device)
        d = viewdir.expand(len(p), 3)

        with torch.no_grad():
            rgb, sigma, semantic = model(p, d)

        all_sigma.append(sigma.cpu())
        all_rgb.append(rgb.cpu())
        all_sem.append(semantic.cpu())

    return torch.cat(all_sigma), torch.cat(all_rgb), torch.cat(all_sem)


def load_sem_nerf_model(path="sem_nerf_model.pth", device="cpu"):
    model = SEM_NGP().to(device)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def load_json(transform_json: str):
    with open(transform_json, "r") as f:
        data = json.load(f)
    return data


def visualize(points, rgbs):
    canvas = scene.SceneCanvas(keys="interactive", bgcolor="black", show=True)
    view = canvas.central_widget.add_view()

    scatter = scene.visuals.Markers()
    scatter.set_data(
        points,
        size=1,
        face_color=rgbs,
        edge_color=rgbs,
    )
    scatter.set_gl_state(blend=False, depth_test=True)
    view.add(scatter)

    axis = scene.visuals.XYZAxis(parent=view.scene)
    view.camera = scene.TurntableCamera(
        fov=45, azimuth=30, elevation=30, distance=3.0, center=(0, 0, 0)
    )

    app.run()


def main():
    args = arg_parse()
    cfg = load_config(args.config_path)

    chkpts = Path(cfg.paths.checkpoints, "sem_nerf_model.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_sem_nerf_model(chkpts, device)

    data = load_json(cfg.files.transforms_json)
    bbox_min = torch.tensor(data["point_bbox"]["min"], dtype=torch.float32)
    bbox_max = torch.tensor(data["point_bbox"]["max"], dtype=torch.float32)

    pts, vol_shape = create_grid_from_bbox(
        bbox_min=bbox_min, bbox_max=bbox_max, res=cfg.data.volume_resolution
    )
    print(pts.shape)
    print(vol_shape)

    sigma, rgb, semantic_logits = sample_density_color_semantic(model, pts, device)

    thr_d = np.percentile(sigma, cfg.data.sigma_threshold_percentile)
    sigma_mask = (sigma > thr_d).squeeze()

    pts_final = pts[sigma_mask]
    rgb_final = rgb[sigma_mask]
    sigma_final = sigma[sigma_mask]
    semantics_final = semantic_logits[sigma_mask]

    pts_final = pts_final.detach().cpu().numpy().astype(np.float32)
    rgb_final = rgb_final.detach().cpu().numpy().astype(np.float32)
    semantics_final = semantics_final.detach().cpu().numpy().astype(np.float32)

    volume_dir = Path(cfg.paths.volume)
    volume_dir.mkdir(parents=True, exist_ok=True)
    volume_path = volume_dir / "volume.npz"
    np.savez(
        volume_path,
        points=pts_final,
        rgbs=rgb_final,
        sigma=sigma_final,
        semantics=semantics_final,
    )
    print(f"Saved volume: {volume_path}")

    visualize(pts_final, rgb_final)


if __name__ == "__main__":
    main()
