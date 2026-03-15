import json
import logging
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.special import softmax
from sklearn.neighbors import KDTree
from skimage.color import rgb2lab
from vispy import scene, app

from src.utils import load_config, arg_parse

from vispy import scene, app

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




def setup_logger(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )
    return logging.getLogger(__name__)


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
    view.camera = scene.TurntableCamera(fov=45, azimuth=30, elevation=30, distance=3.0)

    app.run()


def load_bbox(transform_json_path):
    with open(transform_json_path, "r") as f:
        data = json.load(f)
    bbox = data["point_bbox"]
    return (
        np.array(bbox["min"], dtype=np.float32),
        np.array(bbox["max"], dtype=np.float32),
    )


def load_topdown_mask(mask_path, R):
    img = Image.open(mask_path).convert("L")
    img = img.resize((R, R), resample=Image.NEAREST)
    mask = np.array(img) > 0
    mask = np.rot90(mask, k=3)
    return mask


def get_mask3d(points, mask2d, bbox_min, bbox_max):
    R = mask2d.shape[0]

    x = (points[:, 0] - bbox_min[0]) / (bbox_max[0] - bbox_min[0])
    y = (points[:, 1] - bbox_min[1]) / (bbox_max[1] - bbox_min[1])

    ix = np.clip((x * (R - 1)).astype(np.int32), 0, R - 1)
    iy = np.clip((y * (R - 1)).astype(np.int32), 0, R - 1)

    return mask2d[ix, iy]


def prepare_volume(points, rgbs, semantics, radius, min_L=20.0):
    sem_prob = softmax(semantics, axis=1)

    rgbs = np.clip(rgbs, 0.0, 1.0)
    lab = rgb2lab(rgbs.reshape(-1, 1, 3)).reshape(-1, 3)

    photometric_mask = lab[:, 0] > min_L

    return sem_prob, lab, photometric_mask

def semantic_volume_filter(
    points,
    sigma,
    sem_prob,
    neighbors,
    target_class=1,
    sem_conf_thresh=0.7,
    sigma_percentile=10,
    min_semantic_neighbors=6,
):
    N = len(points)
    mask = np.ones(N, dtype=bool)

    class_prob = sem_prob[:, target_class]
    mask &= class_prob > sem_conf_thresh

    if mask.sum() == 0:
        return mask

    valid_sigma = sigma[mask]
    valid_sigma = valid_sigma[valid_sigma > 0]

    if len(valid_sigma) > 0:
        sigma_thresh = np.percentile(valid_sigma, sigma_percentile)
        mask &= sigma.squeeze() > sigma_thresh

    if mask.sum() == 0:
        return mask

    semantic_votes = class_prob > sem_conf_thresh
    keep = np.zeros(N, dtype=bool)

    for i in np.where(mask)[0]:
        if semantic_votes[neighbors[i]].sum() >= min_semantic_neighbors:
            keep[i] = True

    return keep


def grow_from_seeds(
    neighbors,
    lab,
    seed_mask,
    min_neighbors=10,
    std_factor=2.0,
    max_iters=5,
):
    mask = seed_mask.copy()

    for _ in range(max_iters):
        if mask.sum() == 0:
            break

        lab_obj = lab[mask]
        mean = lab_obj.mean(axis=0)
        std = lab_obj.std(axis=0)

        lower = mean - std_factor * std
        upper = mean + std_factor * std

        color_ok = np.all((lab >= lower) & (lab <= upper), axis=1)
        candidates = np.where(color_ok & ~mask)[0]

        promote = []
        for i in candidates:
            if mask[neighbors[i]].sum() >= min_neighbors:
                promote.append(i)

        if not promote:
            break

        mask[promote] = True

    return mask


def main():
    logger = setup_logger()
    args = arg_parse()
    cfg = load_config(args.config_path)

    volume = np.load(cfg.files.volume)
    points = volume["points"]
    rgbs = volume["rgbs"]
    sigma = volume["sigma"]
    semantics = volume["semantics"]

    visualize(points, rgbs)
    # -------------------------------
    # 1. 3D-MASKE (KRITISCH!)
    # -------------------------------
    bbox_min, bbox_max = load_bbox(cfg.files.transforms_json)
    R = cfg.data.volume_resolution
    mask2d = load_topdown_mask(
        Path(cfg.paths.root) / "axis_renders" / "topdown_filled_mask.png",
        R,
    )
    mask3d = get_mask3d(points, mask2d, bbox_min, bbox_max)

    points = points[mask3d]
    rgbs = rgbs[mask3d]
    sigma = sigma[mask3d]
    semantics = semantics[mask3d]

    visualize(points, rgbs)


    postprocess = True
    if postprocess:

        logger.info(f"Points after 3D mask: {len(points)}")

        ##############
        # APPLY MASK CONSISTENTLY
        sem_prob, lab, photo_mask = prepare_volume(
            points, rgbs, semantics, radius=0.03, min_L=20
        )

        # APPLY MASK CONSISTENTLY (BOOL MASK!)
        points     = points[photo_mask]
        rgbs       = rgbs[photo_mask]
        sigma      = sigma[photo_mask]
        semantics  = semantics[photo_mask]
        sem_prob   = sem_prob[photo_mask]
        lab        = lab[photo_mask]

        # NOW build neighbors
        tree = KDTree(points)
        neighbors = tree.query_radius(points, r=0.03) # default 0.03
        ##############

        seed_mask = semantic_volume_filter(
            points=points,
            sigma=sigma,
            sem_prob=sem_prob,
            neighbors=neighbors,
            target_class=1,
            sem_conf_thresh=0.99,  # 0.99,
            sigma_percentile=30,  # 30,  # 10,
            min_semantic_neighbors=50,
        )
        # wheat_155160.yml
        # sigma_percentile=30
        # sem_conf_thresh=0.6

        final_mask = grow_from_seeds(
            neighbors=neighbors,
            lab=lab,
            seed_mask=seed_mask,
            min_neighbors=20,
            std_factor=2.0,
            max_iters=5,
        )
        # ################
        # wheat_155160.yml
        # min_neighbors=20,
        #
        # wheat_155386.yml
        # min_neighbors=20,
        # max_iters=20,
        #
        # wheat_431983.yml
        # min_neighbors=20,
        # max_iters=20,
        # min_neighbors=150,
        #
        # wheat_716024.yml
        # sem_conf_thresh=0.6

        points_vis = points.copy()
        rgbs_vis = rgbs.copy()
        rgbs_vis[final_mask] = np.array([1.0, 0.0, 0.0])

        points_f = points[final_mask]
        rgbs_f = rgbs[final_mask]

        visualize(points_vis, rgbs_vis)
        visualize(points_f, rgbs_f)

        volume_dir = Path(cfg.paths.volume)
        volume_dir.mkdir(parents=True, exist_ok=True)

        volume_path = volume_dir / "volume_semantic.npz"
        np.savez(volume_path, points=points_f, rgbs=rgbs_f)
        print(f"Saved volume: {volume_path}")

        volume_path = volume_dir / "volume_semantic_red.npz"
        np.savez(volume_path, points=points_vis, rgbs=rgbs_vis)
        print(f"Saved volume: {volume_path}")


if __name__ == "__main__":
    main()
