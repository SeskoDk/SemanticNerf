import cv2
import json
import numpy as np
import open3d as o3d
from tqdm import tqdm
from PIL import Image
from pathlib import Path

from src.utils import load_config


from typing import Dict


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )

    # 1. Calculate ray directions in the camera frame (Z-backward convention)
    # This part is correct for a Z-backward system: -np.ones_like(i)
    dirs = np.stack(
        [(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1
    )

    # 2. *** FIX: Apply the COLMAP-to-NeRF axis flip to the c2w matrix ***
    # This converts the COLMAP-style c2w (Z-forward) to the NeRF-style c2w (Z-backward)
    # by flipping the Y and Z axes of the rotation matrix.
    c2w_flip = c2w.copy()
    # The standard NeRF flip is: c2w[:3, 1:3] *= -1
    # This is equivalent to multiplying the rotation matrix R by [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    c2w_flip[:3, 1:3] *= -1

    # 3. Rotate ray directions from camera frame to the world frame
    # Use the flipped c2w matrix for the rotation
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w_flip[:3, :3], -1)

    # 4. Translate camera frame's origin to the world frame.
    # Use the flipped c2w matrix for the translation
    rays_o = np.broadcast_to(c2w_flip[:3, -1], np.shape(rays_d))

    # 5. Reshape and normalize
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)

    return rays_o, rays_d


def get_intrinsics(transform: Dict, downsample: int = 1) -> Dict:
    W = int(transform["w"])
    H = int(transform["h"])
    fx = float(transform["fl_x"])
    fy = float(transform.get("fl_y", fx))
    cx = float(transform.get("cx", W * 0.5))
    cy = float(transform.get("cy", H * 0.5))

    # downsample intrinsics
    W = W // downsample
    H = H // downsample
    fx = fx / downsample
    fy = fy / downsample
    cx = cx / downsample
    cy = cy / downsample
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    intrinsics = {
        "K": K,
        "W": W,
        "H": H,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }
    return intrinsics


def resize_image(img: np.ndarray, H: int, W: int) -> np.ndarray:
    return cv2.resize(
        img,
        (W, H),
        interpolation=cv2.INTER_AREA,
    )


def resize_masks(mask: np.ndarray, H: int, W: int) -> np.ndarray:
    return cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)


def load_transforms(
    transform_json: Path,
    mask_dir: Path,
    downsample: int = 1,
    use_semantics: bool = True,
):

    with open(transform_json, "r") as f:
        transforms = json.load(f)

    intrinsics = get_intrinsics(transforms, downsample)
    H, W = intrinsics["H"], intrinsics["W"]

    images, masks, poses = [], [], []

    for frame in tqdm(transforms["frames"], desc="Load Data"):
        image_path = frame["file_path"]
        img = np.array(Image.open(image_path))

        if use_semantics:
            stem = Path(image_path).stem
            mask_path = mask_dir / f"{stem}_mask.png"
            mask = np.array(Image.open(mask_path))
            if mask.ndim == 3:
                mask = mask[..., 0]
            mask = (mask > 0).astype(np.float32)

        if downsample > 1:
            img = resize_image(img, H, W)
            if use_semantics:
                mask = resize_masks(mask, H, W)

        img = img.astype(np.float32) / 255.0
        images.append(img)

        if use_semantics:
            masks.append(mask)

        poses.append(np.array(frame["transform_matrix"], dtype=np.float32))

    images = np.stack(images)
    poses = np.stack(poses)
    masks = np.stack(masks) if use_semantics else None

    return images, masks, poses, intrinsics


def intersect_aabb(rays_o, rays_d, aabb_min=-1.0, aabb_max=1.0, eps=1e-6):
    """
    rays_o: (N, 3)
    rays_d: (N, 3)
    returns:
        t_near: (N,)
        t_far:  (N,)
        valid:  (N,) bool
    """

    inv_d = 1.0 / (rays_d + eps)

    t0 = (aabb_min - rays_o) * inv_d
    t1 = (aabb_max - rays_o) * inv_d

    t_min = np.minimum(t0, t1)
    t_max = np.maximum(t0, t1)

    t_near = np.max(t_min, axis=1)
    t_far = np.min(t_max, axis=1)

    valid = t_far >= np.maximum(t_near, 0.0)

    return t_near, t_far, valid


def compute_rays(
    poses: np.ndarray,
    images: np.ndarray,
    masks: np.ndarray,
    H: int,
    W: int,
    K: np.ndarray,
):
    N = len(poses)
    rays_o_list = []
    rays_d_list = []
    rgb_gt_list = []
    mask_gt_list = []

    for idx in tqdm(range(N), desc="Loading rays and rgb_gt", total=N):
        c2w = poses[idx]
        target_image = images[idx]
        rays_o, rays_d = get_rays_np(H, W, K, c2w)
        target_px_values = target_image.reshape(-1, 3)
        rays_o_list.append(rays_o)
        rays_d_list.append(rays_d)
        rgb_gt_list.append(target_px_values)

        if masks is not None:
            target_mask = masks[idx]
            target_mask_values = target_mask.reshape(-1, 1)
            mask_gt_list.append(target_mask_values)

    rays_o_list = np.array(rays_o_list).reshape(-1, 3)
    rays_d_list = np.array(rays_d_list).reshape(-1, 3)
    rgb_gt_list = np.array(rgb_gt_list).reshape(-1, 3)
    print(f"rays_o_all: {rays_o_list.shape}")
    print(f"rays_d_list: {rays_d_list.shape}")
    print(f"rgb_gt_list: {rgb_gt_list.shape}")

    if masks is not None:
        mask_gt_list = np.array(mask_gt_list).reshape(-1, 1)
        print(f"mask_gt_list: {mask_gt_list.shape}")

    return rays_o_list, rays_d_list, rgb_gt_list, mask_gt_list


def compute_eval_rays(
    poses: np.ndarray,
    images: np.ndarray,
    masks: np.ndarray,
    H: int,
    W: int,
    K: np.ndarray,
    N_eval_imgs: int,
    aabb_min: float = -1.0,
    aabb_max: float = 1.0,
):

    rays_o_list = []
    rays_d_list = []
    near_list = []
    far_list = []
    valid_list = []
    rgb_gt_list = []
    mask_gt_list = []

    for idx in tqdm(
        range(N_eval_imgs),
        desc="Computing eval rays",
        total=N_eval_imgs,
    ):
        c2w = poses[idx]
        rays_o, rays_d = get_rays_np(H, W, K, c2w)  # (H*W,3)

        # Compute near/far but DO NOT FILTER
        near, far, valid = intersect_aabb(
            rays_o, rays_d, aabb_min=aabb_min, aabb_max=aabb_max
        )
        near = np.maximum(near, 0.0)

        rays_o_list.append(rays_o)
        rays_d_list.append(rays_d)
        near_list.append(near)
        far_list.append(far)
        valid_list.append(valid)

        if images is not None:
            rgb_gt_list.append(images[idx].reshape(-1, 3))

        if masks is not None:
            mask_gt_list.append(masks[idx].reshape(-1, 1))

    # ---------- INFO PRINTS ----------
    total_rays = N_eval_imgs * H * W
    print("\n[Eval Ray Dataset Summary]")
    print(f"  Images             : {N_eval_imgs}")
    print(f"  Image resolution   : {H} x {W}")
    print(f"  Rays per image     : {H * W}")
    print(f"  Total rays         : {total_rays}")
    print(f"  rays_o shape       : {rays_o.shape}")
    print(f"  rays_d shape       : {rays_d.shape}")
    print(f"  near / far shape   : {near.shape}")
    print(f"  valid mask shape   : {valid.shape}")
    print("----------------------------------\n")

    return (
        np.array(rays_o_list),  # (N,H*W,3)
        np.array(rays_d_list),  # (N,H*W,3)
        np.array(near_list),  # (N,H*W)
        np.array(far_list),  # (N,H*W)
        np.array(valid_list),  # (N,H*W)
        np.array(rgb_gt_list) if rgb_gt_list else None,
        np.array(mask_gt_list) if mask_gt_list else None,
    )


# ============================
# MAIN VISUALIZATION
# ============================


def set_isometric_camera(vis):
    ctr = vis.get_view_control()
    ctr.set_front([1.0, 1.0, 0.5])
    ctr.set_lookat([0.0, 0.0, 0.0])
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_zoom(0.8)


def visualize_rays(
    rays_o: np.ndarray,
    rays_d: np.ndarray,
    near: np.ndarray,
    far: np.ndarray,
    valid: np.ndarray,
    NUM_SAMPLES: int,
    SHOW_ONLY_INSIDE: bool,
    EXTRA_LENGTH: int,
):
    B = len(rays_o)
    N = min(NUM_SAMPLES, B)
    indices = np.random.choice(B, N, replace=False)

    rays_o = rays_o[indices]
    rays_d = rays_d[indices]
    near = near[indices]
    far = far[indices]
    valid = valid[indices]

    # ------------------------
    # Build ray geometry
    # ------------------------
    points = []
    lines = []
    colors = []
    idx = 0

    for o, d, n, f, v in zip(rays_o, rays_d, near, far, valid):

        if not v or f <= n:
            continue

        p_near = o + n * d
        p_far = o + f * d

        if SHOW_ONLY_INSIDE:
            # Inside segment only (green)
            points.extend([p_near, p_far])
            lines.append([idx, idx + 1])
            colors.append([0.0, 1.0, 0.0])
            idx += 2

        else:
            # Before near (red)
            if n > 0:
                points.extend([o, p_near])
                lines.append([idx, idx + 1])
                colors.append([1.0, 0.0, 0.0])
                idx += 2

            # Inside (green)
            points.extend([p_near, p_far])
            lines.append([idx, idx + 1])
            colors.append([0.0, 1.0, 0.0])
            idx += 2

            # After far (red)
            p_far_ext = o + (f + EXTRA_LENGTH) * d
            points.extend([p_far, p_far_ext])
            lines.append([idx, idx + 1])
            colors.append([1.0, 0.0, 0.0])
            idx += 2

    ray_lines = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(points)),
        lines=o3d.utility.Vector2iVector(np.asarray(lines)),
    )
    ray_lines.colors = o3d.utility.Vector3dVector(np.asarray(colors))

    # ------------------------
    # Red reference cube [-1, 1]^3
    # ------------------------
    cube = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(-1, -1, -1),
            max_bound=(1, 1, 1),
        )
    )
    cube.paint_uniform_color([1.0, 0.0, 0.0])

    # Optional coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)

    # ------------------------
    # Visualization
    # ------------------------
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Ray Visualization",
        width=1200,
        height=900,
    )

    vis.add_geometry(ray_lines)
    vis.add_geometry(cube)
    vis.add_geometry(frame)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.0, 0.0, 0.0])
    opt.line_width = 1.0
    opt.light_on = False

    set_isometric_camera(vis)

    vis.run()
    vis.destroy_window()


def arg_parse():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Configuration Loader")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Name of the configuration file (without .yml extension)",
    )
    parser.add_argument(
        "--skip_rays",
        action="store_true",
        help="skips ray computation",
    )
    parser.add_argument(
        "--display_rays",
        action="store_true",
        help="launchs ray renderer",
    )
    parser.add_argument(
        "--max_results",
        type=int,
        default=100,
        help="Number of rays to display",
    )
    parser.add_argument(
        "--only_inside",
        action="store_true",
        help="Show only rays inside the region of interest",
    )

    args = parser.parse_args()
    return args


def main():

    args = arg_parse()
    cfg = load_config(args.config_path)

    do_computations = not args.skip_rays
    N_eval_imgs = 15
    eval_rays = True

    if do_computations:
        downsample = cfg.data.downsample_factor
        transform_json = Path(cfg.files.transforms_json)
        mask_dir = Path(cfg.paths.segmentation_mask)
        use_semantic_model = cfg.model.use_semantics

        images, masks, poses, intrinsics = load_transforms(
            transform_json, mask_dir, downsample, use_semantics=use_semantic_model
        )
        H = intrinsics["H"]
        W = intrinsics["W"]
        K = intrinsics["K"]
        print(f"Image shape: {images.shape}")
        print(f"Mask shape: {masks.shape}")
        print(f"Intrinsic Matrix:\n{K}")

        rays_o_list, rays_d_list, rgb_gt_list, mask_gt_list = compute_rays(
            poses, images, masks, H, W, K
        )

        print("Compute Cube intersection")
        near, far, valid = intersect_aabb(rays_o_list, rays_d_list)
        print("Valid rays:", valid.sum(), "/", len(valid))

        print("Select valid rays")
        rays_o = rays_o_list[valid]
        rays_d = rays_d_list[valid]
        masks = mask_gt_list[valid]
        rgbs = rgb_gt_list[valid]
        near = np.maximum(near[valid], 0.0)
        far = far[valid]

        print(f"valid rays_o: {rays_o.shape}")
        print(f"valid rays_d: {rays_d.shape}")
        print(f"valid masks: {masks.shape}")
        print(f"Near min: {near.min()}, Near max: {near.max()}")
        print(f"Far min: {far.min()}, Far max: {far.max()}")

        ray_path = Path(cfg.paths.root) / "rays.npz"
        np.savez_compressed(
            ray_path,
            rays_o=rays_o.astype(np.float32),
            rays_d=rays_d.astype(np.float32),
            rgb=rgbs.astype(np.float32),
            masks=masks.astype(np.float32) if masks is not None else None,
            H=H,
            W=W,
            K=K.astype(np.float32),
            valid=valid,
            near=near,
            far=far,
        )
        print(f"Saved dataset to: {ray_path}")

    else:
        print("Load Rays")
        ray_path = Path(cfg.paths.root) / "rays.npz"

        data = np.load(ray_path, allow_pickle=True)
        rays_o = data["rays_o"]  # (B, 3)
        rays_d = data["rays_d"]  # (B, 3)
        masks = data["masks"]  # (B, 1)
        near = data["near"]  # (B,)
        far = data["far"]  # (B,)
        valid = data.get("valid", np.ones(len(rays_o), dtype=bool))

        print(f"valid rays_o: {rays_o.shape}")
        print(f"valid rays_d: {rays_d.shape}")
        print(f"valid masks: {masks.shape}")
        print(f"Near min: {near.min()}, Near max: {near.max()}")
        print(f"Far min: {far.min()}, Far max: {far.max()}")

    if eval_rays:
        print("Compute eval rays")
        downsample = cfg.data.downsample_factor
        transform_json = Path(cfg.files.transforms_json)
        mask_dir = Path(cfg.paths.segmentation_mask)
        use_semantic_model = cfg.model.use_semantics

        images, masks, poses, intrinsics = load_transforms(
            transform_json, mask_dir, downsample, use_semantics=use_semantic_model
        )
        H = intrinsics["H"]
        W = intrinsics["W"]
        K = intrinsics["K"]
        print(f"Image shape: {images.shape}")
        print(f"Mask shape: {masks.shape}")
        print(f"Intrinsic Matrix:\n{K}")

        rays_o, rays_d, near, far, valid, rgbs, masks = compute_eval_rays(
            poses,
            images,
            masks,
            H,
            W,
            K,
            N_eval_imgs,
        )

        print(f"rays_o: {rays_o.shape}")
        print(f"rays_d: {rays_d.shape}")
        print(f"masks: {masks.shape}")
        print(f"Near min: {near.min()}, Near max: {near.max()}")
        print(f"Far min: {far.min()}, Far max: {far.max()}")

        eval_ray_path = Path(cfg.paths.root) / "rays_eval.npz"
        np.savez_compressed(
            eval_ray_path,
            rays_o=rays_o.astype(np.float32),
            rays_d=rays_d.astype(np.float32),
            rgb=rgbs.astype(np.float32) if rgbs is not None else None,
            masks=masks.astype(np.float32) if masks is not None else None,
            H=H,
            W=W,
            K=K.astype(np.float32),
            valid=valid,
            near=near,
            far=far,
        )
        print(f"Saved dataset to: {eval_ray_path}")

    if args.display_rays:
        NUM_SAMPLES = args.max_results
        SHOW_ONLY_INSIDE = args.only_inside
        EXTRA_LENGTH = 1

        if eval_rays:
            print("Visualizing eval rays")
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            near   = near.reshape(-1)
            far    = far.reshape(-1)
            valid  = valid.reshape(-1)
            
        visualize_rays(
            rays_o,
            rays_d,
            near,
            far,
            valid,
            NUM_SAMPLES,
            SHOW_ONLY_INSIDE,
            EXTRA_LENGTH,
        )


if __name__ == "__main__":
    main()
