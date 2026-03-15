import os
import numpy as np
import open3d as o3d


# ============================
# USER CONFIG
# ============================
SHOW_ONLY_INSIDE = False 
NUM_SAMPLES = 50
EXTRA_LENGTH = 1
RAY_PATH = r"results/wheat_155160/rays.npz"


# ============================
# CAMERA SETUP
# ============================
def set_isometric_camera(vis):
    ctr = vis.get_view_control()
    ctr.set_front([1.0, 1.0, 0.5])
    ctr.set_lookat([0.0, 0.0, 0.0])
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_zoom(0.8)


# ============================
# MAIN VISUALIZATION
# ============================
def visualize_rays(ray_path: str):
    if not os.path.isfile(ray_path):
        raise FileNotFoundError(ray_path)

    # ------------------------
    # Load ray data
    # ------------------------
    data = np.load(ray_path, allow_pickle=True)

    rays_o = data["rays_o"]    # (B, 3)
    rays_d = data["rays_d"]    # (B, 3)
    near = data["near"]        # (B,)
    far = data["far"]          # (B,)
    valid = data.get("valid", np.ones(len(rays_o), dtype=bool))

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


# ============================
# ENTRY POINT
# ============================
if __name__ == "__main__":
    visualize_rays(RAY_PATH)
