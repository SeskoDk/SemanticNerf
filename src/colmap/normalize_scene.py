import argparse
import numpy as np
import open3d
import json
from pathlib import Path
import pycolmap

from src.utils.config import load_config


class Model:
    def __init__(self):
        self.recon = None
        self.__vis = None
        self.scene_center = None
        self.scene_scale = None
        self.scene_rotation = np.eye(3)

        self.point_bbox_min = None
        self.point_bbox_max = None

    def read_model(self, path):
        self.recon = pycolmap.Reconstruction(path)

    def compute_normalization(self, min_track_len=3):
        xyz = []

        for point in self.recon.points3D.values():
            if point.track.length() < min_track_len:
                continue
            xyz.append(point.xyz)

        xyz = np.asarray(xyz)

        min_xyz = xyz.min(axis=0)
        max_xyz = xyz.max(axis=0)
        # min_xyz = np.percentile(xyz, 1, axis=0)
        # max_xyz = np.percentile(xyz, 99, axis=0)

        center = (min_xyz + max_xyz) / 2.0
        extent = max_xyz - min_xyz
        scale = 2.0 / np.max(extent)

        self.scene_center = center
        self.scene_scale = scale

        print("Scene center:", center)
        print("Scene scale:", scale)

    def _get_normalized_points(self, min_track_len=3):
        pts = []

        for p in self.recon.points3D.values():
            if p.track.length() < min_track_len:
                continue

            x = p.xyz
            x = x - self.scene_center
            x = self.scene_rotation @ x
            x = x * self.scene_scale

            pts.append(x)

        if not pts:
            raise RuntimeError("No valid 3D points for bounding box")

        return np.asarray(pts)

    def compute_point_bounding_box(
        self,
        min_track_len=3,
        lower_percentile=1.0,
        upper_percentile=99.0,
        clamp_xy_to_unit: bool = True,
    ):
        pts = self._get_normalized_points(min_track_len)

        bbox_min = np.percentile(pts, lower_percentile, axis=0)
        bbox_max = np.percentile(pts, upper_percentile, axis=0)

        if clamp_xy_to_unit:
            bbox_min[0] = -1.0
            bbox_min[1] = -1.0
            bbox_max[0] =  1.0
            bbox_max[1] =  1.0

        self.point_bbox_min = bbox_min
        self.point_bbox_max = bbox_max

        print("Point BBox min:", bbox_min)
        print("Point BBox max:", bbox_max)

        return bbox_min, bbox_max

    def add_point_bounding_box(self, color=(0.0, 1.0, 0.0)):
        if self.point_bbox_min is None or self.point_bbox_max is None:
            raise RuntimeError("Point bounding box not computed")

        mn = self.point_bbox_min
        mx = self.point_bbox_max

        corners = np.array(
            [
                [mn[0], mn[1], mn[2]],
                [mx[0], mn[1], mn[2]],
                [mx[0], mx[1], mn[2]],
                [mn[0], mx[1], mn[2]],
                [mn[0], mn[1], mx[2]],
                [mx[0], mn[1], mx[2]],
                [mx[0], mx[1], mx[2]],
                [mn[0], mx[1], mx[2]],
            ]
        )

        lines = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]

        colors = [color for _ in lines]

        bbox = open3d.geometry.LineSet(
            points=open3d.utility.Vector3dVector(corners),
            lines=open3d.utility.Vector2iVector(lines),
        )
        bbox.colors = open3d.utility.Vector3dVector(colors)

        self.__vis.add_geometry(bbox)

    def add_points(self, min_track_len=3, remove_statistical_outlier=True):
        pcd = open3d.geometry.PointCloud()

        xyz = []
        rgb = []
        for point in self.recon.points3D.values():
            if point.track.length() < min_track_len:
                continue
            # xyz.append(point.xyz)
            p = (point.xyz - self.scene_center) * self.scene_scale
            p = self.scene_rotation @ p

            xyz.append(p)
            rgb.append(point.color / 255)

        pcd.points = open3d.utility.Vector3dVector(xyz)
        pcd.colors = open3d.utility.Vector3dVector(rgb)

        # remove obvious outliers
        if remove_statistical_outlier:
            [pcd, _] = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # open3d.visualization.draw_geometries([pcd])
        self.__vis.add_geometry(pcd)
        self.__vis.poll_events()
        self.__vis.update_renderer()

    def add_cameras(self, scale=1):
        frustums = []
        for img in self.recon.images.values():
            # extrinsics
            world_from_cam = img.cam_from_world().inverse()
            R = world_from_cam.rotation.matrix()
            # t = world_from_cam.translation
            R = self.scene_rotation @ world_from_cam.rotation.matrix()
            t = (world_from_cam.translation - self.scene_center) * self.scene_scale
            t = self.scene_rotation @ t

            # intrinsics
            cam = img.camera
            if cam.model in (
                pycolmap.CameraModelId.SIMPLE_PINHOLE,
                pycolmap.CameraModelId.SIMPLE_RADIAL,
                pycolmap.CameraModelId.RADIAL,
            ):
                fx = fy = cam.params[0]
                cx = cam.params[1]
                cy = cam.params[2]
            elif cam.model in (
                pycolmap.CameraModelId.PINHOLE,
                pycolmap.CameraModelId.OPENCV,
                pycolmap.CameraModelId.OPENCV_FISHEYE,
                pycolmap.CameraModelId.FULL_OPENCV,
            ):
                fx = cam.params[0]
                fy = cam.params[1]
                cx = cam.params[2]
                cy = cam.params[3]
            else:
                raise Exception("Camera model not supported")

            # intrinsics
            K = np.identity(3)
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy

            # create axis, plane and pyramid geometries that will be drawn
            cam_model = draw_camera(K, R, t, cam.width, cam.height, scale)
            frustums.extend(cam_model)

        # add geometries to visualizer
        for i in frustums:
            self.__vis.add_geometry(i)

    def create_window(self):
        self.__vis = open3d.visualization.Visualizer()
        # self.__vis.create_window()
        self.__vis.create_window(
            window_name="Reconstruction", width=1200, height=1000, left=100, top=100
        )

        render_opt = self.__vis.get_render_option()
        render_opt.background_color = np.array([0.0, 0.0, 0.0])  # schwarz

    def show(self):
        self.__vis.poll_events()
        self.__vis.update_renderer()
        self.__vis.run()
        self.__vis.destroy_window()

    def compute_average_view_direction(self):
        dirs = []

        for img in self.recon.images.values():
            world_from_cam = img.cam_from_world().inverse()
            R = world_from_cam.rotation.matrix()
            # d = R @ np.array([0.0, 0.0, -1.0])  # camera forward
            d = R @ np.array([0.0, 0.0, -1.0])  # camera backward
            dirs.append(d)

        mean_dir = np.mean(dirs, axis=0)
        mean_dir /= np.linalg.norm(mean_dir)

        print("Average view direction:", mean_dir)
        return mean_dir

    def compute_scene_rotation(self):
        avg_dir = self.compute_average_view_direction()
        target = np.array([0.0, 0.0, 1.0])
        self.scene_rotation = rotation_from_vectors(avg_dir, target)

    def set_z_up_view(self):
        ctr = self.__vis.get_view_control()

        # Blickrichtung (von schräg oben)
        # ctr.set_front([0.0, -1.0, -0.5])
        ctr.set_front([1, 0, 0.5])
        # Z ist oben
        ctr.set_up([0.0, 0.0, 1.0])

        # Fokuspunkt (Ursprung deiner normalisierten Szene)
        ctr.set_lookat([0.0, 0.0, 0.0])

        # Zoom etwas raus
        ctr.set_zoom(0.8)


    #######

    def write_transforms_json(self, out_path, image_dir="images"):
        frames = []

        # Assume shared intrinsics (COLMAP-style)
        cam0 = next(iter(self.recon.cameras.values()))

        if cam0.model not in (
            pycolmap.CameraModelId.SIMPLE_PINHOLE,
            pycolmap.CameraModelId.SIMPLE_RADIAL,
            pycolmap.CameraModelId.RADIAL,
            pycolmap.CameraModelId.PINHOLE,
            pycolmap.CameraModelId.OPENCV,
            pycolmap.CameraModelId.OPENCV_FISHEYE,
            pycolmap.CameraModelId.FULL_OPENCV,
        ):
            raise RuntimeError("Unsupported camera model for transforms.json")

        # intrinsics
        if cam0.model in (
            pycolmap.CameraModelId.SIMPLE_PINHOLE,
            pycolmap.CameraModelId.SIMPLE_RADIAL,
            pycolmap.CameraModelId.RADIAL,
        ):
            fl_x = fl_y = cam0.params[0]
            cx, cy = cam0.params[1:3]
        else:
            fl_x, fl_y = cam0.params[0:2]
            cx, cy = cam0.params[2:4]

        for img in self.recon.images.values():
            world_from_cam = img.cam_from_world().inverse()
            R_wc = world_from_cam.rotation.matrix()
            t_wc = world_from_cam.translation

            # apply normalization
            R_wc = self.scene_rotation @ R_wc
            t_wc = self.scene_rotation @ ((t_wc - self.scene_center) * self.scene_scale)

            c2w = np.eye(4)
            c2w[:3, :3] = R_wc
            c2w[:3, 3] = t_wc

            frames.append(
                {
                    "file_path": str(Path(image_dir) / Path(img.name).name),
                    "transform_matrix": c2w.tolist(),
                }
            )

        transforms = {
            "camera_model": "PINHOLE",
            "fl_x": float(fl_x),
            "fl_y": float(fl_y),
            "cx": float(cx),
            "cy": float(cy),
            "w": int(cam0.width),
            "h": int(cam0.height),
            "point_bbox": {
                "min": self.point_bbox_min.tolist(),
                "max": self.point_bbox_max.tolist(),
            },
            "frames": frames,
        }

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w") as f:
            json.dump(transforms, f, indent=2)

        print(f"Wrote transforms.json to {out_path}")

    def compute_scale_to_unit_cube(
        self,
        min_track_len: int = 3,
        manual_scale: float = 0.85,
    ):
        if not (0.0 < manual_scale <= 1.0):
            raise ValueError("manual_scale must be in (0, 1]")

        xyz = []

        for p in self.recon.points3D.values():
            if p.track.length() < min_track_len:
                continue
            xyz.append(p.xyz - self.scene_center)

        xyz = np.asarray(xyz)

        if len(xyz) == 0:
            raise RuntimeError("No valid 3D points for scaling")

        # robust bounding box
        min_xyz = np.percentile(xyz, 1, axis=0)
        max_xyz = np.percentile(xyz, 99, axis=0)

        extent = max_xyz - min_xyz
        auto_scale = 2.0 / np.max(extent)

        self.scene_scale = auto_scale * manual_scale

        print("Auto scale:", auto_scale)
        print("Manual scale:", manual_scale)
        print("Final scale:", self.scene_scale)

    def compute_normalization_from_cameras(self):
        print("Computing scene center from camera rays (Instant-NGP style)")

        cams = []

        for img in self.recon.images.values():
            world_from_cam = img.cam_from_world().inverse()
            cam_center = world_from_cam.translation
            forward = world_from_cam.rotation.matrix() @ np.array([0.0, 0.0, -1.0])
            cams.append((cam_center, forward))

        totp = np.zeros(3)
        totw = 0.0

        for i in range(len(cams)):
            oa, da = cams[i]
            for j in range(i + 1, len(cams)):
                ob, db = cams[j]
                p, w = closest_point_2_lines(oa, da, ob, db)
                if w > 1e-6:
                    totp += p * w
                    totw += w

        if totw == 0:
            raise RuntimeError("Failed to compute scene center from camera rays")

        center = totp / totw

        # scale like Instant-NGP (average camera distance)
        avg_dist = np.mean([np.linalg.norm(o - center) for o, _ in cams])
        scale = 4.0 / avg_dist

        self.scene_center = center
        self.scene_scale = scale

        print("Scene center (camera rays):", center)
        print("Scene scale:", scale)



    def compute_normalization_from_points(self, min_track_len=3):
        xyz = []

        for p in self.recon.points3D.values():
            if p.track.length() < min_track_len:
                continue
            xyz.append(p.xyz)

        xyz = np.asarray(xyz)

        min_xyz = np.percentile(xyz, 1, axis=0)
        max_xyz = np.percentile(xyz, 99, axis=0)

        center = (min_xyz + max_xyz) / 2.0
        extent = max_xyz - min_xyz
        scale = 2.0 / np.max(extent)

        self.scene_center = center
        self.scene_scale = scale

    # def compute_scale_to_unit_cube(
    #     self,
    #     min_track_len=3,
    #     manual_scale: float = 1.0,
    # ):
    #     xyz = []

    #     for p in self.recon.points3D.values():
    #         if p.track.length() < min_track_len:
    #             continue
    #         xyz.append(p.xyz - self.scene_center)

    #     xyz = np.asarray(xyz)

    #     if len(xyz) == 0:
    #         raise RuntimeError("No valid 3D points for scaling")

    #     min_xyz = np.percentile(xyz, 1, axis=0)
    #     max_xyz = np.percentile(xyz, 99, axis=0)

    #     extent = max_xyz - min_xyz
    #     scale = 2.0 / np.max(extent)

    #     self.scene_scale = scale * 0.8

    #     print("Scene scale (unit cube):", scale)


def draw_camera(K, R, t, w, h, scale=1, color=[0, 0.7, 0]):
    """Create axis, plane and pyramed geometries in Open3D format.
    :param K: calibration matrix (camera intrinsics)
    :param R: rotation matrix
    :param t: translation
    :param w: image width
    :param h: image height
    :param scale: camera model scale
    :param color: color of the image plane and pyramid lines
    :return: camera model geometries (axis, plane and pyramid)
    """

    # intrinsics
    K = K.copy() / scale
    Kinv = np.linalg.inv(K)

    # 4x4 transformation
    T = np.column_stack((R, t))
    T = np.vstack((T, (0, 0, 0, 1)))

    # axis
    axis = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5 * scale)
    axis.transform(T)

    # points in pixel
    points_pixel = [
        [0, 0, 0],
        [0, 0, 1],
        [w, 0, 1],
        [0, h, 1],
        [w, h, 1],
    ]

    # pixel to camera coordinate system
    points = [Kinv @ p for p in points_pixel]

    # image plane
    width = abs(points[1][0]) + abs(points[3][0])
    height = abs(points[1][1]) + abs(points[3][1])
    plane = open3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6)
    plane.paint_uniform_color(color)
    plane.translate([points[1][0], points[1][1], scale])
    plane.transform(T)

    # pyramid
    points_in_world = [(R @ p + t) for p in points]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
    ]
    colors = [color for i in range(len(lines))]
    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points_in_world),
        lines=open3d.utility.Vector2iVector(lines),
    )
    line_set.colors = open3d.utility.Vector3dVector(colors)

    # return as list in Open3D format
    return [axis, plane, line_set]


def draw_unit_cube(color=[1, 0, 0]):
    # 8 Ecken des Würfels [-1,1]^3
    points = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float64,
    )

    # 12 Kanten
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],  # bottom
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],  # top
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],  # vertical
    ]

    colors = [color for _ in lines]

    cube = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points),
        lines=open3d.utility.Vector2iVector(lines),
    )
    cube.colors = open3d.utility.Vector3dVector(colors)

    return cube


def draw_origin(size=0.3):
    return open3d.geometry.TriangleMesh.create_coordinate_frame(
        size=size, origin=[0.0, 0.0, 0.0]
    )


def rotation_from_vectors(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    v = np.cross(a, b)
    c = np.dot(a, b)

    if c < -0.999999:
        # 180° rotation (rare but possible)
        axis = np.array([1, 0, 0])
        if abs(a[0]) > 0.9:
            axis = np.array([0, 1, 0])
        v = np.cross(a, axis)
        v /= np.linalg.norm(v)
        return open3d.geometry.get_rotation_matrix_from_axis_angle(np.pi * v)

    s = np.linalg.norm(v)

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
    return R


def closest_point_2_lines(oa, da, ob, db):
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa + ta * da + ob + tb * db) * 0.5, denom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run COLMAP to generate sparse reconstruction from images."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="wheat_155160",
        help="Name of the configuration file (without .yml extension).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config_path)

    # input_model = "data/weizen_2025/155160/colmap_text"
    colmap_text_path = config.paths.colmap_text
    # read COLMAP model
    model = Model()
    model.read_model(colmap_text_path)

    print("num_cameras:", model.recon.num_cameras())
    print("num_images:", model.recon.num_images())
    print("num_points3D:", model.recon.num_points3D())

    # display using Open3D visualization tools
    model.create_window()
    # model.compute_normalization()

    #TODO: choose one of the two normalization methods
    # model.compute_normalization_from_cameras()
    model.compute_normalization_from_points()
    
    model.compute_scale_to_unit_cube()

    model.compute_scene_rotation()

    model.compute_point_bounding_box()
    model.add_point_bounding_box()

    model.add_points()
    model.add_cameras(scale=0.05)

    cube = draw_unit_cube()
    origin = draw_origin(size=0.3)
    # origin.rotate(model.scene_rotation, center=(0, 0, 0))

    model._Model__vis.add_geometry(cube)
    model._Model__vis.add_geometry(origin)

    model.set_z_up_view()

    model.write_transforms_json(
        out_path=config.files.transforms_json, image_dir=config.paths.image_dir
    )

    model.show()


if __name__ == "__main__":
    main()
