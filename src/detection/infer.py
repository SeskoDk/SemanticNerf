from ultralytics import YOLO
import numpy as np

import cv2
import random
from pathlib import Path

from src.utils import load_config, arg_parse

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_image_files(
    data_dir: Path,
    extensions=IMAGE_EXTENSIONS,
) -> list[Path]:
    exts = {ext.lower() for ext in extensions}

    return [
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in exts
    ]


if __name__ == "__main__":

    args = arg_parse()
    cfg = load_config(args.config_path)
    files = get_image_files(Path(cfg.paths.image_dir))

    # model_chkpt = "tools/pretrained_models/yolo/best.pt"
    model_chkpt = "tools/pretrained_models/yolo/new_model/best.pt"
    model = YOLO(model_chkpt)

    r_idx = random.randint(0, len(files) - 1)
    r_idx = 77
    fname = files[r_idx]

    fname = str("axis_renders\z_rgb.png")

    results = model([fname])

    # Define colors for different elements
    BOX_COLOR = (0, 255, 0)  # Green for bounding boxes
    MASK_COLOR = (0, 0, 255)  # Red for masks
    KEYPOINT_COLOR = (255, 0, 0)  # Blue for keypoints
    OBB_COLOR = (255, 255, 0)  # Cyan for oriented boxes

    for idx, result in enumerate(results):
        img = result.orig_img.copy()  # get the original image

        # Draw bounding boxes
        if hasattr(result, "boxes") and result.boxes is not None:
            for box in result.boxes.xyxy:  # assuming xyxy format
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 2)

        # Draw masks
        if hasattr(result, "masks") and result.masks is not None:
            for mask in result.masks.data:  # mask data as binary 2D array
                colored_mask = np.zeros_like(img)
                colored_mask[mask.astype(bool)] = MASK_COLOR
                img = cv2.addWeighted(img, 1.0, colored_mask, 0.5, 0)

        # Draw keypoints
        if hasattr(result, "keypoints") and result.keypoints is not None:
            for kp_set in result.keypoints.data:
                for kp in kp_set:
                    x, y, conf = kp
                    if conf > 0.5:  # confidence threshold
                        cv2.circle(img, (int(x), int(y)), 3, KEYPOINT_COLOR, -1)

        # Draw oriented bounding boxes (OBB)
        if hasattr(result, "obb") and result.obb is not None:
            for obb_box in result.obb.data:  # assuming obb_box is array of 4 corners
                pts = np.array(obb_box, dtype=np.int32)
                cv2.polylines(img, [pts], isClosed=True, color=OBB_COLOR, thickness=2)

        # Optionally, overlay probabilities/class labels
        if hasattr(result, "probs") and result.probs is not None:
            for box, prob in zip(result.boxes.xyxy, result.probs):
                x1, y1, _, _ = map(int, box)
                label = f"{prob:.2f}"
                cv2.putText(
                    img,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    BOX_COLOR,
                    1,
                )

        # Show using OpenCV
        cv2.imshow("Result", img)
        cv2.waitKey(0)  # small delay to render

        # Save to disk
        cv2.imwrite(f"result_{idx}.jpg", img)

    cv2.destroyAllWindows()
