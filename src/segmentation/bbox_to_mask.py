import json
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from ultralytics import SAM

from src.utils import load_config, arg_parse


def load_bbx_json(path):
    with open(path, "r") as f:
        return json.load(f)


def scale_bboxes(bboxes, scale):
    x_min, y_min, x_max, y_max = bboxes
    x_min = int(x_min * scale)
    y_min = int(y_min * scale)
    x_max = int(x_max * scale)
    y_max = int(y_max * scale)
    return [x_min, y_min, x_max, y_max]


def scale_image(image_path, scale):
    image = cv2.imread(image_path)
    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


def load_sam_model(model_type: str):
    model_path = f"tools/pretrained_models/sam/{model_type}"
    model = SAM(model_path)
    print(f"Loaded SAM model from {model_path}")
    return model

def get_boxes_from_entries(entries, scale):
    bboxes = []
    for e in entries:
        bbox = e["bbox"]
        s_bboxes = scale_bboxes(bbox, scale)
        bboxes.append(s_bboxes)
    return bboxes

def main():

    args = arg_parse()
    cfg = load_config(args.config_path)

    image_dir = Path(cfg.paths.image_dir)
    output_dir = Path(cfg.paths.segmentation_mask)
    output_dir.mkdir(parents=True, exist_ok=True)

    bbox_json_data = load_bbx_json(cfg.files.bbx_results)
    print(f"Found {len(bbox_json_data)} images.")

    model = load_sam_model(model_type=cfg.model.sam_model)

    SCALE = 1.0 / cfg.data.downsample_factor
    IMGSZ = 1036

    for image_name, entries in tqdm(
            bbox_json_data.items(),
            desc="Create Segmentation Masks",
            total=len(bbox_json_data),
        ):

        image_path = image_dir / image_name
        image = scale_image(str(image_path), SCALE)
        bboxes = get_boxes_from_entries(entries, SCALE)

        results = model(image, bboxes=bboxes, verbose=False)

        m = results[0].masks.data.cpu().numpy()
        cm = np.any(m, axis=0).astype(np.uint8)

        # save mask
        cv2.imwrite(output_dir / f"{Path(image_name).stem}_mask.png", cm * 255)

    print(f"Segmentation masks saved at {output_dir}.")

if __name__ == "__main__":
    main()
