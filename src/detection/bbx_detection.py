import json
from pathlib import Path
from typing import Iterable

from tqdm import tqdm
from ultralytics import YOLO

from src.utils import load_config, arg_parse


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_image_files(
    data_dir: Path,
    extensions: Iterable[str] = IMAGE_EXTENSIONS,
) -> list[Path]:
    exts = {ext.lower() for ext in extensions}

    return [
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in exts
    ]


def main():
    print("Loading segmentation configuration...")
    args = arg_parse()
    cfg = load_config(args.config_path)

    image_dir = Path(cfg.paths.image_dir)
    files = get_image_files(image_dir)

    print(f"Found {len(files)} images.")
    print("Load YOLO model...")

    model_chkpt = "tools/pretrained_models/yolo/best.pt"
    model = YOLO(model_chkpt)

    results_dict = {}

    print("Running YOLO inference...")
    for img_path in tqdm(files, desc="Processing images"):

        result = model(img_path, verbose=False)[0]

        image_name = img_path.name
        results_dict[image_name] = []

        if result.boxes is None or len(result.boxes) == 0:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            results_dict[image_name].append(
                {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0]),
                    "class_name": model.names[int(box.cls[0])],
                }
            )

    output_path = Path(cfg.files.bbx_results)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"Saved bounding boxes to {output_path}")


if __name__ == "__main__":
    main()