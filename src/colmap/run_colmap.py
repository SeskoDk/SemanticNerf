import argparse
from pathlib import Path

from src.colmap.runner import ColmapRunner
from src.utils.config import load_config


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

    parser.add_argument(
        "--colmap_bin",
        type=str,
        default="tools/colmap/COLMAP.bat",
        help="Path to the COLMAP binary executable.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config_path)

    runner = ColmapRunner(colmap_bin=Path(args.colmap_bin))

    txt_path = runner.run(
        image_dir=Path(config.paths.data),
        output_dir=Path(config.paths.colmap),
    )

    print(f"\nCOLMAP TXT-Export fertig:\n{txt_path}")


if __name__ == "__main__":
    main()
