from pathlib import Path
import shutil

from ..utils import run_command

class ColmapRunner:
    def __init__(self, colmap_bin: Path):
        if not colmap_bin.exists():
            raise FileNotFoundError(colmap_bin)
        self.colmap_bin = colmap_bin

    def run(self, image_dir: Path, output_dir: Path) -> Path:
        db = output_dir / "colmap.db"
        sparse = output_dir / "sparse"
        text = output_dir / "text"

        self._prepare(output_dir, db, sparse, text)
        self._feature_extraction(db, image_dir)
        self._matching(db)
        self._mapping(db, image_dir, sparse)
        self._bundle_adjustment(sparse)
        self._export_text(sparse, text)

        return text

    def _prepare(self, root: Path, db: Path, *dirs: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        if db.exists():
            db.unlink()
        for d in dirs:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

    def _feature_extraction(self, db: Path, images: Path) -> None:
        run_command([
            str(self.colmap_bin), "feature_extractor",
            "--database_path", str(db),
            "--image_path", str(images),
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", "OPENCV",
        ])

    def _matching(self, db: Path) -> None:
        run_command([
            str(self.colmap_bin), "exhaustive_matcher",
            "--database_path", str(db),
        ])

    def _mapping(self, db: Path, images: Path, sparse: Path) -> None:
        run_command([
            str(self.colmap_bin), "mapper",
            "--database_path", str(db),
            "--image_path", str(images),
            "--output_path", str(sparse),
        ])

    def _bundle_adjustment(self, sparse: Path) -> None:
        run_command([
            str(self.colmap_bin), "bundle_adjuster",
            "--input_path", str(sparse / "0"),
            "--output_path", str(sparse / "0"),
            "--BundleAdjustment.refine_principal_point", "1",
        ])

    def _export_text(self, sparse: Path, text: Path) -> None:
        run_command([
            str(self.colmap_bin), "model_converter",
            "--input_path", str(sparse / "0"),
            "--output_path", str(text),
            "--output_type", "TXT",
        ])
