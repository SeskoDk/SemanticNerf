from pathlib import Path
from omegaconf import OmegaConf, DictConfig


BASE_DIR = Path(__file__).resolve().parents[2]
# CFG_DIR = BASE_DIR / "cfg"
CFG_DIR = BASE_DIR


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

    args = parser.parse_args()
    return args


def load_config(cfg_file: str) -> DictConfig:
    """Load a configuration file using OmegaConf.

    Args:
        cfg_file (str): Name of the configuration file (without .yml extension).
    Returns:
        omegaconf.DictConfig: Loaded configuration.
    """
    config_path = CFG_DIR / cfg_file
    return OmegaConf.load(config_path)


def prepare_output_dirs(cfg: DictConfig) -> DictConfig:
    """
    Resolve all directory paths defined in cfg.paths and create them.

    - resolves OmegaConf interpolations
    - creates only directories (not files)
    - returns updated config
    """

    # 1. Interpolationen auflösen (sehr wichtig!)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)

    if "paths" not in cfg:
        return cfg

    for key, value in cfg.paths.items():
        if not isinstance(value, str):
            continue

        path = Path(value).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        cfg.paths[key] = str(path)

    return cfg