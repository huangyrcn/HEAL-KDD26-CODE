import argparse
import types
import yaml
from pathlib import Path


def get_config(
    dataset=None,
    seed=None,
    cfg_overrides=None,
    encoder_mode=None,
    encoder_ckpt=None,
    diffusion_mode=None,
    diffusion_ckpt=None,
    argv=None,
):
    if dataset is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, required=True)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--cfg", nargs="*", default=[])
        parser.add_argument("--encoder", nargs="?", const="train", default=None)
        parser.add_argument("--diffusion", nargs="?", const="train", default=None)
        args = parser.parse_args(argv)
        dataset = args.dataset
        seed = args.seed if seed is None else seed
        cfg_overrides = args.cfg if cfg_overrides is None else cfg_overrides
        encoder_mode = args.encoder if encoder_mode is None else encoder_mode
        diffusion_mode = args.diffusion if diffusion_mode is None else diffusion_mode

    if dataset is None:
        raise ValueError("dataset is required")
    if cfg_overrides is None:
        cfg_overrides = []

    encoder_mode = encoder_mode or "train"
    diffusion_mode = diffusion_mode or "train"

    if encoder_mode not in ("train", "off") and encoder_ckpt is None:
        encoder_ckpt = encoder_mode
        encoder_mode = "load"
    if diffusion_mode not in ("train", "off") and diffusion_ckpt is None:
        diffusion_ckpt = diffusion_mode
        diffusion_mode = "load"

    cfg_path = Path(__file__).resolve().parent / "configs.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        yaml_cfg = yaml.safe_load(f) or {}

    merged = {}
    for k, v in (yaml_cfg.get("default", {}) or {}).items():
        merged[k] = v
    for k, v in (yaml_cfg.get(dataset, {}) or {}).items():
        merged[k] = v

    for item in cfg_overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override: {item}, expected key=value")
        key, val = item.split("=", 1)
        v = val.strip()
        if v.lower() in ("true", "false"):
            v = v.lower() == "true"
        else:
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    v = val
        merged[key] = v

    merged["dataset"] = dataset
    if seed is not None:
        merged["seed"] = seed
    merged["diffusion_enabled"] = diffusion_mode != "off"

    cfg = types.SimpleNamespace(**merged)
    cfg.runtime = types.SimpleNamespace(
        encoder_mode=encoder_mode,
        encoder_ckpt=encoder_ckpt,
        diffusion_mode=diffusion_mode,
        diffusion_ckpt=diffusion_ckpt,
    )
    return cfg
