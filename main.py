import random
import numpy as np
import torch

from dataset import Dataset
from train import train_encoder
from fs_eval import fewshot_evaluate
from config import get_config


def _load_or_train_encoder(encoder, dataset_obj, cfg, device, encoder_ckpt, skip_train):
    if encoder_ckpt:
        print(f"\n--- Loading Encoder ---")
        state = torch.load(encoder_ckpt, map_location=device)
        encoder.load_state_dict(state)
        print(f"[Encoder] loaded from {encoder_ckpt}")
        return encoder
    if skip_train:
        # Auto-find latest encoder checkpoint
        import os
        savepoint_dir = "./savepoint"
        pattern = f"encoder.{cfg.dataset}.*"
        if os.path.exists(savepoint_dir):
            import glob
            candidates = glob.glob(os.path.join(savepoint_dir, f"encoder.{cfg.dataset}.*.pth"))
            if candidates:
                latest = max(candidates, key=os.path.getmtime)
                print(f"\n--- Auto-loading Encoder ---")
                state = torch.load(latest, map_location=device)
                encoder.load_state_dict(state)
                print(f"[Encoder] loaded from {latest}")
                return encoder
        print(f"\n--- Skip Encoder Training (using random init) ---")
        return encoder
    print(f"\n--- Training Encoder ---")
    return train_encoder(encoder, dataset_obj, cfg, device)


def _get_generator(trained_encoder, dataset_obj, cfg, device, diffusion_ckpt, skip_diffusion_train):
    gen_type = str(getattr(cfg, "diffusion_generator_type", "ddpm")).lower()
    if gen_type == "gaussian":
        from models.ddpm import GaussianGenerator
        scale = float(getattr(cfg, "diffusion_gaussian_scale", 0.1))
        print(f"\n--- Gaussian Generator (scale={scale}) ---")
        return GaussianGenerator(noise_scale=scale, relative=True)
    from train import train_generator, load_generator
    if diffusion_ckpt:
        print(f"\n--- Loading Generator ---")
        return load_generator(trained_encoder, cfg, device, diffusion_ckpt)
    if skip_diffusion_train:
        print(f"\n--- Skip Generator Training ---")
        return None
    print(f"\n--- Training Generator ---")
    return train_generator(trained_encoder, dataset_obj, cfg, device)


def run_experiment(cfg, seed, encoder_mode=None, encoder_ckpt=None, diffusion_mode=None, diffusion_ckpt=None):
    """单个 seed 的实验流程"""
    device = torch.device("cpu") if cfg.gpu < 0 else torch.device("cuda:0")

    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # 创建数据集
    dataset_obj = Dataset(cfg.dataset)
    node_fea_size = dataset_obj.train_graphs[0].x.shape[1]

    print(f"\n[Dataset] {cfg.dataset}: {len(dataset_obj.train_graphs)} graphs, {node_fea_size} features")

    # 创建编码器 (eval_pooling 用于评估时的池化)
    eval_pooling = cfg.model_eval_pooling
    if cfg.model_manifold_type == "euclidean":
        from models.encoder import GIN
        encoder = GIN(
            num_layers=cfg.model_gin_layer,
            num_mlp_layers=2,
            input_dim=node_fea_size,
            hidden_dim=cfg.model_gin_hid,
            learn_eps=cfg.model_learn_eps,
            graph_pooling_type=eval_pooling,
            device=device,
        ).to(device)
    else:
        from models.hgin_encoder import HyperGIN
        _act_map = {"tanh": torch.tanh, "relu": torch.relu, "silu": torch.nn.functional.silu}
        act_fn = _act_map.get(cfg.model_act, torch.tanh)
        encoder = HyperGIN(
            manifold_type=cfg.model_manifold_type,
            num_layers=cfg.model_gin_layer,
            num_mlp_layers=2,
            input_dim=node_fea_size,
            hidden_dim=cfg.model_gin_hid,
            curvature=cfg.model_curvature,
            dropout=cfg.model_dropout,
            training_curvature=cfg.model_training_curvature,
            use_att=cfg.model_use_att,
            use_ptransp=cfg.model_use_ptransp,
            learn_eps=cfg.model_learn_eps,
            graph_pooling_type=eval_pooling,
            small_init=cfg.model_small_init,
            clip_r=cfg.model_clip_r,
            debug_nan=cfg.debug_nan,
            debug_max_calls=cfg.debug_max_calls,
            act=act_fn,
            use_bn=cfg.model_use_bn,
            use_input_proj=cfg.model_use_input_proj,
            device=device,
        ).to(device)
    print(f"[Encoder] type={cfg.model_manifold_type}, pooling={encoder.graph_pooling_type}, dim={encoder.raw_dim}")

    # 训练编码器（可选）
    skip_train = encoder_mode != "train"
    trained_encoder = _load_or_train_encoder(
        encoder, dataset_obj, cfg, device, encoder_ckpt, skip_train
    )

    # 训练生成器（可选）
    generator = None
    if cfg.diffusion_enabled and diffusion_mode != "off":
        skip_diffusion_train = diffusion_mode != "train"
        generator = _get_generator(
            trained_encoder, dataset_obj, cfg, device, diffusion_ckpt, skip_diffusion_train
        )

    # 评估
    print(f"\n--- Evaluation ---")
    results = fewshot_evaluate(trained_encoder, dataset_obj, device, cfg, generator=generator)

    # 输出结果
    if generator is not None:
        b5, b10 = results["baseline"][5]["mean"], results["baseline"][10]["mean"]
        a5, a10 = results["augmented"][5]["mean"], results["augmented"][10]["mean"]
        print(f"[Baseline]  5-shot: {b5:.4f}, 10-shot: {b10:.4f}")
        print(f"[Augmented] 5-shot: {a5:.4f}, 10-shot: {a10:.4f}")
        return {"5_shot": b5, "10_shot": b10, "5_shot_aug": a5, "10_shot_aug": a10}
    else:
        acc5, acc10 = results[5]["mean"], results[10]["mean"]
        print(f"5-shot: {acc5:.4f}, 10-shot: {acc10:.4f}")
        return {"5_shot": acc5, "10_shot": acc10}


def main():
    cfg = get_config()

    seeds = [cfg.seed] if isinstance(cfg.seed, int) else list(cfg.seed)

    print(f"{'=' * 50}")
    print(f"Dataset: {cfg.dataset} | Diffusion: {cfg.diffusion_enabled} | Seeds: {seeds}")
    print(f"{'=' * 50}")
    for k, v in sorted(vars(cfg).items()):
        if k == "runtime":
            continue
        print(f"  {k}: {v}")
    print(f"  runtime: {vars(cfg.runtime)}")
    print(f"{'=' * 50}")

    all_results = {}
    for seed in seeds:
        print(f"\n>>> SEED {seed}")
        all_results[seed] = run_experiment(
            cfg,
            seed,
            encoder_mode=cfg.runtime.encoder_mode,
            encoder_ckpt=cfg.runtime.encoder_ckpt,
            diffusion_mode=cfg.runtime.diffusion_mode,
            diffusion_ckpt=cfg.runtime.diffusion_ckpt,
        )

    if len(seeds) > 1:
        acc5 = np.mean([r["5_shot"] for r in all_results.values()])
        acc10 = np.mean([r["10_shot"] for r in all_results.values()])
        print(f"\n{'=' * 50}")
        print(f"FINAL: 5-shot={acc5:.4f}, 10-shot={acc10:.4f}")
        print(f"{'=' * 50}")

    return all_results


if __name__ == "__main__":
    main()
