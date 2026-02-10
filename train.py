import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader


@torch.no_grad()
def _layer_diagnosis(encoder, train_loader, device, epoch, max_batches=2):
    """Per-layer diagnosis: norm before/after tanh, layer contribution to JK."""
    encoder.eval()
    manifold = encoder.manifold
    c = encoder.c

    # Collect one batch
    batch = next(iter(train_loader)).to(device)
    x = batch.x.to(device)
    edge_index = batch.edge_index.to(device)
    adj = encoder._to_adj(edge_index, x.size(0))
    batch_idx = batch.batch.to(device)

    # Forward through layers, capturing intermediate states
    h = encoder._to_manifold(x)
    input_tan = manifold.logmap0(h, c)
    input_norm = input_tan[..., 1:].norm(dim=-1).mean().item()

    layer_stats = []
    for i, layer in enumerate(encoder.layers):
        # Before this layer
        pre_tan = manifold.logmap0(h, c)
        pre_norm = pre_tan[..., 1:].norm(dim=-1).mean().item()

        h = layer(h, adj)

        # After this layer
        post_tan = manifold.logmap0(h, c)
        post_norm = post_tan[..., 1:].norm(dim=-1).mean().item()

        # Graph-level contribution (mean pool)
        from torch_geometric.nn import global_mean_pool
        h_graph = global_mean_pool(post_tan, batch_idx)
        graph_norm = h_graph.norm(dim=-1).mean().item()

        layer_stats.append({
            'pre': pre_norm, 'post': post_norm, 'graph': graph_norm
        })

    # Print compact summary
    parts = [f"  [LayerDiag] E={epoch:03d} | input_norm={input_norm:.3f}"]
    for i, s in enumerate(layer_stats):
        parts.append(
            f"L{i}({s['pre']:.2f}->{s['post']:.2f}, g={s['graph']:.2f})"
        )
    parts.append(f"| curv={c.item():.4f}")
    print(" | ".join(parts))
    encoder.train()


@torch.no_grad()
def _final_diagnosis(encoder, train_loader, device):
    """Post-training comprehensive diagnosis."""
    encoder.eval()
    manifold = encoder.manifold
    c = encoder.c

    all_tangent, all_labels = [], []
    per_layer_norms = [[] for _ in range(encoder.num_layers)]

    for i, batch in enumerate(train_loader):
        if i >= 5:
            break
        batch = batch.to(device)
        hidden_rep = encoder.forward(batch.x, batch.edge_index)
        batch_idx = batch.batch

        for li, h in enumerate(hidden_rep):
            h_tan = manifold.logmap0(h, c)
            from torch_geometric.nn import global_mean_pool
            h_g = global_mean_pool(h_tan, batch_idx)
            per_layer_norms[li].append(h_g.norm(dim=-1))

        _, h_t, _ = encoder.encode_graph(batch.x, batch.edge_index, batch.batch)
        all_tangent.append(h_t)
        all_labels.append(batch.y)

    all_tangent = torch.cat(all_tangent, dim=0)
    all_labels = torch.cat(all_labels, dim=0).view(-1)

    # Per-layer contribution
    print("  [FinalDiag] Per-layer graph-level norms (mean pool):")
    for li in range(encoder.num_layers):
        norms = torch.cat(per_layer_norms[li])
        print(f"    Layer {li}: mean={norms.mean():.4f}, std={norms.std():.4f}, "
              f"max={norms.max():.4f}, min={norms.min():.4f}")

    # JK tangent stats
    jk_norm = all_tangent.norm(dim=-1)
    print(f"  [FinalDiag] JK tangent: mean_norm={jk_norm.mean():.4f}, "
          f"std_norm={jk_norm.std():.4f}")

    # Effective rank
    sample_n = min(500, all_tangent.shape[0])
    idx = torch.randperm(all_tangent.shape[0])[:sample_n]
    _, s, _ = torch.svd(all_tangent[idx])
    s = s[s > 1e-10]
    p = s / s.sum()
    erank = torch.exp(-(p * p.log()).sum()).item()
    print(f"  [FinalDiag] Effective rank: {erank:.1f}/{all_tangent.shape[1]}")

    # Separation
    h_norm = F.normalize(all_tangent, dim=1)
    sim = h_norm @ h_norm.T
    same_mask = all_labels.unsqueeze(0) == all_labels.unsqueeze(1)
    diag = torch.eye(len(all_labels), dtype=torch.bool, device=device)
    same_mask = same_mask & ~diag
    diff_mask = ~same_mask & ~diag
    same_sim = sim[same_mask].mean().item() if same_mask.any() else 0.0
    diff_sim = sim[diff_mask].mean().item() if diff_mask.any() else 0.0
    print(f"  [FinalDiag] Cosine: same={same_sim:.4f}, diff={diff_sim:.4f}, "
          f"sep={same_sim - diff_sim:.4f}")

    # Curvature
    print(f"  [FinalDiag] Final curvature: {c.item():.6f}")
    encoder.train()


@torch.no_grad()
def _collapse_check(encoder, train_loader, device, epoch, max_batches=3):
    """Quick collapse diagnostic: effective rank + class separation."""
    encoder.eval()
    tangents, labels = [], []
    for i, batch in enumerate(train_loader):
        if i >= max_batches:
            break
        batch = batch.to(device)
        _, h_t, _ = encoder.encode_graph(batch.x, batch.edge_index, batch.batch)
        tangents.append(h_t)
        labels.append(batch.y)
    tangents = torch.cat(tangents, dim=0)
    labels = torch.cat(labels, dim=0).view(-1)

    # Effective rank via singular values
    sample_n = min(500, tangents.shape[0])
    idx = torch.randperm(tangents.shape[0])[:sample_n]
    X = tangents[idx]
    _, s, _ = torch.svd(X)
    s = s[s > 1e-10]
    p = s / s.sum()
    erank = torch.exp(-(p * p.log()).sum()).item()

    # Class separation (cosine)
    h_norm = F.normalize(tangents, dim=1)
    sim = h_norm @ h_norm.T
    same_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    diag = torch.eye(len(labels), dtype=torch.bool, device=device)
    same_mask = same_mask & ~diag
    diff_mask = ~same_mask & ~diag
    same_sim = sim[same_mask].mean().item() if same_mask.any() else 0.0
    diff_sim = sim[diff_mask].mean().item() if diff_mask.any() else 0.0

    # Per-dim std (collapse indicator)
    dim_std = tangents.std(dim=0)
    alive_dims = (dim_std > 0.01).sum().item()

    print(
        f"  [Collapse] E={epoch:03d} | erank={erank:.1f}/{tangents.shape[1]} "
        f"| alive_dims={alive_dims}/{tangents.shape[1]} "
        f"| same_cos={same_sim:.4f} diff_cos={diff_sim:.4f} "
        f"| sep={same_sim - diff_sim:.4f}"
    )
    encoder.train()


def train_encoder(encoder: nn.Module, dataset, cfg, device: torch.device):
    """训练 HypGCL 编码器"""
    # 数据准备
    train_loader = DataLoader(
        dataset.train_graphs, batch_size=cfg.model_batch_size, shuffle=True
    )
    print(
        f"[Data] batch_size={cfg.model_batch_size}, train_graphs={len(dataset.train_graphs)}"
    )

    # 构建 HypGCL 模型
    from models.hypgcl_graph import HypGCLGraph

    model = HypGCLGraph(
        encoder=encoder,
        proj_hidden_dim=cfg.model_proj_hidden_dim,
        proj_layers=cfg.model_proj_layers,
        temperature=cfg.model_temperature,
        hyper_max_weight=cfg.model_hyper_max_weight,
        hyper_start_epoch=cfg.model_hyper_start_epoch,
        hyper_end_epoch=cfg.model_hyper_end_epoch,
        feat_drop_prob=cfg.model_feat_drop_prob,
        edge_drop_prob=cfg.model_edge_drop_prob,
        feat_drop_prob2=cfg.model_feat_drop_prob2,
        edge_drop_prob2=cfg.model_edge_drop_prob2,
        train_pooling=cfg.model_train_pooling,
        vicreg_weight=cfg.model_vicreg_weight,
        vicreg_var_weight=cfg.model_vicreg_var_weight,
        vicreg_cov_weight=cfg.model_vicreg_cov_weight,
        vicreg_target_std=cfg.model_vicreg_target_std,
        vicreg_backbone=cfg.model_vicreg_backbone,
        noise_scale1=cfg.model_noise_scale1,
        noise_scale2=cfg.model_noise_scale2,
        aug_view1=cfg.aug_view1,
        aug_view2=cfg.aug_view2,
        aug_rate1=cfg.aug_rate1,
        aug_rate2=cfg.aug_rate2,
        debug_train=cfg.debug_train,
        debug_max_batches=cfg.debug_max_batches,
        use_supervised_loss=cfg.model_use_supervised_loss,
    ).to(device)
    print(f"[HypGCL] temp={cfg.model_temperature}, curvature={encoder.c}")

    # 优化器
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.optimizer_lr, weight_decay=cfg.optimizer_weight_decay
    )

    # 学习率调度器
    scheduler = None
    sched_type = getattr(cfg, "lr_scheduler_type", None)
    if sched_type == "warmup_cosine":
        T_0 = getattr(cfg, "lr_scheduler_T_0", 50)
        T_mult = getattr(cfg, "lr_scheduler_T_mult", 1)
        eta_min = getattr(cfg, "lr_scheduler_eta_min", 0.0001)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )
        print(
            f"[LR Scheduler] CosineAnnealingWarmRestarts: T_0={T_0}, T_mult={T_mult}, eta_min={eta_min}"
        )
    elif sched_type == "cosine":
        T_max = getattr(cfg, "lr_scheduler_T_max", cfg.train_epoch_num)
        eta_min = getattr(cfg, "lr_scheduler_eta_min", 0.0001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
        print(f"[LR Scheduler] CosineAnnealing: T_max={T_max}, eta_min={eta_min}")

    # 训练
    best_loss, cnt_wait, best_state = float("inf"), 0, None
    start = time.time()

    for epoch in range(cfg.train_epoch_num):
        model.train()
        model.set_epoch(epoch)
        epoch_loss, num_batches = 0.0, 0
        epoch_angle, epoch_dist, epoch_lambda = 0.0, 0.0, 0.0
        epoch_var, epoch_cov, epoch_vicreg = 0.0, 0.0, 0.0

        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            loss, info = model.compute_loss(batch_data)

            if not torch.isfinite(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += info.get("loss", loss.item())
            epoch_angle += info.get("loss_angle", 0.0)
            epoch_dist += info.get("loss_dist", 0.0)
            epoch_lambda += info.get("lambda_d", 0.0)
            epoch_var += info.get("loss_var", 0.0)
            epoch_cov += info.get("loss_cov", 0.0)
            epoch_vicreg += info.get("loss_vicreg", 0.0)
            num_batches += 1

        if num_batches == 0:
            print(f"[ERROR] Epoch {epoch}: All batches NaN/Inf!")
            break

        avg_loss = epoch_loss / num_batches
        avg_angle = epoch_angle / num_batches
        avg_dist = epoch_dist / num_batches
        avg_lambda = epoch_lambda / num_batches
        avg_var = epoch_var / num_batches
        avg_cov = epoch_cov / num_batches
        avg_vicreg = epoch_vicreg / num_batches

        # Step the LR scheduler
        if scheduler is not None:
            scheduler.step()

        if epoch % cfg.train_log_interval == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"(T) Epoch={epoch:03d} | Loss={avg_loss:.4f} "
                f"| Angle={avg_angle:.4f} | Dist={avg_dist:.4f} "
                f"| Var={avg_var:.4f} | Cov={avg_cov:.4f} | VICReg={avg_vicreg:.4f} "
                f"| Lambda_d={avg_lambda:.3f} | LR={cur_lr:.6f} | Time={time.time() - start:.1f}s"
            )

            # --- Collapse tracking: effective rank & separation every log_interval ---
            if epoch % (cfg.train_log_interval * 2) == 0 or epoch == 0:
                _collapse_check(model.encoder, train_loader, device, epoch)
                if model.encoder.manifold.name != "euclidean":
                    _layer_diagnosis(model.encoder, train_loader, device, epoch)

            if avg_loss < best_loss:
                best_loss, cnt_wait = avg_loss, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                cnt_wait += 1

        if cnt_wait > cfg.train_patience:
            print("Early Stopping!")
            break

    # 保存模型
    if best_state:
        model.load_state_dict(best_state)
        import datetime

        os.makedirs("./savepoint", exist_ok=True)
        path = f"./savepoint/encoder.{cfg.dataset}.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        torch.save(model.encoder.state_dict(), path)
        print(f"+++ Saved: {path}")

    # Post-training diagnosis
    print("\n--- Post-Training Diagnosis ---")
    _final_diagnosis(model.encoder, train_loader, device)

    return model.encoder


def _infer_dim_in_from_state(state) -> int:
    residual_mean = state.get("residual_mean", None)
    if isinstance(residual_mean, torch.Tensor) and residual_mean.ndim == 1:
        return int(residual_mean.numel())

    output_proj_w = state.get("backbone.output_proj.weight", None)
    if isinstance(output_proj_w, torch.Tensor) and output_proj_w.ndim == 2:
        return int(output_proj_w.shape[0])

    raise KeyError("Cannot infer dim_in from generator checkpoint state dict")


def _build_diffusion_model(cfg, device: torch.device, dim_in: int):
    from models.ddpm import ResidualDiffusion, DDPMSampler, DDIMSampler, SingleStepSampler, DenoiseSampler

    schedule = str(getattr(cfg, "diffusion_schedule", "cosine"))
    cosine_weight = float(getattr(cfg, "diffusion_cosine_weight", 1.0))
    model_kwargs = dict(
        dim_in=dim_in,
        dim_hidden=cfg.diffusion_hidden_dim,
        num_layers=cfg.diffusion_num_layers,
        T=cfg.diffusion_timesteps,
        beta_1=cfg.diffusion_beta_1,
        beta_T=cfg.diffusion_beta_T,
        anchor_scale=bool(getattr(cfg, "diffusion_anchor_scale", True)),
        cfg_prob=float(getattr(cfg, "diffusion_cfg_prob", 0.0)),
        pred_type=str(getattr(cfg, "diffusion_pred_type", "x0")),
        schedule=schedule,
        cosine_weight=cosine_weight,
        min_snr_gamma=float(getattr(cfg, "diffusion_min_snr_gamma", 5.0)),
    )
    model = ResidualDiffusion(**model_kwargs).to(device)
    cfg_scale = float(getattr(cfg, "diffusion_cfg_scale", 0.0))

    # 选择采样器
    sampler_type = str(getattr(cfg, "diffusion_sampler", "ddim"))
    if sampler_type == "single_step":
        sampler_cls = SingleStepSampler
    elif sampler_type == "denoise":
        t_start = int(getattr(cfg, "diffusion_denoise_t_start", 100))
        denoise_steps = int(getattr(cfg, "diffusion_denoise_steps", 20))
        sampler_cls = lambda m, d, **kw: DenoiseSampler(m, d, t_start=t_start, denoise_steps=denoise_steps, **kw)
    elif sampler_type == "ddim":
        ddim_steps = int(getattr(cfg, "diffusion_ddim_steps", 50))
        ddim_eta = float(getattr(cfg, "diffusion_ddim_eta", 0.0))
        sampler_cls = lambda m, d, **kw: DDIMSampler(m, d, ddim_steps=ddim_steps, eta=ddim_eta, **kw)
    else:
        sampler_cls = DDPMSampler

    return model, sampler_cls, cfg_scale


def _build_generator_sampler(model, device: torch.device, sampler_cls, cfg_scale=0.0, residual_scale=1.0):
    return sampler_cls(model, device, cfg_scale=cfg_scale, residual_scale=residual_scale)


def train_generator(encoder: nn.Module, dataset, cfg, device: torch.device):
    """训练 DDPM 生成器"""
    print("Training DDPM Generator")

    # 提取训练集嵌入
    pairing_method = cfg.diffusion_pairing_method
    need_manifold = pairing_method in ("hyperbolic", "dualrank", "intersect", "lorentz_inner")
    # DDPM ALWAYS uses jk_tangent (384-dim) for training
    # Pairing can use different source for distance calculation
    pairing_source = getattr(cfg, "diffusion_pairing_source", "jk_tangent")

    use_multiview_aug = pairing_method in ("mvaug", "selfpair")
    h1 = h2 = None
    encoder.eval()
    if use_multiview_aug:
        all_tangent_1 = []
        all_tangent_2 = []
    else:
        all_tangent = []
        all_manifold = [] if need_manifold else None
    loader = DataLoader(dataset.train_graphs, batch_size=256, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if use_multiview_aug:
                from models.augmentation import feature_mask_pyg, drop_edge_weighted

                noise_scale1 = cfg.model_noise_scale1
                noise_scale2 = cfg.model_noise_scale2
                feat_drop1 = cfg.diffusion_mv_feat_drop1 if cfg.diffusion_mv_feat_drop1 >= 0 else cfg.model_feat_drop_prob
                edge_drop1 = cfg.diffusion_mv_edge_drop1 if cfg.diffusion_mv_edge_drop1 >= 0 else cfg.model_edge_drop_prob
                feat_drop2 = cfg.diffusion_mv_feat_drop2 if cfg.diffusion_mv_feat_drop2 >= 0 else cfg.model_feat_drop_prob2
                edge_drop2 = cfg.diffusion_mv_edge_drop2 if cfg.diffusion_mv_edge_drop2 >= 0 else cfg.model_edge_drop_prob2

                x1, edge_index1, batch1 = feature_mask_pyg(
                    batch.x,
                    batch.edge_index,
                    batch.batch,
                    mask_rate=feat_drop1,
                    noise_scale=noise_scale1,
                )
                edge_index1 = drop_edge_weighted(edge_index1, drop_prob=edge_drop1)
                x2, edge_index2, batch2 = feature_mask_pyg(
                    batch.x,
                    batch.edge_index,
                    batch.batch,
                    mask_rate=feat_drop2,
                    noise_scale=noise_scale2,
                )
                edge_index2 = drop_edge_weighted(edge_index2, drop_prob=edge_drop2)

                jk_mani1, h_t1, last_mani1 = encoder.encode_graph(
                    x1, edge_index1, batch1
                )
                jk_mani2, h_t2, last_mani2 = encoder.encode_graph(
                    x2, edge_index2, batch2
                )

                all_tangent_1.append(h_t1)
                all_tangent_2.append(h_t2)
            else:
                jk_mani, h_t, last_mani = encoder.encode_graph(
                    batch.x, batch.edge_index, batch.batch
                )
                # DDPM ALWAYS uses jk_tangent
                all_tangent.append(h_t)
                if need_manifold:
                    # For hyperbolic pairing, use the same source as pairing_source
                    if pairing_source in ("last_tangent", "last_manifold"):
                        all_manifold.append(last_mani)
                    else:
                        all_manifold.append(jk_mani)

    if use_multiview_aug:
        h1 = torch.cat(all_tangent_1, dim=0).to(device)
        h2 = torch.cat(all_tangent_2, dim=0).to(device)
        h = h1
    else:
        h = torch.cat(all_tangent, dim=0).to(device)  # jk_tangent for DDPM
    if cfg.diffusion_pair_zscore:
        h = (h - h.mean(dim=0, keepdim=True)) / (h.std(dim=0, keepdim=True) + 1e-6)
        print("[Generator] Applied z-score normalization for pairing")
    print(
        f"[Generator] Extracted {h.shape[0]} embeddings, dim={h.shape[1]}, source=jk_tangent"
    )

    # ========== DEBUG: Encoder 表征质量分析 ==========
    train_labels_dbg = torch.tensor(
        [g.y.item() for g in dataset.train_graphs], device=device
    )
    h_norms = h.norm(dim=1)
    print(
        f"[DEBUG-Encoder] Tangent norm: mean={h_norms.mean():.4f}, std={h_norms.std():.4f}, max={h_norms.max():.4f}"
    )

    # 计算同类/异类平均距离
    h_norm_unit = h / (h.norm(dim=1, keepdim=True) + 1e-8)
    cos_sim = h_norm_unit @ h_norm_unit.T
    same_class_mask = train_labels_dbg.unsqueeze(0) == train_labels_dbg.unsqueeze(1)
    same_class_mask.fill_diagonal_(False)
    diff_class_mask = ~same_class_mask
    diff_class_mask.fill_diagonal_(False)

    if same_class_mask.sum() > 0:
        same_class_sim = cos_sim[same_class_mask].mean().item()
        diff_class_sim = cos_sim[diff_class_mask].mean().item()
        print(f"[DEBUG-Encoder] Same-class cosine sim: {same_class_sim:.4f}")
        print(f"[DEBUG-Encoder] Diff-class cosine sim: {diff_class_sim:.4f}")
        print(
            f"[DEBUG-Encoder] Separation (same - diff): {same_class_sim - diff_class_sim:.4f}"
        )
    # ========== END DEBUG ==========

    # ========== 构建 pairing ==========
    from models.pairing import build_pairs, filter_pairs, _build_pairs_with_weights

    k_nn = cfg.diffusion_k_nn
    N = h.size(0)
    pair_weight_method = str(getattr(cfg, "diffusion_pair_weight", "none"))
    pair_weight_temp = float(getattr(cfg, "diffusion_pair_weight_temp", 1.0))
    pair_ratio = float(getattr(cfg, "diffusion_pair_ratio", 0.0))

    # 参数用于 build_pairs
    mv_noise = float(getattr(cfg, "diffusion_mv_noise", 0.05))
    triad_shared_min = int(getattr(cfg, "diffusion_triad_shared_min", 2))
    hub_degree_q = float(getattr(cfg, "diffusion_hub_degree_q", 0.95))
    rank_sum = int(getattr(cfg, "diffusion_rank_sum", -1))
    dual_rank_sum = int(getattr(cfg, "diffusion_dual_rank_sum", -1))

    # For pairing with different source (e.g., last_tangent), we need to compute those embeddings
    h_pair = h  # Default: use same as DDPM input (jk_tangent)
    if pairing_source in ("last_tangent", "last_manifold"):
        # Extract last_tangent embeddings for pairing
        all_last_tangent = []
        loader_pair = DataLoader(dataset.train_graphs, batch_size=256, shuffle=False)
        with torch.no_grad():
            for batch in loader_pair:
                batch = batch.to(device)
                _, _, last_mani = encoder.encode_graph(
                    batch.x, batch.edge_index, batch.batch
                )
                last_t = encoder.manifold.logmap0(last_mani, encoder.c)
                all_last_tangent.append(last_t)
        h_pair = torch.cat(all_last_tangent, dim=0).to(device)
        print(f"[Pairing] Using {h_pair.shape[0]} {pairing_source} embeddings (dim={h_pair.shape[1]})")

    # Separate anchor/neighbor tensors (selfpair uses h1/h2, others use h for both)
    h_anchor_emb = h
    h_neighbor_emb = h

    if pairing_method == "selfpair":
        # Self-pair: each graph paired with its own augmented view
        # h1 = view1 embeddings, h2 = view2 embeddings (guaranteed same-class)
        anchor_idx = torch.arange(N, device=device)
        neighbor_idx = torch.arange(N, device=device)
        confidence = torch.ones(N, device=device)
        pair_weights = None
        h_anchor_emb = h1
        h_neighbor_emb = h2
        print(f"[SelfPair] Built {N} self-pairs (purity=100%)")
    elif pairing_method == "oracle":
        # Oracle pairing: use true labels to build 100% pure same-class pairs
        labels_oracle = torch.tensor(
            [g.y.item() for g in dataset.train_graphs], device=device
        )
        anchors_o, neighbors_o = [], []
        for i in range(N):
            same_class = (labels_oracle == labels_oracle[i]).nonzero(as_tuple=True)[0]
            same_class = same_class[same_class != i]
            if len(same_class) == 0:
                continue
            perm = torch.randperm(len(same_class), device=device)[:k_nn]
            for j in same_class[perm]:
                anchors_o.append(i)
                neighbors_o.append(j.item())
        anchor_idx = torch.tensor(anchors_o, dtype=torch.long, device=device)
        neighbor_idx = torch.tensor(neighbors_o, dtype=torch.long, device=device)
        confidence = torch.ones(anchor_idx.numel(), device=device)
        pair_weights = torch.ones(anchor_idx.numel(), device=device) if pair_weight_method != "none" else None
        print(f"[Oracle Pairing] Built {anchor_idx.numel()} same-class pairs for {N} samples (k={k_nn})")
    else:
        anchor_idx, neighbor_idx, confidence, pair_weights = _build_pairs_with_weights(
            h=h_pair,  # Use h_pair for distance calculation (jk_tangent or last_tangent)
            pairing_method=pairing_method,
            k_nn=k_nn,
            pair_ratio=pair_ratio,
            cfg=cfg,
            device=device,
            N=N,
            need_manifold=need_manifold,
            encoder=encoder,
            all_manifold=all_manifold,
            mv_noise=mv_noise,
            triad_shared_min=triad_shared_min,
            hub_degree_q=hub_degree_q,
            rank_sum=rank_sum,
            dual_rank_sum=dual_rank_sum,
            compute_weights=(pair_weight_method != "none"),
            weight_method=pair_weight_method,
            weight_temp=pair_weight_temp,
        )
    if anchor_idx.numel() == 0:
        raise ValueError("No pairs mined. Adjust pairing settings.")

    confidence_raw = confidence
    labels = train_labels_dbg
    purity_raw = (labels[anchor_idx] == labels[neighbor_idx]).float().mean().item()
    covered_raw = torch.zeros(N, dtype=torch.bool, device=device)
    covered_raw[anchor_idx] = True
    covered_raw[neighbor_idx] = True
    coverage_raw = covered_raw.float().mean().item()
    print(
        f"[Pairing] method={pairing_method} source={pairing_source} "
        f"k={k_nn} ratio={pair_ratio} pairs={anchor_idx.numel()} "
        f"purity={purity_raw*100:.2f}% mismatch={(1-purity_raw)*100:.2f}% "
        f"coverage={coverage_raw*100:.2f}%"
    )

    conf_topq = float(getattr(cfg, "diffusion_pair_conf_topq", 0.0))
    min_conf = float(getattr(cfg, "diffusion_pair_min_conf", 0.0))
    max_degree = int(getattr(cfg, "diffusion_pair_max_degree", 0))

    anchor_idx, neighbor_idx, confidence, pair_weights = filter_pairs(
        anchor_idx=anchor_idx,
        neighbor_idx=neighbor_idx,
        confidence=confidence_raw,
        num_nodes=N,
        conf_topq=conf_topq,
        min_conf=min_conf,
        max_node_degree=max_degree,
        device=device,
        weights=pair_weights,
    )
    if anchor_idx.numel() == 0:
        raise ValueError("All pairs were filtered out. Relax pair filters.")

    purity = (labels[anchor_idx] == labels[neighbor_idx]).float().mean().item()
    covered = torch.zeros(N, dtype=torch.bool, device=device)
    covered[anchor_idx] = True
    covered[neighbor_idx] = True
    coverage = covered.float().mean().item()
    print(
        f"[PairingFiltered] pairs={anchor_idx.numel()} "
        f"purity={purity*100:.2f}% mismatch={(1-purity)*100:.2f}% "
        f"coverage={coverage*100:.2f}% "
        f"conf_topq={conf_topq} min_conf={min_conf} max_degree={max_degree}"
    )

    # Log pair weights info
    if pair_weights is not None and pair_weights.numel() > 0:
        print(
            f"[Generator] pair_weights: mean={pair_weights.mean():.4f}, "
            f"min={pair_weights.min():.4f}, max={pair_weights.max():.4f}, "
            f"method={pair_weight_method}, temp={pair_weight_temp}"
        )
    # ========== END pairing ==========


    # ========== DEBUG: Pairing 质量分析 ==========
    train_labels = torch.tensor(
        [g.y.item() for g in dataset.train_graphs], device=device
    )
    anchor_labels = train_labels[anchor_idx]
    neighbor_labels = train_labels[neighbor_idx]
    match_mask = anchor_labels == neighbor_labels
    mismatch_rate = 1.0 - match_mask.float().mean().item()
    print(
        f"[DEBUG-Pairing] Total pairs: {len(anchor_idx)}, Mismatch: {mismatch_rate:.4f}"
    )
    # ========== END DEBUG ==========

    # 残差统计仅用于诊断
    residuals = h_neighbor_emb[neighbor_idx] - h_anchor_emb[anchor_idx]

    # ========== DEBUG: 残差分析（同类 vs 跨类）==========
    residual_norms = residuals.norm(dim=1)
    print(
        f"[DEBUG-Residual] Overall norm: mean={residual_norms.mean():.4f}, std={residual_norms.std():.4f}"
    )
    if match_mask.sum() > 0 and (~match_mask).sum() > 0:
        same_res_norm = residual_norms[match_mask].mean().item()
        diff_res_norm = residual_norms[~match_mask].mean().item()
        print(f"[DEBUG-Residual] Same-class residual norm: {same_res_norm:.4f}")
        print(f"[DEBUG-Residual] Diff-class residual norm: {diff_res_norm:.4f}")
    # ========== END DEBUG ==========

    # 创建 DDPM 模型
    model, sampler_cls, cfg_scale = _build_diffusion_model(
        cfg=cfg,
        device=device,
        dim_in=h.size(1),
    )

    # 计算 residual_mean + global_scale
    with torch.no_grad():
        res_for_scale = residuals
        if model.anchor_scale:
            anc_norm = h_anchor_emb[anchor_idx].norm(dim=1, keepdim=True).clamp(min=1e-4)
            res_for_scale = res_for_scale / anc_norm
        # Mean-center
        per_dim_mean = res_for_scale.mean(dim=0)
        model.set_residual_mean(per_dim_mean)
        res_centered = res_for_scale - per_dim_mean.unsqueeze(0)
        per_dim_std = res_centered.std(dim=0).clamp(min=1e-6)
        model.set_global_scale(per_dim_std)
        scaled_norms = (res_centered / per_dim_std.unsqueeze(0)).norm(dim=1)
        print(f"[MeanCenter] residual_mean norm: {per_dim_mean.norm():.4f}")
        print(f"[GlobalScale] per_dim_std: mean={per_dim_std.mean():.4f}, "
              f"scaled x0 norm: mean={scaled_norms.mean():.4f}, target≈{res_for_scale.size(1)**0.5:.1f}")

    # 训练 DDPM
    model.train()
    epochs = cfg.diffusion_epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.diffusion_lr)
    diff_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=cfg.diffusion_lr * 0.01
    )
    n_pairs = len(anchor_idx)
    batch_size = int(getattr(cfg, "diffusion_batch_size", 2048))
    log_interval = int(getattr(cfg, "diffusion_log_interval", 50))

    for epoch in range(epochs):
        perm = torch.randperm(n_pairs, device=device)
        total_loss, n_batches = 0, 0
        for i in range(0, n_pairs, batch_size):
            idx = perm[i : i + batch_size]
            if pair_weights is not None:
                loss = model.loss_fn(
                    h_anchor_emb[anchor_idx[idx]], h_neighbor_emb[neighbor_idx[idx]],
                    weight=pair_weights[idx],
                )
            else:
                loss = model.loss_fn(
                    h_anchor_emb[anchor_idx[idx]], h_neighbor_emb[neighbor_idx[idx]],
                )
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        diff_scheduler.step()
        if epoch % log_interval == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"[Generator] Epoch {epoch:03d} | Loss: {total_loss / max(n_batches, 1):.6f} | LR: {cur_lr:.6f}"
            )

    print(f"[Generator] Training completed")
    # Optional save generator checkpoint
    save_gen = cfg.diffusion_save_generator
    if save_gen:
        import datetime

        os.makedirs("./savepoint", exist_ok=True)
        path = f"./savepoint/generator.{cfg.dataset}.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        torch.save(model.state_dict(), path)
        print(f"+++ Saved generator: {path}")

    residual_scale = float(getattr(cfg, "diffusion_residual_scale", 1.0))
    sampler = _build_generator_sampler(
        model=model,
        device=device,
        sampler_cls=sampler_cls,
        cfg_scale=cfg_scale,
        residual_scale=residual_scale,
    )
    return sampler


def load_generator(encoder: nn.Module, cfg, device: torch.device, ckpt_path: str):
    """Load DDPM generator from checkpoint without retraining."""
    # DDPM always uses jk_tangent (384-dim)
    state = torch.load(ckpt_path, map_location=device)
    try:
        dim_in = _infer_dim_in_from_state(state)
    except KeyError:
        dim_in = encoder.raw_dim  # jk_tangent dimension

    # Infer T from checkpoint (beta shape)
    ckpt_T = state["beta"].shape[0] if "beta" in state else cfg.diffusion_timesteps
    import copy
    cfg_load = copy.copy(cfg)
    cfg_load.diffusion_timesteps = ckpt_T

    model, sampler_cls, cfg_scale = _build_diffusion_model(
        cfg=cfg_load,
        device=device,
        dim_in=dim_in,
    )
    # global_scale / residual_mean buffer may be None in fresh model but tensor in checkpoint
    if "global_scale" in state and state["global_scale"] is not None:
        model.global_scale = torch.zeros_like(state["global_scale"])
    if "residual_mean" in state and state["residual_mean"] is not None:
        model.residual_mean = torch.zeros_like(state["residual_mean"])
    model.load_state_dict(state)
    residual_scale = float(getattr(cfg, "diffusion_residual_scale", 1.0))
    sampler = _build_generator_sampler(
        model=model,
        device=device,
        sampler_cls=sampler_cls,
        cfg_scale=cfg_scale,
        residual_scale=residual_scale,
    )
    print(
        f"[Generator] loaded from {ckpt_path} "
        f"(dim_in={dim_in}, source=jk_tangent, cfg_scale={cfg_scale}, res_scale={residual_scale})"
    )
    return sampler
