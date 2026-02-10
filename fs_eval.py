import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_geometric.data import Batch


def _linear_probe_accuracy(
    support_z, support_y, query_z, query_y,
    input_dim, device, train_steps=500, lr=0.01,
    n_original=0, aug_weight=1.0,
):
    """Linear probe: z-score → SGD + early stopping → accuracy."""
    # z-score from ORIGINAL support only
    base = support_z[:n_original] if n_original > 0 else support_z
    mu = base.mean(dim=0, keepdim=True)
    std = base.std(dim=0, keepdim=True).clamp(min=1e-6)
    support_z = (support_z - mu) / std
    query_z = (query_z - mu) / std

    # Remap labels to 0..C-1 (vectorized, remap both support and query)
    unique_y = torch.unique(support_y)
    target_s = torch.searchsorted(unique_y, support_y)
    target_q = torch.searchsorted(unique_y, query_y)

    model = nn.Linear(input_dim, len(unique_y)).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-2)

    # Per-sample weights (only when augmented samples exist)
    has_aug = 0 < n_original < support_z.size(0) and aug_weight != 1.0
    if has_aug:
        sample_w = torch.ones(support_z.size(0), device=device)
        sample_w[n_original:] = aug_weight

    best_loss, best_state, wait = float("inf"), None, 0
    for step in range(train_steps):
        logits = model(support_z)
        if has_aug:
            loss = (F.cross_entropy(logits, target_s, reduction='none') * sample_w).sum() / sample_w.sum()
        else:
            loss = F.cross_entropy(logits, target_s)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cur_loss = loss.item()
        if cur_loss < best_loss:
            best_loss, wait = cur_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait > 10:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        preds = model(query_z).argmax(dim=-1)
    return (preds == target_q).float().mean().item()


def _knn_accuracy(
    support_z, support_y, query_z, query_y, device="cuda", k=5, distance="euclidean"
):
    support_z = support_z.to(device)
    support_y = support_y.to(device)
    query_z = query_z.to(device)
    query_y = query_y.to(device)

    if support_z.size(0) == 0 or query_z.size(0) == 0:
        return 0.0

    k = max(1, min(int(k), support_z.size(0)))
    distance = str(distance).lower()
    if distance == "cosine":
        q = F.normalize(query_z, dim=1)
        s = F.normalize(support_z, dim=1)
        dist = 1.0 - q @ s.T
    else:
        dist = torch.cdist(query_z, support_z, p=2)

    topk_dist, topk_idx = torch.topk(dist, k, dim=1, largest=False)
    topk_labels = support_y[topk_idx]  # [Q, k]

    unique_labels = torch.unique(support_y)
    weights = 1.0 / (topk_dist + 1e-8)
    scores = torch.zeros(query_z.size(0), unique_labels.numel(), device=device)
    for i, lbl in enumerate(unique_labels):
        mask = (topk_labels == lbl).float()
        scores[:, i] = (weights * mask).sum(dim=1)
    pred_idx = scores.argmax(dim=1)
    preds = unique_labels[pred_idx]
    return (preds == query_y).float().mean().item()


def fewshot_evaluate(encoder, dataset, device, cfg, generator=None, n_aug=3):
    """Few-shot evaluation using Linear Probe classifier."""
    adaptive_aug = cfg.eval_adaptive_aug
    n_aug_5shot = cfg.eval_n_aug_5shot if cfg.eval_n_aug_5shot >= 0 else n_aug
    n_aug_10shot = cfg.eval_n_aug_10shot if cfg.eval_n_aug_10shot >= 0 else max(1, n_aug // 2)
    aug_weight = float(getattr(cfg, "eval_aug_weight", 1.0))
    encoder.eval()

    eval_train_steps = cfg.eval_train_steps
    eval_lr = cfg.eval_lr
    eval_classifier = str(getattr(cfg, "eval_classifier", "linear_probe")).lower()
    eval_knn_k = int(getattr(cfg, "eval_knn_k", 5))
    eval_knn_distance = str(getattr(cfg, "eval_knn_distance", "euclidean")).lower()
    if eval_classifier not in ("linear_probe", "knn"):
        raise ValueError(
            f"Unsupported eval_classifier={eval_classifier}, expected one of "
            "['linear_probe', 'knn']"
        )

    # Evaluation ALWAYS uses jk_tangent for better representation
    eval_pair_source = "jk_tangent"
    input_dim = encoder.raw_dim  # jk_tangent uses raw_dim (384)

    def _encode_task(task, source):
        # Evaluation ALWAYS uses jk_tangent (encoder.encode_task returns jk_tangent)
        # The source parameter is kept for compatibility but ignored
        return encoder.encode_task(task)

    def _eval_tasks(tasks, K_shot, use_aug):
        accs = []
        # Adaptive augmentation count
        if adaptive_aug:
            current_n_aug = n_aug_5shot if K_shot == 5 else n_aug_10shot
        else:
            current_n_aug = n_aug

        for task in tasks:
            support_embs, query_embs = _encode_task(task, eval_pair_source)
            support_label = torch.cat([g.y for g in task["support_set"]]).to(device)
            query_label = torch.cat([g.y for g in task["query_set"]]).to(device)

            support_data = support_embs.detach()
            query_data = query_embs.detach()
            n_original = support_data.size(0)

            if use_aug and generator is not None and current_n_aug > 0:
                with torch.no_grad():
                    n_support = support_data.size(0)
                    unique_lbls = support_label[:n_support].unique()
                    all_aug = generator.sample(
                        support_data, n_samples=current_n_aug,
                    )
                    aug_embs = all_aug
                    aug_labels = support_label.repeat_interleave(current_n_aug)

                # Prototype-based filtering
                proto_map = {}
                for lbl in unique_lbls:
                    mask = support_label[:n_support] == lbl
                    proto_map[lbl.item()] = support_data[:n_support][mask].mean(dim=0)
                proto_stack = torch.stack([proto_map[l.item()] for l in unique_lbls])  # [C, D]

                # For each generated sample, check nearest prototype
                dists = torch.cdist(aug_embs, proto_stack)  # [N_aug, C]
                nearest_proto = unique_lbls[dists.argmin(dim=1)]
                keep_mask = nearest_proto == aug_labels
                n_before = aug_embs.size(0)
                aug_embs = aug_embs[keep_mask]
                aug_labels = aug_labels[keep_mask]

                if len(accs) == 0:
                    print(
                        f"[Proto-Filter] kept {aug_embs.size(0)}/{n_before} "
                        f"({aug_embs.size(0)/max(n_before,1)*100:.1f}%)"
                    )

                support_data = torch.cat([support_data, aug_embs], dim=0)
                support_label = torch.cat([support_label, aug_labels], dim=0)

            if eval_classifier == "knn":
                acc = _knn_accuracy(
                    support_z=support_data,
                    support_y=support_label,
                    query_z=query_data,
                    query_y=query_label,
                    device=device,
                    k=eval_knn_k,
                    distance=eval_knn_distance,
                )
            else:
                acc = _linear_probe_accuracy(
                    support_data, support_label, query_data, query_label,
                    input_dim=input_dim, device=device,
                    train_steps=eval_train_steps, lr=eval_lr,
                    n_original=n_original, aug_weight=aug_weight,
                )
            accs.append(acc)

        return {"mean": np.mean(accs), "std": np.std(accs)}

    if generator is None:
        results = {}
        for K_shot in [5, 10]:
            tasks = (
                dataset.test_tasks_5shot if K_shot == 5 else dataset.test_tasks_10shot
            )
            results[K_shot] = _eval_tasks(tasks, K_shot, use_aug=False)
        return results

    baseline = {}
    augmented = {}
    for K_shot in [5, 10]:
        tasks = dataset.test_tasks_5shot if K_shot == 5 else dataset.test_tasks_10shot
        baseline[K_shot] = _eval_tasks(tasks, K_shot, use_aug=False)
        augmented[K_shot] = _eval_tasks(tasks, K_shot, use_aug=True)

    return {"baseline": baseline, "augmented": augmented}
