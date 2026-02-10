"""
Pairing module: intersect method (cosine ∩ hyperbolic mutual k-NN).
"""

import torch
from typing import Tuple, Optional, Any


def _build_mutual_pairs_from_dist(
    dist_matrix: torch.Tensor, k: int, pair_ratio: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build mutual k-NN pairs from distance matrix.

    Returns:
        anchor_idx: anchor indices
        neighbor_idx: neighbor indices
        kth: k-th neighbor distance for each node
        dist: full distance matrix
    """
    dist = dist_matrix.clone()
    n = dist.size(0)
    dist.fill_diagonal_(float("inf"))

    knn_vals, knn_idx = torch.topk(dist, k=k, dim=1, largest=False)
    kth = knn_vals[:, -1]
    knn_sets = [set(knn_idx[i].tolist()) for i in range(n)]

    # Collect all mutual pairs per node (with distances)
    node_mutual = [[] for _ in range(n)]  # node_mutual[i] = [(j, dist)]
    pair_set = set()
    for i in range(n):
        for j in knn_idx[i].tolist():
            if i not in knn_sets[j]:
                continue
            node_mutual[i].append((j, dist[i, j].item()))
            key = (min(i, j), max(i, j))
            pair_set.add(key)

    if not pair_set:
        empty = torch.empty(0, dtype=torch.long, device=dist.device)
        return empty, empty, kth, dist

    # Step 1: guarantee each node keeps its closest mutual neighbor
    guaranteed = set()
    for i in range(n):
        if not node_mutual[i]:
            continue
        best_j = min(node_mutual[i], key=lambda x: x[1])[0]
        guaranteed.add((min(i, best_j), max(i, best_j)))

    # Step 2: remaining pairs sorted by distance, fill up to N * ratio
    remaining = []
    for key in pair_set:
        if key not in guaranteed:
            remaining.append((key, dist[key[0], key[1]].item()))
    remaining.sort(key=lambda x: x[1])

    max_pairs = int(n * pair_ratio) if pair_ratio > 0 else len(pair_set)
    budget = max(max_pairs - len(guaranteed), 0)
    selected = guaranteed | {p for p, _ in remaining[:budget]}

    # Expand to bidirectional pairs
    anchors, neighbors = [], []
    for i, j in selected:
        anchors.append(i); neighbors.append(j)
        anchors.append(j); neighbors.append(i)

    anchor_idx = torch.tensor(anchors, dtype=torch.long, device=dist.device)
    neighbor_idx = torch.tensor(neighbors, dtype=torch.long, device=dist.device)
    return anchor_idx, neighbor_idx, kth, dist



def _chunked_hyperbolic_dist(
    h_mani: torch.Tensor, manifold: Any, c: float, chunk_size: int = 512
) -> torch.Tensor:
    """Compute hyperbolic distance matrix in chunks to avoid OOM."""
    n = h_mani.size(0)
    device = h_mani.device
    dist_h = torch.zeros(n, n, device=device)
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        for j in range(0, n, chunk_size):
            end_j = min(j + chunk_size, n)
            dist_h[i:end_i, j:end_j] = manifold.dist(
                h_mani[i:end_i].unsqueeze(1),
                h_mani[j:end_j].unsqueeze(0),
                c,
            )
    return dist_h


def build_pairs(
    h: torch.Tensor,
    pairing_method: str,
    k_nn: int,
    pair_ratio: float,
    cfg: Any,
    device: torch.device,
    N: int,
    need_manifold: bool = False,
    encoder: Any = None,
    all_manifold: Optional[list] = None,
    mv_noise: float = 0.05,
    triad_shared_min: int = 2,
    hub_degree_q: float = 0.95,
    rank_sum: int = -1,
    dual_rank_sum: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build training pairs using the specified pairing method.

    Args:
        h: tangent space embeddings [N, D]
        pairing_method: name of pairing method
        k_nn: k for k-NN
        pair_ratio: filter pairs with distance > pair_ratio * kth
        cfg: config object
        device: torch device
        N: number of nodes
        need_manifold: whether method needs manifold embeddings
        encoder: encoder model (for hyperbolic distance)
        all_manifold: list of manifold embeddings
        mv_noise: noise level for multi-view pairing
        triad_shared_min: min shared neighbors for triad consistency
        hub_degree_q: quantile for hub pruning
        rank_sum: threshold for cosine rank sum
        dual_rank_sum: threshold for dual rank sum

    Returns:
        anchor_idx: anchor indices
        neighbor_idx: neighbor indices
        confidence: pair confidence scores
    """
    # Normalize for cosine distance
    h_norm = h / (h.norm(dim=1, keepdim=True) + 1e-8)
    dist_cos = 1.0 - h_norm @ h_norm.T

    # Hyperbolic distance if needed
    use_hyp = pairing_method in ("intersect", "dualrank", "hyperbolic")
    dist_h = None
    if use_hyp:
        if not need_manifold or encoder is None or all_manifold is None:
            raise ValueError(f"{pairing_method} requires manifold embeddings")
        h_mani = torch.cat(all_manifold, dim=0).to(device)
        dist_h = _chunked_hyperbolic_dist(h_mani, encoder.manifold, encoder.c)

    # ========== Distance-based methods ==========
    if pairing_method == "cosine":
        a, b, kth, dist = _build_mutual_pairs_from_dist(dist_cos, k_nn, pair_ratio)
        conf = _pair_confidence(a, b, dist, kth)
        return a, b, conf

    if pairing_method == "euclidean":
        dist_e = torch.cdist(h, h, p=2)
        a, b, kth, dist = _build_mutual_pairs_from_dist(dist_e, k_nn, pair_ratio)
        conf = _pair_confidence(a, b, dist, kth)
        return a, b, conf

    if pairing_method == "eulocalscale":
        dist_e = torch.cdist(h, h, p=2)
        d = dist_e.clone()
        d.fill_diagonal_(float("inf"))
        sigma = torch.topk(d, k_nn, dim=1, largest=False).values[:, -1]
        scale = sigma.unsqueeze(1) * sigma.unsqueeze(0) + 1e-8
        dist_ls = dist_e / scale
        a, b, kth, dist = _build_mutual_pairs_from_dist(dist_ls, k_nn, pair_ratio)
        conf = _pair_confidence(a, b, dist, kth)
        return a, b, conf

    if pairing_method == "localscale":
        d = dist_cos.clone()
        d.fill_diagonal_(float("inf"))
        sigma = torch.topk(d, k_nn, dim=1, largest=False).values[:, -1]
        scale = sigma.unsqueeze(1) * sigma.unsqueeze(0) + 1e-8
        dist_ls = dist_cos / scale
        a, b, kth, dist = _build_mutual_pairs_from_dist(dist_ls, k_nn, pair_ratio)
        conf = _pair_confidence(a, b, dist, kth)
        return a, b, conf

    if pairing_method == "coslocalscale":
        d = dist_cos.clone()
        d.fill_diagonal_(float("inf"))
        sigma = torch.topk(d, k_nn, dim=1, largest=False).values[:, -1]
        scale = sigma.unsqueeze(1) * sigma.unsqueeze(0) + 1e-8
        dist_ls = dist_cos / scale
        a, b, kth, dist = _build_mutual_pairs_from_dist(dist_ls, k_nn, pair_ratio)
        conf = _pair_confidence(a, b, dist, kth)
        return a, b, conf

    if pairing_method == "snn":
        n = h.size(0)
        d = dist_cos.clone()
        d.fill_diagonal_(float("inf"))
        knn = torch.topk(d, k_nn, dim=1, largest=False).indices
        A = torch.zeros(n, n, device=device)
        A.scatter_(1, knn, 1.0)
        shared = A @ A.T
        deg = A.sum(dim=1, keepdim=True)
        union = deg + deg.T - shared
        snn_sim = shared / union.clamp(min=1.0)
        dist_snn = 1.0 - snn_sim
        a, b, kth, dist = _build_mutual_pairs_from_dist(dist_snn, k_nn, pair_ratio)
        conf = _pair_confidence(a, b, dist, kth)
        return a, b, conf

    if pairing_method == "kreciprocal":
        n = h.size(0)
        d = dist_cos.clone()
        d.fill_diagonal_(float("inf"))
        knn = torch.topk(d, k_nn, dim=1, largest=False).indices
        A = torch.zeros(n, n, device=device)
        A.scatter_(1, knn, 1.0)
        R = A * A.T
        shared = R @ R.T
        deg = R.sum(dim=1, keepdim=True)
        union = deg + deg.T - shared
        jac = shared / union.clamp(min=1.0)
        dist_jac = 1.0 - jac
        a, b, kth, dist = _build_mutual_pairs_from_dist(dist_jac, k_nn, pair_ratio)
        conf = _pair_confidence(a, b, dist, kth)
        return a, b, conf

    if pairing_method == "cosrank":
        n = h.size(0)
        d = dist_cos.clone()
        d.fill_diagonal_(float("inf"))
        rank_t = d.argsort(dim=1).argsort(dim=1)
        rank_sym = rank_t + rank_t.T
        thr = rank_sum if rank_sum > 0 else 2 * k_nn
        mask = rank_sym <= thr
        mask.fill_diagonal_(False)
        a, b = mask.nonzero(as_tuple=True)
        conf = (1.0 - rank_sym[a, b].float() / max(thr, 1)).clamp(min=0.0)
        return a, b, conf

    if pairing_method == "mvcos":
        n = h.size(0)
        h2 = h + (mv_noise * torch.randn_like(h) if mv_noise > 0 else 0.0)
        h2_norm = h2 / (h2.norm(dim=1, keepdim=True) + 1e-8)
        dist2 = 1.0 - h2_norm @ h2_norm.T

        a1, b1, kth1, d1 = _build_mutual_pairs_from_dist(dist_cos, k_nn, pair_ratio)
        a2, b2, kth2, d2 = _build_mutual_pairs_from_dist(dist2, k_nn, pair_ratio)
        m1 = torch.zeros(n, n, dtype=torch.bool, device=device)
        m2 = torch.zeros(n, n, dtype=torch.bool, device=device)
        if a1.numel() > 0:
            m1[a1, b1] = True
        if a2.numel() > 0:
            m2[a2, b2] = True
        m = m1 & m2
        a, b = m.nonzero(as_tuple=True)
        if a.numel() == 0:
            return a, b, torch.empty(0, device=device)
        s1 = torch.maximum(kth1[a], kth1[b]).clamp(min=1e-8)
        s2 = torch.maximum(kth2[a], kth2[b]).clamp(min=1e-8)
        conf = torch.exp(
            -0.5 * (d1[a, b].clamp(min=0.0) / s1 + d2[a, b].clamp(min=0.0) / s2)
        ).clamp(max=1.0)
        return a, b, conf

    if pairing_method == "eutriad":
        n = h.size(0)
        dist_e = torch.cdist(h, h, p=2)
        d = dist_e.clone()
        d.fill_diagonal_(float("inf"))
        knn_idx = torch.topk(d, k_nn, dim=1, largest=False).indices
        a, b, kth, dist = _build_mutual_pairs_from_dist(dist_e, k_nn, pair_ratio)
        if a.numel() == 0:
            return a, b, torch.empty(0, device=device)
        shared = _shared_knn_count(knn_idx, a, b)
        keep = shared >= int(max(1, triad_shared_min))
        a = a[keep]
        b = b[keep]
        shared = shared[keep]
        if a.numel() == 0:
            return a, b, torch.empty(0, device=device)
        conf_base = _pair_confidence(a, b, dist, kth)
        triad_boost = (shared.float() / float(max(k_nn, 1))).clamp(min=0.0, max=1.0)
        conf = (conf_base * triad_boost).clamp(max=1.0)
        return a, b, conf

    if pairing_method == "euhub":
        n = h.size(0)
        dist_e = torch.cdist(h, h, p=2)
        a, b, kth, dist = _build_mutual_pairs_from_dist(dist_e, k_nn, pair_ratio)
        if a.numel() == 0:
            return a, b, torch.empty(0, device=device)
        degree = _pair_degree(a, b, n=n, device=device)
        nz = degree[degree > 0]
        if nz.numel() == 0:
            return a, b, torch.empty(0, device=device)
        q = float(min(max(hub_degree_q, 0.0), 1.0))
        hub_thr = torch.quantile(nz, q)
        keep = (degree[a] <= hub_thr) & (degree[b] <= hub_thr)
        a = a[keep]
        b = b[keep]
        if a.numel() == 0:
            return a, b, torch.empty(0, device=device)
        conf = _pair_confidence(a, b, dist, kth)
        return a, b, conf

    if pairing_method == "eucombo":
        n = h.size(0)
        dist_e = torch.cdist(h, h, p=2)
        d = dist_e.clone()
        d.fill_diagonal_(float("inf"))
        knn_idx = torch.topk(d, k_nn, dim=1, largest=False).indices
        a, b, kth, dist = _build_mutual_pairs_from_dist(dist_e, k_nn, pair_ratio)
        if a.numel() == 0:
            return a, b, torch.empty(0, device=device)

        # Step 1: triad consistency filter
        shared = _shared_knn_count(knn_idx, a, b)
        keep_t = shared >= int(max(1, triad_shared_min))
        a = a[keep_t]
        b = b[keep_t]
        shared = shared[keep_t]
        if a.numel() == 0:
            return a, b, torch.empty(0, device=device)

        # Step 2: hub-pruning filter
        degree = _pair_degree(a, b, n=n, device=device)
        nz = degree[degree > 0]
        q = float(min(max(hub_degree_q, 0.0), 1.0))
        hub_thr = torch.quantile(nz, q)
        keep_h = (degree[a] <= hub_thr) & (degree[b] <= hub_thr)
        a = a[keep_h]
        b = b[keep_h]
        shared = shared[keep_h]
        if a.numel() == 0:
            return a, b, torch.empty(0, device=device)

        conf_base = _pair_confidence(a, b, dist, kth)
        triad_boost = (shared.float() / float(max(k_nn, 1))).clamp(min=0.0, max=1.0)
        conf = (conf_base * triad_boost).clamp(max=1.0)
        return a, b, conf

    if pairing_method == "intersect":
        n = h.size(0)
        a_t, b_t, kth_t, d_t = _build_mutual_pairs_from_dist(dist_cos, k_nn, 0)
        a_h, b_h, kth_h, d_h = _build_mutual_pairs_from_dist(dist_h, k_nn, 0)
        m_t = torch.zeros(n, n, dtype=torch.bool, device=device)
        m_h = torch.zeros(n, n, dtype=torch.bool, device=device)
        if a_t.numel() > 0:
            m_t[a_t, b_t] = True
        if a_h.numel() > 0:
            m_h[a_h, b_h] = True
        m = m_t & m_h
        a, b = m.nonzero(as_tuple=True)
        if a.numel() == 0:
            return a, b, torch.empty(0, device=device)
        s_t = torch.maximum(kth_t[a], kth_t[b]).clamp(min=1e-8)
        s_h = torch.maximum(kth_h[a], kth_h[b]).clamp(min=1e-8)
        conf = torch.exp(
            -0.5 * (d_t[a, b].clamp(min=0.0) / s_t + d_h[a, b].clamp(min=0.0) / s_h)
        ).clamp(max=1.0)
        return a, b, conf

    if pairing_method == "dualrank":
        n = h.size(0)
        d_t = dist_cos.clone()
        d_t.fill_diagonal_(float("inf"))
        d_h = dist_h.clone()
        d_h.fill_diagonal_(float("inf"))
        rank_t = d_t.argsort(dim=1).argsort(dim=1)
        rank_h = d_h.argsort(dim=1).argsort(dim=1)
        rank_sum_mat = rank_t + rank_t.T + rank_h + rank_h.T
        thr = dual_rank_sum if dual_rank_sum > 0 else 4 * k_nn
        mask = rank_sum_mat <= thr
        mask.fill_diagonal_(False)
        a, b = mask.nonzero(as_tuple=True)
        conf = (1.0 - rank_sum_mat[a, b].float() / max(thr, 1)).clamp(min=0.0)
        return a, b, conf

    if pairing_method == "hyperbolic":
        a, b, kth, dist = _build_mutual_pairs_from_dist(dist_h, k_nn, pair_ratio)
        conf = _pair_confidence(a, b, dist, kth)
        return a, b, conf

    if pairing_method == "normband":
        delta = getattr(cfg, "diffusion_norm_band", 0.2)
        log_norms = torch.log(h.norm(dim=1, keepdim=True) + 1e-8)
        norm_diff = torch.abs(log_norms - log_norms.T)
        dist_nb = dist_cos.clone()
        dist_nb[norm_diff > delta] = float("inf")
        a, b, kth, dist = _build_mutual_pairs_from_dist(dist_nb, k_nn, pair_ratio)
        conf = _pair_confidence(a, b, dist, kth)
        return a, b, conf

    if pairing_method == "hybrid":
        log_norms = torch.log(h.norm(dim=1, keepdim=True) + 1e-8)
        norm_diff = torch.abs(log_norms - log_norms.T)
        norm_dist = norm_diff / (norm_diff.max() + 1e-8)
        alpha, beta = 0.3, 0.7
        dist_hyb = alpha * dist_cos + beta * norm_dist
        a, b, kth, dist = _build_mutual_pairs_from_dist(dist_hyb, k_nn, pair_ratio)
        conf = _pair_confidence(a, b, dist, kth)
        return a, b, conf

    if pairing_method == "normonly":
        norms = h.norm(dim=1, keepdim=True)
        dist_norm = (norms - norms.T).abs()
        a, b, kth, dist = _build_mutual_pairs_from_dist(dist_norm, k_nn, pair_ratio)
        conf = _pair_confidence(a, b, dist, kth)
        return a, b, conf

    if pairing_method == "lorentz_inner":
        """Pairing using Lorentz inner product directly as similarity.

        For Lorentz manifold: <x,y>_L = -x0*y0 + x_space*y_space^T
        For timelike vectors: <x,y>_L = -k*cosh(d/k) where d is geodesic distance
        Since cosh(d/k) >= 1: <x,y>_L <= -k, with equality only when d=0 (same point)

        For k-NN we need: smaller distance = more similar
        Use arccosh(-<x,y>_L/k) directly (proportional to geodesic distance).
        """
        if not need_manifold or encoder is None or all_manifold is None:
            raise ValueError("lorentz_inner requires manifold embeddings")
        h_mani = torch.cat(all_manifold, dim=0).to(device)
        k = encoder.c

        # Manual computation of Lorentz inner product
        x0 = h_mani[:, 0:1]  # [N, 1]
        x_space = h_mani[:, 1:]  # [N, D-1]

        # Pairwise inner product: <x,y>_L = -x0*y0^T + x_space*y_space^T
        inner = -x0 @ x0.T + x_space @ x_space.T  # [N, N]

        # Convert to geodesic distance: d = sqrt(k) * arccosh(-<x,y>_L/k)
        # Since we only care about relative ordering, we can skip sqrt(k) factor
        ratio = (-inner / k).clamp(min=1.0 + 1e-7)  # -<x,y>_L/k = cosh(d/k) >= 1
        dist_matrix = torch.acosh(ratio)  # Proportional to geodesic distance

        a, b, kth, dist = _build_mutual_pairs_from_dist(dist_matrix, k_nn, pair_ratio)
        conf = _pair_confidence(a, b, dist, kth)
        return a, b, conf

    if pairing_method == "cluster":
        n_clusters = getattr(cfg, "diffusion_n_clusters", 0)
        if n_clusters <= 0:
            n_clusters = max(2, int((N / 2) ** 0.5))
        from sklearn.cluster import KMeans
        h_np = h.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=n_clusters, random_state=72, n_init=10)
        cluster_labels = kmeans.fit_predict(h_np)
        cluster_labels = torch.tensor(cluster_labels, device=device)
        dist_e = torch.cdist(h, h, p=2)
        same_cluster = cluster_labels.unsqueeze(0) == cluster_labels.unsqueeze(1)
        dist_e[~same_cluster] = float("inf")
        a, b, kth, dist = _build_mutual_pairs_from_dist(dist_e, k_nn, pair_ratio)
        conf = _pair_confidence(a, b, dist, kth)
        return a, b, conf

    # Fallback: euclidean
    dist_e = torch.cdist(h, h, p=2)
    a, b, kth, dist = _build_mutual_pairs_from_dist(dist_e, k_nn, pair_ratio)
    conf = _pair_confidence(a, b, dist, kth)
    return a, b, conf


def _build_pairs_with_weights(
    h: torch.Tensor,
    pairing_method: str,
    k_nn: int,
    pair_ratio: float,
    cfg: Any,
    device: torch.device,
    N: int,
    need_manifold: bool = False,
    encoder: Any = None,
    all_manifold: Optional[list] = None,
    mv_noise: float = 0.05,
    triad_shared_min: int = 2,
    hub_degree_q: float = 0.95,
    rank_sum: int = -1,
    dual_rank_sum: int = -1,
    compute_weights: bool = False,
    weight_method: str = "none",
    weight_temp: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Internal function that calls build_pairs and optionally computes weights."""
    # Call the original build_pairs
    a, b, conf = build_pairs(
        h=h,
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
    )

    # Compute weights if requested
    weights = None
    if compute_weights and conf.numel() > 0:
        weights = build_pair_weights(confidence=conf, method=weight_method, temp=weight_temp)

    return a, b, conf, weights


def filter_pairs(
    anchor_idx: torch.Tensor,
    neighbor_idx: torch.Tensor,
    confidence: torch.Tensor,
    num_nodes: int,
    conf_topq: float = 0.0,
    min_conf: float = 0.0,
    max_node_degree: int = 0,
    device: torch.device = torch.device("cpu"),
    weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Filter pairs based on confidence and degree constraints.

    Args:
        anchor_idx: anchor indices
        neighbor_idx: neighbor indices
        confidence: pair confidences
        num_nodes: total number of nodes
        conf_topq: keep only top-q quantile of confidence (0 = disabled)
        min_conf: minimum confidence threshold
        max_node_degree: maximum degree per node (0 = disabled)
        weights: optional pair weights (filtered along with pairs)

    Returns:
        filtered anchor_idx, neighbor_idx, confidence, weights
    """
    if anchor_idx.numel() == 0:
        return anchor_idx, neighbor_idx, confidence, weights

    keep = torch.ones(anchor_idx.numel(), dtype=torch.bool, device=device)

    # Filter by confidence quantile
    if conf_topq > 0 and conf_topq < 1.0:
        thr = torch.quantile(confidence, 1.0 - conf_topq)
        keep &= confidence >= thr

    # Filter by minimum confidence
    if min_conf > 0:
        keep &= confidence >= min_conf

    # Apply confidence filters
    anchor_idx = anchor_idx[keep]
    neighbor_idx = neighbor_idx[keep]
    confidence = confidence[keep]
    if weights is not None:
        weights = weights[keep]

    if anchor_idx.numel() == 0:
        return anchor_idx, neighbor_idx, confidence, weights

    # Filter by max degree
    if max_node_degree > 0:
        degree = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        degree.scatter_add_(
            0, anchor_idx, torch.ones(anchor_idx.numel(), device=device)
        )
        degree.scatter_add_(
            0, neighbor_idx, torch.ones(neighbor_idx.numel(), device=device)
        )
        keep = (
            (degree[anchor_idx] <= max_node_degree)
            & (degree[neighbor_idx] <= max_node_degree)
        )
        anchor_idx = anchor_idx[keep]
        neighbor_idx = neighbor_idx[keep]
        confidence = confidence[keep]
        if weights is not None:
            weights = weights[keep]

    return anchor_idx, neighbor_idx, confidence, weights


def build_pair_weights(
    confidence: torch.Tensor, method: str = "none", temp: float = 1.0
) -> Optional[torch.Tensor]:
    """
    Build pair weights from confidence scores.

    Args:
        confidence: pair confidence scores (higher is better)
        method: "none", "exp", or "linear"
        temp: temperature for exponential weighting (higher = more extreme weights)

    Returns:
        pair weights or None
    """
    if method == "none" or confidence.numel() == 0:
        return None
    if method == "exp":
        # exp权重: w = exp(conf * temp), 高置信度获得更高权重
        return torch.exp(confidence * temp)
    elif method == "linear":
        return confidence
    else:
        return None
