"""
HypGCL for Graphs - Unified Lorentz/Poincaré Implementation

Dual loss mechanism:
  - Distance Loss: hyperbolic distance in manifold space
  - Angle Loss: cosine similarity in tangent space
  - Curriculum learning: gradually increase distance loss weight
  - Anti-collapse regularization (VICReg-style): variance + covariance terms

Reference: https://github.com/Sungwon-Han/HypCD
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool


class HypGCLGraph(nn.Module):
    """
    HypGCL (Hyperbolic Graph Contrastive Learning) for Graphs - supports both Lorentz and Poincaré manifolds.

    Uses encoder's manifold directly for all hyperbolic operations.
    Includes VICReg-style anti-collapse regularization to prevent dimensional collapse.
    """

    def __init__(
        self,
        encoder,
        proj_hidden_dim=256,
        proj_layers=2,
        temperature=0.5,
        hyper_max_weight=1.0,
        hyper_start_epoch=0,
        hyper_end_epoch=100,
        feat_drop_prob=0.2,
        edge_drop_prob=0.2,
        feat_drop_prob2=0.3,
        edge_drop_prob2=0.3,
        train_pooling="mean",
        vicreg_weight=1.0,
        vicreg_var_weight=25.0,
        vicreg_cov_weight=1.0,
        vicreg_target_std=1.0,
        vicreg_backbone=True,
        noise_scale1=0.5,
        noise_scale2=0.7,
        aug_view1="node_drop",
        aug_view2="feature_mask",
        aug_rate1=0.1,
        aug_rate2=0.1,
        debug_train=False,
        debug_max_batches=1,
        use_supervised_loss=False,
    ):
        super().__init__()

        self.encoder = encoder
        self.device = encoder.device
        self.manifold = encoder.manifold
        self.c = encoder.c
        self.temperature = temperature

        self.train_pooling = train_pooling

        # Curriculum learning
        self.hyper_max_weight = hyper_max_weight
        self.hyper_start_epoch = hyper_start_epoch
        self.hyper_end_epoch = hyper_end_epoch
        self.current_epoch = 0

        # Augmentation
        self.feat_drop_prob = feat_drop_prob
        self.edge_drop_prob = edge_drop_prob
        self.feat_drop_prob2 = feat_drop_prob2
        self.edge_drop_prob2 = edge_drop_prob2

        # Anti-collapse (VICReg-style) hyperparameters
        self.vicreg_weight = vicreg_weight
        self.var_weight = vicreg_var_weight
        self.cov_weight = vicreg_cov_weight
        self.var_target_std = vicreg_target_std
        self.vicreg_backbone = vicreg_backbone

        # Projection head (Euclidean MLP for angle loss)
        input_dim = encoder.raw_dim
        layers = []
        in_features = input_dim
        for i in range(proj_layers):
            out_features = proj_hidden_dim if i < proj_layers - 1 else input_dim
            layers.extend([
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(inplace=True),
            ])
            in_features = out_features
        self.projection_head = nn.Sequential(*layers)

        # Augmentation noise scales
        self.noise_scale1 = noise_scale1
        self.noise_scale2 = noise_scale2
        self.aug_view1 = aug_view1
        self.aug_view2 = aug_view2
        self.aug_rate1 = aug_rate1
        self.aug_rate2 = aug_rate2

        # Supervised loss option
        self.use_supervised_loss = use_supervised_loss

        # Debug logging
        self.debug_train = debug_train
        self.debug_max_batches = debug_max_batches
        self._debug_batch_idx = 0

        # Collapse tracking (per-epoch stats)
        self._epoch_tangent_list = []  # accumulate tangent vectors for end-of-epoch stats

        print(
            f"[HypGCLGraph] manifold={self.manifold.name}, temp={temperature}, "
            f"hyper_weight={hyper_max_weight}, epochs=[{hyper_start_epoch},{hyper_end_epoch}]"
        )
        print(
            f"[HypGCLGraph] VICReg: weight={self.vicreg_weight}, "
            f"var_w={self.var_weight}, cov_w={self.cov_weight}, "
            f"target_std={self.var_target_std}, backbone={self.vicreg_backbone}"
        )

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.debug_train:
            self._debug_batch_idx = 0
        self._epoch_tangent_list = []  # reset for new epoch

    def get_distance_weight(self):
        """Curriculum learning: linear ramp from 0 to hyper_max_weight"""
        if self.current_epoch < self.hyper_start_epoch:
            return 0.0
        if self.current_epoch >= self.hyper_end_epoch:
            return self.hyper_max_weight
        progress = (self.current_epoch - self.hyper_start_epoch) / (
            self.hyper_end_epoch - self.hyper_start_epoch
        )
        return progress * self.hyper_max_weight

    def _variance_loss(self, z):
        """VICReg variance term: force std of each dimension > target_std.

        Penalizes dimensions that collapse (std near 0).
        Uses hinge loss: max(0, target_std - std(z_j)) for each dimension j.
        """
        std = torch.sqrt(z.var(dim=0) + 1e-4)  # per-dimension std, batch dim=0
        # Hinge: only penalize when std < target
        var_loss = F.relu(self.var_target_std - std).mean()
        return var_loss

    def _covariance_loss(self, z):
        """VICReg covariance term: decorrelate different dimensions.

        The off-diagonal elements of the covariance matrix should be 0.
        """
        N, D = z.shape
        z_centered = z - z.mean(dim=0)
        cov = (z_centered.T @ z_centered) / (N - 1)  # [D, D]
        # Zero out diagonal (we only penalize off-diagonal)
        cov_offdiag = cov - torch.diag(cov.diag())
        # Mean squared off-diagonal
        cov_loss = (cov_offdiag**2).sum() / D
        return cov_loss

    def _dist_matrix(self, x, y):
        """Compute pairwise distance matrix using manifold.dist"""
        n, m = x.size(0), y.size(0)
        # Expand for pairwise computation
        x_exp = x.unsqueeze(1).expand(n, m, -1)  # [N, M, d]
        y_exp = y.unsqueeze(0).expand(n, m, -1)  # [N, M, d]

        # Flatten, compute distances, reshape
        x_flat = x_exp.reshape(n * m, -1)
        y_flat = y_exp.reshape(n * m, -1)
        dist = self.manifold.dist(x_flat, y_flat, self.c)
        return dist.view(n, m)

    def _info_nce_distance(self, h_manifold):
        """InfoNCE with hyperbolic distance (negative distance as similarity)"""
        batch_size = h_manifold.size(0) // 2

        # Distance matrix
        dist = self._dist_matrix(h_manifold, h_manifold)  # [2B, 2B]

        # Similarity = -distance (smaller distance = higher similarity)
        sim = -dist / self.temperature

        # Numerical stability (HypGCL trick)
        sim = sim - sim.max(dim=1, keepdim=True)[0].detach()

        # Labels: positive pairs are (i, i+B) and (i+B, i)
        labels = torch.cat([torch.arange(batch_size), torch.arange(batch_size)]).to(
            self.device
        )
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # Mask diagonal (self)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=self.device)
        labels = labels[~mask].view(2 * batch_size, -1)
        sim = sim[~mask].view(2 * batch_size, -1)

        # Split positives and negatives
        pos = sim[labels.bool()].view(2 * batch_size, -1)
        neg = sim[~labels.bool()].view(2 * batch_size, -1)

        logits = torch.cat([pos, neg], dim=1)
        targets = torch.zeros(2 * batch_size, dtype=torch.long, device=self.device)

        return F.cross_entropy(logits, targets)

    def _info_nce_angle(self, h_tangent):
        """InfoNCE with cosine similarity in tangent space"""
        batch_size = h_tangent.size(0) // 2

        # Normalize for cosine similarity
        h = F.normalize(h_tangent, p=2, dim=1)
        sim = h @ h.T / self.temperature  # [2B, 2B]

        # Labels
        labels = torch.cat([torch.arange(batch_size), torch.arange(batch_size)]).to(
            self.device
        )
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # Mask diagonal
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=self.device)
        labels = labels[~mask].view(2 * batch_size, -1)
        sim = sim[~mask].view(2 * batch_size, -1)

        # Split
        pos = sim[labels.bool()].view(2 * batch_size, -1)
        neg = sim[~labels.bool()].view(2 * batch_size, -1)

        logits = torch.cat([pos, neg], dim=1)
        targets = torch.zeros(2 * batch_size, dtype=torch.long, device=self.device)

        return F.cross_entropy(logits, targets)

    def _sup_con_loss(self, sim, labels):
        """Proper SupCon loss over a precomputed similarity matrix.

        Args:
            sim: [N, N] raw similarity (cosine or -distance), NOT yet /temperature
            labels: [N] integer class labels
        Returns:
            scalar loss
        """
        N = sim.size(0)
        device = sim.device

        # Scale by temperature
        logits = sim / self.temperature

        # Numerical stability
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

        # Label mask: 1 if same class
        labels = labels.contiguous().view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float()  # [N, N]

        # Exclude self-contrast
        logits_mask = 1.0 - torch.eye(N, device=device)
        mask_pos = mask_pos * logits_mask

        # log_prob = logits - log(sum(exp(logits)) over all non-self)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Mean log-prob over positives
        pos_count = mask_pos.sum(dim=1)  # [N]
        # Skip samples with no positives (singleton classes)
        valid = pos_count > 0
        mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1)[valid] / pos_count[valid]

        return -mean_log_prob_pos.mean()

    def _supervised_angle_loss(self, h_tangent, labels):
        """Supervised contrastive loss (cosine) using true labels."""
        h = F.normalize(h_tangent, p=2, dim=1)
        sim = h @ h.T  # [N, N] cosine similarity
        return self._sup_con_loss(sim, labels)

    def _supervised_distance_loss(self, h_manifold, labels):
        """Supervised contrastive loss (hyperbolic distance) using true labels."""
        dist = self._dist_matrix(h_manifold, h_manifold)  # [N, N]
        sim = -dist  # negative distance as similarity
        return self._sup_con_loss(sim, labels)

    def _pool(self, h_tan, batch_idx):
        """Apply train_pooling to node-level tangent vectors."""
        p = self.train_pooling
        if p == "mean":
            return global_mean_pool(h_tan, batch_idx)
        elif p == "max":
            from torch_geometric.nn import global_max_pool
            return global_max_pool(h_tan, batch_idx)
        elif p == "meansum":
            h_mean = global_mean_pool(h_tan, batch_idx)
            h_sum = global_add_pool(h_tan, batch_idx)
            return torch.cat([h_mean, h_sum], dim=1)
        elif p == "meanmax":
            from torch_geometric.nn import global_max_pool
            h_mean = global_mean_pool(h_tan, batch_idx)
            h_max = global_max_pool(h_tan, batch_idx)
            return torch.cat([h_mean, h_max], dim=1)
        else:  # sum
            return global_add_pool(h_tan, batch_idx)

    def _encode_graph_both(self, x, edge_index, batch_idx):
        """Encode graph for TRAINING using configured train_pooling.

        Returns:
            h_manifold: last layer manifold embedding
            h_tangent: JK tangent embedding (all layers concatenated)
        """
        hidden_rep = self.encoder.forward(x, edge_index)

        graph_features = []
        for h in hidden_rep:
            h_tan = self.manifold.logmap0(h, self.c)
            graph_features.append(self._pool(h_tan, batch_idx))

        h_tangent = torch.cat(graph_features, dim=1)

        # Manifold: last layer, mapped back
        h_last_tan = self.manifold.logmap0(hidden_rep[-1], self.c)
        h_last_pooled = self._pool(h_last_tan, batch_idx)
        # meansum/meanmax 产生 2*dim，取前半（mean 部分）映射回流形
        if self.train_pooling in ("meansum", "meanmax"):
            half = h_last_pooled.size(1) // 2
            h_last_pooled = h_last_pooled[:, :half]
        h_manifold = self.manifold.expmap0(h_last_pooled, self.c)

        return h_manifold, h_tangent

    def compute_loss(self, batch_data):
        """Compute contrastive loss with dual objectives"""
        x, edge_index, batch = batch_data.x, batch_data.edge_index, batch_data.batch
        batch_size = batch.max().item() + 1

        if batch_size < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True), {
                "loss": 0.0
            }

        # Create augmented views
        from models.augmentation import apply_aug

        x1, edge_index1, batch1 = apply_aug(
            self.aug_view1, x, edge_index, batch,
            rate=self.aug_rate1, noise_scale=self.noise_scale1,
        )
        x2, edge_index2, batch2 = apply_aug(
            self.aug_view2, x, edge_index, batch,
            rate=self.aug_rate2, noise_scale=self.noise_scale2,
        )

        # Encode both views with MEAN pooling for training (old behavior)
        # _encode_graph_both uses global_mean_pool (training)
        # encoder.encode_graph uses configured pool method (evaluation)
        h1_manifold, h1_tangent = self._encode_graph_both(x1, edge_index1, batch1)
        h2_manifold, h2_tangent = self._encode_graph_both(x2, edge_index2, batch2)

        # Project tangent embeddings
        z1 = self.projection_head(h1_tangent)
        z2 = self.projection_head(h2_tangent)

        # Concatenate for contrastive loss
        # h_manifold: last-layer manifold (128-dim) — matches old code
        h_manifold = torch.cat([h1_manifold, h2_manifold], dim=0)
        h_tangent = torch.cat([z1, z2], dim=0)

        # Debug: 打印范数/流形偏差信息
        debug_info = {}
        if self.debug_train and self._debug_batch_idx < self.debug_max_batches:
            with torch.no_grad():
                r1 = h1_tangent.norm(dim=1)
                spread = h1_tangent.std(dim=0).mean()
                inner = self.manifold.l_inner(h1_manifold, h1_manifold)
                val = (-inner) / self.c
                dev = (val - 1.0).abs()
                debug_info = {
                    "tan_mean": r1.mean().item(),
                    "tan_max": r1.max().item(),
                    "tan_std": r1.std().item(),
                    "tan_spread": spread.item(),
                    "mani_dev_mean": dev.mean().item(),
                    "mani_dev_max": dev.max().item(),
                }
                print(
                    f"[DEBUG][E{self.current_epoch}B{self._debug_batch_idx}] "
                    f"tan_mean={debug_info['tan_mean']:.3f} "
                    f"tan_max={debug_info['tan_max']:.3f} "
                    f"tan_std={debug_info['tan_std']:.3f} "
                    f"spread={debug_info['tan_spread']:.4f} "
                    f"mani_dev_mean={debug_info['mani_dev_mean']:.4e} "
                    f"mani_dev_max={debug_info['mani_dev_max']:.4e}"
                )
            self._debug_batch_idx += 1
        else:
            # 保留原来的稀疏范数日志
            if self.current_epoch % 20 == 0 or self.current_epoch == 0:
                r1 = h1_tangent.norm(dim=1)
                print(
                    f"[Epoch {self.current_epoch}] h1_tangent norm: "
                    f"mean={r1.mean().item():.2f}, max={r1.max().item():.2f}, std={r1.std().item():.2f}"
                )

        # Compute losses
        # For supervised learning, we need labels
        use_sup = hasattr(batch_data, 'y') and self.use_supervised_loss
        if use_sup:
            graph_labels = batch_data.y  # [batch_size]
            repeated_labels = graph_labels.repeat(2)
            loss_angle = self._supervised_angle_loss(h_tangent, repeated_labels)
            loss_dist = self._supervised_distance_loss(h_manifold, repeated_labels)
        else:
            loss_angle = self._info_nce_angle(h_tangent)
            loss_dist = self._info_nce_distance(h_manifold)

        # Anti-collapse regularization (VICReg-style)
        loss_var = torch.tensor(0.0, device=self.device)
        loss_cov = torch.tensor(0.0, device=self.device)
        if self.vicreg_weight > 0:
            if self.vicreg_backbone:
                # Apply to BACKBONE tangent (before projection head) — more important
                vicreg_input = torch.cat([h1_tangent, h2_tangent], dim=0)
            else:
                # Apply to projected representations
                vicreg_input = h_tangent  # already cat([z1, z2])
            loss_var = self._variance_loss(vicreg_input)
            loss_cov = self._covariance_loss(vicreg_input)

        # Weighted combination
        lambda_d = self.get_distance_weight()
        loss_contrastive = (1 - lambda_d) * loss_angle + lambda_d * loss_dist
        loss_vicreg = self.vicreg_weight * (
            self.var_weight * loss_var + self.cov_weight * loss_cov
        )
        loss = loss_contrastive + loss_vicreg

        info = {
            "loss": loss.item(),
            "loss_angle": loss_angle.item(),
            "loss_dist": loss_dist.item(),
            "loss_var": loss_var.item(),
            "loss_cov": loss_cov.item(),
            "loss_vicreg": loss_vicreg.item(),
            "lambda_d": lambda_d,
        }
        if debug_info:
            info.update(debug_info)

        return loss, info

    def forward(self, batch_data):
        return self.compute_loss(batch_data)

    def encode_graph(self, batch_data):
        """Encode graph for downstream tasks (returns tangent space).

        NOTE: This method is NOT used in the main pipeline. The pipeline
        uses encoder.encode_graph/encode_task directly for evaluation.
        This uses _encode_graph_both (mean pooling) to match training dims.
        """
        x, edge_index, batch = batch_data.x, batch_data.edge_index, batch_data.batch
        _, h_tangent = self._encode_graph_both(x, edge_index, batch)
        return self.projection_head(h_tangent)


# Backward compatibility aliases
HypGCLGraphLorentz = HypGCLGraph
HypGCLGraphTangent = HypGCLGraph


def get_embeddings(model, dataloader):
    """Extract embeddings for all graphs"""
    model.eval()
    embeddings, labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(model.device)
            z = model.encode_graph(batch)
            embeddings.append(z.cpu())
            if hasattr(batch, "y"):
                labels.append(batch.y.cpu())

    model.train()
    return torch.cat(embeddings), torch.cat(labels) if labels else None
