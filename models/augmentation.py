"""
Graph augmentation functions for Contrastive Learning (CL) and CVGGR
"""
import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj


def drop_edge_weighted(edge_index, drop_prob=0.1):
    """随机丢弃边，返回新的edge_index"""
    edge_index, _ = dropout_adj(edge_index, p=drop_prob, force_undirected=True)
    return edge_index


def mask_nodes(x: torch.Tensor,
               batch: torch.Tensor,
               mask_rate: float = 0.1,
               mask_token: torch.Tensor | None = None):
    """
    per-graph 节点特征 mask（更稳）
    x: [num_nodes, feat_dim]
    batch: [num_nodes] 每个节点所属图 id
    mask_token: [1, feat_dim] 若提供则用它替换被mask节点特征
    """
    if mask_rate <= 0:
        return x, torch.empty(0, dtype=torch.long, device=x.device)

    x_masked = x.clone()
    mask_indices = []

    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    for gid in range(num_graphs):
        node_idx = (batch == gid).nonzero(as_tuple=False).view(-1)
        n = int(node_idx.numel())
        if n == 0:
            continue
        m = int(n * mask_rate)
        if m <= 0:
            continue
        perm = torch.randperm(n, device=x.device)[:m]
        mask_indices.append(node_idx[perm])

    if len(mask_indices) == 0:
        return x_masked, torch.empty(0, dtype=torch.long, device=x.device)

    mask_indices = torch.cat(mask_indices, dim=0)

    if mask_token is None:
        x_masked[mask_indices] = 0
    else:
        x_masked[mask_indices] = mask_token.expand(mask_indices.size(0), -1)

    return x_masked, mask_indices


def node_drop_pyg(x, edge_index, batch, drop_rate=0.1):
    """
    SMART-style node drop: disconnect selected nodes (zero their edges)
    but keep them in the graph so pooling still sees all nodes.
    """
    num_nodes = x.size(0)
    drop_mask = torch.rand(num_nodes, device=x.device) < drop_rate
    src, dst = edge_index
    edge_keep = ~drop_mask[src] & ~drop_mask[dst]
    return x, edge_index[:, edge_keep], batch


def apply_aug(name, x, edge_index, batch, rate=0.1, noise_scale=0.5):
    """Dispatch augmentation by name. Returns (x, edge_index, batch)."""
    if name == "node_drop":
        return node_drop_pyg(x, edge_index, batch, drop_rate=rate)
    elif name == "feature_mask":
        return feature_mask_pyg(x, edge_index, batch, mask_rate=rate, noise_scale=noise_scale)
    elif name == "edge_drop":
        return x, drop_edge_weighted(edge_index, drop_prob=rate), batch
    elif name == "none":
        return x, edge_index, batch
    else:
        raise ValueError(f"Unknown augmentation: {name}")


def feature_mask_pyg(x, edge_index, batch, mask_rate=0.1, noise_scale=0.5):
    """
    SMART-style node-level feature mask: select a fraction of nodes
    and replace their entire feature vector with N(0.5, 0.5).
    """
    num_nodes, feat_dim = x.size()
    n_mask = int(num_nodes * mask_rate)
    if n_mask == 0:
        return x, edge_index, batch

    idx = torch.randperm(num_nodes, device=x.device)[:n_mask]
    x_new = x.clone()
    x_new[idx] = torch.normal(0.5, 0.5, size=(n_mask, feat_dim), device=x.device)

    return x_new, edge_index, batch
