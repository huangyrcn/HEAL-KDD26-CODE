"""
Hyperbolic GIN Encoder
支持 Lorentz 和 Poincaré Ball 两种流形，统一层设计

维度约定：
  - hidden_dim: 流形维度（两种流形一致）
  - Lorentz: hidden_dim = (hidden_dim-1)空间 + 1时间
  - Poincaré: hidden_dim = hidden_dim 空间
  - 输出: 两种流形都是 hidden_dim 维（Lorentz 的时间分量保留）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.data import Batch
from .hyperbolic import get_manifold


# ========== 统一的双曲层 ==========

def _clip_tangent(x, clip_r):
    """对数缩放：压缩大范数，保留相对顺序

    原硬裁剪: scale = min(1, clip_r / ||x||)  -> 丢失范数信息
    对数缩放: new_norm = clip_r * log(1 + ||x|| / clip_r)  -> 保留相对顺序
    """
    if clip_r is None:
        return x
    if clip_r <= 0:
        return x
    x_norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(1e-9)
    # 对数缩放：保留范数的相对顺序
    new_norm = clip_r * torch.log1p(x_norm / clip_r)
    return x * (new_norm / x_norm)




class HyperbolicLinear(nn.Module):
    """双曲线性层（统一接口）

    Lorentz: weight 作用于空间分量 (dim-1)
    Poincaré: weight 作用于全部分量 (dim)
    """

    def __init__(
        self, manifold, in_features, out_features, c, dropout=0.0, use_bias=True
    ):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.clip_r = None

        # 根据流形类型确定权重维度
        if manifold.name == "lorentz":
            # Lorentz: 只变换空间分量
            w_in = in_features - 1
            w_out = out_features - 1
        else:
            # Poincaré: 变换全部分量
            w_in = in_features
            w_out = out_features

        self.weight = nn.Parameter(torch.empty(w_out, w_in))
        self.bias = nn.Parameter(torch.empty(w_out)) if use_bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        w = F.dropout(self.weight, self.dropout, training=self.training)

        # Debug: 检查输入
        if torch.isnan(x).any():
            print(f'[HyperbolicLinear] Input x has NaN! max={x.max()}')

        if self.manifold.name == "lorentz":
            return self._forward_lorentz(x, w)
        else:
            return self._forward_poincare(x, w)

    def _forward_lorentz(self, x, w):
        """Lorentz 线性变换: log → 矩阵乘空间分量 → exp"""
        c = self.c
        v = self.manifold.logmap0(x, c)
        v_space = v[:, 1:]

        # 空间分量线性变换
        out_space = v_space @ w.T
        if self.bias is not None:
            out_space = out_space + self.bias

        # 拼接、映射回流形并投影
        out = torch.cat([v[:, :1], out_space], dim=-1)
        if self.clip_r is not None:
            out = _clip_tangent(out, self.clip_r)
        res = self.manifold.expmap0(out, c)

        # Debug
        if torch.isnan(res).any():
            print(f'[HyperbolicLinear] NaN detected!')
            print(f'  v max={v.max()}, v max={v_space.max()}')
            print(f'  out_space max={out_space.max()}')
            print(f'  res max={res.max()}')

        return self.manifold.proj(res, c)

    def _forward_poincare(self, x, w):
        """Poincaré 线性变换: Möbius matvec"""
        c = self.c
        out = self.manifold.mobius_matvec(w, x, c)
        if self.bias is not None:
            bias_hyp = self.manifold.expmap0(self.bias, c)
            out = self.manifold._mobius_add(out, bias_hyp, c)
        return self.manifold.proj(out, c)


class HyperbolicAct(nn.Module):
    """双曲激活层: logmap0 → [BatchNorm →] act → expmap0"""

    def __init__(self, manifold, c, act=torch.tanh, use_bn=False, dim=None):
        super().__init__()
        self.manifold = manifold
        self.c = c
        self.act = act
        self.clip_r = None
        # BatchNorm 作用在空间分量上（Lorentz 去掉第0维）
        if use_bn and dim is not None:
            bn_dim = dim - 1 if manifold.name == "lorentz" else dim
            self.bn = nn.BatchNorm1d(bn_dim)
        else:
            self.bn = None

    def forward(self, x):
        v = self.manifold.logmap0(x, self.c)
        if self.manifold.name == "lorentz":
            v_space = v[..., 1:]
            if self.bn is not None:
                v_space = self.bn(v_space)
            v_space = self.act(v_space)
            if self.clip_r is not None:
                v_space = _clip_tangent(v_space, self.clip_r)
            v = torch.cat([torch.zeros_like(v[..., :1]), v_space], dim=-1)
        else:
            if self.bn is not None:
                v = self.bn(v)
            v = self.act(v)
            if self.clip_r is not None:
                v = _clip_tangent(v, self.clip_r)
        return self.manifold.expmap0(v, self.c)


class HyperbolicAgg(nn.Module):
    """双曲聚合层"""

    def __init__(self, manifold, c, in_features, use_att=False, dropout=0.0):
        super().__init__()
        self.manifold = manifold
        self.c = c
        self.use_att = use_att
        self.clip_r = None

        if use_att:
            self.att = HyperbolicDistAtt(manifold, c, in_features, dropout)

    def forward(self, x, adj):
        # debug flag propagated from encoder/layer
        debug_active = getattr(self, "_debug_active", False)
        if self.use_att:
            self.att._debug_active = debug_active
        if self.use_att:
            weights = self.att(x, adj)
            return self._centroid_with_att(weights, x, adj)
        else:
            weights = self._normalize_adj(adj)
            return self.manifold.centroid(weights, x, self.c)

    def _normalize_adj(self, adj):
        deg = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        D = torch.diag(deg_inv_sqrt).to_sparse()
        return torch.sparse.mm(D, torch.sparse.mm(adj, D))

    def _centroid_with_att(self, att_data, x, adj):
        idx, vals, size = att_data
        att_sparse = torch.sparse_coo_tensor(idx, vals, tuple(size), device=x.device)
        return self.manifold.centroid(att_sparse, x, self.c)

    def reset_parameters(self):
        if self.use_att:
            self.att.linear.reset_parameters()


class HyperbolicDistAtt(nn.Module):
    """基于距离的双曲注意力"""

    def __init__(self, manifold, c, in_features, dropout):
        super().__init__()
        self.manifold = manifold
        self.c = c
        self.linear = HyperbolicLinear(
            manifold, in_features, in_features, c, dropout, True
        )
        self.clip_r = None

    def forward(self, x, adj):
        x = self.linear(x)
        adj = adj.coalesce()
        idx = adj.indices()
        xi, xj = x[idx[0]], x[idx[1]]
        d = self.manifold.dist(xi, xj, self.c)
        if getattr(self, "_debug_active", False):
            if not torch.isfinite(d).all():
                inner = self.manifold.l_inner(xi, xj)
                val = (-inner) / self.c
                print(
                    f"[DEBUG] dist NaN/Inf: val=-inner/k min={val.min().item():.4e} "
                    f"max={val.max().item():.4e} mean={val.mean().item():.4e}"
                )
        att = torch.exp(-d)
        return (idx, att, adj.size())


class HyperbolicMLP(nn.Module):
    """双曲 MLP"""

    def __init__(
        self,
        manifold,
        num_layers,
        c,
        in_features,
        out_features,
        dropout=0.0,
        use_bias=True,
        act=torch.tanh,
        use_bn=False,
    ):
        super().__init__()
        self.manifold = manifold
        self.c = c

        layers = []
        for i in range(num_layers):
            in_dim = in_features if i == 0 else out_features
            layers.append(
                nn.Sequential(
                    HyperbolicLinear(
                        manifold, in_dim, out_features, c, dropout, use_bias
                    ),
                    HyperbolicAct(manifold, c, act=act, use_bn=use_bn, dim=out_features),
                )
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def reset_parameters(self):
        for layer in self.layers:
            for m in layer:
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()


# ========== GIN 层 ==========


class HyperbolicGINLayer(nn.Module):
    """双曲 GIN 层"""

    def __init__(
        self,
        manifold,
        in_features,
        out_features,
        c,
        num_mlp_layers=2,
        eps=0.1,
        use_att=False,
        dropout=0.0,
        use_ptransp=False,
        act=torch.tanh,
        use_bn=False,
    ):
        super().__init__()
        self.manifold = manifold
        self.c = c
        self.eps = nn.Parameter(torch.tensor(eps))
        self.use_ptransp = use_ptransp

        self.agg = HyperbolicAgg(manifold, c, in_features, use_att, dropout)
        self.mlp = HyperbolicMLP(
            manifold, num_mlp_layers, c, in_features, out_features, dropout, act=act, use_bn=use_bn
        )
        self.clip_r = None

    def forward(self, x, adj):
        self.agg._debug_active = getattr(self, "_debug_active", False)
        agg_out = self.agg(x, adj)

        agg_out = self.manifold.proj(agg_out, self.c)
        x = self.manifold.proj(x, self.c)

        v_agg = self.manifold.logmap0(agg_out, self.c)
        v_x = self.manifold.logmap0(x, self.c)

        if self.use_ptransp:
            # 构造原点
            origin = torch.zeros_like(x)
            origin[..., 0] = self.c.sqrt()
            # 将 v_agg 从原点切空间平行传输到 x 的切空间
            v_agg = self.manifold.ptransp(origin, x, v_agg, self.c)

        v_combined = v_agg + (1 + self.eps) * v_x

        if self.clip_r is not None:
            v_combined = _clip_tangent(v_combined, self.clip_r)
        combined = self.manifold.expmap0(v_combined, self.c)
        combined = self.manifold.proj(combined, self.c)
        out = self.mlp(combined)
        return out

    def reset_parameters(self):
        self.agg.reset_parameters()
        self.mlp.reset_parameters()


# ========== HyperbolicGIN Encoder ==========


class HyperGIN(nn.Module):
    """统一的 Hyperbolic GIN Encoder

    输出: [batch_size, hidden_dim * num_layers] 切空间向量
    """

    def __init__(
        self,
        manifold_type="lorentz",
        num_layers=5,
        num_mlp_layers=2,
        input_dim=200,
        hidden_dim=128,
        final_dropout=0.5,
        learn_eps=True,
        graph_pooling_type="sum",
        curvature=1.0,
        dropout=0.0,
        training_curvature=True,
        use_att=True,
        use_ptransp=False,
        small_init=False,
        clip_r=0,
        debug_nan=False,
        debug_max_calls=3,
        act=torch.tanh,
        use_bn=False,
        use_input_proj=True,
        device=None,
    ):
        super().__init__()

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.manifold = get_manifold(manifold_type)
        self.manifold_type = manifold_type
        self.use_input_proj = use_input_proj

        if training_curvature:
            self.c = nn.Parameter(torch.tensor([curvature]))
        else:
            self.register_buffer("c", torch.tensor([curvature]))

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.graph_pooling_type = graph_pooling_type
        # meanmax/meansum 时维度翻倍
        if graph_pooling_type in ("meanmax", "meansum"):
            self.raw_dim = hidden_dim * num_layers * 2
        else:
            self.raw_dim = hidden_dim * num_layers

        # 输入投影层
        if use_input_proj:
            if manifold_type.lower() in ("lorentz", "lorentzian"):
                self.input_proj = nn.Linear(input_dim, hidden_dim - 1)
            else:
                self.input_proj = nn.Linear(input_dim, hidden_dim)
            if small_init:
                nn.init.uniform_(self.input_proj.weight, -0.05, 0.05)
                if self.input_proj.bias is not None:
                    nn.init.zeros_(self.input_proj.bias)
        else:
            self.input_proj = None

        self.clip_r = float(clip_r) if clip_r and clip_r > 0 else None

        # GIN 层
        # 无 input_proj 时，第一层输入维度 = input_dim + 1 (Lorentz) 或 input_dim
        if not use_input_proj:
            if manifold_type.lower() in ("lorentz", "lorentzian"):
                first_in = input_dim + 1
            else:
                first_in = input_dim
        else:
            first_in = hidden_dim

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_feat = first_in if i == 0 else hidden_dim
            layer = HyperbolicGINLayer(
                manifold=self.manifold,
                in_features=in_feat,
                out_features=hidden_dim,
                c=self.c,
                num_mlp_layers=num_mlp_layers,
                eps=0.1 if learn_eps else 0.0,
                use_att=use_att,
                dropout=dropout,
                use_ptransp=use_ptransp,
                act=act,
                use_bn=use_bn,
            )
            layer.clip_r = self.clip_r
            layer.agg.clip_r = self.clip_r
            if layer.agg.use_att:
                layer.agg.att.clip_r = self.clip_r
                layer.agg.att.linear.clip_r = self.clip_r
            # propagate clip to MLP submodules
            for block in layer.mlp.layers:
                for m in block:
                    if hasattr(m, "clip_r"):
                        m.clip_r = self.clip_r
            self.layers.append(layer)

        self.final_dropout = nn.Dropout(final_dropout)
        self.debug_nan = debug_nan
        self.debug_max_calls = debug_max_calls
        self._debug_calls = 0
        self._debug_active = False
        self.to(self.device)

    def _debug_stats(self, name, x):
        if not self._debug_active:
            return
        x_det = x.detach()
        finite = torch.isfinite(x_det)
        if finite.any():
            vals = x_det[finite]
            print(
                f"[DEBUG] {name}: shape={tuple(x_det.shape)} "
                f"finite={finite.float().mean().item():.3f} "
                f"min={vals.min().item():.4e} max={vals.max().item():.4e} "
                f"mean={vals.mean().item():.4e} std={vals.std().item():.4e}"
            )
        else:
            print(f"[DEBUG] {name}: all non-finite")

    def _debug_manifold(self, name, x):
        if not self._debug_active:
            return
        self._debug_stats(name, x)
        inner = self.manifold.l_inner(x, x)
        val = (-inner) / self.c
        self._debug_stats(f"{name}.(-inner/k)", val)

    def _to_manifold(self, x):
        """欧氏输入 → 流形上的点"""
        if self.input_proj is not None:
            x = self.input_proj(x)
            self._debug_stats("input_proj", x)

        if self.manifold.name == "lorentz":
            zeros = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
            x = torch.cat([zeros, x], dim=-1)

        if self.clip_r is not None:
            x = _clip_tangent(x, self.clip_r)
        out = self.manifold.expmap0(x, self.c)
        self._debug_manifold("expmap0", out)
        return out

    def forward(self, x, edge_index):
        """前向传播，返回每层的流形表示"""
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        adj = self._to_adj(edge_index, x.size(0))

        # 映射到流形
        h = self._to_manifold(x)

        # 通过 GIN 层
        hidden_rep = []
        for layer in self.layers:
            layer._debug_active = self._debug_active
            h = layer(h, adj)
            self._debug_manifold("layer_out", h)
            hidden_rep.append(h)

        return hidden_rep

    def _to_adj(self, edge_index, num_nodes):
        return torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.size(1), device=self.device),
            (num_nodes, num_nodes),
        )

    def encode_graph(self, x, edge_index, batch_idx):
        """图编码: 返回流形嵌入和切空间向量

        Returns:
            jk_manifold: Jumping Knowledge 流形嵌入 (用于 hyperbolic pairing)
            jk_tangent: Jumping Knowledge 切空间向量 (用于 DDPM 训练)
            last_manifold: 最后一层流形嵌入 (用于 pairing 备选)
        """
        # enable debug for first few encode calls
        self._debug_calls += 1
        self._debug_active = self.debug_nan and (self._debug_calls <= self.debug_max_calls)
        if self._debug_active:
            print(f"[DEBUG] encode_graph call {self._debug_calls}")

        hidden_rep = self.forward(x, edge_index)

        # 每层: 流形 → 切空间 → 池化
        graph_features = []
        for i, h in enumerate(hidden_rep):
            h_tan = self.manifold.logmap0(h, self.c)
            self._debug_stats(f"layer{i}.logmap0", h_tan)

            if self.graph_pooling_type == "mean":
                h_graph = global_mean_pool(h_tan, batch_idx)
            elif self.graph_pooling_type == "max":
                h_graph = global_max_pool(h_tan, batch_idx)
            elif self.graph_pooling_type == "meanmax":
                h_mean = global_mean_pool(h_tan, batch_idx)
                h_max = global_max_pool(h_tan, batch_idx)
                h_graph = torch.cat([h_mean, h_max], dim=1)
            elif self.graph_pooling_type == "meansum":
                h_mean = global_mean_pool(h_tan, batch_idx)
                h_sum = global_add_pool(h_tan, batch_idx)
                h_graph = torch.cat([h_mean, h_sum], dim=1)
            else:
                h_graph = global_add_pool(h_tan, batch_idx)

            graph_features.append(h_graph)

        # JK: 拼接所有层切空间向量
        jk_tangent = torch.cat(graph_features, dim=1)
        self._debug_stats("jk_tangent", jk_tangent)

        # Debug: 检查 jk_tangent 是否有 NaN
        if torch.isnan(jk_tangent).any():
            print(f'[encode_graph] jk_tangent has NaN! max={jk_tangent.abs().max()}')

        # JK 流形嵌入: 直接 expmap0 回流形（不压缩）
        jk_tangent_m = (
            _clip_tangent(jk_tangent, self.clip_r) if self.clip_r is not None else jk_tangent
        )
        jk_manifold = self.manifold.expmap0(jk_tangent_m, self.c)

        # 最后一层流形嵌入（也不压缩）
        last_in = (
            _clip_tangent(graph_features[-1], self.clip_r)
            if self.clip_r is not None
            else graph_features[-1]
        )
        last_manifold = self.manifold.expmap0(last_in, self.c)

        # reset debug flag
        self._debug_active = False

        return jk_manifold, jk_tangent, last_manifold

    def encode_task(self, task):
        """Few-shot 任务编码"""
        support_batch = Batch.from_data_list(task["support_set"]).to(self.device)
        query_batch = Batch.from_data_list(task["query_set"]).to(self.device)

        _, support_embs, _ = self.encode_graph(
            support_batch.x, support_batch.edge_index, support_batch.batch
        )
        _, query_embs, _ = self.encode_graph(
            query_batch.x, query_batch.edge_index, query_batch.batch
        )

        return support_embs, query_embs

    def get_curvature(self):
        return self.c


# ========== 向后兼容别名 ==========

HyperbolicGIN = HyperGIN
LorentzGIN = HyperGIN
PoincareGIN = HyperGIN
