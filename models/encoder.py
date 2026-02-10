import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from .hyperbolic import EuclideanManifold


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
        num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
        input_dim: dimensionality of input features
        hidden_dim: dimensionality of hidden units at ALL layers
        output_dim: number of classes for prediction
        device: which device to use
        """
        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class GIN(nn.Module):
    def __init__(
        self,
        num_layers=5,
        num_mlp_layers=2,
        input_dim=200,
        hidden_dim=128,
        final_dropout=0.5,
        learn_eps=True,
        graph_pooling_type="sum",
        neighbor_pooling_type="sum",
        device=None,
        latent_dim=128,
        **kwargs,
    ):
        super(GIN, self).__init__()

        self.final_dropout = final_dropout
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # HyperGIN-compatible attributes
        self.manifold = EuclideanManifold()
        self.manifold_type = "euclidean"
        self.register_buffer("c", torch.tensor([1.0]))
        self.hidden_dim = hidden_dim
        self.debug_nan = False
        self._debug_calls = 0
        self._debug_active = False

        self.num_layers = num_layers  # number of conv layers (same convention as HyperGIN)
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.latent_dim = latent_dim
        self.learn_eps = learn_eps

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers):
            in_dim = input_dim if layer == 0 else hidden_dim
            mlp = MLP(num_mlp_layers, in_dim, hidden_dim, hidden_dim)
            self.convs.append(GINConv(mlp, train_eps=learn_eps, eps=0.1))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Alias for compatibility with diagnostics that iterate encoder.layers
        self.layers = self.convs

        self.raw_dim = self.num_layers * hidden_dim
        self.to(self.device)

    def forward(self, x, edge_index):
        """前向传播，返回每层隐藏表示（与HyperGIN接口一致）"""
        hidden_rep = []
        h = x
        for layer in range(self.num_layers):
            h = self.convs[layer](h, edge_index)
            h = self.batch_norms[layer](h)
            h = F.relu(h)
            hidden_rep.append(h)
        return hidden_rep

    def _to_adj(self, edge_index, num_nodes):
        """兼容 HyperGIN 的 _to_adj 方法"""
        return torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.size(1), device=self.device),
            (num_nodes, num_nodes),
        )

    def _to_manifold(self, x):
        """欧氏空间无需映射"""
        return x

    def encode_graph(self, x, edge_index, batch_idx):
        """编码图数据，返回3元组（与HyperGIN接口一致）

        Returns:
            jk_manifold: 欧氏空间中等同于 jk_tangent
            jk_tangent: JK拼接所有层池化结果
            last_manifold: 最后一层池化结果
        """
        hidden_rep = self.forward(x, edge_index)

        pooled_layers = []
        for h in hidden_rep:
            if self.graph_pooling_type == "mean":
                pooled = global_mean_pool(h, batch_idx)
            elif self.graph_pooling_type == "max":
                pooled = global_max_pool(h, batch_idx)
            else:
                pooled = global_add_pool(h, batch_idx)
            pooled_layers.append(pooled)

        jk_tangent = torch.cat(pooled_layers, dim=1)
        last_manifold = pooled_layers[-1]
        jk_manifold = jk_tangent

        return jk_manifold, jk_tangent, last_manifold

    def encode_task(self, task):
        support_graphs = task["support_set"]
        query_graphs = task["query_set"]

        if isinstance(support_graphs, Batch):
            sb = support_graphs
        elif isinstance(support_graphs, list):
            sb = Batch.from_data_list(support_graphs).to(self.device)
        else:
            sb = support_graphs

        if isinstance(query_graphs, Batch):
            qb = query_graphs
        elif isinstance(query_graphs, list):
            qb = Batch.from_data_list(query_graphs).to(self.device)
        else:
            qb = query_graphs

        _, support_embs, _ = self.encode_graph(sb.x, sb.edge_index, sb.batch)
        _, query_embs, _ = self.encode_graph(qb.x, qb.edge_index, qb.batch)

        return support_embs, query_embs

    def get_curvature(self):
        return self.c
