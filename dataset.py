import json
import torch
import numpy as np
import pickle as pkl
from collections import defaultdict
from torch_geometric.data import Data, Batch

class Dataset:
    def __init__(self, name):
        self.dataset_name = name
        self.train_graphs = []
        self.test_graphs = []

        all_graphs, label_dict, tagset = load_data(self.dataset_name, True)

        self.tagset = tagset

        with open("data/{}/train_test_classes.json".format(self.dataset_name), "r") as f:
            all_class_splits = json.load(f)
            self.train_classes = all_class_splits["train"]
            self.test_classes = all_class_splits["test"]

        train_classes_mapping = {}
        for cl in self.train_classes:
            train_classes_mapping[cl] = len(train_classes_mapping)
        self.train_classes_num = len(train_classes_mapping)

        test_classes_mapping = {}
        for cl in self.test_classes:
            test_classes_mapping[cl] = len(test_classes_mapping)
        self.test_classes_num = len(test_classes_mapping)

        for i in range(len(all_graphs)):
            if all_graphs[i].y.item() in self.train_classes:
                self.train_graphs.append(all_graphs[i])

            if all_graphs[i].y.item() in self.test_classes:
                self.test_graphs.append(all_graphs[i])

        for graph in self.train_graphs:
            graph.y = torch.tensor([train_classes_mapping[graph.y.item()]], dtype=torch.long)
        for graph in self.test_graphs:
            graph.y = torch.tensor([test_classes_mapping[graph.y.item()]], dtype=torch.long)

        np.random.shuffle(self.train_graphs)

        self.train_tasks = defaultdict(list)
        for graph in self.train_graphs:
            self.train_tasks[graph.y.item()].append(graph)

        np.random.shuffle(self.test_graphs)

        self.test_tasks = defaultdict(list)
        for graph in self.test_graphs:
            self.test_tasks[graph.y.item()].append(graph)


        # 静态生成测试任务列表
        self.test_tasks_5shot = self._build_test_tasks(K_shot=5, query_size=10)
        self.test_tasks_10shot = self._build_test_tasks(K_shot=10, query_size=10)

    def _build_test_tasks(self, K_shot, query_size):
        """构建测试任务列表"""
        class_indices = list(range(self.test_classes_num))
        total_query_size = len(class_indices) * query_size

        # Support set: 每个类别前 K_shot 个（所有 task 共用）
        support_set = []
        query_pool = []
        for class_idx in class_indices:
            g_list = list(self.test_tasks[class_idx])
            support_set.extend(g_list[:K_shot])
            query_pool.extend(g_list[K_shot:])

        # shuffle query pool 一次
        from numpy.random import RandomState
        rd = RandomState(0)
        rd.shuffle(query_pool)

        # 按顺序生成 tasks
        tasks = []
        pool_len = len(query_pool)
        num_full_tasks = pool_len // total_query_size
        remainder = pool_len % total_query_size

        # 生成完整任务
        for i in range(num_full_tasks):
            start_idx = i * total_query_size
            query_set = query_pool[start_idx:start_idx + total_query_size]
            tasks.append({'support_set': support_set, 'query_set': query_set})

        # 最后一个任务：如果有剩余，从开头补样本
        if remainder > 0:
            start_idx = num_full_tasks * total_query_size
            query_set = query_pool[start_idx:] + query_pool[:total_query_size - remainder]
            tasks.append({'support_set': support_set, 'query_set': query_set})

        return tasks




def load_data(dataset, degree_as_tag):
    """加载图数据集，返回 PyG Data 对象列表"""
    print('loading data')
    def _build_edge_index(edges):
        """从边列表构建双向 edge_index"""
        if not edges:
            return torch.empty(2, 0, dtype=torch.long)
        edges = edges + [[j, i] for i, j in edges]  # 添加反向边
        return torch.LongTensor(edges).t().contiguous()

    if dataset in ['Letter_high', 'ENZYMES', 'Reddit', 'TRIANGLES']:
        data_list = []
        label_dict = {}
        all_node_tags = []  # 存储每个图的 node_tags

        with open(f'data/{dataset}/{dataset}.txt', 'r') as f:
            n_g = int(f.readline().strip())
            for _ in range(n_g):
                n, label = [int(w) for w in f.readline().strip().split()]
                if label not in label_dict:
                    label_dict[label] = len(label_dict)
                
                edges = []
                node_degrees = []
                for node_id in range(n):
                    row = [int(w) for w in f.readline().strip().split()]
                    degree, neighbors = row[1], row[2:2+row[1]]
                    node_degrees.append(degree)
                    edges.extend([[node_id, nb] for nb in neighbors])
                
                all_node_tags.append(node_degrees if degree_as_tag else None)
                data_list.append(Data(
                    edge_index=_build_edge_index(edges),
                    y=torch.tensor([label], dtype=torch.long)
                ))

        # 构建 node features (one-hot of degree)
        tagset = sorted(set(tag for tags in all_node_tags for tag in tags))
        tag2index = {tag: i for i, tag in enumerate(tagset)}
        
        for data, node_tags in zip(data_list, all_node_tags):
            data.x = torch.zeros(len(node_tags), len(tagset))
            data.x[range(len(node_tags)), [tag2index[t] for t in node_tags]] = 1

        print(f'# classes: {len(label_dict)}')
        print(f'# node features: {len(tagset)}')
        print(f'# graphs: {len(data_list)}\n')

        return data_list, label_dict, tagset

    elif dataset in ['R52', 'COIL']:
        print(dataset)
        node_attrs = pkl.load(open(f'data/{dataset}/{dataset}_node_attributes.pickle', 'rb'))
        
        data_list = []
        for split in ['train', 'val', 'test']:
            split_data = pkl.load(open(f'data/{dataset}/{dataset}_{split}_set.pickle', 'rb'))
            graph2nodes = split_data["graph2nodes"]
            graph2edges = split_data['graph2edges']
            
            for label, graph_ids in split_data['label2graphs'].items():
                for gid in graph_ids:
                    nodes = list(graph2nodes[gid])
                    node_map = {n: i for i, n in enumerate(nodes)}
                    
                    edges = [[node_map[e[0]], node_map[e[1]]] for e in graph2edges[gid]]
                    x = torch.FloatTensor(node_attrs[graph2nodes[gid]])
                    if dataset == 'R52':
                        x = x.unsqueeze(-1)
                    
                    data_list.append(Data(
                        x=x,
                        edge_index=_build_edge_index(edges),
                        y=torch.tensor([label], dtype=torch.long)
                    ))

        print(f'# graphs: {len(data_list)}\n')
        return data_list, None, None
    
    elif dataset == 'ogbg-ppa':
        from ogb.graphproppred import GraphPropPredDataset
        ogb_dataset = GraphPropPredDataset(name=dataset)

        data_list = [
            Data(
                x=torch.FloatTensor(graph['node_feat']),
                edge_index=torch.LongTensor(graph['edge_index']),
                y=torch.tensor([label], dtype=torch.long)
            )
            for graph, label in ogb_dataset
        ]

        return data_list, {i: i for i in range(37)}, list(range(37))
