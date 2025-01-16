import os
import copy
import math
import pickle
import random
import collections
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

TRAIN_MODE = "train"
VAL_MODE = "val"
TEST_MODE = "test"
BROAD_TEST_MODE = "Extra_test_300"

GMN_DATA_TYPE = "gmn"
PYG_DATA_TYPE = "pyg"

GraphCollection = collections.namedtuple(
    'GraphCollection', 
    ['from_idx', 'to_idx', 'node_features', 'edge_features', 'graph_idx', 'num_graphs']
)

class SubgraphIsomorphismDataset:
    def __init__(self, mode, dataset_name, dataset_size, batch_size, data_type, dataset_base_path, experiment, dataset_path_override=None, device=None):
        assert mode in [TRAIN_MODE, VAL_MODE, TEST_MODE, BROAD_TEST_MODE]
        self.mode = mode
        self.dataset_name = dataset_name
        self.dataset_size = dataset_size
        self.max_node_set_size = {"small": 15, "large": 20}[dataset_size]
        self.batch_size = batch_size
        self.data_type = data_type
        self.dataset_base_path = dataset_base_path
        self.device = experiment.device if experiment else (device if device else 'cuda:0')
        self.batch_setting = None
        self.dataset_path_override = dataset_path_override

        self.load_graphs(experiment=experiment)
        self.preprocess_subgraphs_to_pyG_data()
        self.build_adjacency_info()

        self.max_edge_set_size = max(
            max([graph.number_of_edges() for graph in self.query_graphs]),
            max([graph.number_of_edges() for graph in self.corpus_graphs])
        )

    def load_graphs(self, experiment):
        dataset_accessor = lambda file_name: os.path.join(
            self.dataset_base_path, self.dataset_path_override or f"{self.dataset_size}_dataset",
            "splits", self.mode, file_name
        )
        
        # Load query graphs
        pair_count = f"{80 if self.dataset_size == 'small' else 240}k"
        mode_prefix = "test" if "test" in self.mode else self.mode
        query_graph_file = dataset_accessor(f"{mode_prefix}_{self.dataset_name}{pair_count}_query_subgraphs.pkl")
        self.query_graphs = pickle.load(open(query_graph_file, 'rb'))
        num_query_graphs = len(self.query_graphs)
        if experiment:
            experiment.log("loaded %s query graphs from %s", self.mode, query_graph_file)

        # Load subgraph isomorphism relationships of query vs corpus graphs
        relationships_file = query_graph_file.replace("query_subgraphs", "rel_nx_is_subgraph_iso")
        self.relationships = pickle.load(open(relationships_file, 'rb'))
        if experiment:
            experiment.log("loaded %s relationships from %s", self.mode, relationships_file)

        assert list(self.relationships.keys()) == list(range(num_query_graphs))

        # Load corpus graphs
        corpus_graph_file = os.path.join(
            os.path.dirname(os.path.dirname(query_graph_file)),
            f"{self.dataset_name}{pair_count}_corpus_subgraphs.pkl"
        )
        self.corpus_graphs = pickle.load(open(corpus_graph_file, 'rb'))
        if experiment:
            experiment.log("loaded corpus graphs from %s", corpus_graph_file)

        self.pos_pairs, self.neg_pairs = [], []
        for query_idx in range(num_query_graphs):
            for corpus_idx in self.relationships[query_idx]['pos']:
                self.pos_pairs.append((query_idx, corpus_idx))
            for corpus_idx in self.relationships[query_idx]['neg']:
                self.neg_pairs.append((query_idx, corpus_idx))

    def create_pyG_object(self, graph):
        num_nodes = graph.number_of_nodes()
        features = torch.ones(num_nodes, 1, dtype=torch.float, device=self.device)

        edges = list(graph.edges)
        doubled_edges = [[x, y] for (x, y) in edges] + [[y, x] for (x, y) in edges]
        edge_index = torch.tensor(np.array(doubled_edges).T, dtype=torch.int64, device=self.device)
        return Data(x = features, edge_index = edge_index), num_nodes

    def preprocess_subgraphs_to_pyG_data(self):
        self.query_graph_data, self.query_graph_sizes = zip(
            *[self.create_pyG_object(query_graph) for query_graph in self.query_graphs]
        )
        self.corpus_graph_data, self.corpus_graph_sizes = zip(
            *[self.create_pyG_object(corpus_graph) for corpus_graph in self.corpus_graphs]
        )
    
    def build_adjacency_info(self):
        def adj_list_from_graph_list(graphs):
            adj_list = []
            for graph in graphs:
                unpadded_adj = torch.tensor(nx.adjacency_matrix(graph).todense(), dtype=torch.float, device=self.device)
                assert unpadded_adj.shape[0] == unpadded_adj.shape[1]
                num_nodes = len(unpadded_adj)
                padded_adj = F.pad(unpadded_adj, pad = (0, self.max_node_set_size - num_nodes, 0, self.max_node_set_size - num_nodes))
                adj_list.append(padded_adj)
            return adj_list

        self.query_adj_list = adj_list_from_graph_list(self.query_graphs)
        self.corpus_adj_list = adj_list_from_graph_list(self.corpus_graphs)
    
    def _pack_batch(self, graphs):
        from_idx = []
        to_idx = []
        graph_idx = []
        all_graphs = [individual_graph for graph_tuple in graphs for individual_graph in graph_tuple]

        total_nodes, total_edges = 0, 0
        for idx, graph in enumerate(all_graphs):
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            edges = np.array(graph.edges(), dtype=np.int32)
            
            from_idx.append(edges[:, 0] + total_nodes)
            to_idx.append(edges[:, 1] + total_nodes)
            graph_idx.append(np.ones(num_nodes, dtype=np.int32) * idx)

            total_nodes += num_nodes
            total_edges += num_edges
        
        return GraphCollection(
            from_idx = torch.tensor(np.concatenate(from_idx, axis=0), dtype=torch.int64, device=self.device),
            to_idx = torch.tensor(np.concatenate(to_idx, axis=0), dtype=torch.int64, device=self.device),
            graph_idx = torch.tensor(np.concatenate(graph_idx, axis=0), dtype=torch.int64, device=self.device),
            num_graphs = len(all_graphs),
            node_features = torch.ones(total_nodes, 1, dtype=torch.float, device=self.device),
            edge_features = torch.ones(total_edges, 1, dtype=torch.float, device=self.device)
        )

    def create_stratified_batches(self):
        self.batch_setting = 'stratified'
        random.shuffle(self.pos_pairs), random.shuffle(self.neg_pairs)
        pos_to_neg_ratio = len(self.pos_pairs) / len(self.neg_pairs)

        num_pos_per_batch = math.ceil(pos_to_neg_ratio/(1 + pos_to_neg_ratio) * self.batch_size)
        num_neg_per_batch = self.batch_size - num_pos_per_batch

        batches_pos, batches_neg = [], []
        labels_pos, labels_neg = [], []
        for idx in range(0, len(self.pos_pairs), num_pos_per_batch):
            elements_remaining = len(self.pos_pairs) - idx
            elements_chosen = min(num_pos_per_batch, elements_remaining)
            batches_pos.append(self.pos_pairs[idx : idx + elements_chosen])
            labels_pos.append([1.0] * elements_chosen)
        for idx in range(0, len(self.neg_pairs), num_neg_per_batch):
            elements_remaining = len(self.neg_pairs) - idx
            elements_chosen = min(num_neg_per_batch, elements_remaining)
            batches_neg.append(self.neg_pairs[idx : idx + elements_chosen])
            labels_neg.append([0.0] * elements_chosen)

        self.num_batches = min(len(batches_pos), len(batches_neg))
        self.batches = [pos + neg for (pos, neg) in zip(batches_pos[:self.num_batches], batches_neg[:self.num_batches])]
        self.labels = [pos + neg for (pos, neg) in zip(labels_pos[:self.num_batches], labels_neg[:self.num_batches])]

        return self.num_batches

    def create_custom_batches(self, pair_list):
        self.batch_setting = 'custom'
        self.batches = []
        for idx in range(0, len(pair_list), self.batch_size):
            self.batches.append(pair_list[idx : idx + self.batch_size])
        
        self.num_batches = len(self.batches)
        return self.num_batches

    def fetch_batch_by_id(self, idx):
        assert idx < self.num_batches
        batch = self.batches[idx]

        query_graph_idxs, corpus_graph_idxs = zip(*batch)
        
        if self.data_type == GMN_DATA_TYPE:
            query_graphs = [self.query_graphs[idx] for idx in query_graph_idxs]
            corpus_graphs = [self.corpus_graphs[idx] for idx in corpus_graph_idxs]
            all_graphs = self._pack_batch(zip(query_graphs, corpus_graphs))
        elif self.data_type == PYG_DATA_TYPE:
            query_graphs = [self.query_graph_data[idx] for idx in query_graph_idxs]
            corpus_graphs = [self.corpus_graph_data[idx] for idx in corpus_graph_idxs]
            all_graphs = list(zip(query_graphs, corpus_graphs))

        query_graph_sizes = [self.query_graph_sizes[idx] for idx in query_graph_idxs]
        corpus_graph_sizes = [self.corpus_graph_sizes[idx] for idx in corpus_graph_idxs]
        all_graph_sizes = list(zip(query_graph_sizes, corpus_graph_sizes))

        query_graph_adjs = [self.query_adj_list[idx] for idx in query_graph_idxs]
        corpus_graph_adjs = [self.corpus_adj_list[idx] for idx in corpus_graph_idxs]
        all_graph_adjs = list(zip(query_graph_adjs, corpus_graph_adjs))

        if self.batch_setting == 'stratified':
            target = torch.tensor(np.array(self.labels[idx]), dtype=torch.float, device=self.device)
            return all_graphs, all_graph_sizes, target, all_graph_adjs
        elif self.batch_setting == 'custom':
            return all_graphs, all_graph_sizes, None, all_graph_adjs
        else:
            raise NotImplementedError

def get_datasets(dataset_config, experiment, data_type, modes=['train', 'val']):
    return {
        mode: SubgraphIsomorphismDataset(
            mode = mode, experiment = experiment,
            data_type = data_type, **dataset_config
        ) for mode in modes
    }