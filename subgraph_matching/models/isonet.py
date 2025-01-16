import torch
import torch.nn.functional as F
from utils import model_utils
from subgraph_matching.models._template import AlignmentModel
from utils.tooling import ReadOnlyConfig
import GMN.graphembeddingnetwork as gmngen

class ISONET(torch.nn.Module):
    def __init__(
        self,
        max_node_set_size,
        max_edge_set_size,
        encoder_config: ReadOnlyConfig,
        propagation_layer_config: ReadOnlyConfig,
        propagation_steps,
        sinkhorn_config: ReadOnlyConfig,
        sinkhorn_feature_dim,
        device
    ):
        super(ISONET, self).__init__()
        self.max_edge_set_size = max_edge_set_size
        self.device = device

        self.graph_size_to_mask_map = model_utils.graph_size_to_mask_map(
            max_set_size=max_edge_set_size, lateral_dim=sinkhorn_feature_dim, device=self.device
        )

        self.encoder = gmngen.GraphEncoder(**encoder_config)
        self.prop_layer = gmngen.GraphPropLayer(**propagation_layer_config)
        self.propagation_steps = propagation_steps

        self.sinkhorn_config = sinkhorn_config
        self.sinkhorn_feature_layers = torch.nn.Sequential(
            torch.nn.Linear(propagation_layer_config.edge_hidden_sizes[-1], sinkhorn_feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(sinkhorn_feature_dim, sinkhorn_feature_dim)
        )

    def forward(self, graphs, graph_sizes, graph_adj_matrices):
        node_features, edge_features, from_idx, to_idx, graph_idx = model_utils.get_graph_features(graphs)

        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for _ in range(self.propagation_steps) :
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx, edge_features_enc)
        
        edge_features_enc = model_utils.propagation_messages(
            propagation_layer=self.prop_layer,
            node_features=node_features_enc,
            edge_features=edge_features_enc,
            from_idx=from_idx,
            to_idx=to_idx
        )

        paired_edge_counts = model_utils.get_paired_edge_counts(
            from_idx=from_idx, to_idx=to_idx,
            graph_idx=graph_idx, num_graphs=2*len(graph_sizes)
        )

        stacked_features_query, stacked_features_corpus = model_utils.split_and_stack(edge_features_enc, paired_edge_counts, self.max_edge_set_size)
        transformed_features_query = self.sinkhorn_feature_layers(stacked_features_query)
        transformed_features_corpus = self.sinkhorn_feature_layers(stacked_features_corpus)

        def mask_graphs(features, graph_sizes):
            mask = torch.stack([self.graph_size_to_mask_map[i] for i in graph_sizes])
            return mask * features
        num_edges_query = map(lambda pair: pair[0], paired_edge_counts)
        masked_features_query = mask_graphs(transformed_features_query, num_edges_query)
        num_edges_corpus = map(lambda pair: pair[1], paired_edge_counts)
        masked_features_corpus = mask_graphs(transformed_features_corpus, num_edges_corpus)

        sinkhorn_input = torch.matmul(masked_features_query, masked_features_corpus.permute(0, 2, 1))
        transport_plan = model_utils.sinkhorn_iters(log_alpha=sinkhorn_input, device=self.device, **self.sinkhorn_config)
        
        return model_utils.feature_alignment_score(stacked_features_query, stacked_features_corpus, transport_plan)