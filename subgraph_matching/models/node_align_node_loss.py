import torch
from utils import model_utils
from utils.tooling import ReadOnlyConfig
import GMN.graphembeddingnetwork as gmngen
from subgraph_matching.models._template import AlignmentModel

class NodeAlignNodeLoss(AlignmentModel):
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
        super(NodeAlignNodeLoss, self).__init__()
        self.max_node_set_size = max_node_set_size
        self.device = device

        self.graph_size_to_mask_map = model_utils.graph_size_to_mask_map(
            max_set_size=max_node_set_size, lateral_dim=sinkhorn_feature_dim, device=self.device
        )

        self.encoder = gmngen.GraphEncoder(**encoder_config)
        self.prop_layer = gmngen.GraphPropLayer(**propagation_layer_config)
        self.propagation_steps = propagation_steps

        self.sinkhorn_config = sinkhorn_config
        self.sinkhorn_feature_layers = torch.nn.Sequential(
            torch.nn.Linear(propagation_layer_config.node_state_dim, sinkhorn_feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(sinkhorn_feature_dim, sinkhorn_feature_dim)
        )

    def forward_with_alignment(self, graphs, graph_sizes, graph_adj_matrices):
        query_sizes, corpus_sizes = zip(*graph_sizes)
        query_sizes = torch.tensor(query_sizes, device=self.device)
        corpus_sizes = torch.tensor(corpus_sizes, device=self.device)

        node_features, edge_features, from_idx, to_idx, _ = model_utils.get_graph_features(graphs)

        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for _ in range(self.propagation_steps) :
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx, edge_features_enc)
        
        stacked_features_query, stacked_features_corpus = model_utils.split_and_stack(
            node_features_enc, graph_sizes, self.max_node_set_size
        )
        transformed_features_query = self.sinkhorn_feature_layers(stacked_features_query)
        transformed_features_corpus = self.sinkhorn_feature_layers(stacked_features_corpus)

        def mask_graphs(features, graph_sizes):
            mask = torch.stack([self.graph_size_to_mask_map[i] for i in graph_sizes])
            return mask * features
        masked_features_query = mask_graphs(transformed_features_query, query_sizes)
        masked_features_corpus = mask_graphs(transformed_features_corpus, corpus_sizes)

        sinkhorn_input = torch.matmul(masked_features_query, masked_features_corpus.permute(0, 2, 1))
        transport_plan = model_utils.sinkhorn_iters(log_alpha=sinkhorn_input, device=self.device, **self.sinkhorn_config)
        
        return model_utils.feature_alignment_score(stacked_features_query, stacked_features_corpus, transport_plan), [transport_plan, transport_plan.transpose(-1, -2)]
