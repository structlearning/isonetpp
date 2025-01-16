import torch
import torch.nn.functional as F
from utils import model_utils
from subgraph_matching.models._template import AlignmentModel
from utils.tooling import ReadOnlyConfig
import GMN.graphembeddingnetwork as gmngen
from subgraph_matching.models.gmn_baseline import INTERACTION_POST, INTERACTION_PRE

class EdgeEarlyInteractionBaseline(AlignmentModel):
    def __init__(
        self,
        max_node_set_size,
        max_edge_set_size,
        encoder_config: ReadOnlyConfig,
        propagation_layer_config: ReadOnlyConfig,
        propagation_steps,
        sinkhorn_config: ReadOnlyConfig,
        sinkhorn_feature_dim,
        device,
        interaction_when: str = INTERACTION_POST,
    ):
        super(EdgeEarlyInteractionBaseline, self).__init__()
        self.max_node_set_size = max_node_set_size
        self.max_edge_set_size = max_edge_set_size
        self.interaction_when = interaction_when
        self.device = device

        self.graph_size_to_mask_map = model_utils.graph_size_to_mask_map(
            max_set_size=max_edge_set_size, lateral_dim=sinkhorn_feature_dim, device=self.device
        )

        self.encoder = gmngen.GraphEncoder(**encoder_config)
        self.prop_layer = gmngen.GraphPropLayer(**propagation_layer_config)
        self.propagation_steps = propagation_steps

        self.message_dim = propagation_layer_config.edge_hidden_sizes[-1]
        assert self.message_dim == encoder_config.edge_hidden_sizes[-1], (
            "if message_dim != dim of edge encoding, can't use initial edge encoding directly"
        )
        interaction_input_dim = self.message_dim * 2
        interaction_output_dim = propagation_layer_config.edge_embedding_dim
        self.interaction_encoder = torch.nn.Sequential(
            torch.nn.Linear(interaction_input_dim, interaction_input_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(interaction_input_dim, interaction_output_dim)
        )

        self.sinkhorn_config = sinkhorn_config
        self.sinkhorn_feature_layers = torch.nn.Sequential(
            torch.nn.Linear(self.message_dim, sinkhorn_feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(sinkhorn_feature_dim, sinkhorn_feature_dim)
        )

    def forward_with_alignment(self, graphs, graph_sizes, graph_adj_matrices):
        query_sizes, corpus_sizes = zip(*graph_sizes)
        query_sizes = torch.tensor(query_sizes, device=self.device)
        corpus_sizes = torch.tensor(corpus_sizes, device=self.device)

        node_features, edge_features, from_idx, to_idx, graph_idx = model_utils.get_graph_features(graphs)

        paired_edge_counts = model_utils.get_paired_edge_counts(
            from_idx, to_idx, graph_idx, 2*len(graph_sizes)
        )

        # Encode node and edge features
        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        num_edges, edge_feature_dim = edge_features_enc.shape

        padded_edge_indices = model_utils.get_padded_indices(paired_edge_counts, self.max_edge_set_size, self.device)

        interaction_features = torch.zeros_like(edge_features_enc)

        transport_plans = []
        for prop_idx in range(1, self.propagation_steps + 1):
            if self.interaction_when == INTERACTION_PRE:
                edge_features_enc = self.interaction_encoder(torch.cat([edge_features_enc, interaction_features], dim=-1))

            # Message propagation on combined features
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx, edge_features_enc)

            edge_features_enc = model_utils.propagation_messages(
                propagation_layer=self.prop_layer,
                node_features=node_features_enc,
                edge_features=edge_features_enc,
                from_idx=from_idx,
                to_idx=to_idx
            )

            if self.interaction_when == INTERACTION_POST:
                edge_features_enc = self.interaction_encoder(torch.cat([edge_features_enc, interaction_features], dim=-1))

            stacked_edge_features_query, stacked_edge_features_corpus = model_utils.split_and_stack(
                edge_features_enc, paired_edge_counts, self.max_edge_set_size
            )

            transformed_features_query = self.sinkhorn_feature_layers(stacked_edge_features_query)
            transformed_features_corpus = self.sinkhorn_feature_layers(stacked_edge_features_corpus)

            def mask_graphs(features, graph_sizes):
                mask = torch.stack([self.graph_size_to_mask_map[i] for i in graph_sizes])
                return mask * features
            num_edges_query = map(lambda pair: pair[0], paired_edge_counts)
            masked_features_query = mask_graphs(transformed_features_query, num_edges_query)
            num_edges_corpus = map(lambda pair: pair[1], paired_edge_counts)
            masked_features_corpus = mask_graphs(transformed_features_corpus, num_edges_corpus)

            sinkhorn_input = torch.matmul(masked_features_query, masked_features_corpus.permute(0, 2, 1))
            transport_plan = model_utils.sinkhorn_iters(log_alpha=sinkhorn_input, device=self.device, **self.sinkhorn_config)

            # Compute interaction-based features
            interleaved_edge_features = model_utils.get_interaction_feature_store(
                transport_plan, stacked_edge_features_query, stacked_edge_features_corpus
            )
            interaction_features = interleaved_edge_features[padded_edge_indices, :]
            
            transport_plans.append(transport_plan)

        score = model_utils.feature_alignment_score(stacked_edge_features_query, stacked_edge_features_corpus, transport_plan)

        return score, torch.stack(transport_plans, dim=1)
