import torch
import functools
from utils import model_utils
from subgraph_matching.models.gmn_baseline import (
    GMNBaseline,
    AGGREGATED, SET_ALIGNED,
    INTERACTION_POST, INTERACTION_PRE, INTERACTION_MSG_ONLY, INTERACTION_UPD_ONLY
)

class GMNIterativeRefinement(GMNBaseline):
    def __init__(self, refinement_steps: int, **kwargs):
        super(GMNIterativeRefinement, self).__init__(**kwargs)
        self.refinement_steps = refinement_steps

    def propagation_step_with_pre_interaction(
        self, from_idx, to_idx, node_features_enc, edge_features_enc, interaction_features
    ):
        combined_features = self.interaction_layer(
            torch.cat([node_features_enc, interaction_features], dim=-1)
        )

        features_input_to_msg = combined_features
        features_input_to_upd = combined_features
        # set the features for the other path as un-combined features
        if self.interaction_when == INTERACTION_MSG_ONLY:
            features_input_to_upd = node_features_enc
        elif self.interaction_when == INTERACTION_UPD_ONLY:
            features_input_to_msg = node_features_enc

        aggregated_messages = self.prop_layer._compute_aggregated_messages(
            features_input_to_msg, from_idx, to_idx, edge_features_enc
        )
        node_features_enc = self.prop_layer._compute_node_update(features_input_to_upd, [aggregated_messages])
        return node_features_enc

    def propagation_step_with_post_interaction(
        self, from_idx, to_idx, node_features_enc, edge_features_enc, interaction_features
    ):
        aggregated_messages = self.prop_layer._compute_aggregated_messages(
            node_features_enc, from_idx, to_idx, edge_features_enc
        )
        node_features_enc = self.prop_layer._compute_node_update(
            node_features_enc, [aggregated_messages, node_features_enc - interaction_features]
        )
        return node_features_enc

    def forward_with_alignment(self, graphs, graph_sizes, graph_adj_matrices):
        query_sizes, corpus_sizes = zip(*graph_sizes)
        query_sizes = torch.tensor(query_sizes, device=self.device)
        corpus_sizes = torch.tensor(corpus_sizes, device=self.device)
        padded_node_indices = model_utils.get_padded_indices(graph_sizes, self.max_node_set_size, self.device)

        features_to_transport_plan = functools.partial(
            model_utils.features_to_transport_plan,
            query_sizes=query_sizes, corpus_sizes=corpus_sizes,
            graph_size_to_mask_map=self.graph_size_to_mask_map
        )

        node_features, edge_features, from_idx, to_idx, graph_idx = model_utils.get_graph_features(graphs)

        encoded_node_features, encoded_edge_features = self.encoder(node_features, edge_features)
        num_nodes, node_feature_dim = encoded_node_features.shape

        node_feature_store = torch.zeros(num_nodes, node_feature_dim * (self.propagation_steps + 1), device=self.device)
        updated_node_feature_store = torch.zeros_like(node_feature_store)

        transport_plans = []

        for refine_idx in range(self.refinement_steps):
            node_features_enc, edge_features_enc = torch.clone(encoded_node_features), torch.clone(encoded_edge_features)

            for prop_idx in range(1, self.propagation_steps + 1):
                interaction_idx = node_feature_dim * prop_idx
                interaction_features = node_feature_store[:, interaction_idx - node_feature_dim : interaction_idx]

                node_features_enc = self.propagation_function(
                    from_idx, to_idx, node_features_enc, edge_features_enc, interaction_features
                )

                updated_node_feature_store[:, interaction_idx : interaction_idx + node_feature_dim] = torch.clone(node_features_enc)

            stacked_feature_store_query, stacked_feature_store_corpus = model_utils.split_and_stack(
                updated_node_feature_store, graph_sizes, self.max_node_set_size
            )
            final_features_query = stacked_feature_store_query[:, :, -node_feature_dim:]
            final_features_corpus = stacked_feature_store_corpus[:, :, -node_feature_dim:]

            transport_plan = features_to_transport_plan(
                final_features_query, final_features_corpus,
                preprocessor = self.interaction_alignment_preprocessor,
                alignment_function = self.interaction_alignment_function,
                what_for = 'interaction'
            )
            transport_plans.append(transport_plan)

            interleaved_node_features = model_utils.get_interaction_feature_store(
                transport_plan[0], stacked_feature_store_query, stacked_feature_store_corpus,
                reverse_transport_plan=transport_plan[1]
            )
            node_feature_store[:, node_feature_dim:] = interleaved_node_features[padded_node_indices, node_feature_dim:]

        ############################## SCORING ##############################
        if self.scoring == AGGREGATED:
            return self.aggregated_scoring(node_features_enc, graph_idx, graph_sizes)#, torch.stack(transport_plans, dim=1)
        elif self.scoring == SET_ALIGNED:
            return self.set_aligned_scoring(node_features_enc, graph_sizes, features_to_transport_plan)
