import torch
import GMN.utils as gmnutils
import torch.nn.functional as F
from GMN.loss import euclidean_distance
import GMN.graphembeddingnetwork as gmngen
from utils.tooling import ReadOnlyConfig
from utils import model_utils


class GMN_embed_hinge(torch.nn.Module):
    def __init__(
        self,
        encoder_config: ReadOnlyConfig,
        propagation_layer_config: ReadOnlyConfig,
        aggregator_config: ReadOnlyConfig,
        max_edge_set_size,
        max_node_set_size,
        propagation_steps,
        device
    ):
        super(GMN_embed_hinge, self).__init__()
        self.propagation_steps = propagation_steps        
        self.encoder = gmngen.GraphEncoder(**encoder_config)
        self.prop_layer = gmngen.GraphPropLayer(**propagation_layer_config)      
        self.aggregator = gmngen.GraphAggregator(**aggregator_config)

    def forward(self, graphs, graph_sizes, graph_adj_matrices):
        """
        """
        node_features, edge_features, from_idx, to_idx, graph_idx = model_utils.get_graph_features(graphs)

        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for i in range(self.propagation_steps) :
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)
            
        graph_vectors = self.aggregator(node_features_enc,graph_idx,2*len(graph_sizes) )
        x, y = gmnutils.reshape_and_split_tensor(graph_vectors, 2)
        scores = -torch.sum(torch.nn.ReLU()(x-y),dim=-1)
        return scores
