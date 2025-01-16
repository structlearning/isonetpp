import subgraph_matching.dataset as dataset

from subgraph_matching.models.node_align_node_loss import NodeAlignNodeLoss
from subgraph_matching.models.isonet import ISONET
from subgraph_matching.models.edge_early_interaction_baseline import EdgeEarlyInteractionBaseline
from subgraph_matching.models.edge_early_interaction import EdgeEarlyInteraction
from subgraph_matching.models.gmn_baseline import GMNBaseline
from subgraph_matching.models.gmn_iterative_refinement import GMNIterativeRefinement
from subgraph_matching.models.graphsim import GraphSim
from subgraph_matching.models.egsc import EGSC
from subgraph_matching.models.h2mn import H2MN
from subgraph_matching.models.greed import Greed
from subgraph_matching.models.gotsim import GOTSim
from subgraph_matching.models.gmn_embed import GMN_embed_hinge
from subgraph_matching.models.greed import Greed
from subgraph_matching.models.neuromatch import NeuroMatch
from subgraph_matching.models.simgnn import SimGNN
from subgraph_matching.models.gmn_embed import GMN_embed_hinge
from subgraph_matching.models.greed import Greed

model_name_to_class_mappings = {
    'node_align_node_loss': NodeAlignNodeLoss,
    'isonet': ISONET,
    'edge_early_interaction': EdgeEarlyInteraction,
    'edge_early_interaction_baseline': EdgeEarlyInteractionBaseline,
    'graphsim': GraphSim,
    'egsc': EGSC,
    'H2MN': H2MN,
    'gotsim': GOTSim,
    'gmn_embed': GMN_embed_hinge,
    'neuromatch': NeuroMatch,
    'greed': Greed,
    'simgnn': SimGNN,
    'gmn_embed': GMN_embed_hinge,
}

def get_model_names():
    return list(model_name_to_class_mappings.keys())

def get_model(model_name, config, max_node_set_size, max_edge_set_size, device):
    if model_name.startswith('gmn_baseline'):
        model_class = GMNBaseline
    elif model_name.startswith('gmn_iterative_refinement'):
        model_class = GMNIterativeRefinement
    elif model_name.startswith('edge_early_interaction_baseline'):
        model_class = EdgeEarlyInteractionBaseline
    elif model_name.startswith('edge_early_interaction'):
        model_class = EdgeEarlyInteraction
    elif model_name.startswith('isonet'):
        model_class = ISONET
    else:
        model_class = model_name_to_class_mappings[model_name]

    return model_class(
        max_node_set_size=max_node_set_size,
        max_edge_set_size=max_edge_set_size,
        device=device,
        **config
    )

def get_data_type_for_model(model_name):
    if model_name in ['graphsim', 'egsc', 'gotsim', 'H2MN', 'greed', 'neuromatch', 'simgnn']:
        return dataset.PYG_DATA_TYPE
    return dataset.GMN_DATA_TYPE
