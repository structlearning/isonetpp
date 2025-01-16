import torch
from abc import ABC, abstractmethod

class AlignmentModel(torch.nn.Module, ABC):
    def __init__(self):
        super(AlignmentModel, self).__init__()

    @abstractmethod
    def forward_with_alignment(self, graphs, graph_sizes, graph_adj_matrices):
        pass

    def forward(self, graphs, graph_sizes, graph_adj_matrices):
        return self.forward_with_alignment(graphs, graph_sizes, graph_adj_matrices)[0]

    def forward_for_alignment(self, graphs, graph_sizes, graph_adj_matrices):
        return self.forward_with_alignment(graphs, graph_sizes, graph_adj_matrices)[1]

class EncodingLayer(torch.nn.Module, ABC):
    def __init__(self):
        super(EncodingLayer, self).__init__()

    @abstractmethod
    def forward(self, graphs, batch_size):
        pass

class InteractionLayer(torch.nn.Module, ABC):
    def __init__(self):
        super(InteractionLayer, self).__init__()

    @abstractmethod
    def forward(self, query_features, corpus_features, batch_size):
        pass

# TODO: think of a better name; supposed to be the RQ0 class
class EncodeThenInteractModel(torch.nn.Module, ABC):
    def __init__(self):
        super(EncodeThenInteractModel, self).__init__()
        self.encoding_layer: EncodingLayer = None
        self.interaction_layer: InteractionLayer = None

    @abstractmethod
    def forward(self, graphs, graph_sizes, graph_adj_matrices):
        pass