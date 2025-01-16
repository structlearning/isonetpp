import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU
from torch_geometric.nn import GINConv
from torch_geometric.data import Batch
from utils import model_utils

class AttentionModule(torch.nn.Module):
    def __init__(self, dim_size):
        super(AttentionModule, self).__init__()
        self.dim_size = dim_size
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.dim_size, self.dim_size)) 
        self.weight_matrix1 = torch.nn.Parameter(torch.Tensor(self.dim_size, self.dim_size))

        channel = self.dim_size*1
        reduction = 4
        self.fc = nn.Sequential(
                        nn.Linear(channel,  channel // reduction),
                        nn.ReLU(inplace = True),
                        nn.Linear(channel // reduction, channel),
                        nn.Tanh()
                )

        self.fc1 =  nn.Linear(channel,  channel)

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, x, batch, size=None):
        attention = self.fc(x)
        x = attention * x + x

        size = batch[-1].item() + 1 if size is None else size # size is the quantity of batches: 128 eg
        mean = model_utils.unsorted_segment_sum(x, batch, size) # dim of mean: 128 * 16
        mean /= model_utils.unsorted_segment_sum(torch.ones_like(x), batch, size) # dim of mean: 128 * 16

        transformed_global = torch.tanh(torch.mm(mean, self.weight_matrix)) 
        coefs = torch.sigmoid((x * transformed_global[batch]).sum(dim=1)) # transformed_global[batch]: 1128 * 16; coefs: 1128 * 0
        weighted = coefs.unsqueeze(-1) * x 

        return model_utils.unsorted_segment_sum(weighted, batch, size) # 128 * 16


class SETensorNetworkModule(torch.nn.Module):
    def __init__(self, dim_size):
        super(SETensorNetworkModule, self).__init__()
        channel = dim_size*2
        reduction = 4
        self.fc_se = nn.Sequential(
                        nn.Linear(channel,  channel // reduction),
                        nn.ReLU(inplace = True),
                        nn.Linear(channel // reduction, channel),
                        nn.Sigmoid()
                )

        self.fc0 = nn.Sequential(
                        nn.Linear(channel,  channel),
                        nn.ReLU(inplace = True),
                        nn.Linear(channel, channel),
                        nn.ReLU(inplace = True)
                )

        self.fc1 = nn.Sequential(
                        nn.Linear(channel,  channel),
                        nn.ReLU(inplace = True),
                         nn.Linear(channel, dim_size // 2),
                        nn.ReLU(inplace = True)
                )

    def forward(self, embedding_1, embedding_2):

        combined_representation = torch.cat((embedding_1, embedding_2), 1)
        se_feat_coefs = self.fc_se(combined_representation)
        se_feat = se_feat_coefs * combined_representation + combined_representation
        scores = self.fc1(se_feat)

        return scores


class SEAttentionModule(torch.nn.Module):
    def __init__(self, dim_size):
        super(SEAttentionModule, self).__init__()
        channel = dim_size*1
        reduction = 4
        self.fc = nn.Sequential(
                        nn.Linear(channel,  channel // reduction),
                        nn.ReLU(inplace = True),
                        nn.Linear(channel // reduction, channel),
                        nn.Sigmoid()
                )

    def forward(self, x):
        x = self.fc(x)
        return x


class EGSC(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        max_node_set_size,
        max_edge_set_size,
        filters_1,
        filters_2,
        filters_3,
        bottle_neck_neurons,
        dropout,
        device
    ):
        super(EGSC, self).__init__()
        self.device = device
        self.filters_1 = filters_1
        self.filters_2 = filters_2
        self.filters_3 = filters_3
        self.bottle_neck_neurons = bottle_neck_neurons
        self.dropout = dropout

        self.feature_count = (self.filters_1 + self.filters_2 + self.filters_3 ) // 2

        nn1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, self.filters_1), 
            torch.nn.ReLU(), 
            torch.nn.Linear(self.filters_1, self.filters_1),
            torch.nn.BatchNorm1d(self.filters_1))

        nn2 = torch.nn.Sequential(
            torch.nn.Linear(self.filters_1, self.filters_2), 
            torch.nn.ReLU(), 
            torch.nn.Linear(self.filters_2, self.filters_2),
            torch.nn.BatchNorm1d(self.filters_2))

        nn3 = torch.nn.Sequential(
            torch.nn.Linear(self.filters_2, self.filters_3), 
            torch.nn.ReLU(), 
            torch.nn.Linear(self.filters_3, self.filters_3),
            torch.nn.BatchNorm1d(self.filters_3))

        self.convolution_1 = GINConv(nn1, train_eps=True)
        self.convolution_2 = GINConv(nn2, train_eps=True)
        self.convolution_3 = GINConv(nn3, train_eps=True)

        self.attention_level3 = AttentionModule(self.filters_3)

        self.attention_level2 = AttentionModule(self.filters_2)

        self.attention_level1 = AttentionModule(self.filters_1)

        self.tensor_network_level3 = SETensorNetworkModule(dim_size=self.filters_3)
        self.tensor_network_level2 = SETensorNetworkModule(dim_size=self.filters_2)
        self.tensor_network_level1 = SETensorNetworkModule(dim_size=self.filters_1)
        self.fully_connected_first = torch.nn.Linear(self.feature_count, self.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.bottle_neck_neurons, 1)

        self.score_attention = SEAttentionModule(self.feature_count)


    def convolutional_pass_level1(self, edge_index, features):
        """
        Making convolutional pass.
        """
        features = self.convolution_1(features, edge_index)
        features = F.relu(features)
        features_1 = F.dropout(features, p=self.dropout, training=self.training)
        return features_1

    def convolutional_pass_level2(self, edge_index, features):
        features_2 = self.convolution_2(features, edge_index)
        features_2 = F.relu(features_2)
        features_2 = F.dropout(features_2, p=self.dropout, training=self.training)
        return features_2

    def convolutional_pass_level3(self, edge_index, features):
        features_3 = self.convolution_3(features, edge_index)
        features_3 = F.relu(features_3)
        features_3 = F.dropout(features_3, p=self.dropout, training=self.training)
        return features_3

    def convolutional_pass_level4(self, edge_index, features):
        features_out = self.convolution_4(features, edge_index)
        return features_out

    def forward(self, graphs, graph_sizes, graph_adj_matrices):
        query_sizes, corpus_sizes = zip(*graph_sizes)
        query_sizes = torch.tensor(query_sizes, device=self.device)
        corpus_sizes = torch.tensor(corpus_sizes, device=self.device)

        query_graphs, corpus_graphs = zip(*graphs)
        query_batch = Batch.from_data_list(query_graphs)
        corpus_batch = Batch.from_data_list(corpus_graphs)


        edge_index_1 = query_batch.edge_index
        edge_index_2 = corpus_batch.edge_index
        features_1 = query_batch.x
        features_2 = corpus_batch.x
        batch_1 = query_batch.batch
        batch_2 = corpus_batch.batch

        features_level1_1 = self.convolutional_pass_level1(edge_index_1, features_1)
        features_level1_2 = self.convolutional_pass_level1(edge_index_2, features_2)
        pooled_features_level1_1 = self.attention_level1(features_level1_1, batch_1) # 128 * 64
        pooled_features_level1_2 = self.attention_level1(features_level1_2, batch_2) # 128 * 64
        scores_level1 = self.tensor_network_level1(pooled_features_level1_1, pooled_features_level1_2)

        features_level2_1 = self.convolutional_pass_level2(edge_index_1, features_level1_1)
        features_level2_2 = self.convolutional_pass_level2(edge_index_2, features_level1_2)

        pooled_features_level2_1 = self.attention_level2(features_level2_1, batch_1) # 128 * 32
        pooled_features_level2_2 = self.attention_level2(features_level2_2, batch_2) # 128 * 32
        scores_level2 = self.tensor_network_level2(pooled_features_level2_1, pooled_features_level2_2)

        features_level3_1 = self.convolutional_pass_level3(edge_index_1, features_level2_1)
        features_level3_2 = self.convolutional_pass_level3(edge_index_2, features_level2_2)
        pooled_features_level3_1 = self.attention_level3(features_level3_1, batch_1) # 128 * 16
        pooled_features_level3_2 = self.attention_level3(features_level3_2, batch_2) # 128 * 16
        scores_level3 = self.tensor_network_level3(pooled_features_level3_1, pooled_features_level3_2)

        scores = torch.cat((scores_level3, scores_level2, scores_level1), dim=1)

        scores = F.relu(self.fully_connected_first(self.score_attention(scores)*scores + scores))
        score = self.scoring_layer(scores).view(-1) # dim of score: 128 * 0

        return  score