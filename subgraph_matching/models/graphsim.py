import torch
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Batch
from torch.nn.utils.rnn import  pad_sequence


class CNNLayerV1(torch.nn.Module):
    def __init__(self, kernel_size, stride, in_channels, out_channels, num_similarity_matrices):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_similarity_matrices = num_similarity_matrices
        padding_temp = (self.kernel_size - 1)//2
        if self.kernel_size%2 == 0:
            self.padding = torch.nn.ZeroPad2d((padding_temp, padding_temp+1, padding_temp, padding_temp+1))
        else:
            self.padding = torch.nn.ZeroPad2d((padding_temp, padding_temp, padding_temp, padding_temp))
        self.layers = torch.nn.ModuleList([torch.nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                               kernel_size=self.kernel_size, stride=stride) for i in range(num_similarity_matrices)])
        
    def forward(self, similarity_matrices_list):
        result = []
        for i in range(self.num_similarity_matrices):
            result.append(self.layers[i](self.padding(similarity_matrices_list[i])))
        return result
    
class MaxPoolLayerV1(torch.nn.Module):
    def __init__(self, stride, pool_size, num_similarity_matrices):
        super().__init__()
        self.stride = stride
        self.pool_size = pool_size
        self.num_similarity_matrices = num_similarity_matrices
        padding_temp = (self.pool_size - 1)//2
        if self.pool_size%2 == 0:
            self.padding = torch.nn.ZeroPad2d((padding_temp, padding_temp+1, padding_temp, padding_temp+1))
        else:
            self.padding = torch.nn.ZeroPad2d((padding_temp, padding_temp, padding_temp, padding_temp))
        self.layers = torch.nn.ModuleList([torch.nn.MaxPool2d(kernel_size=self.pool_size, stride=stride) for i in range(num_similarity_matrices)])
        
    def forward(self, similarity_matrices_list):
        result = []
        for i in range(self.num_similarity_matrices):
            result.append(self.layers[i](self.padding(similarity_matrices_list[i])))
        return result    


class GraphSim(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        gcn_size,
        conv_kernel_size,
        conv_pool_size,
        conv_out_channels,
        linear_size,
        dropout,
        max_node_set_size,
        device,
        max_edge_set_size
    ):
        super(GraphSim, self).__init__()
        self.input_dim = input_dim

        self.gcn_layers = torch.nn.ModuleList([])
        self.conv_layers = torch.nn.ModuleList([])
        self.pool_layers = torch.nn.ModuleList([])
        self.linear_layers = torch.nn.ModuleList([])
        self.num_conv_layers = len(conv_kernel_size)
        self.num_linear_layers = len(linear_size)
        self.num_gcn_layers = len(gcn_size)
        self.dropout = dropout
        self.linear_size = linear_size
        self.max_node_set_size = max_node_set_size

        num_ftrs = self.input_dim
        for i in range(self.num_gcn_layers):
            self.gcn_layers.append(
                pyg_nn.GCNConv(num_ftrs, gcn_size[i]))
            num_ftrs = gcn_size[i]

        in_channels = 1
        for i in range(self.num_conv_layers):
            self.conv_layers.append(CNNLayerV1(kernel_size=conv_kernel_size[i],
                stride=1, in_channels=in_channels, out_channels=conv_out_channels[i],
                num_similarity_matrices=self.num_gcn_layers))
            self.pool_layers.append(MaxPoolLayerV1(pool_size=conv_pool_size[i],
                stride=conv_pool_size[i], num_similarity_matrices=self.num_gcn_layers))
            in_channels = conv_out_channels[i]

        for i in range(self.num_linear_layers-1):
            self.linear_layers.append(torch.nn.Linear(linear_size[i],
                linear_size[i+1]))

        self.scoring_layer = torch.nn.Linear(linear_size[-1], 1)

    def GCN_pass(self, data):
        features, edge_index = data.x, data.edge_index
        abstract_feature_matrices = []
        for i in range(self.num_gcn_layers-1):
            features = self.gcn_layers[i](features, edge_index)
            abstract_feature_matrices.append(features)
            features = torch.nn.functional.relu(features)
            features = torch.nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)


        features = self.gcn_layers[-1](features, edge_index)
        abstract_feature_matrices.append(features)
        return abstract_feature_matrices

    def Conv_pass(self, similarity_matrices_list):
        features = [_.unsqueeze(1) for _ in similarity_matrices_list]
        for i in range(self.num_conv_layers):
            features = self.conv_layers[i](features)
            features = [torch.relu(_)  for _ in features]
            features = self.pool_layers[i](features)

            features = [torch.nn.functional.dropout(_,
                                               p=self.dropout,
                                               training=self.training)  for _ in features]
        return features

    def linear_pass(self, features):
        for i in range(self.num_linear_layers-1):
            features = self.linear_layers[i](features)
            features = torch.nn.functional.relu(features)
            features = torch.nn.functional.dropout(features,p=self.dropout,
                                               training=self.training)
        return features
    
    def pad_matrix(self, matrix):
        M = self.max_node_set_size
        if matrix.shape[-1] < M or matrix.shape[-2] < M:
            pad_x = max(0, M - matrix.shape[-1])
            pad_y = max(0, M - matrix.shape[-2])
            padded_tensor = torch.nn.functional.pad(matrix, (0, pad_x, 0, pad_y))
            return padded_tensor
        return matrix

    def forward(self, batch_data,batch_data_sizes, batch_adj):
        # q_graphs,c_graphs = zip(*batch_data)
        # a,b = zip(*batch_data_sizes)
        query_graphs, corpus_graphs = zip(*batch_data)

        query_batch = Batch.from_data_list(query_graphs)
        corpus_batch = Batch.from_data_list(corpus_graphs)

        a,b = zip(*batch_data_sizes)

        query_abstract_features_list = self.GCN_pass(query_batch)
        query_abstract_features_list = [pad_sequence(torch.split(query_abstract_features_list[i], list(a), dim=0), batch_first=True) \
                                        for i in range(self.num_gcn_layers)]


        corpus_abstract_features_list = self.GCN_pass(corpus_batch)
        corpus_abstract_features_list = [pad_sequence(torch.split(corpus_abstract_features_list[i], list(b), dim=0), batch_first=True) \
                                          for i in range(self.num_gcn_layers)]

        similarity_matrices_list = [torch.matmul(query_abstract_features_list[i],\
                                    corpus_abstract_features_list[i].permute(0,2,1))
                                    for i in range(self.num_gcn_layers)]
        
        similarity_matrices_list = [self.pad_matrix(_) for _ in similarity_matrices_list]
        
        # print(f'similarity_matrices_list shape: {similarity_matrices_list[0].shape} && length = {len(similarity_matrices_list)}')

        features = torch.cat(self.Conv_pass(similarity_matrices_list), dim=1).view(-1,
                              self.linear_size[0])
        
        features = self.linear_pass(features)

        score_logits = self.scoring_layer(features)

        return score_logits.view(-1)
