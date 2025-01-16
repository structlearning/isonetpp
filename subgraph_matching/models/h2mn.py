import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import random_walk
from torch_geometric.data import Data
from torch.nn.parameter import Parameter
from torch_sparse import spspmm, coalesce
from torch_geometric.utils import softmax, degree
from torch_scatter import scatter, scatter_add, scatter_max
from torch_geometric.nn import HypergraphConv, GCNConv, MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, HypergraphConv
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, dense_to_sparse
from torch_geometric.data import Batch


def topk(x, ratio, batch, min_score=None, tol=1e-7):
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter_max(x, batch)[0][batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero(as_tuple=False).view(-1)
    else:
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
             num_nodes.cumsum(dim=0)[:-1]], dim=0)

        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes, ),
                             torch.finfo(x.dtype).min)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)

        _, perm = dense_x.sort(dim=-1, descending=True)

        perm = perm + cum_num_nodes.view(-1, 1)
        perm = perm.view(-1)

        k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
        mask = [
            torch.arange(k[i], dtype=torch.long, device=x.device) +
            i * max_num_nodes for i in range(batch_size)
        ]
        mask = torch.cat(mask, dim=0)

        perm = perm[mask]

    return perm


class TwoHopNeighbor(object):
    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        N = data.num_nodes

        value = edge_index.new_ones((edge_index.size(1), ), dtype=torch.float)

        index, value = spspmm(edge_index, value, edge_index, value, N, N, N, True)
        value.fill_(0)
        index, value = remove_self_loops(index, value)

        edge_index = torch.cat([edge_index, index], dim=1)
        if edge_attr is None:
            data.edge_index, _ = coalesce(edge_index, None, N, N)
        else:
            value = value.view(-1, *[1 for _ in range(edge_attr.dim() - 1)])
            value = value.expand(-1, *list(edge_attr.size())[1:])
            edge_attr = torch.cat([edge_attr, value], dim=0)
            data.edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
            data.edge_attr = edge_attr

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


def hypergraph_construction(edge_index, edge_attr, num_nodes, k=2, mode='RW'):
    if mode == 'RW':
        # Utilize random walk to construct hypergraph
        row, col = edge_index
        start = torch.arange(num_nodes, device=edge_index.device)
        walk = random_walk(row, col, start, walk_length=k)
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float, device=edge_index.device)
        adj[walk[start], start.unsqueeze(1)] = 1.0
        edge_index, _ = dense_to_sparse(adj)
    else:
        # Utilize neighborhood to construct hypergraph
        if k == 1:
            edge_index, edge_attr = add_remaining_self_loops(edge_index, edge_attr)
        else:
            neighbor_augment = TwoHopNeighbor()
            hop_data = Data(edge_index=edge_index, edge_attr=edge_attr)
            hop_data.num_nodes = num_nodes
            for _ in range(k-1):
                hop_data = neighbor_augment(hop_data)
            hop_edge_index = hop_data.edge_index
            hop_edge_attr = hop_data.edge_attr
            edge_index, edge_attr = add_remaining_self_loops(hop_edge_index, hop_edge_attr, num_nodes=num_nodes)

    return edge_index, edge_attr

def hyperedge_representation(x, edge_index):
    gloabl_edge_rep = x[edge_index[0]]
    gloabl_edge_rep = scatter(gloabl_edge_rep, edge_index[1], dim=0, reduce='mean')

    x_rep = x[edge_index[0]]
    gloabl_edge_rep = gloabl_edge_rep[edge_index[1]]

    coef = softmax(torch.sum(x_rep * gloabl_edge_rep, dim=1), edge_index[1], num_nodes=x_rep.size(0))
    weighted = coef.unsqueeze(-1) * x_rep

    hyperedge = scatter(weighted, edge_index[1], dim=0, reduce='sum')

    return hyperedge

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

class HyperedgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super(HyperedgeConv, self).__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def message(self, x_j, edge_index_i, norm):
        out = norm[edge_index_i].view(-1, 1) * x_j.view(-1, self.out_channels)

        return out

    def forward(self, x, hyperedge_index, hyperedge_weight=None):
        x = torch.matmul(x, self.weight)

        num_nodes = hyperedge_index[0].max().item() + 1
        if hyperedge_weight is None:
            D = degree(hyperedge_index[0], num_nodes, x.dtype)
        else:
            D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                            hyperedge_index[0], dim=0, dim_size=num_nodes)
        D = 1.0 / D
        D[D == float("inf")] = 0

        if hyperedge_index.numel() == 0:
            num_edges = 0
        else:
            num_edges = hyperedge_index[1].max().item() + 1
        B = 1.0 / degree(hyperedge_index[1], num_edges, x.dtype)
        B[B == float("inf")] = 0
        if hyperedge_weight is not None:
            B = B * hyperedge_weight

        out = B.view(-1, 1) * x
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, size=(num_edges, num_nodes))

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)


class HyperedgePool(MessagePassing):
    def __init__(self, nhid, ratio):
        super(HyperedgePool, self).__init__()
        self.ratio = ratio
        self.nhid = nhid
        self.alpha = 0.1
        self.K = 10
        self.hypergnn = HypergraphConv(self.nhid, 1)

    def message(self, x_j, edge_index_i, norm):
        out = norm[edge_index_i].view(-1, 1) * x_j.view(-1, 1)

        return out
    
    def forward(self, x, batch, edge_index, edge_weight):
        # Init pagerank values
        pr = torch.sigmoid(self.hypergnn(x, edge_index, edge_weight))

        if edge_weight is None:
            D = degree(edge_index[0], x.size(0), x.dtype)
        else:
            D = scatter_add(edge_weight[edge_index[1]], edge_index[0], dim=0, dim_size=x.size(0))
        D = 1.0 / D
        D[D == float("inf")] = 0

        if edge_index.numel() == 0:
            num_edges = 0
        else:
            num_edges = edge_index[1].max().item() + 1 
        B = 1.0 / degree(edge_index[1], num_edges, x.dtype)
        B[B == float("inf")] = 0
        if edge_weight is not None:
            B = B * edge_weight

        hidden = pr
        for k in range(self.K):
            self.flow = 'source_to_target'
            out = self.propagate(edge_index, x=pr, norm=B)
            self.flow = 'target_to_source'
            pr = self.propagate(edge_index, x=out, norm=D)
            pr = pr * (1 - self.alpha)
            pr += self.alpha * hidden

        score = self.calc_hyperedge_score(pr, edge_index)
        score = score.view(-1)
        perm = topk(score, self.ratio, batch)
        
        x_hyperedge = hyperedge_representation(x, edge_index)
        x_hyperedge = x_hyperedge[perm] * score[perm].view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = self.filter_hyperedge(edge_index, edge_weight, perm, num_nodes=score.size(0))

        return x_hyperedge, edge_index, edge_attr, batch
    
    def calc_hyperedge_score(self, x, edge_index):
        x = x[edge_index[0]]
        score = scatter(x, edge_index[1], dim=0, reduce='mean')

        return score
    
    def filter_hyperedge(self, edge_index, edge_attr, perm, num_nodes):
        mask = perm.new_full((num_nodes, ), -1)
        i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
        mask[perm] = i

        row, col = edge_index
        mask = (mask[col] >= 0)
        row, col = row[mask], col[mask]

        # ID re-mapping operation, which makes the ids become continuous 
        unique_row = torch.unique(row)
        unique_col = torch.unique(col)
        combined = torch.cat((unique_row, unique_col))
        uniques, counts = combined.unique(return_counts=True)
        difference = uniques[counts == 1]

        new_perm = torch.cat((unique_col, difference))
        max_id = new_perm.max().item() + 1
        new_mask = new_perm.new_full((max_id,), -1)
        j = torch.arange(new_perm.size(0), dtype=torch.long, device=new_perm.device)
        new_mask[new_perm] = j

        row, col = new_mask[row], new_mask[col]

        if edge_attr is not None:
            edge_attr = edge_attr[mask]

        return torch.stack([row, col], dim=0), edge_attr


class CrossGraphConvolutionOperator(MessagePassing):
    def __init__(self, out_nhid, in_nhid):
        super(CrossGraphConvolutionOperator, self).__init__('add')
        self.out_nhid = out_nhid
        self.in_nhid = in_nhid
        self.weight = torch.nn.Parameter(torch.Tensor(self.out_nhid, self.in_nhid))
        nn.init.xavier_uniform_(self.weight.data)

    def forward(self, x, assign_index, N, M):
        global_x = self.propagate(assign_index, size=(N, M), x=x)
        target_x = x[1]
        target_x = torch.unsqueeze(target_x, dim=1)
        global_x = torch.unsqueeze(global_x, dim=1)
        weight = torch.unsqueeze(self.weight, dim=0)
        target_x = target_x * weight
        global_x = global_x * weight
        numerator = torch.sum(target_x * global_x, dim=-1)
        target_x_denominator = torch.sqrt(torch.sum(torch.square(target_x), dim=-1) + 1e-6)
        global_x_denominator = torch.sqrt(torch.sum(torch.square(global_x), dim=-1) + 1e-6)
        denominator = torch.clamp(target_x_denominator * global_x_denominator, min=1e-6)

        return numerator / denominator

    def message(self, x_i, x_j, edge_index):
        x_i_norm = torch.norm(x_i, dim=-1, keepdim=True)
        x_j_norm = torch.norm(x_j, dim=-1, keepdim=True)
        x_norm = torch.clamp(x_i_norm * x_j_norm, min=1e-6)
        x_product = torch.sum(x_i * x_j, dim=1, keepdim=True)
        coef = F.relu(x_product / x_norm)
        coef_sum = scatter(coef + 1e-6, edge_index[1], dim=0, reduce='sum')
        normalized_coef = coef / coef_sum[edge_index[1]]

        return normalized_coef * x_j


class CrossGraphConvolution(torch.nn.Module):
    def __init__(self, out_nhid, in_nhid):
        super(CrossGraphConvolution, self).__init__()
        self.out_nhid = out_nhid
        self.in_nhid = in_nhid
        self.cross_conv = CrossGraphConvolutionOperator(self.out_nhid, self.in_nhid)
    
    def forward(self, x_left, batch_left, x_right, batch_right):
        num_nodes_x_left = scatter_add(batch_left.new_ones(x_left.size(0)), batch_left, dim=0)
        shift_cum_num_nodes_x_left = torch.cat([num_nodes_x_left.new_zeros(1), num_nodes_x_left.cumsum(dim=0)[:-1]], dim=0)
        cum_num_nodes_x_left = num_nodes_x_left.cumsum(dim=0)

        num_nodes_x_right = scatter_add(batch_right.new_ones(x_right.size(0)), batch_right, dim=0)
        shift_cum_num_nodes_x_right = torch.cat([num_nodes_x_right.new_zeros(1), num_nodes_x_right.cumsum(dim=0)[:-1]], dim=0)
        cum_num_nodes_x_right = num_nodes_x_right.cumsum(dim=0)

        adj = torch.zeros((x_left.size(0), x_right.size(0)), dtype=torch.float, device=x_left.device)
        # Construct batch fully connected graph in block diagonal matirx format
        for idx_i, idx_j, idx_x, idx_y in zip(shift_cum_num_nodes_x_left, cum_num_nodes_x_left, shift_cum_num_nodes_x_right, cum_num_nodes_x_right):
            adj[idx_i:idx_j, idx_x:idx_y] = 1.0
        new_edge_index, _ = self.dense_to_sparse(adj)
        row, col = new_edge_index

        assign_index1 = torch.stack([col, row], dim=0)
        out1 = self.cross_conv((x_right, x_left), assign_index1, N=x_right.size(0), M=x_left.size(0))

        assign_index2 = torch.stack([row, col], dim=0)
        out2 = self.cross_conv((x_left, x_right), assign_index2, N=x_left.size(0), M=x_right.size(0))

        return out1, out2

    def dense_to_sparse(self, adj):
        assert adj.dim() == 2
        index = adj.nonzero(as_tuple=False).t().contiguous()
        value = adj[index[0], index[1]]
        return index, value


class ReadoutModule(torch.nn.Module):
    def __init__(self, nhid):
        """
        :param args: Arguments object.
        """
        super(ReadoutModule, self).__init__()

        self.weight = torch.nn.Parameter(torch.Tensor(nhid, nhid))
        nn.init.xavier_uniform_(self.weight.data)

    def forward(self, x, batch):
        """
        Making a forward propagation pass to create a graph level representation.
        :param x: Result of the GNN.
        :param batch: Batch vector, which assigns each node to a specific example
        :param size: Size
        :return representation: A graph level representation matrix.
        """
        mean_pool = global_mean_pool(x, batch)
        transformed_global = torch.tanh(torch.mm(mean_pool, self.weight))
        coefs = torch.sigmoid((x * transformed_global[batch]).sum(dim=1))
        weighted = coefs.unsqueeze(-1) * x

        return global_add_pool(weighted, batch)


class MLPModule(torch.nn.Module):
    def __init__(
        self,
        nhid,
        dropout
    ):
        super(MLPModule, self).__init__()
        self.dropout = dropout

        self.lin0 = torch.nn.Linear(nhid * 2 * 4, nhid * 2)
        nn.init.xavier_uniform_(self.lin0.weight.data)
        nn.init.zeros_(self.lin0.bias.data)

        self.lin1 = torch.nn.Linear(nhid * 2, nhid)
        nn.init.xavier_uniform_(self.lin1.weight.data)
        nn.init.zeros_(self.lin1.bias.data)

        self.lin2 = torch.nn.Linear(nhid, nhid // 2)
        nn.init.xavier_uniform_(self.lin2.weight.data)
        nn.init.zeros_(self.lin2.bias.data)

        self.lin3 = torch.nn.Linear(nhid // 2, 1)
        nn.init.xavier_uniform_(self.lin3.weight.data)
        nn.init.zeros_(self.lin3.bias.data)

    def forward(self, scores):
        scores = F.relu(self.lin0(scores))
        scores = F.dropout(scores, p=self.dropout, training=self.training)
        scores = F.relu(self.lin1(scores))
        scores = F.dropout(scores, p=self.dropout, training=self.training)
        scores = F.relu(self.lin2(scores))
        scores = F.dropout(scores, p=self.dropout, training=self.training)
        scores = self.lin3(scores).view(-1)

        return scores

class H2MN(nn.Module):
    def __init__(
        self,
        nhid,
        k,
        mode,
        input_dim,
        ratio1,
        ratio2,
        ratio3,
        dropout,
        max_node_set_size,
        max_edge_set_size,
        device
    ):
        super(H2MN, self).__init__()
        self.nhid = nhid
        self.k = k
        self.mode = mode

        self.conv0 = GCNConv(input_dim, self.nhid)
        self.conv1 = HypergraphConv(self.nhid, self.nhid)
        self.cross_conv1 = CrossGraphConvolution(self.nhid, self.nhid)
        self.pool1 = HyperedgePool(self.nhid, ratio1)

        self.conv2 = HyperedgeConv(self.nhid, self.nhid)
        self.cross_conv2 = CrossGraphConvolution(self.nhid, self.nhid)
        self.pool2 = HyperedgePool(self.nhid, ratio2)

        self.conv3 = HyperedgeConv(self.nhid, self.nhid)
        self.cross_conv3 = CrossGraphConvolution(self.nhid, self.nhid)
        self.pool3 = HyperedgePool(self.nhid, ratio3)

        self.readout0 = ReadoutModule(nhid)
        self.readout1 = ReadoutModule(nhid)
        self.readout2 = ReadoutModule(nhid)
        self.readout3 = ReadoutModule(nhid)

        self.mlp = MLPModule(nhid, dropout)

    def forward(self, graphs, graph_sizes, graph_adj_matrices):
        query_graphs, corpus_graphs = zip(*graphs)

        query_batch = Batch.from_data_list(query_graphs)
        corpus_batch = Batch.from_data_list(corpus_graphs)
        features_1, edge_index_1, edge_attr_1, batch_1 = query_batch.x, query_batch.edge_index, query_batch.edge_attr, query_batch.batch
        features_2, edge_index_2, edge_attr_2, batch_2 = corpus_batch.x, corpus_batch.edge_index, corpus_batch.edge_attr, corpus_batch.batch

        # Layer 0
        # Graph Convolution Operation
        f1_conv0 = F.leaky_relu(self.conv0(features_1, edge_index_1, edge_attr_1), negative_slope=0.2)
        f2_conv0 = F.leaky_relu(self.conv0(features_2, edge_index_2, edge_attr_2), negative_slope=0.2)

        att_f1_conv0 = self.readout0(f1_conv0, batch_1)
        att_f2_conv0 = self.readout0(f2_conv0, batch_2)
        score0 = torch.cat([att_f1_conv0, att_f2_conv0], dim=1)

        edge_index_1, edge_attr_1 = hypergraph_construction(edge_index_1, edge_attr_1, num_nodes=features_1.size(0), k=self.k, mode=self.mode)
        edge_index_2, edge_attr_2 = hypergraph_construction(edge_index_2, edge_attr_2, num_nodes=features_2.size(0), k=self.k, mode=self.mode)

        # Layer 1
        # Hypergraph Convolution Operation
        f1_conv1 = F.leaky_relu(self.conv1(f1_conv0, edge_index_1, edge_attr_1), negative_slope=0.2)
        f2_conv1 = F.leaky_relu(self.conv1(f2_conv0, edge_index_2, edge_attr_2), negative_slope=0.2)

        # Hyperedge Pooling
        edge1_conv1, edge1_index_pool1, edge1_attr_pool1, edge1_batch_pool1 = self.pool1(f1_conv1, batch_1, edge_index_1, edge_attr_1)
        edge2_conv1, edge2_index_pool1, edge2_attr_pool1, edge2_batch_pool1 = self.pool1(f2_conv1, batch_2, edge_index_2, edge_attr_2)

        # Cross Graph Convolution
        hyperedge1_cross_conv1, hyperedge2_cross_conv1 = self.cross_conv1(edge1_conv1, edge1_batch_pool1, edge2_conv1, edge2_batch_pool1)

        # Readout Module
        att_f1_conv1 = self.readout1(hyperedge1_cross_conv1, edge1_batch_pool1)
        att_f2_conv1 = self.readout1(hyperedge2_cross_conv1, edge2_batch_pool1)
        score1 = torch.cat([att_f1_conv1, att_f2_conv1], dim=1)

        # Layer 2
        # Hypergraph Convolution Operation
        f1_conv2 = F.leaky_relu(self.conv2(hyperedge1_cross_conv1, edge1_index_pool1, edge1_attr_pool1), negative_slope=0.2)
        f2_conv2 = F.leaky_relu(self.conv2(hyperedge2_cross_conv1, edge2_index_pool1, edge2_attr_pool1), negative_slope=0.2)

        # Hyperedge Pooling
        edge1_conv2, edge1_index_pool2, edge1_attr_pool2, edge1_batch_pool2 = self.pool2(f1_conv2, edge1_batch_pool1, edge1_index_pool1, edge1_attr_pool1)
        edge2_conv2, edge2_index_pool2, edge2_attr_pool2, edge2_batch_pool2 = self.pool2(f2_conv2, edge2_batch_pool1, edge2_index_pool1, edge2_attr_pool1)

        # Cross Graph Convolution
        hyperedge1_cross_conv2, hyperedge2_cross_conv2 = self.cross_conv2(edge1_conv2, edge1_batch_pool2, edge2_conv2, edge2_batch_pool2)

        # Readout Module
        att_f1_conv2 = self.readout2(hyperedge1_cross_conv2, edge1_batch_pool2)
        att_f2_conv2 = self.readout2(hyperedge2_cross_conv2, edge2_batch_pool2)
        score2 = torch.cat([att_f1_conv2, att_f2_conv2], dim=1)

        # Layer 3
        # Hypergraph Convolution Operation
        f1_conv3 = F.leaky_relu(self.conv3(hyperedge1_cross_conv2, edge1_index_pool2, edge1_attr_pool2), negative_slope=0.2)
        f2_conv3 = F.leaky_relu(self.conv3(hyperedge2_cross_conv2, edge2_index_pool2, edge2_attr_pool2), negative_slope=0.2)

        # Hyperedge Pooling
        edge1_conv3, edge1_index_pool3, edge1_attr_pool3, edge1_batch_pool3 = self.pool3(f1_conv3, edge1_batch_pool2, edge1_index_pool2, edge1_attr_pool2)
        edge2_conv3, edge2_index_pool3, edge2_attr_pool3, edge2_batch_pool3 = self.pool3(f2_conv3, edge2_batch_pool2, edge2_index_pool2, edge2_attr_pool2)

        # Cross Graph Convolution
        hyperedge1_cross_conv3, hyperedge2_cross_conv3 = self.cross_conv3(edge1_conv3, edge1_batch_pool3, edge2_conv3, edge2_batch_pool3)

        # Readout Module
        att_f1_conv3 = self.readout3(hyperedge1_cross_conv3, edge1_batch_pool3)
        att_f2_conv3 = self.readout3(hyperedge2_cross_conv3, edge2_batch_pool3)
        score3 = torch.cat([att_f1_conv3, att_f2_conv3], dim=1)

        scores = torch.cat([score0, score1, score2, score3], dim=1)
        scores = self.mlp(scores)

        return scores
