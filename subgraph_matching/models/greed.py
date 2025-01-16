import torch
import torch_geometric as pyg
import torch.nn.functional as F
from torch_geometric.data import Batch


class EmbedModel(torch.nn.Module):
    def __init__(
        self,
        n_layers,
        hidden_dim,
        output_dim,
        input_dim
    ):
        super().__init__()
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.pre = torch.nn.Linear(self.input_dim, self.hidden_dim)

        self.convs = torch.nn.ModuleList()
        for l in range(self.n_layers):
            self.convs.append(
                pyg.nn.GINConv(torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_dim, self.hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_dim, self.hidden_dim)
                ))
            )

        self.post = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim*(self.n_layers+1), self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.output_dim)
        )

        self.pool = pyg.nn.global_add_pool

    def forward(self, g):
        x = g.x
        edge_index = g.edge_index

        x = self.pre(x)
        emb = x
        xres = x
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            if i&1:
                x += xres
                xres = x
            x = torch.nn.functional.relu(x)
            emb = torch.cat((emb, x), dim=1)

        x = emb
        x = self.pool(x, g.batch)
        x = self.post(x)
        return x


class Greed(torch.nn.Module):
    def __init__(
        self,
        n_layers,
        hidden_dim,
        output_dim,
        input_dim,
        max_node_set_size,
        max_edge_set_size,
        device,
    ):
        super().__init__()
        self.embed_model = EmbedModel(
            n_layers,
            hidden_dim,
            output_dim,
            input_dim
        )
        self.device = device

    def forward_emb(self, x, y):
        return -torch.sum(torch.nn.ReLU()(x-y),dim=-1)

    def forward(self, graphs, graph_sizes, graph_adj_matrices):
        query_graphs, corpus_graphs = zip(*graphs)
        query_batch = Batch.from_data_list(query_graphs)
        corpus_batch = Batch.from_data_list(corpus_graphs)

        gx = self.embed_model(query_batch)
        hx = self.embed_model(corpus_batch)
        return self.forward_emb(gx, hx)
