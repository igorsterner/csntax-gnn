
import torch
from torch.nn import Sequential
from torch_geometric.nn import GATConv, GINEConv, global_mean_pool


class GraphEmbedder(torch.nn.Module):
    def __init__(self, hidden_channels, node_attr_dim, edge_attr_dim, conv_type):
        super(GraphEmbedder, self).__init__()

        self.convs = torch.nn.ModuleList()

        for layer in range(2):
            if layer == 0:
                in_size = node_attr_dim
            else:
                in_size = hidden_channels
            out_size = hidden_channels
            if conv_type == "gat":
                self.convs.append(
                    GATConv(
                        in_size,
                        out_size,
                        edge_dim=edge_attr_dim,
                        add_self_loops=False,
                    )
                )
            elif conv_type == "gine":
                nn = Sequential(
                    torch.nn.Linear(in_size, out_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(out_size, out_size),
                )
                self.convs.append(
                    GINEConv(
                        nn,
                        edge_dim=edge_attr_dim,
                        train_eps=True,
                    )
                )
            else:
                raise NotImplementedError

    def forward(self, x, edge_index, edge_attr, batch):

        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        batch = batch.to(device)

        for i, conv in enumerate(self.convs):
            if i == 0:
                x = conv(x, edge_index, edge_attr)
            else:
                x = x + conv(x, edge_index, edge_attr)

        x = global_mean_pool(x, batch)

        return x


class GraphClassifier(torch.nn.Module):
    def __init__(self, hidden_channels, node_attr_dim, edge_attr_dim, conv_type):
        super(GraphClassifier, self).__init__()

        self.embedder = GraphEmbedder(
            hidden_channels, node_attr_dim, edge_attr_dim, conv_type
        )

        self.lin_out = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels, 2 * hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * hidden_channels, 2),
        )

        # not log
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(
        self, x1, edge_index1, edge_attr1, batch1, x2, edge_index2, edge_attr2, batch2
    ):

        embed1 = self.embedder(x1, edge_index1, edge_attr1, batch1)
        embed2 = self.embedder(x2, edge_index2, edge_attr2, batch2)

        x = torch.cat([embed1, embed2], dim=-1)
        x = self.lin_out(x).squeeze(1)

        # softmax
        x = self.softmax(x)

        return x
