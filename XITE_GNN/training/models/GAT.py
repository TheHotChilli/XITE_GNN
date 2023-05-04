"""!
@file
@ingroup Training
@brief GAT (Graph Attention Network) model definition.
"""

"""!
@addtogroup Training
@{
"""

import torch
from torch.nn import Linear, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool


class GAT(torch.nn.Module):
    """!
    GAT (Graph Attention Network) neural network PyTorch module that includes multiple GAT layers and a readout layer.
    The module is designed to take in graph data in the PyTorch Geometric format and output a graph classification label.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, dropout, dropout_G):
        """!
        Class constructor. 

        @param in_channels      the number of input channels for the first GAT layer.
        @param hidden_channels  a list containing the number of hidden channels for each GAT layer.
        @param out_channels     the number of output channels for the final linear layer.
        @param dropout          dropout probability in the final FC layer.
        @param dropout_G        dropout probability in the GAT layers.
        """
        super(GAT, self).__init__()
        self.num_layers = len(hidden_channels)
        for i in range(self.num_layers):
            self.__setattr__("GATConv"+str(i+1), GATConv(in_channels, hidden_channels[i], dropout=dropout_G, edge_dim=1))
            in_channels = hidden_channels[i]
        self.dropout = Dropout(dropout)
        self.lin = Linear(hidden_channels[-1], out_channels)

    def forward(self, data):
        """!
        The forward function of the GAT module.

        @param data     PyTorch Geometric Data object containing the graph data.
        @retval x       Output of the GAT module, which is a tensor containing node classification labels.
        """
        x = data.x
        edge_index, edge_weight = data.edge_index, data.edge_attr
        batch = data.batch
        # 1. GAT Layers --> Node Embeddings
        for i in range(self.num_layers):
            x = self.__getattr__("GATConv"+str(i+1))(x, edge_index, edge_attr=edge_weight)
            x = x.relu()
        # 2. Readout layer (map node embeddings to graph embedding)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # 3. Traditional classifier
        x = self.dropout(x)
        x = self.lin(x)
        x = torch.sigmoid(x)
        return x


# class GAT(torch.nn.Module):
#     def __init__(self, in_channels, hidden1_channels, hidden2_channels , out_channels, dropout):
#         super(GAT, self).__init__()
#         self.GATConv1 = GATConv(in_channels, hidden1_channels, dropout=dropout, edge_dim=1)
#         self.GATConv2 = GATConv(hidden1_channels, hidden2_channels, dropout=dropout, edge_dim=1)
#         self.dropout = Dropout(dropout)
#         self.lin = Linear(hidden2_channels, out_channels)

#     def forward(self, data):
#         x = data.x
#         edge_index, edge_weight = data.edge_index, data.edge_attr
#         batch = data.batch
#         # 1. Node embedding
#         x = self.GATConv1(x, edge_index, edge_attr=edge_weight)
#         x = x.relu()
#         x = self.GATConv2(x, edge_index, edge_attr=edge_weight)
#         x = x.relu()
#         # 2. Readout layer (map node embeddings to graph embedding)
#         x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
#         # 3. Traditional classifier
#         # x = F.dropout(x, p=0.5, training=self.training)
#         x = self.dropout(x)
#         x = self.lin(x)
#         x = torch.sigmoid(x)
#         return x


"""!
@}
"""