"""!
@file
@ingroup Training
@brief GCN (Graph Convolutional Network) model definition.
"""

"""!
@addtogroup Training
@{
"""

import torch
from torch.nn import Linear, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


# class GCN(torch.nn.Module):
#     def __init__(self, in_channels, hidden1_channels, hidden2_channels , out_channels, dropout):
#         super(GCN, self).__init__()
#         self.GCNConv1 = GCNConv(in_channels, hidden1_channels)
#         self.GCNConv2 = GCNConv(hidden1_channels, hidden2_channels)
#         self.dropout = Dropout(dropout)
#         self.lin = Linear(hidden2_channels, out_channels)

#     def forward(self, data):
#         x = data.x
#         edge_index, edge_weight = data.edge_index, data.edge_attr
#         batch = data.batch
#         # 1. Node embedding
#         x = self.GCNConv1(x, edge_index, edge_weight)
#         x = x.relu()
#         x = self.GCNConv2(x, edge_index, edge_weight)
#         x = x.relu()
#         # 2. Readout layer (map node embeddings to graph embedding)
#         x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
#         # 3. Traditional classifier
#         # x = F.dropout(x, p=0.5, training=self.training)
#         x = self.dropout(x)
#         x = self.lin(x)
#         x = torch.sigmoid(x)
#         return x


class GCN(torch.nn.Module):
    """!
    GCN (Graph Convolutional Network) neural network PyTorch module that includes multiple GCN layers, 
    a readout layer and FC classifier. The module is designed to take in graph data in the PyTorch Geometric 
    format and output a graph classification label.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, dropout, dropout_G):
        """!
        Class constructor. 

        @param in_channels      the number of input channels for the first GCN layer.
        @param hidden_channels  a list containing the number of hidden channels for each GCN layer.
        @param out_channels     the number of output channels for the final linear layer.
        @param dropout          dropout probability in the final FC layer.
        @param dropout_G        dropout probability in the GCN layers.
        """

        super(GCN, self).__init__()
        self.num_layers = len(hidden_channels)
        for i in range(self.num_layers):
            self.__setattr__("GCNConv"+str(i+1), GCNConv(in_channels, hidden_channels[i]))
            in_channels = hidden_channels[i]
        self.dropout_G = Dropout(dropout_G)
        self.dropout = Dropout(dropout)
        self.lin = Linear(hidden_channels[-1],out_channels)

    def forward(self, data):
        """!
        The forward function of the GCN module.

        @param data     PyTorch Geometric Data object containing the graph data.
        @retval x       Output of the GCN module, which is a tensor containing node classification labels.
        """
        x = data.x
        edge_index, edge_weight = data.edge_index, data.edge_attr
        batch = data.batch
        # 1. GCN Layers --> Node Embeddings
        for i in range(self.num_layers):
            x = self.__getattr__("GCNConv"+str(i+1))(x, edge_index, edge_weight)
            x = x.relu()
            if i < self.num_layers:
                x = self.dropout_G(x)  
        # 2. Readout layer (map node embeddings to graph embedding)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # 3. Traditional classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.dropout(x)
        x = self.lin(x)
        x = torch.sigmoid(x)
        return x
    

"""!
@}
"""