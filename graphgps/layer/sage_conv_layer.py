import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class SAGEConvLayer(nn.Module):
    """GraphSAGE convolution layer."""
    def __init__(self, dim_in, dim_out, dropout, residual):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual

        self.model = SAGEConv(dim_in, dim_out, project=True)

    def forward(self, batch):
        x_in = batch.x

        batch.x = self.model(batch.x, batch.edge_index)

        batch.x = F.elu(batch.x)
        batch.x = F.dropout(batch.x, p=self.dropout, training=self.training)
        batch.x = F.normalize(batch.x, p=2., dim=-1)

        if self.residual:
            batch.x = x_in + batch.x  # residual connection

        return batch
