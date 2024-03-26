import torch
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_node_encoder
from torch_geometric.graphgym.models.layer import new_layer_config, MLP


@register_node_encoder('MusicNodeEncoder')
class LinearNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.encoder = MLP(
            new_layer_config(
                cfg.share.dim_in - 3,
                emb_dim-3,
                1,
                has_act=True,
                has_bias=True,
                cfg=cfg,
            ))


    def forward(self, batch):
        batch.x = torch.cat((batch.x[:,:3], self.encoder(batch.x[:,3:])), dim=1)
        return batch
