import numpy as np
import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import new_layer_config, MLP, Linear
from torch_geometric.graphgym.register import register_head
import torch.nn.functional as F


@register_head('transductive_edge')
class GNNTransductiveEdgeHead(torch.nn.Module):
    r"""A GNN prediction head for edge-level/link-level prediction tasks.

    Args:
        dim_in (int): The input feature dimension.
        dim_out (int): The output feature dimension.
    """
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        # Module to decode edges from node embeddings:
        if cfg.model.edge_decoding == 'concat':
            self.layer_post_mp = MLP(
                new_layer_config(
                    dim_in * 2,
                    dim_out,
                    cfg.gnn.layers_post_mp,
                    has_act=False,
                    has_bias=True,
                    cfg=cfg,
                ))
            self.decode_module = lambda v1, v2: \
                self.layer_post_mp(torch.cat((v1, v2), dim=-1))
        else:
            if dim_out > 1:
                raise ValueError(f"Binary edge decoding "
                                 f"'{cfg.model.edge_decoding}' is used for "
                                 f"multi-class classification")
            self.layer_post_mp = MLP(
                new_layer_config(
                    dim_in,
                    dim_in,
                    cfg.gnn.layers_post_mp,
                    has_act=True,
                    has_bias=True,
                    cfg=cfg,
                ))
            if cfg.gnn.linear_output_layer != -1:
                self.output_layer = Linear(
                    new_layer_config(
                        dim_in, 
                        cfg.gnn.linear_output_layer,
                        num_layers=1,
                        has_act=False,
                        has_bias=True,
                        cfg=cfg
                    ))
            if cfg.model.edge_decoding == 'dot':
                self.decode_module = lambda v1, v2: torch.sum(v1 * v2, dim=-1)
            elif cfg.model.edge_decoding == 'euclidean':
                # self.decode_module = lambda v1, v2: torch.sqrt(torch.sum((v1 - v2) ** 2, dim=-1))
                self.decode_module = lambda v1, v2: 1 / (1 + torch.sqrt(torch.sum((v1 - v2) ** 2, dim=-1)))
                # self.decode_module = lambda v1, v2: torch.exp(1 + torch.sqrt(torch.sum((v1 - v2) ** 2, dim=-1)))
            elif cfg.model.edge_decoding == 'cosine_similarity':
                self.decode_module = torch.nn.CosineSimilarity(dim=-1)
            else:
                raise ValueError(f"Unknown edge decoding "
                                 f"'{cfg.model.edge_decoding}'")

    def _apply_index(self, batch):
        if cfg.model.loss_fun == 'triplet':
            triplets = batch[f'{batch.split}_triplet']
            if cfg.dataset.triplets_per_edge == 'two':
                edges = torch.cat((triplets[[0, 1]][:, :int(triplets.size(1)/2)], triplets[[0, 2]]), 1)
                labels = torch.cat((torch.ones(int(triplets.size(1)/2), dtype=int), 
                                    torch.zeros(triplets.size(1), dtype=int)))
            else:
                edges = torch.cat((triplets[[0, 1]], triplets[[0, 2]]), 1)
                labels = torch.cat((torch.ones(triplets.size(1), dtype=int), 
                                    torch.zeros(triplets.size(1), dtype=int)))
            if cfg.optim.normalize_embds:
              return triplets, F.normalize(batch.x[edges], p=2, dim=-1), labels
            else:
              return triplets, batch.x[edges], labels
        else:
            index = f'{batch.split}_edge_index'
            label = f'{batch.split}_edge_label'
            return batch.x[batch[index]], batch[label]

    def forward(self, batch):
        if cfg.model.edge_decoding != 'concat':
            batch = self.layer_post_mp(batch)
            if cfg.gnn.linear_output_layer != -1:
                batch = self.output_layer(batch)
        if cfg.model.loss_fun == 'triplet':
            triplets, pred, label = self._apply_index(batch)
            nodes_first = pred[0]
            nodes_second = pred[1]
            pred = self.decode_module(nodes_first, nodes_second)
            if cfg.optim.normalize_embds:
              return F.normalize(batch.x, p=2, dim=-1), triplets, pred, label
            else:
              return batch.x, triplets, pred, label
        else:
            pred, label = self._apply_index(batch)
            nodes_first = pred[0]
            nodes_second = pred[1]
            pred = self.decode_module(nodes_first, nodes_second)
            return pred, label
