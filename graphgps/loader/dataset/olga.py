import os
import os.path as osp
from itertools import product
from typing import Callable, List, Optional

import numpy as np
import scipy.sparse
from random import sample
import torch

from torch_geometric.utils import to_networkx

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
    Data,
)


class OLGA(InMemoryDataset):
    r"""
    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """
    url = 'https://polybox.ethz.ch/index.php/s/ZXCdjkCJns9qlMY/download'
    
    def __init__(
        self,
        root: str,
        embedding,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])
        if embedding is None:
            pass
        else:
            print('embedding not None')
            exit(0)

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'acousticbrainz.npy',
            'artist_connections.npz',
            'train_mask.npz',
            'val_mask.npz',
            'test_mask.npz'
        ]

    @property
    def processed_file_names(self) -> str:
        return 'data.pth'

    def download(self) -> None:
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def process(self) -> None:
        # get edges
        ac = np.load(os.path.join(self.raw_dir, 'olga_data/artist_connections.npz'))
        indices = ac['indices']
        indptr = ac['indptr']
        data = ac['data']
        shape = ac['shape']
        sparse_matrix = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
        edges = sparse_matrix.nonzero()
        edges_torch = torch.tensor(np.array((edges[0], edges[1])), dtype=int)

        # get train/val/test node split
        train_m = np.load(os.path.join(self.raw_dir, 'olga_data/train_mask.npz')) 
        train_indices = train_m['indices']
        num_train_nodes = len(train_indices)
        val_m = np.load(os.path.join(self.raw_dir, 'olga_data/val_mask.npz')) 
        val_indices = val_m['indices']
        num_val_nodes = len(val_indices)
        test_m = np.load(os.path.join(self.raw_dir, 'olga_data/test_mask.npz')) 
        test_indices = test_m['indices']
        num_test_nodes = len(test_indices)
        train_edge_indices = []
        val_graph_edge_indices = []
        val_check_edge_indices = []
        test_graph_edge_indices = []
        test_check_edge_indices = []
        for i in range(len(edges[0])):
            # train
            if edges[0][i] in train_indices and edges[1][i] in train_indices:
                train_edge_indices.append(i)
            # val graph
            elif (edges[0][i] in train_indices and edges[1][i] in val_indices) or \
                 (edges[0][i] in val_indices and edges[1][i] in train_indices):
                val_graph_edge_indices.append(i)
            # val check
            elif (edges[0][i] in val_indices and edges[1][i] in val_indices):
                val_check_edge_indices.append(i)
            # test graph
            elif (edges[0][i] in train_indices and edges[1][i] in test_indices) or \
                 (edges[0][i] in val_indices and edges[1][i] in test_indices) or \
                 (edges[0][i] in test_indices and edges[1][i] in train_indices) or \
                 (edges[0][i] in test_indices and edges[1][i] in val_indices):
                test_graph_edge_indices.append(i)
            # test check
            elif (edges[0][i] in test_indices and edges[1][i] in test_indices):
                test_check_edge_indices.append(i)
        train_edges_torch = edges_torch[:, torch.tensor(train_edge_indices)]
        val_graph_edges_torch = edges_torch[:, torch.tensor(val_graph_edge_indices)]
        val_check_edges_torch = edges_torch[:, torch.tensor(val_check_edge_indices)]
        test_graph_edges_torch = edges_torch[:, torch.tensor(test_graph_edge_indices)]
        test_check_edges_torch = edges_torch[:, torch.tensor(test_check_edge_indices)]

        # create Data
        data = Data()

        # graphs
        data['edge_index_train'] = train_edges_torch
        data['edge_index_val'] = torch.cat((train_edges_torch, val_graph_edges_torch), 1)
        data['edge_index_test'] = torch.cat((train_edges_torch, val_graph_edges_torch, val_check_edges_torch, test_graph_edges_torch), 1)

        # features
        data['x_train'] = torch.nn.functional.one_hot((torch.zeros(num_train_nodes, dtype=int)), num_classes=3).float()
        data['x_val'] = torch.nn.functional.one_hot((torch.zeros(num_train_nodes + num_val_nodes, dtype=int)), num_classes=3).float()
        data['x_test'] = torch.nn.functional.one_hot((torch.zeros(num_train_nodes + num_val_nodes + num_test_nodes, dtype=int)), num_classes=3).float()

        # training data (sampled from in main)
        neg_train_edges_tuples = []
        for i in train_indices:
            for j in train_indices:
                if (i < j) and (torch.tensor([i, j]) not in train_edges_torch.T):
                    neg_train_edges_tuples.append((i, j))
        neg_train_edges_torch = torch.tensor(neg_train_edges_tuples, dtype=int).T
        data['train_edge_index'] = torch.cat((train_edges_torch, neg_train_edges_torch), 1)
        data['train_edge_label'] = torch.cat((torch.ones(len(train_edge_indices), dtype=int), torch.zeros(len(neg_train_edges_tuples), dtype=int)))

        # validation data
        neg_val_check_edges_tuples = []
        for i in val_indices:
            for j in val_indices:
                if (i < j) and (torch.tensor([i, j]) not in val_check_edges_torch.T):
                    neg_val_check_edges_tuples.append((i, j))
        neg_val_check_edges_torch = torch.tensor(sample(neg_val_check_edges_tuples, len(val_check_edge_indices)), dtype=int).T
        data['val_edge_index'] = torch.cat((val_check_edges_torch, neg_val_check_edges_torch), 1)
        data['val_edge_label'] = torch.cat((torch.ones(len(val_check_edge_indices), dtype=int), torch.zeros(len(val_check_edge_indices), dtype=int)))

        # testing data
        neg_test_check_edges_tuples = []
        for i in test_indices:
            for j in test_indices:
                if (i < j) and (torch.tensor([i, j]) not in test_check_edges_torch.T):
                    neg_test_check_edges_tuples.append((i, j))
        neg_test_check_edges_torch = torch.tensor(sample(neg_test_check_edges_tuples, len(test_check_edge_indices)), dtype=int).T
        data['test_edge_index'] = torch.cat((test_check_edges_torch, neg_test_check_edges_torch), 1)
        data['test_edge_label'] = torch.cat((torch.ones(len(test_check_edge_indices), dtype=int), torch.zeros(len(test_check_edge_indices), dtype=int)))

        # save data
        torch.save(data, self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
