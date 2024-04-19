import os
import os.path as osp
from itertools import product
from typing import Callable, List, Optional

import numpy as np
import scipy.sparse
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
        # find all actual edges
        ac = np.load(os.path.join(self.raw_dir, 'olga_data/artist_connections.npz'))
        indices = ac['indices']
        indptr = ac['indptr']
        data = ac['data']
        shape = ac['shape']
        sparse_matrix = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
        edges = torch.tensor(np.array((sparse_matrix.nonzero()[0], sparse_matrix.nonzero()[1])), dtype=int)

        # create train/val/test split for edges
        train_ratio = 0.72
        test_ratio = 0.2
        val_ratio = 0.08       
        total_edges = edges.size(1)
        shuffled_index = torch.randperm(total_edges)
        train_size = int(train_ratio * total_edges)
        test_size = int(test_ratio * total_edges)
        val_size = total_edges - train_size - test_size
        train_index, rest_index = shuffled_index.split(train_size)
        test_index, val_index = rest_index.split([test_size, val_size])
        train_edges = edges[:, train_index]
        test_edges = edges[:, test_index]
        val_edges = edges[:, val_index]

        # get negative edges
        num_nodes = shape[0]
        pos_edges_set = set(zip(*sparse_matrix.nonzero()))
        # neg_edges_tups = [edge for edge in ((i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j) if edge not in pos_edges_set]
        neg_edges_tups = [edge for edge in ((i, j) for j in range(num_nodes) for i in range(j)) if edge not in pos_edges_set]
        neg_edges = torch.tensor(neg_edges_tups, dtype=int).T

        # create data
        data = Data()
        data['x'] = torch.nn.functional.one_hot((torch.zeros(num_nodes, dtype=int)), num_classes=3).float()
        data['edge_index'] = train_edges
        neg_size = len(neg_edges_tups)
        neg_randperm = torch.randperm(neg_size)
        data['val_edge_index'] = torch.cat((val_edges, neg_edges[:, neg_randperm[:val_size]]), 1)
        data['val_edge_label'] = torch.cat((torch.ones(val_size, dtype=int), torch.zeros(val_size, dtype=int)))
        data['test_edge_index'] = torch.cat((test_edges, neg_edges[:, neg_randperm[val_size:val_size+test_size]]), 1)
        data['test_edge_label'] = torch.cat((torch.ones(test_size, dtype=int), torch.zeros(test_size, dtype=int)))
        data['train_edge_index'] = torch.cat((train_edges, neg_edges, val_edges, test_edges), 1)
        data['train_edge_label'] = torch.cat((torch.ones(train_size, dtype=int), torch.zeros(neg_size + val_size + test_size, dtype=int)))

        # save data
        torch.save(data, self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
