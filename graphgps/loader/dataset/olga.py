import os
import os.path as osp
from itertools import product
from typing import Callable, List, Optional

import numpy as np
import scipy.sparse
from random import sample
import torch

import time
import logging

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
        num_nodes = shape[0]

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

        # create mapping:
        #   - train nodes should have indices [0, num_train_nodes - 1]
        #   - val nodes should have indices [num_train_nodes, num_train_nodes + num_val_nodes - 1]
        #   - test nodes should have indices [num_train_nodes + num_val_nods , num_train_nodes + num_val_nodes + num_test_nodes - 1]
        # mapping holds a mapping from indices from dataset to indices as described above
        mapping = [float('nan')] * num_nodes
        for i in range(len(train_indices)):
            mapping[train_indices[i]] = i
        for i in range(len(val_indices)):
            mapping[val_indices[i]] = num_train_nodes + i
        for i in range(len(test_indices)):
            mapping[test_indices[i]] = num_train_nodes + num_val_nodes + i
        mapping_torch = torch.tensor(mapping)

        # apply mapping to all edges
        mapped_train_edges_torch = mapping_torch[train_edges_torch].long()
        mapped_val_graph_edges_torch = mapping_torch[val_graph_edges_torch].long()
        mapped_val_check_edges_torch = mapping_torch[val_check_edges_torch].long()
        mapped_test_graph_edges_torch = mapping_torch[test_graph_edges_torch].long()
        mapped_test_check_edges_torch = mapping_torch[test_check_edges_torch].long()

        # create Data
        data = Data()

        # features (trivial at the moment)
        # Comment: x_test is called x such that graphgym derives correct cfg.share.dim_in
        data['x_train'] = torch.nn.functional.one_hot((torch.zeros(num_train_nodes, dtype=int)), num_classes=3).float()
        data['x_val'] = torch.nn.functional.one_hot((torch.zeros(num_train_nodes + num_val_nodes, dtype=int)), num_classes=3).float()
        data['x'] = torch.nn.functional.one_hot((torch.zeros(num_train_nodes + num_val_nodes + num_test_nodes, dtype=int)), num_classes=3).float()

        # graph for message passing
        data['edge_index_train'] = mapped_train_edges_torch
        data['edge_index_val'] = torch.cat((mapped_train_edges_torch, 
                                            mapped_val_graph_edges_torch), 1)
        data['edge_index_test'] = torch.cat((mapped_train_edges_torch, 
                                             mapped_val_graph_edges_torch, 
                                             mapped_val_check_edges_torch, 
                                             mapped_test_graph_edges_torch), 1)
        
        # edges to predict with label
        #   - train: only positive edges, negative edges are sampled in OLGASampler in main
        #   - val: positive edges and same number of negative edges randomly sampled here
        #   - test: positive edges and same number of negative edges randomly sampled here

        # data['pos_train_edge_index'] = mapped_train_edges_torch
        # data['train_edge_label'] = torch.ones(len(train_edge_indices), dtype=int)
        start_t = time.time()
        mapped_train_edges_set = {(int(edge[0]), int(edge[1])) for edge in mapped_train_edges_torch.T}
        mapped_neg_train_edges= [edge for edge in ((i, j) for j in range(num_train_nodes) for i in range(j)) if edge not in mapped_train_edges_set]
        mapped_neg_train_edges_torch = torch.tensor(mapped_neg_train_edges, dtype=int).T
        data['train_edge_index'] = torch.cat((mapped_train_edges_torch, mapped_neg_train_edges_torch), 1)
        data['train_edge_label'] = torch.cat((torch.ones(len(mapped_train_edges_set), dtype=int), 
                                              torch.zeros(len(mapped_neg_train_edges), dtype=int)))
        logging.info(f"time for neg edges: {time.time() - start_t} seconds")


        mapped_neg_val_check_edges_tuples = []
        mapped_val_indices = range(num_train_nodes, num_train_nodes + num_val_nodes)
        num_val_edges_each = len(val_check_edge_indices)
        mapped_val_check_edges_lst = mapped_val_check_edges_torch.T.tolist()
        for i in range(num_val_edges_each):
            v1, v2 = np.random.choice(mapped_val_indices, 2, replace=False)
            while [min(v1, v2), max(v1, v2)] in mapped_val_check_edges_lst:
                v1, v2 = np.random.choice(mapped_val_indices, 2, replace=False)
            mapped_neg_val_check_edges_tuples.append((min(v1, v2), max(v1, v2)))
        mapped_neg_val_check_edges_torch = torch.tensor(mapped_neg_val_check_edges_tuples, dtype=int).T
        data['val_edge_index'] = torch.cat((mapped_val_check_edges_torch, mapped_neg_val_check_edges_torch), 1)
        data['val_edge_label'] = torch.cat((torch.ones(num_val_edges_each, dtype=int), 
                                            torch.zeros(num_val_edges_each, dtype=int)))

        mapped_neg_test_check_edges_tuples = []
        mapped_test_indices = range(num_train_nodes + num_val_nodes, num_train_nodes + num_val_nodes + num_test_nodes)
        num_test_edges_each = len(test_check_edge_indices)
        mapped_test_check_edges_lst = mapped_test_check_edges_torch.T.tolist()
        for i in range(num_test_edges_each):
            v1, v2 = np.random.choice(mapped_test_indices, 2, replace=False)
            while [min(v1, v2), max(v1, v2)] in mapped_test_check_edges_lst:
                v1, v2 = np.random.choice(mapped_test_indices, 2, replace=False)
            mapped_neg_test_check_edges_tuples.append((min(v1, v2), max(v1, v2)))
        mapped_neg_test_check_edges_torch = torch.tensor(mapped_neg_test_check_edges_tuples, dtype=int).T
        data['test_edge_index'] = torch.cat((mapped_test_check_edges_torch, mapped_neg_test_check_edges_torch), 1)
        data['test_edge_label'] = torch.cat((torch.ones(num_test_edges_each, dtype=int), 
                                            torch.zeros(num_test_edges_each, dtype=int)))

        # save data
        torch.save(data, self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
