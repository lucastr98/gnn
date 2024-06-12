import os
import os.path as osp
from itertools import product
from typing import Callable, List, Optional
from torch_geometric.graphgym.config import cfg

import numpy as np
import scipy.sparse
from random import sample
import torch

import logging

from torch_geometric.utils import to_networkx, structured_negative_sampling

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
    Data,
)


class DISCOOLGATriplet(InMemoryDataset):
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
    url = 'https://polybox.ethz.ch/index.php/s/c2tHBz7P2HefGUl/download'
    
    def __init__(
        self,
        root: str,
        embedding,
        triplets_per_edge,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ) -> None:
        self.triplets_per_edge = triplets_per_edge
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
            'similar_to.npz',
            'influenced_by.npz',
            'followed_by.npz',
            'associated_with.npz'
            'collaborated_with.npz',
            'train_mask.npy',
            'val_mask.npy',
            'test_mask.npy',
            'acousticbrainz.npy',
            'moods_themes.npy',
            'clap.npy' #,
            # 'val_triplets_one',
            # 'val_triplets_two',
            # 'test_triplets_one',
            # 'test_triplets_two'
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
        ac = np.load(os.path.join(self.raw_dir, 'disco-olga_data/similar_to.npz'))
        indices = ac['indices']
        indptr = ac['indptr']
        data = ac['data']
        shape = ac['shape']
        sparse_matrix = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
        edges = sparse_matrix.nonzero()
        edges_torch = torch.tensor(np.array((edges[0], edges[1])), dtype=int)
        num_nodes = shape[0]

        # get train/val/test node split
        train_m = np.load(os.path.join(self.raw_dir, 'disco-olga_data/train_mask.npy')) 
        train_indices = train_m['indices']
        num_train_nodes = len(train_indices)
        val_m = np.load(os.path.join(self.raw_dir, 'disco-olga_data/val_mask.npy')) 
        val_indices = val_m['indices']
        num_val_nodes = len(val_indices)
        test_m = np.load(os.path.join(self.raw_dir, 'disco-olga_data/test_mask.npy')) 
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
        which_features = 'acousticbrainz_clap' # rand, acousticbrainz, clap, acousticbrainz_clap
        acousticbrainz_features = np.load(os.path.join(self.raw_dir, 'disco-olga_data/acousticbrainz.npy'))
        clap_features = np.load(os.path.join(self.raw_dir, 'disco-olga_data/clap.npy'))
        moods_themes_features = np.load(os.path.join(self.raw_dir, 'disco-olga_data/moods_themes.npy'))
        rand_features = np.random.rand(num_nodes, 2613).astype(np.float32)
        if cfg.dataset.features == 'rand':
            features = rand_features
        elif cfg.dataset.features == 'acousticbrainz':
            features = acousticbrainz_features
        elif cfg.dataset.features == 'clap':
            features = clap_features
        elif cfg.dataset.features == 'acousticbrainz_clap':
            features = np.hstack((acousticbrainz_features, clap_features))
        elif cfg.dataset.features == 'acousticbrainz_moods-themes':
            features = np.hstack((acousticbrainz_features, moods_themes_features))
        elif cfg.dataset.features == 'clap_moods-themes':
            features = np.hstack((clap_features, moods_themes_features))
        elif cfg.dataset.features == 'acousticbrainz_clap_moods-themes':
            features = np.hstack((clap_features, moods_themes_features))
        else:
            logging.info(f'cfg.dataset.features is incorrect: {cfg.dataset.features}')
            exit(0)
        data['x_train'] = torch.tensor(features[:num_train_nodes]).float()
        data['x_val'] = torch.tensor(features[:(num_train_nodes + num_val_nodes)]).float()
        data['x'] = torch.tensor(features).float()

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
        #   - val: triplets randomly sampled here
        #   - test: triplets randomly sampled here
        # train
        data['train_edge_index'] = mapped_train_edges_torch
        
        resample_eval_triplets = True

        # val
        if resample_eval_triplets:
            map_2_zero_val_check_edges_torch = mapped_val_check_edges_torch - num_train_nodes
            if self.triplets_per_edge == "two":
                map_2_zero_val_check_edges = torch.cat((map_2_zero_val_check_edges_torch,
                                                        map_2_zero_val_check_edges_torch[[1, 0]]), 1)
            else:
                swap_mask = torch.rand(map_2_zero_val_check_edges_torch.size(1)) > 0.5
                map_2_zero_val_check_edges = map_2_zero_val_check_edges_torch.clone()
                map_2_zero_val_check_edges[0, swap_mask], map_2_zero_val_check_edges[1, swap_mask] = \
                    map_2_zero_val_check_edges_torch[1, swap_mask], map_2_zero_val_check_edges_torch[0, swap_mask]
            a, p, n = structured_negative_sampling(map_2_zero_val_check_edges, 
                                                  num_nodes=num_val_nodes, 
                                                  contains_neg_self_loops=False)
            data['val_triplets'] = torch.stack((a, p, n)) + num_train_nodes
        else:
            if self.triplets_per_edge == "two":
                data['val_triplets'] = torch.load(os.path.join(self.raw_dir, 'olga_data/val_triplets_two.pt'))
            else:
                data['val_triplets'] = torch.load(os.path.join(self.raw_dir, 'olga_data/val_triplets_one.pt'))
                
            

        # test
        if resample_eval_triplets:
            map_2_zero_test_check_edges_torch = mapped_test_check_edges_torch - (num_train_nodes + num_val_nodes)
            if self.triplets_per_edge == "two":
                map_2_zero_test_check_edges = torch.cat((map_2_zero_test_check_edges_torch,
                                                        map_2_zero_test_check_edges_torch[[1, 0]]), 1)
            else:
                swap_mask = torch.rand(map_2_zero_test_check_edges_torch.size(1)) > 0.5
                map_2_zero_test_check_edges = map_2_zero_test_check_edges_torch.clone()
                map_2_zero_test_check_edges[0, swap_mask], map_2_zero_test_check_edges[1, swap_mask] = \
                    map_2_zero_test_check_edges_torch[1, swap_mask], map_2_zero_test_check_edges_torch[0, swap_mask]
            a, p, n = structured_negative_sampling(map_2_zero_test_check_edges, 
                                                  num_nodes=num_test_nodes, 
                                                  contains_neg_self_loops=False)
            data['test_triplets'] = torch.stack((a, p, n)) + (num_train_nodes + num_val_nodes)
        else:
            if self.triplets_per_edge == "two":
                data['test_triplets'] = torch.load(os.path.join(self.raw_dir, 'olga_data/test_triplets_two.pt'))
            else:
                data['test_triplets'] = torch.load(os.path.join(self.raw_dir, 'olga_data/test_triplets_one.pt'))


        # save data
        torch.save(data, self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
