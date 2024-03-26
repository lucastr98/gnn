import os
import os.path as osp
from itertools import product
from typing import Callable, List, Optional

import numpy as np
import scipy.sparse as sp
import torch

from torch_geometric.utils import to_networkx

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
    Data,
)


class LastFM(InMemoryDataset):
    r"""A subset of the last.fm music website keeping track of users' listining
    information from various sources, as collected in the
    `"MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph
    Embedding" <https://arxiv.org/abs/2002.01680>`_ paper.
    last.fm is a heterogeneous graph containing three types of entities - users
    (1,892 nodes), artists (17,632 nodes), and artist tags (1,088 nodes).
    This dataset can be used for link prediction, and no labels or features are
    provided.

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
    url = 'https://polybox.ethz.ch/index.php/s/EeQrarTtTXz1fv1/download'
    
    url_embeddings = 'https://polybox.ethz.ch/index.php/s/dkQW1wLywyT1NuV'

    def __init__(
        self,
        root: str,
        embedding,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])
        N_u = 1892
        N_a = 17632
        N_t = 1088
        if embedding == 'clap':
            embeddings = np.load('clap_embeddings_17632artists.npy')
            self.data.x = torch.cat((self.data.x, torch.cat((torch.zeros((N_u, embeddings.shape[1])), torch.from_numpy(embeddings), torch.zeros((N_t, embeddings.shape[1]))), dim=0)), dim=1).to(torch.float32)
        elif embedding == 'beats':
            embeddings = np.load('beats_embedding.npy')
            self.data.x = torch.cat((self.data.x, torch.cat((torch.zeros((N_u, embeddings.shape[1])), torch.from_numpy(embeddings), torch.zeros((N_t, embeddings.shape[1]))), dim=0)), dim=1).to(torch.float32)
        elif embedding is None:
            pass
        else:
            print('Could not load embedding ' + embedding)
            exit(0)

        print(self.data['edge_index'].shape)
        print(self.data['train_edge_label'].sum())
        print(self.data['val_edge_label'].sum())
        print(self.data['test_edge_label'].sum())


    @property
    def raw_file_names(self) -> List[str]:
        return [
            'data.pth'
        ]

    @property
    def processed_file_names(self) -> str:
        return 'data.pth'

    def download(self) -> None:
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def process(self) -> None:
        data = Data()
        node_nums = {}

        data_dict = torch.load(osp.join(self.raw_dir, 'data.pth'))

        s = {}
        N_u = 1892
        N_a = 17632
        N_t = 1088

        s['user'] = (0, N_u)
        s['artist'] = (N_u, N_u + N_a)
        s['tag'] = (N_u + N_a, N_u + N_a + N_t)

        print(f'N_u = {N_u}, N_a = {N_a}, N_t = {N_t}')


        node_offsets = {'user': 0, 'artist': N_u, 'tag': N_u + N_a}

        data['x'] = torch.nn.functional.one_hot(torch.cat((torch.zeros(N_u, dtype=int), torch.ones(N_a, dtype=int), torch.ones(N_t, dtype=int)*2)), num_classes=3).float()

        data['edge_index'] = data_dict['edge_index']
        data['train_edge_index'] = data_dict['train_edge_index']
        data['train_edge_label'] = data_dict['train_edge_label']
        data['val_edge_index'] = data_dict['val_edge_index']
        data['val_edge_label'] = data_dict['val_edge_label']
        data['test_edge_index'] = data_dict['test_edge_index']
        data['test_edge_label'] = data_dict['test_edge_label']



        if self.pre_transform is not None:
            data = self.pre_transform(data)

        save_data = data
        torch.save(save_data, self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
