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
    url = 'https://www.dropbox.com/s/jvlbs09pz6zwcka/LastFM_processed.zip?dl=1'
    
    url_embeddings = 'https://polybox.ethz.ch/index.php/s/dkQW1wLywyT1NuV'

    def __init__(
        self,
        root: str,
        with_embeddings: bool,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform, pre_transform)
        if with_embeddings:
            self.data = torch.load(self.processed_paths[0])[1]#, data_cls=HeteroData)
        else:
            self.data = torch.load(self.processed_paths[0])[0]

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'adjM.npz', 'node_types.npy', 'train_val_test_neg_user_artist.npz',
            'train_val_test_pos_user_artist.npz'
        ]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def process(self) -> None:
        data = Data()
        node_nums = {}

        new_data = torch.load('data.pth')

        node_type_idx = np.load(osp.join(self.raw_dir, 'node_types.npy'))
        node_type_idx = torch.from_numpy(node_type_idx).to(torch.long)

        node_types = ['user', 'artist', 'tag']
        for i, node_type in enumerate(node_types):
            node_nums[node_type] = int((node_type_idx == i).sum())

        s = {}
        N_u = node_nums['user']
        N_a = node_nums['artist']
        N_t = node_nums['tag']
        s['user'] = (0, N_u)
        s['artist'] = (N_u, N_u + N_a)
        s['tag'] = (N_u + N_a, N_u + N_a + N_t)

        # Build the actual edge_index
        # TODO: Combine edges for different relationships into one

        edge_indices = []

        node_offsets = {'user': 0, 'artist': N_u, 'tag': N_u + N_a}

        data['num_nodes'] = N_u + N_a + N_t

        data.x = torch.nn.functional.one_hot(torch.cat((torch.zeros(N_u, dtype=int), torch.ones(N_a, dtype=int), torch.ones(N_t, dtype=int)*2)), num_classes=3).float()
        print(data.x)

        A = sp.load_npz(osp.join(self.raw_dir, 'adjM.npz'))
        for src, dst in product(node_types, node_types):
            A_sub = A[s[src][0]:s[src][1], s[dst][0]:s[dst][1]].tocoo()
            if A_sub.nnz > 0:
                row = torch.from_numpy(A_sub.row).to(torch.long) + node_offsets[src]
                col = torch.from_numpy(A_sub.col).to(torch.long) + node_offsets[dst]
                print(f"{src}_{dst}")
                print(torch.stack([row, col], dim=0))
                print(torch.max(torch.from_numpy(A_sub.row).to(torch.long)))
                print(torch.max(torch.from_numpy(A_sub.col).to(torch.long)))
                edge_indices.append(torch.stack([row, col], dim=0))
                #if src == 'user' and dst == 'artist':
                #    user_artist = torch.stack([row, col], dim=0)
                #data[src, dst].edge_index = torch.stack([row, col], dim=0)
                #print(f"{src}, {dst}")
                #print(torch.stack([row, col], dim=0))
        data['edge_index'] = torch.cat(edge_indices, dim=1)
        #print(data.edge_index)


        # Compute the train, val and test edges and labels
        pos_split = np.load(
            osp.join(self.raw_dir, 'train_val_test_pos_user_artist.npz'))
        neg_split = np.load(
            osp.join(self.raw_dir, 'train_val_test_neg_user_artist.npz'))

        for name in ['train', 'val', 'test']:
            #if name != 'train':
            edge_index_pos = pos_split[f'{name}_pos_user_artist']
            edge_index_pos = torch.from_numpy(edge_index_pos)
            edge_index_pos = edge_index_pos.t().to(torch.long).contiguous()
            #data['user', 'artist'][f'{name}_pos_edge_index'] = edge_index
            #else:
            #    edge_index_pos = user_artist


            edge_index_neg = neg_split[f'{name}_neg_user_artist']
            edge_index_neg = torch.from_numpy(edge_index_neg)
            edge_index_neg = edge_index_neg.t().to(torch.long).contiguous()
            
            edge_index_neg[1,:] = edge_index_neg[1,:] + N_u
            edge_index_pos[1,:] = edge_index_pos[1,:] + N_u
            #data['user', 'artist'][f'{name}_neg_edge_index'] = edge_index

            edge_label = torch.cat((torch.ones(edge_index_pos.shape[1], dtype=int), torch.zeros(edge_index_neg.shape[1], dtype=int)))
            edge_index = torch.cat((edge_index_pos, edge_index_neg), dim=1)
            print(name)
            print(edge_index)
            data[f"{name}_edge_index"] = edge_index
            data[f"{name}_edge_label"] = edge_label

            #nx_graph = to_networkx(data, node_attrs=["x"])
            #print([n for n in nx_graph.neighbors(N_u + 836)])
        
        #embeddings = np.load('clap_embeddings_17632artists.npy')
        embeddings = np.load('beats_embedding.npy')
        print('Loaded embeddings')
        print(torch.from_numpy(embeddings).shape)
        print(embeddings)
        print(N_a)
        print(data.x.dtype)
        data['edge_index'] = new_data['edge_index']
        data['train_edge_index'] = new_data['train_edge_index']
        data['train_edge_label'] = new_data['train_edge_label']
        data['val_edge_index'] = new_data['val_edge_index']
        data['val_edge_label'] = new_data['val_edge_label']
        data['test_edge_index'] = new_data['test_edge_index']
        data['test_edge_label'] = new_data['test_edge_label']

        data_with_embeddings = data.clone()
        data_with_embeddings.x = torch.cat((data_with_embeddings.x, torch.cat((torch.zeros((N_u, embeddings.shape[1])), torch.from_numpy(embeddings), torch.zeros((N_t, embeddings.shape[1]))), dim=0)), dim=1).to(torch.float32)

        if self.pre_transform is not None:
            data = self.pre_transform(data)
            data_with_embeddings = self.pre_transform(data_with_embeddings)

        save_data = [data, data_with_embeddings]
        torch.save(save_data, self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
