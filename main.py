import datetime
import os
import torch
import logging
import math
import numpy as np
from typing import Union
from torch_geometric.data import Data, HeteroData

import graphgps  # noqa, register custom modules
from graphgps.agg_runs import agg_runs
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_dataset, get_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule, train
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything
from torch import Tensor

from graphgps.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from graphgps.logger import create_logger


torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True


def new_optimizer_config(cfg):
    return OptimizerConfig(optimizer=cfg.optim.optimizer,
                           base_lr=cfg.optim.base_lr,
                           weight_decay=cfg.optim.weight_decay,
                           momentum=cfg.optim.momentum)


def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch, reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode, eval_period=cfg.train.eval_period)


class LastFMSampler(torch.utils.data.DataLoader):
    r"""A data loader that randomly samples nodes within a graph and returns
    their induced subgraph.

    .. note::

        For an example of using
        :class:`~torch_geometric.loader.RandomNodeLoader`, see
        `examples/ogbn_proteins_deepgcn.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        ogbn_proteins_deepgcn.py>`_.

    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
            The :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` graph object.
        num_parts (int): The number of partitions.
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`num_workers`.
    """
    def __init__(
        self,
        data: Data,
        batch_size: int,
        **kwargs,
    ):
        self.data = data

        edge_index = data.edge_index

        self.edge_index = edge_index
        self.num_nodes = data.num_nodes
        self.num_pos_edges = (self.data.train_edge_label == 1).sum(dim=0)

        super().__init__(
            range(batch_size),
            batch_size=1,
            collate_fn=self.collate_fn,
            **kwargs,
        )

    def collate_fn(self, index):
        if not isinstance(index, Tensor):
            index = torch.tensor(index)


        edge_mask = torch.cat((torch.randperm(self.num_pos_edges)[:cfg.dataset.num_pos_samples], torch.randperm(self.data.train_edge_index.shape[1]-self.num_pos_edges)[:cfg.dataset.num_neg_samples]+self.num_pos_edges))
        edge_index = self.data.train_edge_index[:, edge_mask]
        edge_label = self.data.train_edge_label[edge_mask]

        return Data(x=self.data.x, edge_index = self.edge_index, train_edge_index = edge_index, train_edge_label = edge_label, num_nodes=self.num_nodes, val_edge_index=self.data.val_edge_index, val_edge_label=self.data.val_edge_label, test_edge_index=self.data.test_edge_index, test_edge_label=self.data.test_edge_label)


class OLGASampler(torch.utils.data.DataLoader):
    r"""A data loader that randomly samples nodes within a graph and returns
    their induced subgraph.

    .. note::

        For an example of using
        :class:`~torch_geometric.loader.RandomNodeLoader`, see
        `examples/ogbn_proteins_deepgcn.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        ogbn_proteins_deepgcn.py>`_.

    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
            The :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` graph object.
        num_parts (int): The number of partitions.
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`num_workers`.
    """
    def __init__(
        self,
        data: Data,
        split: str,
        batch_size: int,
        **kwargs,
    ):
        self.data = data
        self.split = split
        if split == 'train':
            self.num_nodes = len(data.x_train)
        elif split == 'val':
            self.num_nodes = len(data.x_val)
        elif split == 'test':
            self.num_nodes = len(data.x_test)
        else:
            logging.warning(f"OLGASampler: split={split} not known")

        super().__init__(
            range(batch_size),
            batch_size=1,
            collate_fn=self.collate_fn,
            **kwargs,
        )

    def collate_fn(self, index):
        if not isinstance(index, Tensor):
            index = torch.tensor(index)

        if self.split == 'train':
            num_pos_edges = (self.data.train_edge_label == 1).sum(dim=0)
            pos_samples = self.data.train_edge_index[:, torch.randperm(num_pos_edges)[:cfg.dataset.num_pos_samples]]
            neg_train_edges_tuples = []
            for i in range(cfg.dataset.num_neg_samples):
                v1, v2 = np.random.choice(self.data.train_indices, 2, replace=False)
                while torch.tensor([min(v1, v2), max(v1, v2)]) in self.data.train_edge_index.T:
                    v1, v2 = np.random.choice(self.data.train_indices, 2, replace=False)
                neg_train_edges_tuples.append((min(v1, v2), max(v1, v2)))
            neg_samples = torch.tensor(neg_train_edges_tuples, dtype=int).T
            return Data(num_nodes=self.num_nodes, 
                        x=self.data.x_train, 
                        edge_index=self.data.edge_index_train,
                        train_edge_index=pos_samples, 
                        train_edge_label=neg_samples,
                        mapping=self.data.train_mapping)

            # num_pos_edges = (self.data.train_edge_label == 1).sum(dim=0)
            # edge_mask = torch.cat((torch.randperm(num_pos_edges)[:cfg.dataset.num_pos_samples], 
            #                        torch.randperm(self.data.train_edge_index.shape[1] - num_pos_edges)[:cfg.dataset.num_neg_samples] + num_pos_edges))
            # edge_index = self.data.train_edge_index[:, edge_mask]
            # edge_label = self.data.train_edge_label[edge_mask]
            # return Data(num_nodes=self.num_nodes, 
            #             x=self.data.x_train, 
            #             edge_index=self.data.edge_index_train,
            #             train_edge_index=edge_index, 
            #             train_edge_label=edge_label,
            #             mapping=self.data.train_mapping)
        elif self.split == 'val':
            return Data(num_nodes=self.num_nodes, 
                        x=self.data.x_val, 
                        edge_index=self.data.edge_index_val,
                        val_edge_index=self.data.val_edge_index, 
                        val_edge_label=self.data.val_edge_label,
                        mapping=self.data.val_mapping)
        elif self.split == 'test':
            return Data(num_nodes=self.num_nodes, 
                        x=self.data.x_test, 
                        edge_index=self.data.edge_index_test,
                        test_edge_index=self.data.test_edge_index, 
                        test_edge_label=self.data.test_edge_label,
                        mapping=self.data.test_mapping)


def custom_create_loader(cfg):
    """Create data loader object.

    Returns: List of PyTorch data loaders

    """
    dataset = create_dataset()
    # train loader
    if cfg.dataset.task == 'graph':
        id = dataset.data['train_graph_index']
        loaders = [
            custom_get_loader(dataset[id], cfg.train.sampler, cfg.train.batch_size,
                       shuffle=True, split='train')
        ]
        delattr(dataset.data, 'train_graph_index')
    else:
        loaders = [
            custom_get_loader(dataset, cfg.train.sampler, cfg.train.batch_size,
                       shuffle=True, split='train')
        ]

    if True:
        val_test_size = 1

    # val and test loaders
    for i in range(cfg.share.num_splits - 1):
        split = 'val' if i == 0 else 'test'
        if cfg.dataset.task == 'graph':
            split_names = ['val_graph_index', 'test_graph_index']
            id = dataset.data[split_names[i]]
            loaders.append(
                custom_get_loader(dataset[id], cfg.val.sampler, val_test_size,
                           shuffle=False, split=split))
            delattr(dataset.data, split_names[i])
        else:
            loaders.append(
                custom_get_loader(dataset, cfg.val.sampler, val_test_size,
                           shuffle=False, split=split))

    return loaders

def custom_get_loader(dataset, sampler, batch_size, shuffle=True, split='train'):
    pw = cfg.num_workers > 0
    if False:
        loader_train = get_loader(dataset, sampler, batch_size, shuffle)
    else:
        loader_train = OLGASampler(dataset[0], split=split,
                                        shuffle=shuffle,
                                        batch_size=batch_size,
                                        pin_memory=True, persistent_workers=pw)

    return loader_train



def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


def custom_set_run_dir(cfg, run_id):
    """Custom output directory naming for each experiment run.

    Args:
        cfg (CfgNode): Configuration node
        run_id (int): Main for-loop iter id (the random seed or dataset split)
    """
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)


def run_loop_settings():
    """Create main loop execution settings based on the current cfg.

    Configures the main execution loop to run in one of two modes:
    1. 'multi-seed' - Reproduces default behaviour of GraphGym when
        args.repeats controls how many times the experiment run is repeated.
        Each iteration is executed with a random seed set to an increment from
        the previous one, starting at initial cfg.seed.
    2. 'multi-split' - Executes the experiment run over multiple dataset splits,
        these can be multiple CV splits or multiple standard splits. The random
        seed is reset to the initial cfg.seed value for each run iteration.

    Returns:
        List of run IDs for each loop iteration
        List of rng seeds to loop over
        List of dataset split indices to loop over
    """
    if len(cfg.run_multiple_splits) == 0:
        # 'multi-seed' run mode
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = seeds
    else:
        # 'multi-split' run mode
        if args.repeat != 1:
            raise NotImplementedError("Running multiple repeats of multiple "
                                      "splits in one run is not supported.")
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices


if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    dump_cfg(cfg)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    # Repeat for multiple experiment runs
    for run_id, seed, split_index in zip(*run_loop_settings()):
        # Set configurations for each run
        custom_set_run_dir(cfg, run_id)
        set_printing()
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        auto_select_device()
        if cfg.pretrained.dir:
            cfg = load_pretrained_model_cfg(cfg)
        logging.info(f"[*] Run ID {run_id}: seed={cfg.seed}, "
                     f"split_index={cfg.dataset.split_index}")
        logging.info(f"    Starting now: {datetime.datetime.now()}")
        # Set machine learning pipeline
        loaders = custom_create_loader(cfg)
        loggers = create_logger()
        model = create_model()
        if cfg.pretrained.dir:
            model = init_model_from_pretrained(
                model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
                cfg.pretrained.reset_prediction_head, seed=cfg.seed
            )
        optimizer = create_optimizer(model.parameters(),
                                     new_optimizer_config(cfg))
        scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)
        # Start training
        if cfg.train.mode == 'standard':
            if cfg.wandb.use:
                logging.warning("[W] WandB logging is not supported with the "
                                "default train.mode, set it to `custom`")
            datamodule = GraphGymDataModule()
            train(model, datamodule, logger=True)
        else:
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
                                       scheduler)
    # Aggregate results from different seeds
    try:
        agg_runs(cfg.out_dir, cfg.metric_best)
    except Exception as e:
        logging.info(f"Failed when trying to aggregate multiple runs: {e}")
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
    logging.info(f"[*] All done: {datetime.datetime.now()}")
