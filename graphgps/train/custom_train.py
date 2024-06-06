import logging
import time

import numpy as np
import torch
from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch

from graphgps.loss.subtoken_prediction_loss import subtoken_cross_entropy
from graphgps.utils import cfg_to_dict, flatten_dict, make_wandb_name

import torch.nn.functional as F

def calculate_ndcg_at_k(smallest_idx, num_nodes, x, triplets):
    k = cfg.ndcg_metric.k

    # calculate pairwise similarities and get closest k points
    x_eval = x[-num_nodes:]
    x_eval_normalized = F.normalize(x_eval, p=2, dim=-1)
    if cfg.model.edge_decoding == "cosine_similarity":
        # cosine similarity is large if vectors are close
        x_cosine_similarity = F.cosine_similarity(x_eval_normalized[None,:,:], x_eval_normalized[:,None,:], dim=-1)
        top_similarities_with_self, top_indices_with_self = torch.topk(x_cosine_similarity, k + 1)
    elif cfg.model.edge_decoding == "euclidean":
        # euclidean distance is small if vectors are close
        x_euclidean = F.pairwise_distance(x_eval_normalized[None,:,:], x_eval_normalized[:,None,:])
        top_similarities_with_self, top_indices_with_self = torch.topk(x_euclidean, k + 1, largest=False)
    elif cfg.model.edge_decoding == "dot":
        # dot product is large large if vectors are close
        x_dot = torch.sum(x_eval_normalized[None, :, :] * x_eval_normalized[:, None, :], dim=-1)
        top_similarities_with_self, top_indices_with_self = torch.topk(x_dot, k + 1)
    else:
        logging.info(f"edge decoding {cfg.model.edge_decoding} not supported for calculating NDCG")

    # best similarity is always similarity to self --> remove
    top_indices = top_indices_with_self[:, 1:]
    top_similarities = top_similarities_with_self[:, 1:]

    # create predictions of top k closest nodes
    top_predictions = torch.sigmoid(top_similarities)
    # prediction_mat = (top_predictions < cfg.model.thresh)
    prediction_mat = (top_predictions > cfg.model.thresh)

    # create set of all positive edges in graph
    # set contains both configurations: (v1, v2) and (v2, v1)
    if cfg.dataset.triplets_per_edge == 'two':
        all_edges = triplets[[0, 1]]
    else:
        all_edges = torch.cat((triplets[[0, 1]], triplets[[1, 0]]), 1)
    all_edges_mapped = all_edges - smallest_idx
    edge_set = set(map(tuple, all_edges_mapped.T.tolist())) 

    # create edges from the top k (can be either way around)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    row_indices = torch.arange(num_nodes).to(device).view(-1, 1).expand_as(top_indices)
    top_edges = torch.stack((row_indices, top_indices), dim=-1).reshape(-1, 2)
    
    # create mask indicating if top edges are actual positive edges or not
    # mask: 
    #   - has size (num_nodes, k) --> for each node there's 200 fields for the 200 closest nodes
    #   - is 1 if there is a positive edge, is 0 if there is no positive edge --> no edge
    mask = torch.tensor([tuple(edge.tolist()) in edge_set for edge in top_edges], device=device).view(num_nodes, k)

    # compute NDCG
    # discounting factor
    discounting = 1.0 / torch.arange(2, k + 2, dtype=torch.get_default_dtype()).log2().to(device)

    # denominator: 
    #   - count the number of actual neighbors per node --> y_count
    #   - clamp the count s.t. it's at most 200 --> y_count_clamped
    #   - the denominator is sum_k{d(k)} with k <= y_count_clamped
    #     therefore we calculate cumsum of discounting and index with
    #     y_count_clamped --> idcg
    y_count = torch.zeros(num_nodes, dtype=torch.long)
    for edge in all_edges_mapped.T:
        y_count[edge[0]] += 1
    y_count.to(device)
    y_count_clamped = y_count.clamp(max=k)
    discounting_summed = torch.cumsum(discounting, dim=0).to(device)
    idcg = discounting_summed[y_count_clamped]

    # nominator:
    #  - take AND of mask and prediction matrices to get a matrix that
    #    is 1 if the edge was predicted and is actually an edge and that
    #    is 0 if the edge was not predicted OR is actually not and edge
    #    --> pred_index_mat
    #  - make a column vector out of discounting --> discounting.view(1, -1)
    #  - multiply and sum the matrix and the column vector to get the nominator
    pred_index_mat = (mask & prediction_mat)
    dcg = (pred_index_mat * discounting.view(1, -1)).sum(dim=-1)
    
    # calculate ndcg
    ndcg_values = dcg / idcg
    ndcg = torch.mean(ndcg_values)

    # exclude nodes that have no positive edge in the graph
    nz_sum = 0.0
    nz_count = 0
    for i in range(num_nodes):
        if y_count[i] > 0:
            nz_sum +=  ndcg_values[i]
            nz_count += 1
    ndcg_without_zeroes = nz_sum / nz_count
    return ndcg.item(), ndcg_without_zeroes.item()

def process_model_output(x, triplets, pred, true):
    _true = true.detach().to('cpu', non_blocking=True)
    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    if pred.ndim > 1 and true.ndim == 1:
        pred = torch.nn.functional.log_softmax(pred, dim=-1)
    else:
        pred = torch.sigmoid(pred)
    _pred = pred.detach().to('cpu', non_blocking=True)

    x_triplets = x[triplets]
    x_triplets_normalized = F.normalize(x_triplets, p=2, dim=-1)
    anchor = x_triplets_normalized[0]
    positive = x_triplets_normalized[1]
    negative = x_triplets_normalized[2]

    return _pred, _true, anchor, positive, negative

def train_epoch(logger, loader, model, optimizer, scheduler, batch_accumulation, triplet_loss=None):
    model.train()
    optimizer.zero_grad()
    time_start = time.time()
    for iter, batch in enumerate(loader):
        batch.split = 'train'
        batch.to(torch.device(cfg.accelerator))
        if cfg.model.loss_fun == 'triplet':
            x, triplets, pred, true = model(batch)
            _pred, _true, anchor, positive, negative = process_model_output(x, triplets, pred, true)
            np.savetxt('pred.txt', _pred.numpy())
            loss = triplet_loss(anchor, positive, negative)
        else:
            pred, true = model(batch) # of type LightningModule
            if cfg.dataset.name == 'ogbg-code2':
                loss, pred_score = subtoken_cross_entropy(pred, true)
                _true = true
                _pred = pred_score
            else:
                loss, pred_score = compute_loss(pred, true)
                _true = true.detach().to('cpu', non_blocking=True)
                _pred = pred_score.detach().to('cpu', non_blocking=True)
        loss.backward()
        # Parameters update after accumulating gradients for given num. batches.
        if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg.optim.clip_grad_norm_value)
            optimizer.step()
            optimizer.zero_grad()
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name)
        time_start = time.time()


@torch.no_grad()
def eval_epoch(logger, loader, model, split='val', triplet_loss=None, calculate_ndcg=False, split_num_nodes=[None]*3):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.accelerator))
        if cfg.model.loss_fun == 'triplet':
            x, triplets, pred, true = model(batch)
            extra_stats = {}
            _pred, _true, anchor, positive, negative = process_model_output(x, triplets, pred, true)
            loss = triplet_loss(anchor, positive, negative)
            if calculate_ndcg:
                num_nodes_prev = split_num_nodes[0] if (split == 'val') else split_num_nodes[1]
                num_nodes = (split_num_nodes[1] - split_num_nodes[0]) if (split == 'val') else (split_num_nodes[2] - split_num_nodes[1])
                _, ndcg = calculate_ndcg_at_k(num_nodes_prev, num_nodes, x, triplets)
        else:
            if cfg.gnn.head == 'inductive_edge':
                pred, true, extra_stats = model(batch)
            else:
                #print(batch)
                #batch.x = torch.zeros((batch.num_nodes,1), device=batch.edge_index.device)
                pred, true = model(batch)
                extra_stats = {}
            if cfg.dataset.name == 'ogbg-code2':
                loss, pred_score = subtoken_cross_entropy(pred, true)
                _true = true
                _pred = pred_score
            else:
                loss, pred_score = compute_loss(pred, true)
                _true = true.detach().to('cpu', non_blocking=True)
                _pred = pred_score.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name,
                            ndcg=(ndcg if calculate_ndcg else None),
                            **extra_stats)
        time_start = time.time()


@register_train('custom')
def custom_train(loggers, loaders, model, optimizer, scheduler, loss=None, split_num_nodes=[None]*3):
    """
    Customized training pipeline.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler,
                                cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch %s', start_epoch)

    if cfg.wandb.use:
        try:
            import wandb
        except:
            raise ImportError('WandB is not installed.')
        if cfg.wandb.name == '':
            wandb_name = make_wandb_name(cfg)
        else:
            wandb_name = cfg.wandb.name
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                         name=wandb_name)
        run.config.update(cfg_to_dict(cfg))

    num_splits = len(loggers)
    split_names = ['val', 'test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        calculate_ndcg = ((cur_epoch + 1) % cfg.ndcg_metric.rate == 0) if cfg.ndcg_metric.use else False
        start_time = time.perf_counter()
        if cfg.dataset.name == 'PyG-OLGA_triplet':
            train_epoch(loggers[0], loaders[0], model, optimizer, scheduler,
                        cfg.optim.batch_accumulation, loss)
        else:
            train_epoch(loggers[0], loaders[0], model, optimizer, scheduler,
                        cfg.optim.batch_accumulation)
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                if cfg.dataset.name == 'PyG-OLGA_triplet':
                    eval_epoch(loggers[i], loaders[i], model,
                              split=split_names[i - 1], triplet_loss=loss, 
                              calculate_ndcg=calculate_ndcg,
                              split_num_nodes=split_num_nodes)
                else:
                    eval_epoch(loggers[i], loaders[i], model,
                              split=split_names[i - 1])
                perf[i].append(loggers[i].write_epoch(cur_epoch))
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        val_perf = perf[1]
        if cfg.optim.scheduler == 'reduce_on_plateau':
            scheduler.step(val_perf[-1]['loss'])
        else:
            scheduler.step()
        full_epoch_times.append(time.perf_counter() - start_time)
        # Checkpoint with regular frequency (if enabled).
        if cfg.train.enable_ckpt and not cfg.train.ckpt_best \
                and is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

        if cfg.wandb.use:
            run.log(flatten_dict(perf), step=cur_epoch)

        # Log current best stats on eval epoch.
        if is_eval_epoch(cur_epoch):
            best_epoch = np.array([vp['loss'] for vp in val_perf]).argmin()
            best_train = best_val = best_test = ""
            if cfg.metric_best != 'auto':
                # Select again based on val perf of `cfg.metric_best`.
                m = cfg.metric_best
                best_epoch = getattr(np.array([vp[m] for vp in val_perf]),
                                     cfg.metric_agg)()
                if m in perf[0][best_epoch]:
                    best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
                else:
                    # Note: For some datasets it is too expensive to compute
                    # the main metric on the training set.
                    best_train = f"train_{m}: {0:.4f}"
                best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
                best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

                if cfg.wandb.use:
                    bstats = {"best/epoch": best_epoch}
                    for i, s in enumerate(['train', 'val', 'test']):
                        bstats[f"best/{s}_loss"] = perf[i][best_epoch]['loss']
                        if m in perf[i][best_epoch]:
                            bstats[f"best/{s}_{m}"] = perf[i][best_epoch][m]
                            run.summary[f"best_{s}_perf"] = \
                                perf[i][best_epoch][m]
                        for x in ['hits@1', 'hits@3', 'hits@10', 'mrr']:
                            if x in perf[i][best_epoch]:
                                bstats[f"best/{s}_{x}"] = perf[i][best_epoch][x]
                    run.log(bstats, step=cur_epoch)
                    run.summary["full_epoch_time_avg"] = np.mean(full_epoch_times)
                    run.summary["full_epoch_time_sum"] = np.sum(full_epoch_times)
            # Checkpoint the best epoch params (if enabled).
            if cfg.train.enable_ckpt and cfg.train.ckpt_best and \
                    best_epoch == cur_epoch:
                save_ckpt(model, optimizer, scheduler, cur_epoch)
                if cfg.train.ckpt_clean:  # Delete old ckpt each time.
                    clean_ckpt()
            logging.info(
                f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                f"(avg {np.mean(full_epoch_times):.1f}s) | "
                f"Best so far: epoch {best_epoch}\t"
                f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
                f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
            )
            if hasattr(model, 'trf_layers'):
                # Log SAN's gamma parameter values if they are trainable.
                for li, gtl in enumerate(model.trf_layers):
                    if torch.is_tensor(gtl.attention.gamma) and \
                            gtl.attention.gamma.requires_grad:
                        logging.info(f"    {gtl.__class__.__name__} {li}: "
                                     f"gamma={gtl.attention.gamma.item()}")
    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    # close wandb
    if cfg.wandb.use:
        run.finish()
        run = None

    logging.info('Task done, results saved in %s', cfg.run_dir)


@register_train('inference-only')
def inference_only(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to run inference only.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    num_splits = len(loggers)
    split_names = ['train', 'val', 'test']
    perf = [[] for _ in range(num_splits)]
    cur_epoch = 0
    start_time = time.perf_counter()

    for i in range(0, num_splits):
        eval_epoch(loggers[i], loaders[i], model,
                   split=split_names[i])
        perf[i].append(loggers[i].write_epoch(cur_epoch))

    best_epoch = 0
    best_train = best_val = best_test = ""
    if cfg.metric_best != 'auto':
        # Select again based on val perf of `cfg.metric_best`.
        m = cfg.metric_best
        if m in perf[0][best_epoch]:
            best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
        else:
            # Note: For some datasets it is too expensive to compute
            # the main metric on the training set.
            best_train = f"train_{m}: {0:.4f}"
        best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
        best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

    logging.info(
        f"> Inference | "
        f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
        f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
        f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
    )
    logging.info(f'Done! took: {time.perf_counter() - start_time:.2f}s')
    for logger in loggers:
        logger.close()


@register_train('PCQM4Mv2-inference')
def ogblsc_inference(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to run inference on OGB-LSC PCQM4Mv2.

    Args:
        loggers: Unused, exists just for API compatibility
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    from ogb.lsc import PCQM4Mv2Evaluator
    evaluator = PCQM4Mv2Evaluator()

    num_splits = 3
    split_names = ['valid', 'test-dev', 'test-challenge']
    assert len(loaders) == num_splits, "Expecting 3 particular splits."

    # Check PCQM4Mv2 prediction targets.
    logging.info(f"0 ({split_names[0]}): {len(loaders[0].dataset)}")
    assert(all([not torch.isnan(d.y)[0] for d in loaders[0].dataset]))
    logging.info(f"1 ({split_names[1]}): {len(loaders[1].dataset)}")
    assert(all([torch.isnan(d.y)[0] for d in loaders[1].dataset]))
    logging.info(f"2 ({split_names[2]}): {len(loaders[2].dataset)}")
    assert(all([torch.isnan(d.y)[0] for d in loaders[2].dataset]))

    model.eval()
    for i in range(num_splits):
        all_true = []
        all_pred = []
        for batch in loaders[i]:
            batch.to(torch.device(cfg.accelerator))
            pred, true = model(batch)
            all_true.append(true.detach().to('cpu', non_blocking=True))
            all_pred.append(pred.detach().to('cpu', non_blocking=True))
        all_true, all_pred = torch.cat(all_true), torch.cat(all_pred)

        if i == 0:
            input_dict = {'y_pred': all_pred.squeeze(),
                          'y_true': all_true.squeeze()}
            result_dict = evaluator.eval(input_dict)
            logging.info(f"{split_names[i]}: MAE = {result_dict['mae']}")  # Get MAE.
        else:
            input_dict = {'y_pred': all_pred.squeeze()}
            evaluator.save_test_submission(input_dict=input_dict,
                                           dir_path=cfg.run_dir,
                                           mode=split_names[i])


@ register_train('log-attn-weights')
def log_attn_weights(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to inference on the test set and log the attention
    weights in Transformer modules.

    Args:
        loggers: Unused, exists just for API compatibility
        loaders: List of loaders
        model (torch.nn.Module): GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    import os.path as osp
    from torch_geometric.loader.dataloader import DataLoader
    from graphgps.utils import unbatch, unbatch_edge_index

    start_time = time.perf_counter()

    # The last loader is a test set.
    l = loaders[-1]
    # To get a random sample, create a new loader that shuffles the test set.
    loader = DataLoader(l.dataset, batch_size=l.batch_size,
                        shuffle=True, num_workers=0)

    output = []
    # batch = next(iter(loader))  # Run one random batch.
    for b_index, batch in enumerate(loader):
        bsize = batch.batch.max().item() + 1  # Batch size.
        if len(output) >= 128:
            break
        print(f">> Batch {b_index}:")

        X_orig = unbatch(batch.x.cpu(), batch.batch.cpu())
        batch.to(torch.device(cfg.accelerator))
        model.eval()
        model(batch)

        # Unbatch to individual graphs.
        X = unbatch(batch.x.cpu(), batch.batch.cpu())
        edge_indices = unbatch_edge_index(batch.edge_index.cpu(),
                                          batch.batch.cpu())
        graphs = []
        for i in range(bsize):
            graphs.append({'num_nodes': len(X[i]),
                           'x_orig': X_orig[i],
                           'x_final': X[i],
                           'edge_index': edge_indices[i],
                           'attn_weights': []  # List with attn weights in layers from 0 to L-1.
                           })

        # Iterate through GPS layers and pull out stored attn weights.
        for l_i, (name, module) in enumerate(model.model.layers.named_children()):
            if hasattr(module, 'attn_weights'):
                print(l_i, name, module.attn_weights.shape)
                for g_i in range(bsize):
                    # Clip to the number of nodes in this graph.
                    # num_nodes = graphs[g_i]['num_nodes']
                    # aw = module.attn_weights[g_i, :num_nodes, :num_nodes]
                    aw = module.attn_weights[g_i]
                    graphs[g_i]['attn_weights'].append(aw.cpu())
        output += graphs

    logging.info(
        f"[*] Collected a total of {len(output)} graphs and their "
        f"attention weights for {len(output[0]['attn_weights'])} layers.")

    # Save the graphs and their attention stats.
    save_file = osp.join(cfg.run_dir, 'graph_attn_stats.pt')
    logging.info(f"Saving to file: {save_file}")
    torch.save(output, save_file)

    logging.info(f'Done! took: {time.perf_counter() - start_time:.2f}s')
