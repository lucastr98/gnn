from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_ndcg_metric')
def set_cfg_ndcg_metric(cfg):
    # NDCG group
    cfg.ndcg_metric = CN()

    # Use ndcg or not
    cfg.ndcg_metric.use = False

    # k of NDCG@k
    cfg.ndcg_metric.k = 200

    # rate of NDCG calculation --> every {cfg.ndcg.rate} epochs ndcg is calculated
    cfg.ndcg_metric.rate = 10
