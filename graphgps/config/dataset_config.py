from torch_geometric.graphgym.register import register_config


@register_config('dataset_cfg')
def dataset_cfg(cfg):
    """Dataset-specific config options.
    """

    # The number of node types to expect in TypeDictNodeEncoder.
    cfg.dataset.node_encoder_num_types = 0

    # The number of edge types to expect in TypeDictEdgeEncoder.
    cfg.dataset.edge_encoder_num_types = 0

    # VOC/COCO Superpixels dataset version based on SLIC compactness parameter.
    cfg.dataset.slic_compactness = 10

    # infer-link parameters (e.g., edge prediction task)
    cfg.dataset.infer_link_label = "None"

    cfg.dataset.num_pos_samples = 10000

    cfg.dataset.num_neg_samples = 100000

    cfg.dataset.num_samples = 100000

    # can be one or two
    #   - one: samples one negative edge randomly from all edges involving one 
    #          of the two edges of the positive edge
    #   - two: samples two negative edges randomly, one from all edges involving
    #          the first node and one from all edges involving the second node
    #          of the positive edge
    cfg.dataset.triplets_per_edge = "two"
    
    cfg.dataset.embedding = None

    cfg.dataset.features = 'random'
