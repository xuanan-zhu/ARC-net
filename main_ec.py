import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import sys
import math
import pprint
import random
import time
import logging
import yaml
import numpy as np
from torch import nn
from torch.optim import lr_scheduler
from torchdrug import core, models, tasks, datasets, utils, layers, data, metrics
from torchdrug.utils import comm
from torchdrug.layers import geometry
from easydict import EasyDict
from collections.abc import Sequence
from torchdrug.layers import functional
import jinja2
from jinja2 import meta
import easydict
from torch import distributed as dist
import torch
import torch.nn.functional as F
from torchdrug.data import Graph
from torch.nn.utils.rnn import pad_sequence
from ARC_net import ARC_net
from datetime import time, datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
logger = logging.getLogger(__file__)

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def get_root_logger(file=True):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    if file:
        handler = logging.FileHandler("log.txt")
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def create_working_directory(cfg):
    file_name = "working_dir.tmp"
    world_size = comm.get_world_size()
    if world_size > 1 and not dist.is_initialized():
        comm.init_process_group("nccl", init_method="env://")

    current_time = datetime.now()
    time_str = current_time.strftime("%Y-%m-%d-%H-%M-%S")

    working_dir = os.path.join(os.path.expanduser(cfg.output_dir),
                               "tasks",
                               "paper",
                               "ARC-net",
                               time_str)

    # synchronize working directory
    if comm.get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(working_dir)
        os.makedirs(working_dir)
    comm.synchronize()
    if comm.get_rank() != 0:
        with open(file_name, "r") as fin:
            working_dir = fin.read()
    comm.synchronize()
    if comm.get_rank() == 0:
        os.remove(file_name)

    os.chdir(working_dir)
    return working_dir



def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    return cfg


def build_downstream_solver(cfg, dataset):
    train_set, valid_set, test_set = dataset.split()
    if comm.get_rank() == 0:
        logger.warning(dataset)
        logger.warning("#train: %d, #valid: %d, #test: %d" % (len(train_set), len(valid_set), len(test_set)))

    task = MultipleBinaryClassification(model=gearnet, graph_construction_model=graph_construction_model,
                                        num_mlp_layer=3,
                                        criterion="bce",
                                        metric=["auprc@micro", "f1_max"], task=[_ for _ in range(len(dataset.tasks))])

    cfg.optimizer.params = task.parameters()
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)

    if "scheduler" not in cfg:
        scheduler = None
    elif cfg.scheduler["class"] == "ReduceLROnPlateau":
        cfg.scheduler.pop("class")
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **cfg.scheduler)
    else:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)
        cfg.engine.scheduler = scheduler

    solver = core.Engine(task, train_set, valid_set, test_set, optimizer, gpus=[0, 1, 2, 3], batch_size=2, log_interval=1000)

    if "lr_ratio" in cfg:
        cfg.optimizer.params = [
            {'params': solver.model.model.parameters(), 'lr': cfg.optimizer.lr * cfg.lr_ratio},
            {'params': solver.model.mlp.parameters(), 'lr': cfg.optimizer.lr}
        ]
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)
        solver.optimizer = optimizer
    elif "sequence_model_lr_ratio" in cfg:
        assert cfg.task.model["class"] == "FusionNetwork"
        cfg.optimizer.params = [
            {'params': solver.model.model.sequence_model.parameters(),
             'lr': cfg.optimizer.lr * cfg.sequence_model_lr_ratio},
            {'params': solver.model.model.structure_model.parameters(), 'lr': cfg.optimizer.lr},
            {'params': solver.model.mlp.parameters(), 'lr': cfg.optimizer.lr}
        ]
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)
        solver.optimizer = optimizer

    if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **cfg.scheduler)
    elif scheduler is not None:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)
        solver.scheduler = scheduler

    if cfg.get("checkpoint") is not None:
        solver.load(cfg.checkpoint)

    if cfg.get("model_checkpoint") is not None:
        if comm.get_rank() == 0:
            logger.warning("Load checkpoint from %s" % cfg.model_checkpoint)
        cfg.model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
        model_dict = torch.load(cfg.model_checkpoint, map_location=torch.device('cpu'))
        task.model.load_state_dict(model_dict)

    return solver, scheduler

class MultipleBinaryClassification(tasks.Task, core.Configurable):
    """
    Multiple binary classification task for graphs / molecules / proteins.

    Parameters:
        model (nn.Module): graph representation model
        task (list of int, optional): training task id(s).
        criterion (list or dict, optional): training criterion(s). For dict, the keys are criterions and the values
            are the corresponding weights. Available criterions are ``bce``.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``auroc@macro``, ``auprc@macro``, ``auroc@micro``, ``auprc@micro`` and ``f1_max``.
        num_mlp_layer (int, optional): number of layers in the MLP prediction head
        normalization (bool, optional): whether to normalize the target
        reweight (bool, optional): whether to re-weight tasks according to the number of positive samples
        graph_construction_model (nn.Module, optional): graph construction model
        verbose (int, optional): output verbose level
    """

    eps = 1e-10
    _option_members = {"criterion", "metric"}

    def __init__(self, model, task=(), criterion="bce", metric=("auprc@micro", "f1_max"), num_mlp_layer=1,
                 normalization=True, reweight=False, graph_construction_model=None, verbose=0):
        super(MultipleBinaryClassification, self).__init__()
        self.model = model
        self.task = task
        self.register_buffer("task_indices", torch.LongTensor(task))
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        self.normalization = normalization
        self.reweight = reweight
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose

        hidden_dims1 = [self.model.output_dim] * (self.num_mlp_layer - 1)
        hidden_dims2 = [self.model.output_dim_p] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(self.model.output_dim, hidden_dims1 + [len(task)])
        self.mlp2 = layers.MLP(self.model.output_dim_p, hidden_dims2 + [len(task)])

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the weight for each task on the training set.
        """
        values = []
        for data in train_set:
            values.append(data["targets"][self.task_indices])
        values = torch.stack(values, dim=0)

        if self.reweight:
            num_positive = values.sum(dim=0)
            weight = (num_positive.mean() / num_positive).clamp(1, 10)
        else:
            weight = torch.ones(len(self.task), dtype=torch.float)

        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, pred2 = self.predict(batch, all_loss, metric, training=True)
        target = self.target(batch)

        for criterion, weight in self.criterion.items():
            if criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                loss2 = F.binary_cross_entropy_with_logits(pred2, target, reduction="none")
                loss = loss + loss2
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)
            loss = loss.mean(dim=0)
            loss = (loss * self.weight).sum() / self.weight.sum()

            name = tasks._get_criterion_name(criterion)
            metric[name] = loss
            all_loss += loss * weight

        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None, training=False):
        # torch.cuda.empty_cache()
        graph = batch["graph"]
        graph_ori = None
        if self.graph_construction_model:
            graph, graph_ori = self.graph_construction_model(batch)
        output = self.model(graph=graph, graph_ori=graph_ori, all_loss=all_loss, metric=metric)
        pred = self.mlp(output["graph_feature"])
        if training:
            pred2 = self.mlp2(output["point_result"])
            return pred, pred2
        else:
            return pred

    def target(self, batch):
        target = batch["targets"][:, self.task_indices]
        return target

    def evaluate(self, pred, target):
        metric = {}
        for _metric in self.metric:
            if _metric == "auroc@micro":
                score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
            elif _metric == "auroc@macro":
                score = metrics.variadic_area_under_roc(pred, target.long(), dim=0).mean()
            elif _metric == "auprc@micro":
                score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
            elif _metric == "auprc@macro":
                score = metrics.variadic_area_under_prc(pred, target.long(), dim=0).mean()
            elif _metric == "f1_max":
                score = metrics.f1_max(pred, target)
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            metric[name] = score

        return metric


class GraphConstruction(nn.Module, core.Configurable):
    """
    Construct a new graph from an existing graph.

    See `torchdrug.layers.geometry` for a full list of available node and edge layers.

    Parameters:
        node_layers (list of nn.Module, optional): modules to construct nodes of the new graph
        edge_layers (list of nn.Module, optional): modules to construct edges of the new graph
        edge_feature (str, optional): edge features in the new graph.
            Available features are ``residue_type``, ``gearnet``.

            1. For ``residue_type``, the feature of the edge :math:`e_{ij}` between residue :math:`i` and residue
                :math:`j` is the concatenation ``[residue_type(i), residue_type(j)]``.
            2. For ``gearnet``, the feature of the edge :math:`e_{ij}` between residue :math:`i` and residue :math:`j`
                is the concatenation ``[residue_type(i), residue_type(j), edge_type(e_ij),
                sequential_distance(i,j), spatial_distance(i,j)]``.

    .. note::
        You may customize your own edge features by inheriting this class and define a member function
        for your features. Use ``edge_feature="my_feature"`` to call the following feature function.

        .. code:: python

            def edge_my_feature(self, graph, edge_list, num_relation):
                ...
                return feature # the first dimension must be ``graph.num_edge``
    """

    max_seq_dist = 10

    def __init__(self, node_layers=None, edge_layers=None, edge_feature="residue_type"):
        super(GraphConstruction, self).__init__()
        if node_layers is None:
            self.node_layers = nn.ModuleList()
        else:
            self.node_layers = nn.ModuleList(node_layers)
        if edge_layers is None:
            edge_layers = nn.ModuleList()
        else:
            edge_layers = nn.ModuleList(edge_layers)
        self.edge_layers = edge_layers
        self.edge_feature = edge_feature

    def edge_residue_type(self, graph, edge_list, num_relation):
        node_in, node_out, _ = edge_list.t()
        residue_in, residue_out = graph.atom2residue[node_in], graph.atom2residue[node_out]
        in_residue_type = graph.residue_type[residue_in]
        out_residue_type = graph.residue_type[residue_out]

        return torch.cat([
            functional.one_hot(in_residue_type, len(data.Protein.residue2id)),
            functional.one_hot(out_residue_type, len(data.Protein.residue2id))
        ], dim=-1)

    def edge_gearnet(self, graph, edge_list, num_relation):
        node_in, node_out, r = edge_list.t()
        residue_in, residue_out = graph.atom2residue[node_in], graph.atom2residue[node_out]
        in_residue_type = graph.residue_type[residue_in]
        out_residue_type = graph.residue_type[residue_out]
        sequential_dist = torch.abs(residue_in - residue_out)
        spatial_dist = (graph.node_position[node_in] - graph.node_position[node_out]).norm(dim=-1)

        return torch.cat([
            functional.one_hot(in_residue_type, len(data.Protein.residue2id)),
            functional.one_hot(out_residue_type, len(data.Protein.residue2id)),
            functional.one_hot(r, num_relation),
            functional.one_hot(sequential_dist.clamp(max=self.max_seq_dist), self.max_seq_dist + 1),
            spatial_dist.unsqueeze(-1)
        ], dim=-1)

    def apply_node_layer(self, graph):
        for layer in self.node_layers:
            graph = layer(graph)
        return graph

    def apply_edge_layer(self, graph):
        if not self.edge_layers:
            return graph

        edge_list = []
        num_edges = []
        num_relations = []
        for layer in self.edge_layers:
            edges, num_relation = layer(graph)
            edge_list.append(edges)
            num_edges.append(len(edges))
            num_relations.append(num_relation)

        edge_list = torch.cat(edge_list)
        num_edges = torch.tensor(num_edges, device=graph.device)
        num_relations = torch.tensor(num_relations, device=graph.device)
        num_relation = num_relations.sum()
        offsets = (num_relations.cumsum(0) - num_relations).repeat_interleave(num_edges)
        edge_list[:, 2] += offsets

        # reorder edges into a valid PackedGraph
        node_in = edge_list[:, 0]
        edge2graph = graph.node2graph[node_in]
        order = edge2graph.argsort()
        edge_list = edge_list[order]
        num_edges = edge2graph.bincount(minlength=graph.batch_size)
        offsets = (graph.num_cum_nodes - graph.num_nodes).repeat_interleave(num_edges)

        if hasattr(self, "edge_%s" % self.edge_feature):
            edge_feature = getattr(self, "edge_%s" % self.edge_feature)(graph, edge_list, num_relation)
        elif self.edge_feature is None:
            edge_feature = None
        else:
            raise ValueError("Unknown edge feature `%s`" % self.edge_feature)
        data_dict, meta_dict = graph.data_by_meta(include=(
            "node", "residue", "node reference", "residue reference", "graph"
        ))

        if isinstance(graph, data.PackedProtein):
            data_dict["num_residues"] = graph.num_residues
        if isinstance(graph, data.PackedMolecule):
            data_dict["bond_type"] = torch.zeros_like(edge_list[:, 2])
        return type(graph)(edge_list, num_nodes=graph.num_nodes, num_edges=num_edges, num_relation=num_relation,
                           view=graph.view, offsets=offsets, edge_feature=edge_feature,
                           meta_dict=meta_dict, **data_dict)

    def forward(self, batch):
        """
        Generate a new graph based on the input graph and pre-defined node and edge layers.

        Parameters:
            graph (Graph): :math:`n` graph(s)

        Returns:
            graph (Graph): new graph(s)
        """
        graph_ori = batch["graph"]
        graph = self.apply_node_layer(graph_ori)
        graph = self.apply_edge_layer(graph)
        return graph, graph_ori


def div_input(layer_input, num_residues):
    input = []
    end_index = 0
    for i in num_residues:
        start_index = end_index
        end_index = i + end_index
        input.append(layer_input[start_index:end_index])

    return input

class Readout(nn.Module):

    def __init__(self, type="node"):
        super(Readout, self).__init__()
        self.type = type

    def get_index2graph(self, graph):
        if self.type == "node":
            input2graph = graph.node2graph
        elif self.type == "edge":
            input2graph = graph.edge2graph
        elif self.type == "residue":
            input2graph = graph.residue2graph
        else:
            raise ValueError("Unknown input type `%s` for readout functions" % self.type)
        return input2graph





def train_and_validate(cfg, solver, scheduler):
    if cfg.train.num_epoch == 0:
        return

    step = math.ceil(2.0)
    best_result = float("-inf")
    best_epoch = -1

    for i in range(0, cfg.train.num_epoch, step):
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)
        solver.train(**kwargs)
        solver.save("model_epoch_%d.pth" % solver.epoch)
        print("learning rate:")
        print(scheduler.optimizer.param_groups[0]['lr'])
        print("scheduler.best:", scheduler.best)
        solver.evaluate("train")
        metric = solver.evaluate("valid")
        solver.evaluate("test")

        result = metric[cfg.metric]
        if result > best_result:
            best_result = result
            best_epoch = solver.epoch
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(result)

    solver.load("model_epoch_%d.pth" % best_epoch)
    return solver


def test(cfg, solver):
    solver.evaluate("valid")
    return solver.evaluate("test")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='ARC-net')
    parser.add_argument('--seed', type=int, default = 1024)
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("seed:",seed)

    gearnet = ARC_net(input_dim=64, hidden_dims=[256, 256, 256, 256, 256, 256],
                                                        num_relation=7,
                                                        edge_input_dim=52+7, num_angle_bin=8,
                                                        batch_norm=True, concat_hidden=False, short_cut=True,
                                                        readout="sum")
    graph_construction_model = GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
                                                 edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                              geometry.KNNEdge(k=10, min_distance=5),
                                                              geometry.SequentialEdge(max_distance=2)],
                                                 edge_feature="gearnet")
    cfg = {'output_dir': '~/scratch/protein_output',
           'dataset': {'class': 'EnzymeCommission', 'path': '~/scratch/protein-datasets/', 'test_cutoff': 0.95,
                       'atom_feature': None, 'bond_feature': None,
                       'transform': {'class': 'ProteinView', 'view': 'residue'}},
           'task': MultipleBinaryClassification(model=gearnet, graph_construction_model=graph_construction_model,
                                                num_mlp_layer=3,
                                                criterion="bce",
                                                metric=["auprc@micro", "f1_max"]),
           'optimizer': {'class': 'AdamW', 'lr': 0.0001, 'weight_decay': 0},
           'scheduler': {'class': 'ReduceLROnPlateau', 'factor': 0.5, 'patience': 2, 'verbose': True, 'mode': 'max'},
           'engine': {'gpus': None, 'batch_size': 2, 'log_interval': 1000}, 'model_checkpoint': None,
           'metric': 'f1_max',
           'train': {'num_epoch': 200}}
    cfg = EasyDict(cfg)
    working_dir = create_working_directory(cfg)

    logger = get_root_logger()
    if comm.get_rank() == 0:
        logger.warning(pprint.pformat(cfg))

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver, scheduler = build_downstream_solver(cfg, dataset)

    train_and_validate(cfg, solver, scheduler)
    test(cfg, solver)


