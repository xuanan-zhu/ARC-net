from torch import nn
from torchdrug import core, layers, data
from collections.abc import Sequence
from typing import List, Tuple, Union, Any
import torch
from AEP import AEP, GA_layer
from torch.nn.utils.rnn import pad_sequence
from torch_scatter import scatter_add

def div_input(layer_input, num_residues):
    input = []
    end_index = 0
    for i in num_residues:
        start_index = end_index
        end_index = i + end_index
        input.append(layer_input[start_index:end_index])

    return input

def augment_point_cloud(point_cloud, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    device = point_cloud.device

    # random rotation
    angles = torch.rand(3) * 2 * 3.141592653589793  # [0, 2π]
    rotation_matrix = torch.tensor([
        [torch.cos(angles[2]) * torch.cos(angles[1]),
         torch.cos(angles[2]) * torch.sin(angles[1]) * torch.sin(angles[0]) - torch.sin(angles[2]) * torch.cos(
             angles[0]),
         torch.cos(angles[2]) * torch.sin(angles[1]) * torch.cos(angles[0]) + torch.sin(angles[2]) * torch.sin(
             angles[0])],

        [torch.sin(angles[2]) * torch.cos(angles[1]),
         torch.sin(angles[2]) * torch.sin(angles[1]) * torch.sin(angles[0]) + torch.cos(angles[2]) * torch.cos(
             angles[0]),
         torch.sin(angles[2]) * torch.sin(angles[1]) * torch.cos(angles[0]) - torch.cos(angles[2]) * torch.sin(
             angles[0])],

        [-torch.sin(angles[1]),
         torch.cos(angles[1]) * torch.sin(angles[0]),
         torch.cos(angles[1]) * torch.cos(angles[0])]
    ])
    rotation_matrix = rotation_matrix.to(device)
    rotated_point_cloud = point_cloud @ rotation_matrix.T

    # random translation
    translation_vector = (torch.rand(3) * 2 - 1).to(device)  # [-1, 1]
    translated_point_cloud = rotated_point_cloud + translation_vector

    # random flip
    should_flip = torch.rand(1).item() > 0.5
    flipped_point_cloud = torch.flip(translated_point_cloud, dims=[1]) if should_flip else translated_point_cloud

    return flipped_point_cloud

class MLP(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 activation: nn.Module = nn.ReLU(),
                 use_bn: bool = False,
                 use_dropout: bool = False):
        super().__init__()
        self.layers = nn.ModuleList()

        current_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, h_dim))
            if use_bn:
                self.layers.append(nn.BatchNorm1d(h_dim))
            self.layers.append(activation)
            if use_dropout:
                self.layers.append(nn.Dropout(0.5))  
            current_dim = h_dim

        self.layers.append(nn.Linear(current_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def pad_pointclouds(pointclouds, all_indices_feature):
    lengths = torch.tensor([pc.shape[0] for pc in pointclouds], device=pointclouds[0].device)

    padded_pointclouds = pad_sequence(pointclouds, batch_first=True, padding_value=0.0)
    padded_atom_types = pad_sequence(all_indices_feature, batch_first=True, padding_value=38.0)

    max_N = padded_pointclouds.size(1)

    assert padded_pointclouds.size(1) == padded_atom_types.size(1), "Lengths of pointclouds and all_indices_feature must match."

    masks = (torch.arange(max_N, device=padded_pointclouds.device)[None, :] < lengths[:, None])

    return padded_pointclouds, padded_atom_types, masks

class ARC_net(nn.Module, core.Configurable):
    """
    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        num_relation (int): number of relations
        edge_input_dim (int, optional): dimension of edge features
        num_angle_bin (int, optional): number of bins to discretize angles between edges.
            The discretized angles are used as relations in edge message passing.
            If not provided, edge message passing is disabled.
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """

    def __init__(self, input_dim, hidden_dims, num_relation, edge_input_dim=None, num_angle_bin=None,
                 short_cut=False, batch_norm=False, activation="relu", concat_hidden=False, readout="sum"):
        super(ARC_net, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = 256 * len(hidden_dims)
        self.output_dim_p = 256
        self.dims = [input_dim] + list(hidden_dims)
        self.edge_dims = [edge_input_dim] + list(hidden_dims)

        self.num_relation = num_relation
        self.num_angle_bin = num_angle_bin
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.batch_norm = batch_norm

        self.AEPs = nn.ModuleList()
        self.mlp_structures2env = nn.ModuleList()
        self.mlp_env2structures = nn.ModuleList()
        in_channels = [64, 128, 128, 128, 128, 128]
        mlp_list = [[128], [128], [128], [128], [128], [128]]
        outputs = [256, 256, 256, 256, 256, 256]
        nsamples = [5, 5, 5, 5, 5, 5]
        npoints = [64, 64, 32, 32, 8, 8]

        for i in range(len(nsamples)):
            self.AEPs.append(
                AEP(in_channel=in_channels[i], output=outputs[i], mlp_list=mlp_list[i], nsample=nsamples[i],
                    npoint=npoints[i]))
            self.mlp_structures2env.append(
                MLP(input_dim=hidden_dims[i - 1] + mlp_list[i - 1][-1], hidden_dims=[hidden_dims[i - 1]],
                    output_dim=mlp_list[i - 1][-1],
                    activation=nn.ReLU(), use_bn=True, use_dropout=False))
            self.mlp_env2structures.append(
                MLP(input_dim=hidden_dims[i - 1] + outputs[i - 1], hidden_dims=[hidden_dims[i - 1]],
                    output_dim=hidden_dims[i - 1],
                    activation=nn.ReLU(), use_bn=True, use_dropout=False))
        self.atom_type_embedding = nn.Embedding(38 + 1, 64, padding_idx=38) 
        self.res_type_embedding = nn.Linear(21, 64) 

        self.Last_GA_layer = GA_layer(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=mlp_list[-1][-1] + 3,
            mlp=[outputs[-1]],
            group_all=True,
        )

        self.layers = nn.ModuleList()
        self.point_batch_norms = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.point_batch_norms.append(nn.BatchNorm1d(outputs[i]))
            self.layers.append(layers.GeometricRelationalGraphConv(self.dims[i], self.dims[i + 1], num_relation,
                                                                   None, batch_norm, activation))

        if self.num_angle_bin:
            self.spatial_line_graph = layers.SpatialLineGraph(num_angle_bin)
            self.edge_layers = nn.ModuleList()
            self.mixture = nn.ModuleList()
            for i in range(len(self.edge_dims) - 1):
                self.edge_layers.append(
                    layers.GeometricRelationalGraphConv(self.edge_dims[i], self.edge_dims[i + 1], num_angle_bin,
                                                        None, batch_norm, activation))
                self.mixture.append(nn.Linear(self.edge_dims[i + 1] * num_relation, self.edge_dims[i + 1]))

        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)

        self.activate = nn.ReLU()
        self.last_point_batch_norms = nn.BatchNorm1d(outputs[-1])

    def forward(self, graph, graph_ori, all_loss=None, metric=None):
        """
        Compute the node representations and the graph representation(s).

        Parameters:
            graph (Graph): :math:`n` graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``node_feature`` and ``graph_feature`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        layer_input1 = graph.node_feature.float()

        hiddens_list = []

        if self.num_angle_bin:
            line_graph = self.spatial_line_graph(graph)
            edge_input = line_graph.node_feature.float()
            res_keep = None

        # The point cloud part
        # Point cloud randomization
        mask = (graph_ori.atom_name == data.Protein.atom_name2id["CA"])
        all_atom_pos = augment_point_cloud(graph_ori.node_position)
        all_atom_feature = graph_ori.atom_name
        ca_pos = all_atom_pos[mask]
        all_atom_pos_list = div_input(all_atom_pos, graph_ori.num_atoms)
        all_atom_feature_list = div_input(all_atom_feature, graph_ori.num_atoms)
        ca_pos_list = div_input(ca_pos, graph.num_residues)

        all_indices_xyz = []
        all_indices_feature = []
        for i in range(len(ca_pos_list)):
            # Calculate the distance matrix
            sqrdists = torch.cdist(ca_pos_list[i], all_atom_pos_list[i], p=2)
            mask = sqrdists < 10  

            # Vectorize all atoms
            rows, cols = mask.nonzero(as_tuple=True)
            selected_ca = ca_pos_list[i][rows]
            selected_atom = all_atom_pos_list[i][cols]
            delta = selected_atom - selected_ca

            counts = mask.sum(dim=1).tolist()
            indices_xyz = torch.split(delta, counts)
            indices_feature = torch.split(all_atom_feature_list[i][cols], counts)

            all_indices_xyz.extend(indices_xyz)
            all_indices_feature.extend(indices_feature)

        padded_coords, padded_atom_types, coord_masks = pad_pointclouds(all_indices_xyz, all_indices_feature)
        padded_atom_types = self.atom_type_embedding(padded_atom_types)
        layer_input1 = self.res_type_embedding(layer_input1)

        cur_xyz = padded_coords
        cur_points = padded_atom_types

        for i in range(len(self.layers)):
            if i > 0:
                gcn_context = layer_input1
                expanded_gcn_context = gcn_context.unsqueeze(1).expand(-1, cur_points.shape[1], -1)
                mix_cur_points = torch.cat([cur_points, expanded_gcn_context], dim=-1)
                n1, n2, d = mix_cur_points.shape
                mix_cur_points = mix_cur_points.reshape(n1 * n2, d)
                cur_points = self.mlp_structures2env[i](mix_cur_points).reshape(n1, n2, -1) + cur_points

                mix_feature = torch.cat([layer_input1, all_points], dim=-1)
                layer_input1 = self.mlp_env2structures[i](mix_feature) + layer_input1

            if i == len(self.layers) - 1:
                cur_xyz, cur_points, last_all_points = self.AEPs[i](cur_xyz, cur_points, None, extract_features=False)
                all_points = self.Last_GA_layer(cur_xyz, cur_points)[1].squeeze(-1)
            else:
                cur_xyz, cur_points, all_points = self.AEPs[i](cur_xyz, cur_points, None, extract_features=False)
            all_points = self.point_batch_norms[i](all_points)

            hidden1 = self.layers[i](graph, layer_input1)
            if self.short_cut and hidden1.shape == layer_input1.shape:
                hidden1 = hidden1 + layer_input1

            if self.num_angle_bin:
                edge_hidden = self.edge_layers[i](line_graph, edge_input)

                edge_weight = graph.edge_weight.unsqueeze(-1)
                node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]

                update = scatter_add(edge_hidden * edge_weight, node_out, dim=0,
                                     dim_size=graph.num_node * self.num_relation)
                update = update.view(graph.num_node, self.num_relation * edge_hidden.shape[1])

                update = self.mixture[i](update)
                update = self.edge_layers[i].activation(update)

                if res_keep and update.shape == res_keep.shape:
                    update = update + res_keep
                    res_keep = update

                edge_input = edge_hidden

                hidden1 = hidden1 + update

            if self.batch_norm:
                hidden1 = self.batch_norms[i](hidden1)

            hiddens_list.append(hidden1)
            layer_input1 = hidden1

        node_feature = torch.cat(hiddens_list, dim=-1)
        graph_feature = self.readout(graph, node_feature)

        point_result = self.readout(graph, self.last_point_batch_norms(last_all_points))

        return {
            "graph_feature": graph_feature,
            "node_feature": None,
            "point_result": point_result

        }