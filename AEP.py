from torch_cluster import fps
import torch.nn.functional as F
import torch
from torch import nn

def knn_point(nsample, xyz, new_xyz, mask=None):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
        mask: mask for xyz, [B, N]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    if mask is not None:
        mask = mask.unsqueeze(1).expand_as(sqrdists)
        sqrdists[~mask] = float('inf')

    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points, mask=None):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    xyz = xyz.contiguous()

    fps_idx = farthest_point_sample(xyz, npoint, mask).long() # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)
    new_points = index_points(points, fps_idx)
    # new_xyz = xyz[:]
    # new_points = points[:]

    idx = knn_point(nsample, xyz, new_xyz, mask)
    #idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points

from torch.nn.utils.rnn import pad_sequence
def pad_pointclouds(pointclouds):
    lengths = torch.tensor([pc.shape[0] for pc in pointclouds], device=pointclouds[0].device)

    padded_pointclouds = pad_sequence(pointclouds, batch_first=True, padding_value=0.0)

    max_N = padded_pointclouds.size(1)
    masks = (torch.arange(max_N, device=padded_pointclouds.device)[None, :] < lengths[:, None])

    return padded_pointclouds, masks

def farthest_point_sample(xyz: torch.Tensor, npoint: int, mask: torch.Tensor = None):
    B, N, C = xyz.shape
    device = xyz.device

    if mask is None:
        mask = torch.ones(B, N, dtype=torch.bool, device=device)

    mask_flatten = mask.view(-1)
    xyz_flatten = xyz.view(-1, C)
    xyz_flatten = xyz_flatten[mask_flatten]  # [total_valid, 3]
    batch_indices = torch.arange(mask.shape[0]).unsqueeze(1) * torch.ones((1, mask.shape[1]), dtype=torch.int64)
    batch_indices = batch_indices.to(device).view(-1)[mask_flatten]  # [total_valid]

    # if not valid_points:
    #     return torch.zeros(B, npoint, dtype=torch.long, device=device)


    counts = torch.bincount(batch_indices, minlength=B).float()
    # ratios = torch.clamp(npoint / counts, max=1)  # [B]
    ratios = npoint / counts
    # npoint = torch.round((npoint / counts) * 100.0) / 100 * counts
    # ratios = npoint / counts
    sampled_idx = fps(
        src=xyz_flatten,
        batch=batch_indices,
        ratio=ratios,
        random_start=True
    )  # [total_sampled]
    all_index = torch.tile(torch.arange(mask.shape[1]), (mask.shape[0], 1)).to(sampled_idx.device)
    all_index_flatten = all_index.view(-1)
    mask_flatten = mask.view(-1)
    valid_index = all_index_flatten[mask_flatten]
    fix_sampled_idx = valid_index[sampled_idx]
    fix_batch_indices = batch_indices[sampled_idx]
    sorted_indices = torch.argsort(fix_batch_indices)
    sorted_sampled_idx = fix_sampled_idx[sorted_indices]
    output = sorted_sampled_idx.view(B, -1)
    return output

def square_distance(src, dst, src_mask=None, dst_mask=None):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).unsqueeze(2)
    dist += torch.sum(dst ** 2, -1).unsqueeze(1)
    if src_mask is not None and dst_mask is not None:
        src_mask = src_mask.unsqueeze(2)
        dst_mask = dst_mask.unsqueeze(1)
        mask = ~(src_mask & dst_mask)
        dist[mask] = float('inf')
    return dist

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points

def sample_and_group_all(xyz, points):
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat(
            [grouped_xyz, points.view(B, 1, N, -1)], dim=-1
        )
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.activation = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.activation(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        out += residual
        # out = self.activation(out)
        return out

class GA_layer(nn.Module):
    def __init__(
        self, npoint, radius, nsample, in_channel, mlp, group_all
    ):
        super(GA_layer, self).__init__()
        self.activation = Swish()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        last_channel = in_channel
        #res_block
        self.res_blocks = nn.ModuleList()
        for out_channel in mlp:
            self.res_blocks.append(ResidualBlock(last_channel, out_channel))
            last_channel = out_channel


        self.conv = nn.Conv1d(
            in_channels=in_channel,  
            out_channels=mlp[0],  
            kernel_size=3,  
            padding=1 
        )

        self.bn = nn.BatchNorm1d(mlp[0]) 
        self.relu = nn.ReLU()  

    def forward(self, xyz, points, xyz_mask=None):
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint,
                self.radius,
                self.nsample,
                xyz,
                points,
                xyz_mask,
            )
        new_points = new_points.permute(0, 3, 2, 1)  # [B, D, nsample, npoint]


        for res_block in self.res_blocks:
            new_points = res_block(new_points)
        new_points = torch.max(new_points, 2)[0]  # [B, D, npoint]


        new_points = new_points.squeeze(-1) 

        return new_xyz, new_points

class little_SA(nn.Module):
    def __init__(
        self, npoint, nsample, in_channel, mlp_list
    ):
        super(little_SA, self).__init__()
        self.activation = Swish()
        self.npoint = npoint
        self.nsample = nsample
        self.res_blocks = nn.ModuleList()


        blocks = nn.ModuleList()

        if in_channel is not None:
            last_channel = in_channel + 3  # +3 for grouped_xyz_norm
        else:
            last_channel = 3  # Only grouped_xyz_norm
        self.res_blocks = nn.ModuleList()
        for out_channel in mlp_list:
            self.res_blocks.append(ResidualBlock(last_channel, out_channel))
            last_channel = out_channel


    def forward(self, xyz, points, xyz_mask=None):
        assert xyz.size(0) == points.size(0), print("error, batch size not match", xyz.size(), points.size())
        B, N, C = xyz.shape
        S = self.npoint
        fps_idx = farthest_point_sample(xyz, S, mask=None).long()
        new_xyz = index_points(xyz, fps_idx)
        new_points_list = []
        if 1:
            idx = knn_point(self.nsample, xyz, new_xyz, xyz_mask)
            grouped_xyz = index_points(xyz, idx)
            grouped_xyz -= new_xyz.unsqueeze(2)
            if points is not None:
                grouped_points = index_points(points, idx)
                grouped_points = torch.cat(
                    [grouped_xyz, grouped_points], dim=-1
                )
            else:
                grouped_points = grouped_xyz
            grouped_points = grouped_points.permute(
                0, 3, 2, 1
            )  # [B, D, nsample, npoint]


            # res_block
            # blocks = self.res_blocks[0]
            for res_block in  self.res_blocks:
                grouped_points = res_block(grouped_points)
            new_points = torch.max(grouped_points, 2)[0]
            new_points_list.append(new_points)


        new_points_concat = torch.cat(new_points_list, dim=1)  # [B, D', npoint]

        return new_xyz, new_points_concat


class AEP(nn.Module):
    def __init__(self, in_channel, output, mlp_list=[64,64], nsample=5, npoint=128):
        super(AEP, self).__init__()
        self.output = output

        self.sa1 = little_SA(
            npoint=npoint,  
            nsample = nsample,
            in_channel=in_channel,
            mlp_list=mlp_list
        )


        self.sa3 = GA_layer(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=mlp_list[-1]+3,
            mlp=[self.output],
            group_all=True,
        )



    def forward(self, xyzs, points, xyz_masks=None, extract_features=False):



        # print(xyzs.shape)
        l1_xyz, l1_points = self.sa1(xyzs, points, xyz_masks)
        l1_points = l1_points.permute(0, 2, 1)
        #
        if extract_features:
            return l1_xyz, l1_points, self.sa3(l1_xyz.detach(), l1_points.detach())[1].squeeze(-1)
        l2_xyz, l2_points = self.sa3(l1_xyz, l1_points)
        l2_points = l2_points.squeeze(-1)
        return l1_xyz, l1_points, l2_points




if __name__ == '__main__':
    device = "cuda:0"
    # test

    xyz_list = [torch.randn(400, 3).to(device) for _ in range(1000)] 
    xyz_list.extend([torch.randn(100, 3).to(device) for _ in range(1000)])  
    points_list = [torch.randn(400, 38).to(device) for _ in range(1000)] 
    points_list.extend([torch.randn(100, 38).to(device) for _ in range(1000)])
    node_num_list = [100] * 20
    xyzs, xyz_masks = pad_pointclouds(xyz_list)

    points, _ = pad_pointclouds(points_list)
    import time
    print("xyzs shape:", xyzs.shape)
    print("points shape:", points.shape)
    start_time = time.time()
    model3 = AEP(in_channel=38, output=32, mlp_list=[32,32], nsample=5, npoint=5).to(device)
    model32 = AEP(in_channel=32, output=64, mlp_list=[64,64], nsample=5, npoint=5).to(device)
    l1_xyz, l1_points, l2_points = model3(xyzs, points, None)
    end_time = time.time()
    print("l1_xyz:", l1_xyz.shape)
    print("l1_points:", l1_points.shape)
    print("l2_points:", l2_points.shape)
    print(f"Time taken for forward pass: {end_time - start_time:.4f} seconds")
    start_time = time.time()
    l1_xyz, l1_points, l2_points = model32(l1_xyz, l1_points, None)
    end_time = time.time()
    print("l1_xyz:", l1_xyz.shape)
    print("l1_points:", l1_points.shape)
    print("l2_points:", l2_points.shape)
    print(f"Time taken for forward pass: {end_time - start_time:.4f} seconds")

