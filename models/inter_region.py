import torch
from torch import nn
from models.dynamic_region_partition import DRP
from utils.util import farthest_point_sample, index_points, square_distance


def interpolation(neighbor_weights, idx, region_feature, N):
    """
    Interpolation by 3 nearest neighbors with weights got from *get_interpolation_weights* function.
    :param neighbor_weights: (B,N)
        The weights of interpolation points.
    :param idx:(B,N,3)
        The index of interpolation points.
    :param region_feature: (B,C,m,S)

    :param N: int
        The number of input points.
    :return:

    region_feature: (B,C,N)
        feature after interpolation
    """
    batch_size, C, _, _ = region_feature.shape
    region_feature = torch.gather(region_feature.view(batch_size, C, -1), 2,
                                  idx.contiguous().view(batch_size, 1, -1).expand(batch_size, C, N * 3))
    region_feature = region_feature.view(batch_size, C, N, 3)
    region_feature = neighbor_weights.unsqueeze(1) * region_feature
    region_feature = region_feature.sum(-1, keepdim=False)
    return region_feature


class IRL(nn.Module):
    def __init__(self, channel, sample_ratio, m, region_global=None, ire_fp='None'):
        super(IRL, self).__init__()
        self.channel = channel
        self.sample_ratio = sample_ratio
        self.region_global = region_global
        
        self.m = m
        
        self.group = m
        self.ire_fp = ire_fp
        self.DRP_layer = DRP(in_channel=self.channel * 2, out_channel=self.channel, k_hat=20).cuda()
        self.kconv = nn.Conv2d(self.channel, self.channel, kernel_size=1, bias=False)
        self.qconv = nn.Conv2d(self.channel, self.channel, kernel_size=1, bias=False)
        self.vconv = nn.Conv2d(self.channel, self.channel, kernel_size=1, bias=False)
        self.outconv = nn.Conv2d(self.channel, self.channel, kernel_size=1)

    def sample_rep_points(self, x, xyz, LFE_idx):
        """
        Get representative points points from origin point cloud.
        :param xyz: (B, C, 3)
            The coordinate of the input point cloud.
        :return:
        rep_points_idx: (B,S*m,1)
            The indices of representative points.
        interpolation_idx: (B,N,3)
            The index of interpolation points
        interpolation_weights: (B,N)
            The weights of interpolation points
        """

        batch_size, _, N = xyz.size()
        S = N // self.sample_ratio
        # Get the center point of sampled regions.
        # center_index:(B, S)
        center_index, att_score = self.DRP_layer(x, S, idx_=LFE_idx)
        center_xyz = index_points(xyz.permute(0, 2, 1), center_index.long())
        
        center_to_all = square_distance(center_xyz, xyz.permute(0, 2, 1))  # center_to_all: (B, S, N)

        # Representative points index got by kNN.
        # (B,S,m)
        _, rep_points_idx = torch.topk(center_to_all, self.m, largest=False)

        # Get center of a region
        idx2 = rep_points_idx[:, :, :1].expand(batch_size, S, 3)
        rep_points_idx = rep_points_idx.view(batch_size, -1).unsqueeze(-1)
        representative_xyz = torch.gather(xyz.permute(0, 2, 1), 1, idx2).view(batch_size, S, 1, 3)
        representative_xyz = representative_xyz.permute(0, 2, 1, 3).contiguous()
        
        representative_xyz = representative_xyz.view(batch_size, -1, 3)
        
        all_to_rep = square_distance(xyz.permute(0, 2, 1), representative_xyz)
        
        dis, interpolation_idx = torch.topk(all_to_rep, 3, largest=False)
        dis_inv = 1. / (dis + 1e-5)
        interpolation_weights = dis_inv / dis_inv.sum(-1, keepdim=True)

        return rep_points_idx, interpolation_idx, interpolation_weights, att_score

    def region_relation(self, x, att_score=None):
        """
        Enhance feature by modeling the region relation.
        :param x: (B,C,N)

        :return: Enhanced feature
        """

        batch_size, C, N = x.size()
        S = N // self.sample_ratio

        # sample regions
        idx_ = self.f_idx.expand(batch_size, S * self.m, C)
        center_feature = torch.gather(x.permute(0, 2, 1), 1, idx_).view(batch_size, S, self.m, C).permute(0, 3, 1, 2)
        if att_score is not None:
            att_score = att_score.view(batch_size, 1, S, 1).expand(batch_size, C, S, self.m)
            center_feature = center_feature * att_score

        K = self.kconv(center_feature)
        K = K.permute(0, 3, 2, 1)
        Q = self.qconv(center_feature)
        Q = Q.permute(0, 3, 2, 1)
        V = self.vconv(center_feature)
        V = V.permute(0, 3, 2, 1)
        K = torch.split(K, split_size_or_sections=1, dim=1)
        Q = torch.split(Q, split_size_or_sections=1, dim=1)
        V = torch.split(V, split_size_or_sections=1, dim=1)
        res = []

        # Modeling relation between different regions.
        for i in range(self.group):
            K_ = K[i].view(batch_size, S, C)
            Q_ = Q[i].view(batch_size, S, C)
            V_ = V[i].view(batch_size, S, C)
            att_w = torch.bmm(K_, Q_.permute(0, 2, 1))

            att_w = torch.softmax(att_w.div((S * C) ** 0.5), dim=-1)
            
            out = torch.matmul(att_w, V_).unsqueeze(1)
            res.append(out)
        
        enhanced_feature = torch.cat(res, dim=1).view(batch_size, self.group, S, C). \
            permute(0, 3, 1, 2)
        enhanced_feature = self.outconv(enhanced_feature)
        enhanced_feature = enhanced_feature + center_feature.transpose(-1, -2)


        enhanced_feature = enhanced_feature.mean(dim=2, keepdim=True)
        
        enhanced_feature = interpolation(self.neighbor_weights, self.idx, enhanced_feature, N)
        return enhanced_feature

    def init_weights(self):
        """
        Initial weights.
        """
        nn.init.constant_(self.outconv.weight, 0)
        nn.init.constant_(self.outconv.bias, 0)

    def forward(self, x, xyz, LFE_idx):
        """
        :param x: (B,C,N)
            Feature
        :param xyz: (B,3,N)
            The coordinate of the input point cloud.
        :return: enhanced feature with the same shape of x
        """

        self.f_idx, self.idx, self.neighbor_weights, att_score = self.sample_rep_points(x, xyz, LFE_idx)
        x = x * att_score.unsqueeze(1)
        enhanced_feature = self.region_relation(x)
        x = x + enhanced_feature
        return x


def knn(x, k):
    """
    :param x: (B,3,N)
    :param k: int
    :return: (B,N,k_hat)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k_hat)
    return idx


if __name__ == '__main__':
    irl = IRL(channel=64, sample_ratio=8, m=4, ire_fp='avg').cuda()
    xyz = torch.rand(2, 3, 1024).cuda()
    feature = torch.rand(2, 64, 1024).cuda()
    kNN_Graph_idx = knn(xyz, 20)
    y = irl(feature, xyz, kNN_Graph_idx)
    print(y.shape)
