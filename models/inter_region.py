import torch
from torch import nn

from models.dynamic_region_partition import DRP
from utils.util import index_points, knn, square_distance


def interpolation(neighbor_weights, idx, region_feature, num_points):
    """Interpolate region features to all points using three neighbors.

    :param neighbor_weights: Interpolation weights with shape ``(B, N, 3)``.
    :param idx: Neighbor indices with shape ``(B, N, 3)``.
    :param region_feature: Region features with shape ``(B, C, 1, S)``.
    :param num_points: Number of input points.
    :return: Interpolated features with shape ``(B, C, N)``.
    """
    batch_size, channels, _, _ = region_feature.shape
    gather_idx = idx.contiguous().view(batch_size, 1, -1)
    gather_idx = gather_idx.expand(batch_size, channels, num_points * 3)
    region_feature = torch.gather(
        region_feature.view(batch_size, channels, -1), 2, gather_idx
    )
    region_feature = region_feature.view(batch_size, channels, num_points, 3)
    region_feature = neighbor_weights.unsqueeze(1) * region_feature
    return region_feature.sum(-1, keepdim=False)


class IRL(nn.Module):
    def __init__(self, channel, sample_ratio, m, region_global=None, ire_fp='None'):
        super(IRL, self).__init__()
        if sample_ratio <= 0:
            raise ValueError("sample_ratio must be greater than zero.")
        if m <= 0:
            raise ValueError("m must be greater than zero.")

        self.channel = channel
        self.sample_ratio = sample_ratio
        # These attributes are retained for compatibility with historical
        # configurations; the current IRL path does not read them.
        self.region_global = region_global
        self.m = m
        self.group = m
        self.ire_fp = ire_fp
        # Device placement is controlled by the parent model's .to(device).
        self.DRP_layer = DRP(
            in_channel=self.channel * 2,
            out_channel=self.channel,
            k_hat=20,
        )
        self.kconv = nn.Conv2d(self.channel, self.channel, kernel_size=1, bias=False)
        self.qconv = nn.Conv2d(self.channel, self.channel, kernel_size=1, bias=False)
        self.vconv = nn.Conv2d(self.channel, self.channel, kernel_size=1, bias=False)
        self.outconv = nn.Conv2d(self.channel, self.channel, kernel_size=1)

    def sample_rep_points(self, x, xyz, LFE_idx):
        """Select representative points and build interpolation metadata.

        :param x: Point features with shape ``(B, C, N)``.
        :param xyz: Coordinates with shape ``(B, 3, N)``.
        :param LFE_idx: Shared local-neighborhood graph indices.
        :return:
            Representative indices, interpolation indices, interpolation
            weights, and point attention scores.
        """
        batch_size, _, num_points = xyz.size()
        region_num = num_points // self.sample_ratio
        xyz_bnc = xyz.permute(0, 2, 1)

        center_index, att_score = self.DRP_layer(
            x, region_num, idx_=LFE_idx
        )
        center_xyz = index_points(xyz_bnc, center_index.long())
        center_to_all = square_distance(center_xyz, xyz_bnc)

        # Representative points index got by kNN.
        # Shape: (B, S, m).
        _, rep_points_idx = torch.topk(center_to_all, self.m, largest=False)

        # The nearest representative serves as the geometric proxy used to
        # compute interpolation neighborhoods.
        proxy_idx = rep_points_idx[:, :, :1].expand(batch_size, region_num, 3)
        rep_points_idx = rep_points_idx.view(batch_size, -1).unsqueeze(-1)
        representative_xyz = torch.gather(xyz_bnc, 1, proxy_idx).view(
            batch_size, region_num, 1, 3
        )
        representative_xyz = representative_xyz.permute(0, 2, 1, 3).contiguous()
        representative_xyz = representative_xyz.view(batch_size, -1, 3)

        all_to_rep = square_distance(xyz_bnc, representative_xyz)
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

        batch_size, channels, num_points = x.size()
        region_num = num_points // self.sample_ratio

        # sample regions
        idx_ = self.f_idx.expand(batch_size, region_num * self.m, channels)
        center_feature = torch.gather(x.permute(0, 2, 1), 1, idx_)
        center_feature = center_feature.view(
            batch_size, region_num, self.m, channels
        ).permute(0, 3, 1, 2)
        if att_score is not None:
            att_score = att_score.view(
                batch_size, 1, region_num, 1
            ).expand(batch_size, channels, region_num, self.m)
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
            K_ = K[i].view(batch_size, region_num, channels)
            Q_ = Q[i].view(batch_size, region_num, channels)
            V_ = V[i].view(batch_size, region_num, channels)
            att_w = torch.bmm(K_, Q_.permute(0, 2, 1))

            # Preserve the original PRA-Net scaling factor.
            att_w = torch.softmax(
                att_w.div((region_num * channels) ** 0.5), dim=-1
            )
            out = torch.matmul(att_w, V_).unsqueeze(1)
            res.append(out)

        enhanced_feature = torch.cat(res, dim=1).view(
            batch_size, self.group, region_num, channels
        ).permute(0, 3, 1, 2)
        enhanced_feature = self.outconv(enhanced_feature)
        enhanced_feature = enhanced_feature + center_feature.transpose(-1, -2)

        enhanced_feature = enhanced_feature.mean(dim=2, keepdim=True)
        enhanced_feature = interpolation(
            self.neighbor_weights, self.idx, enhanced_feature, num_points
        )
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

        self.f_idx, self.idx, self.neighbor_weights, att_score = (
            self.sample_rep_points(x, xyz, LFE_idx)
        )
        x = x * att_score.unsqueeze(1)
        enhanced_feature = self.region_relation(x)
        x = x + enhanced_feature
        return x

if __name__ == '__main__':
    irl = IRL(channel=64, sample_ratio=8, m=4, ire_fp='avg').cuda()
    xyz = torch.rand(2, 3, 1024).cuda()
    feature = torch.rand(2, 64, 1024).cuda()
    kNN_Graph_idx = knn(xyz, 20)
    y = irl(feature, xyz, kNN_Graph_idx)
    print(y.shape)
