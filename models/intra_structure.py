import torch
from torch import nn
from utils.util import knn
from models.dynamic_feature_aggregation import DFA


def get_graph_feature(x, xyz=None, idx=None, k_hat=20):
    """
    Get graph features by minus the k_hat nearest neighbors' feature.
    :param x: (B,C,N)
        input features
    :param xyz: (B,3,N) or None
        xyz coordinate
    :param idx: (B,N,k_hat)
        kNN graph index
    :param k_hat: (int)
        the neighbor number
    :return: graph feature (B,C,N,k_hat)
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(xyz, k=k_hat)  # (batch_size, num_points, k_hat)


    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k_hat, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims)
    feature = feature - x
    feature = feature.permute(0, 3, 1, 2)
    return feature


class ISL(nn.Module):

    def __init__(self, in_channel, out_channel_list, k_hat=20, bias=False,):

        """
        :param in_channel:
            input feature channel type:int
        :param out_channel_list: int or list of int
            out channel of MLPs
        :param k_hat: int
            k_hat in ISL
        :param bias: bool
            use bias or not
        """
        super(ISL, self).__init__()

        out_channel = out_channel_list[0]

        self.self_feature_learning = nn.Conv1d(in_channel // 2, out_channel, kernel_size=1, bias=bias)
        self.neighbor_feature_learning = nn.Conv2d(in_channel // 2, out_channel, kernel_size=1, bias=bias)
        self.k = k_hat

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        last_layer_list = []

        for i in range(len(out_channel_list) - 1):
            in_channel = out_channel_list[i]
            out_channel = out_channel_list[i + 1]
            last_layer_list.append(nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=bias))
            last_layer_list.append(nn.BatchNorm2d(out_channel))
            last_layer_list.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.last_layers = nn.Sequential(*last_layer_list)
        
        self.bn = nn.BatchNorm2d(out_channel)
        
        

        self.bn2 = nn.BatchNorm1d(out_channel)
        self.bn = nn.BatchNorm2d(out_channel)

        
        self.DFA_layer = DFA(features=out_channel, M=2, r=1)


    def forward(self, x, idx_):
        """
        :param x: (B,3,N)
            Input point cloud
        :param idx_: (B,N,k_hat)
            kNN graph index
        :return: graph feature: (B,C,N,k_hat)
        """

        x_minus = get_graph_feature(x, idx=idx_, k_hat=self.k)
        # (B,C,N,K)
        a1 = self.neighbor_feature_learning(x_minus)
        # (B,C,N)
        a2 = self.self_feature_learning(x)
    
    
        a1 = self.leaky_relu(self.bn(a1))
        # (B,C,N)
        a1 = a1.max(dim=-1, keepdim=False)[0]
        a2 = self.leaky_relu(self.bn2(a2))
        res = self.DFA_layer([a1, a2])
        
        res = self.last_layers(res)

        return res


if __name__ == '__main__':
    x = torch.rand((2, 3, 100))
    isl = ISL(in_channel=6, out_channel_list=[16], k_hat=20, bias=False)
    idx = knn(x, k=20)
    out = isl(x, idx)
