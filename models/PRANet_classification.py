import torch
import torch.nn as nn
import torch.nn.functional as F

from models.intra_structure import ISL
from models.inter_region import IRL
from utils.util import knn


class PRANet_classification(nn.Module):
    def __init__(self, args, output_channels=40, input_channel=3):
        super(PRANet_classification, self).__init__()
        self.args = args
        self.k_hat = args.k_hat
        self.sample_ratio = args.sample_ratio_list
        self.m_list = args.m_list
        self.start_layer = args.start_layer

        if not 0 <= self.start_layer <= 4:
            raise ValueError("start_layer must be between 0 and 4.")

        in_channel = input_channel
        # ISL layers
        self.ISL1 = ISL(in_channel=in_channel * 2, out_channel_list=[64], k_hat=self.k_hat, bias=False)
        self.ISL2 = ISL(in_channel=64 * 2, out_channel_list=[64], k_hat=self.k_hat, bias=False)
        self.ISL3 = ISL(in_channel=64 * 2, out_channel_list=[128], k_hat=self.k_hat, bias=False)
        self.ISL4 = ISL(in_channel=128 * 2, out_channel_list=[256], k_hat=self.k_hat, bias=False)

        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(args.emb_dims),
                                   nn.LeakyReLU(negative_slope=0.2))

        # FC layers
        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)

        self.bn18 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn19 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

        # IRL modules are stored from start_layer onward. The negative indices
        # used in forward keep each module aligned with its corresponding ISL
        # stage. Setting start_layer to 4 disables IRL without changing ISL.
        channels = [64, 64, 128, 256]
        self.irl_list = nn.ModuleList()
        for i in range(self.start_layer, 4):
            irl = IRL(channel=channels[i], sample_ratio=self.sample_ratio[i], m=self.m_list[i])
            self.irl_list.append(irl)

        if args.init:
            print("Init Conv")
            for m in self.modules():
                if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                    print(m)
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            print("End Init Conv")
        else:
            print("Don't Init Conv")

    def forward(self, x, train_mode=True):
        """
        :param x: (B, 3, N)
            Input point cloud.
        :param train_mode:
            Retained for backward compatibility. Module training behavior is
            controlled by ``model.train()`` and ``model.eval()``.
        :return:
            Classification logits.
        """
        batch_size = x.size(0)
        # Neighborhoods are constructed from the original XYZ coordinates and
        # are intentionally detached from the feature-learning graph.
        xyz = x[:, :3, :].clone().detach()

        # Compute the graph once and reuse it in all ISL/IRL stages.
        knn_graph_idx = knn(xyz, self.k_hat)

        # (ISL1)
        x1 = self.ISL1(x, knn_graph_idx)

        # (IRL)
        if self.start_layer <= 0:
            x1 = self.irl_list[-4](x1, xyz, knn_graph_idx)

        # (ISL2)
        x2 = self.ISL2(x1, knn_graph_idx)

        # (IRL)
        if self.start_layer <= 1:
            x2 = self.irl_list[-3](x2, xyz, knn_graph_idx)

        # (ISL3)
        x3 = self.ISL3(x2, knn_graph_idx)

        if self.start_layer <= 2:
            x3 = self.irl_list[-2](x3, xyz, knn_graph_idx)

        # (ISL4)
        x4 = self.ISL4(x3, knn_graph_idx)

        # (IRL)
        if self.start_layer <= 3:
            x4 = self.irl_list[-1](x4, xyz, knn_graph_idx)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)

        # Pooling
        x_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_avg = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x_max, x_avg), 1)

        # FC
        x = F.leaky_relu(self.bn18(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn19(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


if __name__ == '__main__':
    # Training settings
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description='PRA-Net Classification Network')
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    print("\n**************************")
    for k, v in config['common'].items():
        setattr(args, k, v)
        print('\n[%s]:' % (k), v)
    print("\n**************************\n")
    model = PRANet_classification(args, output_channels=15)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points = torch.randn([8, 3, 4096], device=device)
    model = model.to(device)
    out = model(points)
    print(out)

