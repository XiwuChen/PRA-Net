import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch_cluster import random_walk
from utils.util import knn
from models.intra_structure import ISL
from models.dynamic_feature_aggregation import DFA
from models.inter_region import IRL


class PRANet_classification(nn.Module):
    def __init__(self, args, output_channels=40, input_channel=3):
        super(PRANet_classification, self).__init__()
        self.args = args
        self.k_hat = args.k_hat
        self.sample_ratio = args.sample_ratio_list
        self.m_list = args.m_list
        self.start_layer = args.start_layer
        

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

        # IRL layers
        channels = [64, 64, 128, 256]
        
        self.irl_list = nn.ModuleList()
        for i in range(self.start_layer, 4):
            irl = IRL(channel=channels[i], sample_ratio=self.sample_ratio[i], m=self.m_list[i])

            self.irl_list.append(irl)

        if args.init:
            print("Init Conv")
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    print(m)
                    nn.init.xavier_normal_(m.weight.data)
                    try:
                        m.bias.data.fill_(0)
                    except:
                        print("No bias here")
            print("End Init Conv")
        else:
            print("Don't Init Conv")



    def forward(self, x, train_mode=True):
        """
        :param x: (B,3,N)
            input point cloud
        :return:
            cls predication
        """
        batch_size, _, N = x.size()
        xyz = x[:, :3, :].clone().detach()  # xyz:(B, 3, N)

        # The kNN graph index used in ISL
        kNN_Graph_idx = knn(xyz, self.k_hat)

        # (ISL1)
        x1 = self.ISL1(x, kNN_Graph_idx)

        # (IRL)
        if self.start_layer <= 0:
            x1 = self.irl_list[-4](x1, xyz, kNN_Graph_idx)

        # (ISL2)
        x2 = self.ISL2(x1, kNN_Graph_idx)

        # (IRL)
        if self.start_layer <= 1:
            x2 = self.irl_list[-3](x2, xyz, kNN_Graph_idx)


        # (ISL3)
        x3 = self.ISL3(x2, kNN_Graph_idx)

        if self.start_layer <= 2:
            x3 = self.irl_list[-2](x3, xyz, kNN_Graph_idx)

        # (ISL4)
        x4 = self.ISL4(x3, kNN_Graph_idx)

        # (IRL)
        if self.start_layer <= 3:
            x4 = self.irl_list[-1](x4, xyz, kNN_Graph_idx)


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
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='PR-Net Classification Network')
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    print("\n**************************")
    for k, v in config['common'].items():
        setattr(args, k, v)
        print('\n[%s]:' % (k), v)
    print("\n**************************\n")
    model = PRANet_classification(args, output_channels=15)

    points = torch.randn([8, 3, 4096])
    points = points.cuda()
    model = model.cuda()
    out = model(points)
    print(out)

