import torch
from torch import nn


class DFA(nn.Module):
    def __init__(self, features, M=2, r=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(DFA, self).__init__()

        self.M = M
        self.features = features
        d = max(int(self.features / r), L)

        self.fc = nn.Sequential(nn.Conv1d(self.features, d, kernel_size=1),
                                nn.BatchNorm1d(d))
                                
        self.fc_out = nn.Sequential(nn.Conv1d(d, self.features, kernel_size=1),
                                nn.BatchNorm1d(self.features))


    def forward(self, x):
        """
        :param x: [x1,x2] (B,C,N)
        :return:
        """
        
        shape = x[0].shape
        if len(shape) > 3:
            assert NotImplemented('Don not support len(shape)>=3.')

        # (B,MC,N)
        fea_U = x[0] + x[1]
        
        fea_z = self.fc(fea_U)
        # B，C，N
        fea_cat = self.fc_out(fea_z)
        
        attention_vectors = torch.sigmoid(fea_cat)
        fea_v = attention_vectors * x[0] + (1 - attention_vectors) * x[1]

        return fea_v
