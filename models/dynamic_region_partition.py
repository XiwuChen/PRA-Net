import torch
from torch import nn
from utils.util import knn



class DRP(nn.Module):
    def __init__(self, in_channel, out_channel, k_hat, bias=False):
        """
        :param in_channel:
            input feature channel type:int
        :param out_channel_list: int or list of int
            out channel of MLPs
        :param k_hat: int
            k_hat in LFE
        :param bias: bool
            use bias or not
        """
        super(DRP, self).__init__()
        self.att_mlp = nn.Conv1d(out_channel, 1, kernel_size=1)

    def forward(self, x, region_num, idx_=None):
        batch_size, _, N = x.size()
        
        att_score = self.att_mlp(x).squeeze(dim=1)
        
        att_score = torch.sigmoid(att_score)
        
        gap = N // region_num
        _, idx = torch.sort(att_score, dim=-1, descending=True)
        idx = idx[:, ::gap]
        
        return idx, att_score

