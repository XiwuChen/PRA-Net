import torch
from torch import nn



class DRP(nn.Module):
    def __init__(self, in_channel, out_channel, k_hat, bias=False):
        """Construct the Dynamic Region Partition scoring layer.

        ``in_channel``, ``k_hat``, and ``bias`` are retained for compatibility
        with the original constructor. Region scores are computed from
        ``out_channel`` input features.
        """
        super(DRP, self).__init__()
        self.att_mlp = nn.Conv1d(out_channel, 1, kernel_size=1)

    def forward(self, x, region_num, idx_=None):
        """Select region centers by dilated sampling over sorted scores.

        :param x: Point features with shape ``(B, C, N)``.
        :param region_num: Requested number of regions.
        :param idx_: Reserved for compatibility with the original API.
        :return: Selected center indices and per-point attention scores.
        """
        del idx_  # The graph index is intentionally unused by this DRP variant.
        num_points = x.size(2)
        if not 0 < region_num <= num_points:
            raise ValueError("region_num must be in the range [1, N].")

        att_score = self.att_mlp(x).squeeze(dim=1)
        att_score = torch.sigmoid(att_score)

        # This is dilated top-S sampling: sort all points by score, then select
        # evenly spaced ranks. Replacing it with topk would change the method.
        gap = num_points // region_num
        _, idx = torch.sort(att_score, dim=-1, descending=True)
        idx = idx[:, ::gap]

        return idx, att_score

