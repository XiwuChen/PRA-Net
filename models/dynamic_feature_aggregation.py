import torch
from torch import nn


class DFA(nn.Module):
    def __init__(self, features, M=2, r=1, L=32):
        """Construct a two-branch Dynamic Feature Aggregation module.

        Args:
            features: Input channel dimensionality.
            M: Number of feature branches. Retained for configuration and
                checkpoint compatibility; the current implementation uses two.
            r: Reduction ratio used to compute the latent dimension.
            L: Minimum latent dimension described in the paper.
        """
        super(DFA, self).__init__()

        self.M = M
        self.features = features
        d = max(int(self.features / r), L)

        self.fc = nn.Sequential(
            nn.Conv1d(self.features, d, kernel_size=1),
            nn.BatchNorm1d(d),
        )
        self.fc_out = nn.Sequential(
            nn.Conv1d(d, self.features, kernel_size=1),
            nn.BatchNorm1d(self.features),
        )

    def forward(self, x):
        """Fuse two feature tensors with learned channel-point attention.

        :param x: Two tensors, each with shape ``(B, C, N)``.
        :return: Fused features with shape ``(B, C, N)``.
        """
        shape = x[0].shape
        if len(shape) > 3:
            raise NotImplementedError("DFA only supports tensors shaped (B, C, N).")

        # Summarize both branches, then predict a complementary mixing weight.
        aggregated_features = x[0] + x[1]
        latent_features = self.fc(aggregated_features)
        attention_vectors = torch.sigmoid(self.fc_out(latent_features))
        fused_features = attention_vectors * x[0] + (1 - attention_vectors) * x[1]

        return fused_features
