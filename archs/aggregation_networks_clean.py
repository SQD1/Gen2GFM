import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from archs.detectron2.resnet import ResNet, BottleneckBlock

# One-timestep aggregated Features
class GlobalWeightedFuser(nn.Module):
    """
    Global weighted fusion for the pyramid features, applying learnable module-wise scalars
    """

    def __init__(
            self,
            feature_dims,
            projection_dim=384,
            num_norm_groups=32,
            num_res_blocks=1,
    ):
        """
        feature_dims is a dict {w1:[c1, c2, ...], w2:[c1', c2', ...]}
        """
        super().__init__()

        self.feature_dims = feature_dims
        self.scales = sorted(feature_dims.keys(), reverse=True)  # e.g. [64, 32, 16, 8]
        self.num_scales = len(self.scales)  # e.g. 4

        self.bottleneck_layers = nn.ModuleList()

        for scale_idx, scale in enumerate(self.scales):
            for fd_idx, fd in enumerate(self.feature_dims[scale]):
                bottleneck_layer = nn.Sequential(
                    *ResNet.make_stage(
                        BottleneckBlock,
                        num_blocks=num_res_blocks,
                        in_channels=fd,
                        bottleneck_channels=projection_dim // 4,
                        out_channels=projection_dim,
                        norm="GN",
                        num_norm_groups=num_norm_groups
                    )
                )
                self.bottleneck_layers.append(bottleneck_layer)

        mixing_weights = torch.ones(len(self.bottleneck_layers))
        self.mixing_weights = nn.Parameter(mixing_weights)

    def weights_init(self, m):

        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def get_bottleneck_index(self, stride_id, l, feat_dict):
        """Sum the #layers in all preceding strides + offset by l."""
        return sum(len(feat_dict[s]) for s in self.scales[:stride_id]) + l

    def forward(self, batch_feats):
        """
        Args:
        pyramid batch_feats:
        [(B, L1, H1, W1), (B, L2, H2, W2), (B, L3, H3, W3), (B, L4, H4, W4)]
        Returns:
        scale_feats: [(B, L1, H1, W1), (B, L2, H2, W2), (B, L3, H3, W3), (B, L4, H4, W4)]

        H1=64, H2=32, H3=16, H4=8
        """

        mixing_weights = torch.nn.functional.softmax(self.mixing_weights, dim=0)

        scale_feats = []

        for scale_idx, scale in enumerate(self.scales):
            scale_feat = None
            t_feats = batch_feats[scale_idx].float()
            start_channel = 0
            for fd_idx, fd in enumerate(self.feature_dims[scale]):
                end_channel = start_channel + fd
                feats = t_feats[:, start_channel:end_channel, :, :]

                bn_layer_idx = self.get_bottleneck_index(scale_idx, fd_idx, self.feature_dims)

                bottleneck_layer = self.bottleneck_layers[bn_layer_idx]
                weight_idx = bn_layer_idx

                bottlenecked_feature = bottleneck_layer(feats)
                bottlenecked_feature = mixing_weights[weight_idx] * bottlenecked_feature

                if scale_feat is None:
                    scale_feat = bottlenecked_feature
                else:
                    scale_feat += bottlenecked_feature

                start_channel = end_channel

            scale_feats.append(scale_feat)

        return scale_feats, mixing_weights