from torch import nn

from .dynamic_ops import DynamicLinear, DynamicBatchNorm2d
from .dynamic_primitives import DynaConvBN, DynaResNetBasicBlock, DynaCellWS


class DynaJAHSBench201Network(nn.Module):
    def __init__(
        self,
        stem_out_channels=16,
        num_modules_per_stack=5,
        bn_affine=True,
        bn_momentum=0.1,
        bn_track_running_stats=True,
        cell_type="shared",
        grayscale=True,
    ):
        super(DynaJAHSBench201Network, self).__init__()
        self.sample_config = None
        self.channels = (
            C
        ) = stem_out_channels  # size_of_input channels to the first cell channels = [16, 32, 64]
        self.num_modules = N = num_modules_per_stack
        self.num_labels = 10

        self.bn_momentum = bn_momentum
        self.bn_affine = bn_affine
        self.bn_track_running_stats = bn_track_running_stats

        if grayscale:
            C_in = 1
        else:
            C_in = 3
        self.DynaStem = DynaConvBN(
            C_in=C_in,
            C_out=C,
            kernel_size=3,
            affine=self.bn_affine,
            bn_momentum=self.bn_momentum,
            bn_track_running_stats=self.bn_track_running_stats,
        )

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev = C
        self.cells = nn.ModuleList()
        cell_repeat = 1
        for i, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if reduction:
                cell = DynaResNetBasicBlock(
                    i,
                    C_prev,
                    C_curr,
                    3,
                    2,
                    self.bn_affine,
                    self.bn_momentum,
                    self.bn_track_running_stats,
                )
                cell_repeat = 1

            else:
                if cell_type == "shared":
                    cell = DynaCellWS(
                        i,
                        C_curr,
                        cell_repeat,
                        self.bn_affine,
                        self.bn_momentum,
                        self.bn_track_running_stats,
                    )
                elif cell_type == "entangled":
                    cell = DynaCellWS(
                        i,
                        C_curr,
                        cell_repeat,
                        self.bn_affine,
                        self.bn_momentum,
                        self.bn_track_running_stats,
                    )
                cell_repeat += 1
            self.cells.append(cell)
            C_prev = C_curr

        self.postproc_bn = DynamicBatchNorm2d(C_prev, momentum=self.bn_momentum)
        self.postproc_ac = nn.ModuleDict(
            {
                "ReLU": nn.ReLU(inplace=False),
                "Hardswish": nn.Hardswish(inplace=False),
                "Mish": nn.Mish(inplace=False),
            }
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = DynamicLinear(C_prev, self.num_labels)

    def set_sample_cfg(self, config):
        self.sample_config = config
        self.C_out = C = self.sample_config["W"]

        # set input stem channels
        self.DynaStem.set_sample_cfg(config, C)
        # Calculates the Dynamic Channels for each cell
        N = self.num_modules
        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        for cell, C_out in zip(self.cells, layer_channels):
            cell.set_sample_cfg(config, C_out)

    def forward(self, inputs):
        feature = self.DynaStem(inputs)
        for cell in self.cells:
            feature = cell(feature)

        out = self.postproc_bn(feature)
        out = self.postproc_ac[self.sample_config["Activation"]](out)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return logits
