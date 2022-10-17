from torch import nn
from .dynamic_ops import DynamicLinear, DynamicConv2d, DynamicBatchNorm2d


class Zero(nn.Module):
    def __init__(self, stride=1):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        return x.mul(0.0)


class DynaConvBN(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride=1,
        affine=True,
        bn_momentum=0.1,
        bn_track_running_stats=True,
        **kwargs
    ):
        super(DynaConvBN, self).__init__()
        self.kernel_size = kernel_size
        self.activation_fns = nn.ModuleDict(
            {
                "ReLU": nn.ReLU(inplace=False),
                "Hardswish": nn.Hardswish(inplace=False),
                "Mish": nn.Mish(inplace=False),
            }
        )
        self.conv = DynamicConv2d(C_in, C_out, kernel_size, stride=stride)
        self.bn = DynamicBatchNorm2d(
            C_out,
            affine=affine,
            momentum=bn_momentum,
            track_running_stats=bn_track_running_stats,
        )

    def set_sample_cfg(self, config, C_out):
        self.sample_config = config
        self.C_out = C_out

    def forward(self, x):
        x = self.activation_fns[self.sample_config["Activation"]](x)
        x = self.conv(x, self.C_out)
        x = self.bn(x)
        return x


class DynaResNetBasicBlock(nn.Module):
    def __init__(
        self,
        id,
        C_in,
        C_out,
        kernel_size=3,
        stride=2,
        affine=True,
        bn_momentum=0.1,
        bn_track_running_stats=True,
    ):
        super(DynaResNetBasicBlock, self).__init__()
        self.conv_a = DynaConvBN(
            C_in,
            C_out,
            kernel_size,
            stride=stride,
            affine=affine,
            momentum=bn_momentum,
            track_running_stats=bn_track_running_stats,
        )
        self.conv_b = DynaConvBN(
            C_out,
            C_out,
            kernel_size,
            stride=1,
            affine=affine,
            momentum=bn_momentum,
            track_running_stats=bn_track_running_stats,
        )

        self.downsample = False

        if stride == 2:
            self.downsample_avg = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            self.downsample_conv = DynamicConv2d(C_in, C_out, kernel_size=1, stride=1)
            self.downsample = True

        self.cell_id = id

    def set_sample_cfg(self, config, C_out):
        self.sample_config = config
        self.conv_a.set_sample_cfg(config, C_out)
        self.conv_b.set_sample_cfg(config, C_out)
        self.C_out = C_out

    def forward(self, x):
        basicblock = self.conv_a(x)
        basicblock = self.conv_b(basicblock)
        if self.downsample:
            x = self.downsample_avg(x)
            residual = self.downsample_conv(x, self.C_out)
        else:
            residual = x
        return residual + basicblock


# in JAHSbench affine and track_running set stats are left as true


class DynaCellWS(nn.Module):
    def __init__(
        self,
        id,
        C,
        cell_repeat,
        affine=True,
        bn_momentum=0.1,
        bn_track_running_stats=True,
    ):
        super(DynaCellWS, self).__init__()
        self.channels = C
        self.node_op = {1: nn.Identity(), 2: nn.Identity(), 3: sum, 4: sum}
        self.ops = nn.ModuleDict()
        for i in ["Op1", "Op2", "Op3", "Op4", "Op5", "Op6"]:
            self.ops[i] = nn.ModuleList(
                [
                    nn.Identity(),
                    Zero(),
                    DynaConvBN(
                        C_in=C,
                        C_out=C,
                        kernel_size=3,
                        stride=1,
                        affine=affine,
                        momentum=bn_momentum,
                        track_running_stats=bn_track_running_stats,
                    ),
                    DynaConvBN(
                        C_in=C,
                        C_out=C,
                        kernel_size=1,
                        stride=1,
                        affine=affine,
                        momentum=bn_momentum,
                        track_running_stats=bn_track_running_stats,
                    ),
                    nn.AvgPool2d(
                        kernel_size=3, stride=1, padding=1, count_include_pad=False
                    ),
                ]
            )

        self.cell_id = id
        self.cell_repeat = cell_repeat

    def set_sample_cfg(self, config, C_out):
        self.sample_config = config
        self.C_out = C_out
        for k, v in self.ops.items():
            v[2].set_sample_cfg(config, C_out)
            v[3].set_sample_cfg(config, C_out)

    def forward(self, x):
        # Handles Dynamic Depth
        if self.cell_repeat > self.sample_config["N"]:
            return x

        x = self.node_op[1](x)

        x_1_2 = self.ops["Op1"][self.sample_config["Op1"]](x)
        x_1_3 = self.ops["Op2"][self.sample_config["Op2"]](x)
        x_1_4 = self.ops["Op3"][self.sample_config["Op3"]](x)

        x_2 = self.node_op[2](x_1_2)
        x_2_3 = self.ops["Op4"][self.sample_config["Op4"]](x_2)
        x_2_4 = self.ops["Op5"][self.sample_config["Op5"]](x_2)

        x_3 = self.node_op[3]((x_1_3, x_2_3))
        x_3_4 = self.ops["Op6"][self.sample_config["Op6"]](x_3)

        x_4 = self.node_op[4]((x_1_4, x_2_4, x_3_4))

        return x_4
