import time
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

from utils import AverageMeter, accuracy


def conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    use_BN: bool = True,
):
    """
    Simple convolutional block

    :param in_channels: int
        number of input channels
    :param out_channels: int
        number of output channels
    :param kernel_size: int
        kernel size
    :param stride: int
        Stride of the convolution
    :param padding: int
        padded value
    :param use_BN: bool
        if BN is applied
    :return: conv_block: torch.nn.Module
        a convolutional block layer
    """
    c = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    a = nn.ReLU(inplace=False)
    if use_BN:
        b = nn.BatchNorm2d(out_channels)
        return nn.Sequential(c, a, b)
    else:
        return nn.Sequential(c, a)


class torchModel(nn.Module):
    """
    The model to optimize
    """

    def __init__(
        self,
        config: Dict[str, Any],
        input_shape: Tuple[int, int, int] = (1, 28, 28),
        num_classes: int = 10,
    ):
        super(torchModel, self).__init__()
        layers = []
        n_conv_layers = config["n_conv_layers"] if "n_conv_layers" in config else 3
        kernel_size = config["kernel_size"] if "kernel_size" in config else 5
        use_BN = config["use_BN"] if "use_BN" in config else False
        glob_av_pool = (
            config["global_avg_pooling"] if "global_avg_pooling" in config else True
        )
        in_channels = input_shape[0]
        key_conv = "n_channels_conv_"
        out_channels = config.get(key_conv + "0")
        out_channels = out_channels if out_channels is not None else 16
        dropout_rate = config["dropout_rate"] if "dropout_rate" in config else 0.2

        for i in range(n_conv_layers):
            padding = (kernel_size - 1) // 2
            conv_block_0 = conv_block(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                use_BN=use_BN,
            )
            p = nn.MaxPool2d(kernel_size=2, stride=2)
            layers.extend([conv_block_0, p])
            in_channels = out_channels
            out_channels_tmp = config.get(key_conv + str((i + 1)))
            out_channels = out_channels_tmp if out_channels_tmp else out_channels * 2

        self.conv_layers = nn.Sequential(*layers)
        self.pooling = nn.AdaptiveAvgPool2d(1) if glob_av_pool else nn.Identity()
        self.output_size = num_classes

        self.fc_layers = nn.ModuleList()
        n_in = self._get_conv_output(input_shape)
        key_fc = "n_channels_fc_"
        n_out = config.get(key_fc + "0")
        n_out = n_out if n_out else 256

        if "n_fc_layers" in config:
            n_fc_layers = config["n_fc_layers"]
        else:
            n_fc_layers = 3
            config = {
                "n_channels_fc_0": 27,
                "n_channels_fc_1": 17,
                "n_channels_fc_2": 273,
            }
        for i in range(n_fc_layers):
            fc = nn.Linear(int(n_in), int(n_out))
            self.fc_layers += [fc]
            n_in = n_out
            n_out_tmp = config.get(key_fc + str((i + 1)))
            n_out = n_out_tmp if n_out_tmp else n_out / 2

        self.last_fc = nn.Linear(int(n_in), self.output_size)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.time_train = 0

    # generate input sample and forward to get shape
    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self.conv_layers(input)
        output_feat = self.pooling(output_feat)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        for fc_layer in self.fc_layers:
            x = self.dropout(F.relu(fc_layer(x)))
        x = self.last_fc(x)
        return x

    def train_fn(self, optimizer, criterion, loader, device, train=True):
        """
        Training method
        :param optimizer: optimization algorithm
        :criterion: loss function
        :param loader: data loader for either training or testing set
        :param device: torch device
        :param train: boolean to indicate if training or test set is used
        :return: (accuracy, loss) on the data
        """
        time_begin = time.time()
        score = AverageMeter()
        objs = AverageMeter()
        self.train()

        t = tqdm(loader)
        for images, labels in t:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = self(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            acc = accuracy(logits, labels, topk=(1,))[0]  # accuracy given by top 3
            n = images.size(0)
            objs.update(loss.item(), n)
            score.update(acc.item(), n)

            t.set_description("(=> Training) Loss: {:.4f}".format(objs.avg))

        self.time_train += time.time() - time_begin
        print("training time: " + str(self.time_train))
        return score.avg, objs.avg

    def eval_fn(self, loader, device, train=False):
        """
        Evaluation method
        :param loader: data loader for either training or testing set
        :param device: torch device
        :param train: boolean to indicate if training or test set is used
        :return: accuracy on the data
        """
        score = AverageMeter()
        self.eval()

        t = tqdm(loader)
        with torch.no_grad():  # no gradient needed
            for images, labels in t:
                images = images.to(device)
                labels = labels.to(device)

                outputs = self(images)
                acc = accuracy(outputs, labels, topk=(1,))[0]
                score.update(acc.item(), images.size(0))

                t.set_description("(=> Test) Score: {:.4f}".format(score.avg))

        return score.avg
