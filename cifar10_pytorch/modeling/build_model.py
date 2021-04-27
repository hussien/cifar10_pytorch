import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math

class swish(nn.Module):
    """
    swish activation. https://arxiv.org/abs/1710.05941
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class hswish(nn.Module):
    """
    h-swish activation. https://arxiv.org/abs/1905.02244
    """
    def __init__(self):
        super(hswish, self).__init__()
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x):
        return x * self.relu6(x + 3) / 6

class Drop_Connect:
    def __init__(self, drop_connect_rate):
        self.keep_prob = 1.0 - torch.tensor(drop_connect_rate, requires_grad=False)

    def __call__(self, x):
        drop_mask = torch.rand(x.shape[0], 1, 1, 1) + self.keep_prob
        drop_mask = drop_mask.type_as(x)
        drop_mask.floor_()
        return drop_mask * x / self.keep_prob

def get_model_urls():
    # model_urls = {
    #     'EfficientNetB0': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b0-355c32eb.pth',
    #     'EfficientNetB1': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b1-f1951068.pth',
    #     'EfficientNetB2': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b2-8bb594d6.pth',
    #     'EfficientNetB3': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b3-5fb5a3c3.pth',
    #     'EfficientNetB4': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b4-6ed6700e.pth',
    #     'EfficientNetB5': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b5-b6417697.pth',
    #     'EfficientNetB6': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b6-c76e70fd.pth',
    #     'EfficientNetB7': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b7-dcc49843.pth',
    # }
    model_urls = {
        'EfficientNetB0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth',
        'EfficientNetB1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth',
        'EfficientNetB2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth',
        'EfficientNetB3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth',
        'EfficientNetB4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth',
        'EfficientNetB5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth',
        'EfficientNetB6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth',
        'EfficientNetB7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth'
    }
    return model_urls

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block.
    """
    def __init__(self, in_ch, out_ch, expansion, kernel_size, stride, drop_connect_rate=0.2):
        """
        Args:
            in_ch (int): number of input channels
            out_ch (int): number of output channels
            expansion (int): channel expansion rate
            kernel_size (int): kernel size of depthwise conv layer
            stride (int): stride of depthwise conv layer
            drop_connect_rate:
        """
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.expansion = expansion
        self.id_skip = True if stride == 1 and in_ch == out_ch else False
        self.drop_connect_rate = drop_connect_rate
        self.swish = swish()

        ch = expansion * in_ch

        if expansion != 1:
            self._expand_conv = nn.Conv2d(
                in_ch, ch,
                kernel_size=1, stride=1,
                padding=0, bias=False)
            self._bn0 = nn.BatchNorm2d(ch, eps=1e-3, momentum=0.01)

        self._depthwise_conv = nn.Conv2d(ch, ch, kernel_size=kernel_size,
                               stride=stride, padding=(kernel_size-1)//2, groups=ch, bias=False)
        self._bn1 = nn.BatchNorm2d(ch, eps=1e-3, momentum=0.01)

        # SE layers
        self._se_reduce = nn.Conv2d(ch, in_ch//4, kernel_size=1)
        self._se_expand = nn.Conv2d(in_ch//4, ch, kernel_size=1)

        self._project_conv = nn.Conv2d(
            ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self._bn2 = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01)

        self._drop_connect = Drop_Connect(self.drop_connect_rate)

    def forward(self, inputs):
        x = inputs
        if self.expansion != 1:
            x = self.swish(self._bn0(self._expand_conv(x)))
        h = self.swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        se = F.avg_pool2d(h, h.size(2))
        se = self.swish(self._se_reduce(se))
        se = self._se_expand(se).sigmoid()
        h = h * se

        h = self._bn2(self._project_conv(h))

        # Skip Connection
        if self.id_skip:
            if self.training and self.drop_connect_rate > 0:
                h = self._drop_connect(h)
            h = h + inputs
        return h


class EfficientNet(nn.Module):
    """
    EfficientNet model. https://arxiv.org/abs/1905.11946
    """
    def __init__(self, block_args, num_classes=10):
        super(EfficientNet, self).__init__()
        self.block_args = block_args
        self._conv_stem = nn.Conv2d(
            3,
            block_args["stem_ch"],
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self._bn0 = nn.BatchNorm2d(block_args["stem_ch"], eps=1e-3, momentum=0.01)
        self._blocks = self._make_blocks()
        self._conv_head = nn.Conv2d(
            block_args["head_in_ch"],
            block_args["head_out_ch"],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self._bn1 = nn.BatchNorm2d(block_args["head_out_ch"], eps=1e-3, momentum=0.01)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self.block_args["dropout_rate"])
        self._fc = nn.Linear(block_args["head_out_ch"], num_classes)
        self.swish = swish()

    def _make_blocks(self):
        layers = []
        for n in range(7):
            strides = [self.block_args["stride"][n]] \
                     + [1] * (self.block_args["num_repeat"][n] - 1)
            in_chs = [self.block_args["input_ch"][n]] \
                     + [self.block_args["output_ch"][n]] * (self.block_args["num_repeat"][n] - 1)
            for stride, in_ch in zip(strides, in_chs):
                layers.append(
                    MBConvBlock(
                        in_ch,
                        self.block_args["output_ch"][n],
                        self.block_args["expand_ratio"][n],
                        self.block_args["kernel_size"][n],
                        stride,
                        drop_connect_rate=self.block_args["drop_connect_rate"],
                    )
                )
        return nn.Sequential(*layers)

    def forward(self, x):
        h = self.swish(self._bn0(self._conv_stem(x)))
        h = self._blocks(h)
        h = self.swish(self._bn1(self._conv_head(h)))
        h = self._avg_pooling(h)
        h = h.view(h.size(0), -1)
        h = self._dropout(h)
        h = self._fc(h)
        return h


def round_filters(ch, multiplier, divisor=8):
    """
    channel number scaling for EfficientNet.
    Args:
        ch (int): number of channel to scale
        multiplier (float): scaling factor
        divisor (int): divisor of scaled number of channels
    Returns:
        ch_scaled (int): scaled number of channels

    """
    ch *= multiplier
    ch_scaled = int(ch + divisor / 2) // divisor * divisor
    if ch_scaled < 0.9 * ch:
        ch_scaled += divisor
    return ch_scaled


def build_EfficientNet(cfg):
    """
    build EfficientNetB0-B7 from the input configuration.
    """
    block_args = {
        "num_repeat": [1, 2, 2, 3, 3, 4, 1],
        "kernel_size": [3, 3, 5, 3, 5, 5, 3],
        "stride": [1, 2, 2, 2, 1, 2, 1],
        "expand_ratio": [1, 6, 6, 6, 6, 6, 6],
        "input_ch": [32, 16, 24, 40, 80, 112, 192],
        "output_ch": [16, 24, 40, 80, 112, 192, 320],
        "dropout_rate": cfg.MODEL.DROPOUT_RATE,
        "drop_connect_rate": cfg.MODEL.DROPCONNECT_RATE,
        "stem_ch": round_filters(32, cfg.MODEL.CHANNEL_MULTIPLIER),
        "head_in_ch": round_filters(320, cfg.MODEL.CHANNEL_MULTIPLIER),
        "head_out_ch": round_filters(1280, cfg.MODEL.CHANNEL_MULTIPLIER),
    }

    # Scale number of blocks
    if cfg.MODEL.DEPTH_MULTIPLIER > 1.0:
        block_args["num_repeat"] = [
            math.ceil(n * cfg.MODEL.DEPTH_MULTIPLIER) for n in block_args["num_repeat"]
        ]

    # Scale number of channels
    if cfg.MODEL.CHANNEL_MULTIPLIER > 1.0:
        block_args["input_ch"] = [
            round_filters(n, cfg.MODEL.CHANNEL_MULTIPLIER) for n in block_args["input_ch"]
        ]

        block_args["output_ch"] = [
            round_filters(n, cfg.MODEL.CHANNEL_MULTIPLIER) for n in block_args["output_ch"]
        ]

    # Load ImageNet-pretrained model
    if cfg.MODEL.PRETRAINED:
        model_urls = get_model_urls()
        state_dict = model_zoo.load_url(model_urls[cfg.MODEL.PRETRAINED])
        model = EfficientNet(block_args, num_classes=1000)
        model.load_state_dict(state_dict)
        model._fc = nn.Linear(
            round_filters(1280, cfg.MODEL.CHANNEL_MULTIPLIER),
            10
        )
    else:
        model = EfficientNet(block_args)
    return model



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1,
                               stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1,
                               stride=stride, groups=groups,
                               dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10,
                 groups=1, width_per_group=64):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = list()
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def build_ResNet(cfg):
    _blocks = {
        "Basic": BasicBlock,
        "Bottleneck": Bottleneck,
    }
    return ResNet(_blocks[cfg.MODEL.RESNET_BLOCK],
                  cfg.MODEL.RESNET_LAYERS)






"""VGG in PyTorch"""



_LAYER_DEFINITION = {
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, cfg):
        super(VGG, self).__init__()
        self.features = self._make_layers(_LAYER_DEFINITION[cfg.MODEL.VGG_NUM])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, definition):
        layers = []
        in_channels = 3
        for x in definition:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



_META_MODELS = {
    "VGG": VGG,
    "ResNet": build_ResNet,
    "EfficientNet": build_EfficientNet,
}


def build_model(cfg):
    model = _META_MODELS[cfg.MODEL.NAME](cfg)
    return model