import numpy as np
from absl import logging

import trtlite
import trtlite.nn as nn
import trtlite.nn.functional as F

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50']
logging.set_verbosity(logging.INFO)


class AvgPool2dPad(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.avgpool2d = nn.AvgPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=False)
        self.forward = self._forward

    def _forward(self, input):
        input = self.avgpool2d(input)
        return input


class Conv2dSame(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = (stride, 1)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 
            padding=0, stride=self.stride, dilation=dilation, groups=groups, bias=bias)

        if self.stride[0] == 1 and self.kernel_size == 1:
            self.pad_info = [0, 0, 0, 0]
        elif self.stride[0] == 1 and self.kernel_size == 3:
            self.pad_info = [1, 1, 1, 1]
        elif self.stride[0] == 2 and self.kernel_size == 3:
            self.pad_info = [1, 1, 0, 2]
        else:
            raise ValueError('Unsupported stride and kernel size')

        self.forward = self._forward0

    def _forward0(self, input):
        input = F.pad(input, self.pad_info)
        return self.conv(input)
    
    def _forward1(self, input):
        return self.conv(input)
    

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return Conv2dSame(in_planes, out_planes, kernel_size=3, stride=stride,
                     groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Conv2dSame(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, debug=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers 
        # downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.debug = debug

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        # out += identity
        out = F.add(out, identity)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers 
        # downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
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

        # out += identity
        out = F.add(out, identity)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """A Modified Resnet Version. height will get 32 downsample rate
    width will only get the 4 downsample rate.
    """

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 deep_stem=True, norm_layer=None, input_channels=1):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.deep_stem = deep_stem
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        self.groups = groups
        self.base_width = width_per_group
        self.stage_n = [1, 2, 3, 4]

        print('self.deep_stem:', self.deep_stem)
        print('input_channels:', input_channels)
        if self.deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(input_channels, self.inplanes // 2, 
                          kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(self.inplanes // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inplanes // 2, self.inplanes // 2, 
                          kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(self.inplanes // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inplanes // 2, self.inplanes, 
                          kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(self.inplanes),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, self.inplanes, 
                          kernel_size=7, stride=1, padding=3, bias=False),
                norm_layer(self.inplanes),
                nn.ReLU(inplace=True)
            )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=False)
        self.layer1 = self._make_layer(block, 64, layers[0], stage=self.stage_n[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2,
            dilate=replace_stride_with_dilation[0], stage=self.stage_n[1])
        
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2,
            dilate=replace_stride_with_dilation[1], stage=self.stage_n[2])
        
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2,
            dilate=replace_stride_with_dilation[2], stage=self.stage_n[3])
        
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, ceil_mode=False)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, stage=1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stage == 1:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride=1),
                    norm_layer(planes * block.expansion),
                    nn.ReLU(inplace=True),
                )
            else:
                downsample = nn.Sequential(
                    AvgPool2dPad(kernel_size=(stride, 1), 
                                 stride=(stride, 1), padding=(0,0)),
                    conv1x1(self.inplanes, planes * block.expansion, stride=1),
                    norm_layer(planes * block.expansion),
                    #nn.ReLU(inplace=True),
                )
        elif stage == 1:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=1),
                norm_layer(planes * block.expansion),
                nn.ReLU(inplace=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.maxpool1(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

def resnet10(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet10', BasicBlock, [1, 1, 1, 1], pretrained, progress,
                   **kwargs)

def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def test():
    config = {
        'engine_file': './output/models/model.plan',
        'weight_file': './output/models/weights.pkl',
        'inputs': [('pixel_value', (-1, 1, 32, -1))],
        'input_profiles': [((1, 1, 32, 40), (4, 1, 32, 800), (8, 1, 32, 800))],
    }
    
    model = resnet34()
    model.build_engine(config)
    
    session = trtlite.InferenceSession(config['engine_file'])
    x = np.random.rand(1, 1, 32, 224).astype(np.float32)
    output = session.run({'pixel_value': x})
    print('output', output['output_0'].shape)


if __name__ == '__main__':
    test()  