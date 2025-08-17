import math

import numpy as np
import tensorrt as trt
from absl import logging

import trtlite
import trtlite.nn as nn
import trtlite.nn.functional as F
from trtlite.plugins.layers import TransformerDecoderPlugin, TransformerEncoderLayerV1

logging.set_verbosity(logging.INFO)


class AvgPool2dPad(nn.Module):

    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.avgpool2d = nn.AvgPool2d(kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      ceil_mode=False)
        self.forward = self._forward

    def _forward(self, input):
        input = self.avgpool2d(input)
        return input


class Conv2dSame(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = (stride, 1)
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              padding=0,
                              stride=self.stride,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)

        if self.stride[0] == 1 and self.kernel_size == 1:
            self.pad_info = [0, 0, 0, 0]
        elif self.stride[0] == 1 and self.kernel_size == 3:
            self.pad_info = [1, 1, 1, 1]
        elif self.stride[0] == 2 and self.kernel_size == 3:
            self.pad_info = [1, 1, 0, 2]
        else:
            raise ValueError('Unsupported stride and kernel size')

        # self.pad_info = [0, 0, 0, 0]
        # self.forward = self._forward1
        self.forward = self._forward0

    def _forward0(self, input):
        input = F.pad(input, self.pad_info)
        return self.conv(input)

    def _forward1(self, input):
        input_rows = input.size(2)
        filter_rows = self.kernel_size
        out_rows = (input_rows + self.stride[0] - 1) // self.stride[0]
        padding_rows = max(0, (out_rows - 1) * self.stride[0] +
                           (filter_rows - 1) * self.dilation + 1 - input_rows)
        rows_odd = (padding_rows % 2 != 0)
        padding_cols = max(0, (out_rows - 1) * self.stride[1] +
                           (filter_rows - 1) * self.dilation + 1 - input_rows)
        cols_odd = (padding_cols % 2 != 0)

        pad_total = self.kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        self.pad_info = [pad_beg, pad_end, pad_beg, pad_end]
        if rows_odd:
            self.pad_info = [pad_beg, pad_end, 0, pad_total]
        elif cols_odd:
            self.pad_info = [0, pad_total, pad_beg, pad_end]

        logging.info('conv2d info: %s,%s,%s', self.pad_info, self.stride,
                     self.kernel_size)
        input = F.pad(input, self.pad_info)
        input = self.conv(input)
        return input

    def _forward2(self, input):
        return self.conv(input)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return Conv2dSame(in_planes,
                      out_planes,
                      kernel_size=3,
                      stride=stride,
                      groups=groups,
                      bias=False,
                      dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Conv2dSame(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None,
                 debug=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
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

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
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

    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 zero_init_residual=True,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 deep_stem=True,
                 norm_layer=None,
                 input_channels=1):
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
                nn.Conv2d(input_channels,
                          self.inplanes // 2,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False),
                norm_layer(self.inplanes // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inplanes // 2,
                          self.inplanes // 2,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False),
                norm_layer(self.inplanes // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inplanes // 2,
                          self.inplanes,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False),
                norm_layer(self.inplanes),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3,
                          self.inplanes,
                          kernel_size=7,
                          stride=1,
                          padding=3,
                          bias=False), norm_layer(self.inplanes), nn.ReLU(inplace=True))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=False)
        self.layer1 = self._make_layer(block, 64, layers[0], stage=self.stage_n[0])
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       stage=self.stage_n[1])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       stage=self.stage_n[2])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       stage=self.stage_n[3])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2,
                                     stride=2,
                                     padding=0,
                                     ceil_mode=False)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    dilate=False,
                    stage=1,
                    debug=False):
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
                                 stride=(stride, 1),
                                 padding=(0, 0)),
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
        layers.append(
            block(self.inplanes,
                  planes,
                  stride,
                  downsample,
                  self.groups,
                  self.base_width,
                  previous_dilation,
                  norm_layer,
                  debug=debug))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
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

def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


class ProjectLayer(nn.Module):
    def __init__(self, cnn, hidden_dim):
        super(ProjectLayer, self).__init__()
        self.backbone = cnn

        num_filters = 512
        self.post_conv = nn.Conv2d(num_filters, hidden_dim, 1, 1)
        self.post_bn = nn.BatchNorm2d(hidden_dim, eps=1e-3)
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.pe = PositionalEncoding(hidden_dim)

    def forward(self, x):
        # x: (b, c, h, w)
        x = self.backbone(x)
        x = self.post_conv(x)
        x = self.post_bn(x)
        x = nn.ReLU(inplace=True)(x)
        x = F.permute(x, (0, 3, 2, 1))
        shape = F.get_shape(x)
        b, w, h, c = F.split(shape, [1, 1, 1, 1], dim=0)
        c = F.mul(h, c)
        shape = F.cat([b, w, c])
        x = F.reshape(x, shape)
        x = self.dense(x)
        x = F.relu(x)
        x = self.pe(x)
        return x
    

class Embeddings(nn.Module):
    def __init__(self, word_vec_size,
                 word_vocab_size,
                 word_padding_idx,
                 sparse=True):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(
            word_vocab_size, word_vec_size, 
            padding_idx=word_padding_idx, sparse=sparse)

    def forward(self, source):
        return self.embedding(source)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.
    Args:
       dim (int): embedding size
    """

    def __init__(self, dim, max_len=1000):
        super(PositionalEncoding, self).__init__()
        assert dim % 2 == 0, ("dim should be even")
        position = np.arange(0, max_len)
        num_timescales = dim // 2
        log_timescale_increment = math.log(10000) / (num_timescales - 1)
        inv_timescales = np.exp(
            np.arange(num_timescales, dtype=np.float32) * -log_timescale_increment)
        scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)
        signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
        self.pe = signal.astype(np.float32)

    def forward(self, emb, step=None):
        pe = F.add_constant(self.pe.shape, self.pe)
        emb_shape = F.get_shape(emb)
        bs = F.gather(emb_shape, 0, 0)
        if step is None:
            emb_len = F.gather(emb_shape, 0, 1)
            pe = F.slice(pe, shape=emb_len, dim=0)
            pe_part = F.add_tile(pe, bs, 0)
            emb = F.add(emb, pe_part)
        else:
            pe_part = F.gather(pe, 0, step)
            pe_part = F.add_tile(pe_part, bs, 0)
            emb = F.add(emb, pe_part)
        return emb


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff, 
            max_relative_positions=0):
        super(TransformerEncoder, self).__init__()

        self.transformer = nn.ModuleList(
            [TransformerEncoderLayerV1(d_model, heads, d_ff, i,
                int(i==num_layers-1))
             for i in range(num_layers)])
        
        # last layer_norm is merged into TransformerEncoderLayer
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    @classmethod
    def from_opt(cls, opt):
        """Alternate constructor."""
        return cls(
            opt.encode_num_layer,
            opt.encode_hidden_dim,
            opt.encode_head_num,
            opt.intermediate_size,
            )

    def forward(self, src, attn_mask, ffn_mask):
        # out = src.contiguous()
        # Run the forward pass of every layer of the tranformer.
        for layer in self.transformer:
            src = layer([src, attn_mask, ffn_mask])
            # break

        # out = self.layer_norm(out)s
        # ALERT: get rid of transpose
        return src

def show_tensor(t, name='tensor'):
    print(f'INFO tensor name=[{name}] shape=[{t.shape}] dtype=[{t.dtype}]')

class TransformerDecoder(nn.Module):
    def __init__(self, 
                 layer_num, head_num, head_size, 
                 vocab_size, start_id, end_id, beam_size):
        super().__init__()
        self.decoder = TransformerDecoderPlugin(
            layer_num, head_num, head_size, vocab_size, start_id, end_id, beam_size)
        self.beam_size = beam_size

    def getExtendSeqLen(self, memory_seq_lens, beam_size):
        ONE = F.add_constant([1], 1, np.int64)
        BEAM = F.add_constant([1], beam_size, np.int64)
        memory_seq_lens = F.add_tile(memory_seq_lens, F.cat([ONE, BEAM]))
        return memory_seq_lens

    def getExtendMemory(self, inputs, beam_size):
        shape = F.get_shape(inputs)
        B = F.gather(shape, 0, 0)
        W = F.gather(shape, 0, 1)
        C = F.gather(shape, 0, 2)
        ONE = F.add_constant([1], 1, np.int64)
        BEAM = F.add_constant([1], beam_size, np.int64)

        inputs = F.reshape(inputs, F.cat([B, ONE, W, C]))
        inputs = F.add_tile(inputs, F.cat([ONE, BEAM, ONE, ONE]))
        B = F.prod_(B, BEAM)
        inputs = F.reshape(inputs, F.cat([B, W, C]))
        return inputs

    def forward(self, memory, memory_seq_lens, beam_size=None):
        if beam_size is None:
            beam_size = self.beam_size

        ext_mem_seq_len = self.getExtendSeqLen(memory_seq_lens, beam_size)
        ext_memory = self.getExtendMemory(memory, beam_size)
        xs = [ext_memory, ext_mem_seq_len]
        return self.decoder.forward(xs)


class VisionTransformer(nn.Module):
    def __init__(self, is_resnet_vd, backbone, encoder, decoding):
        super(VisionTransformer, self).__init__()
        self.downsample = 4 if is_resnet_vd else 8
        self.backbone = backbone
        self.encoder = encoder
        self.decoding = decoding

    def getInputsPadding(self, inputs, inputs_shape):
        L = F.gather(inputs_shape, 1, 1)
        shape = F.get_shape(inputs)
        B = F.gather(shape, 0, 0)
        W = F.gather(shape, 0, 1)

        ONE = F.add_constant([1], 1, np.int64)

        DIV = F.add_constant([1], self.downsample, np.float32)
        DIV = F.add_tile(DIV, B, expand_dim=0)
        L = F.div_(L, DIV)
        
        extended_seq_len = F.identity(L, trt.DataType.INT32)

        L = F.add_tile(L, F.cat([ONE, W]))

        indexs = F.arange(W, np.float32)
        indexs = F.add_tile(indexs, B, expand_dim=0)

        outputs = F.less_(indexs, L)
        outputs = F.identity(outputs, trt.DataType.FLOAT)
        outputs = F.reshape(outputs, F.cat([B, ONE, W]))
        return outputs, extended_seq_len

    def getAttBias(self, inputs, inputs_padding):
        shape = F.get_shape(inputs)
        W = F.gather(shape, 0, 1)
        ONE = F.add_constant([1], 1, np.int64)
        outputs = F.add_tile(inputs_padding, F.cat([ONE, W, ONE]))
        return outputs

    def getFFNBias(self, inputs, inputs_padding):
        shape = F.get_shape(inputs)
        H = F.gather(shape, 0, 2)
        ONE = F.add_constant([1], 1, np.int64)
        perm = (0, 2, 1)
        outputs = F.transpose(inputs_padding, perm)
        outputs = F.add_tile(outputs, F.cat([ONE, ONE, H]))
        return outputs

    def forward(self, image, image_shape):
        x = self.backbone(image)
        inputs_padding, mem_seq_len = self.getInputsPadding(x, image_shape)
        attn_mask = self.getAttBias(x, inputs_padding)
        ffn_mask = self.getFFNBias(x, inputs_padding)
        encoder_out = self.encoder(x, attn_mask, ffn_mask)
        output_ids, parent_ids, seq_len = self.decoding(encoder_out, mem_seq_len)
        return output_ids, parent_ids, seq_len

def test():
    config = {
        'engine_file': './output/models/vis_transformer_fp32.plan',
        'weight_file': './output/models/model_trt.pkl',
        'inputs': [('image', (-1, 1, 32, -1)), ('image_shape', (-1, 2))],
        'input_profiles': [
            ((1, 1, 32, 40), (64, 1, 32, 832), (128, 1, 32, 1032)),
            ((1, 2), (64, 2), (128, 2))
        ],
        'output_names': ['output_ids', 'parent_ids', 'seq_len'],
        'precision': 'fp32',
        'max_workspace_size': 8 * (1 << 30)
    }

    vocab_size = 6411
    backbone = resnet34()
    project_layer = ProjectLayer(backbone, 512)
    encoder = TransformerEncoder(3, 512, 8, 1024)
    decoder = TransformerDecoder(3, 8, 64, vocab_size, 0, 1, 5)
    model = VisionTransformer(True, project_layer, encoder, decoder)
    # model.build_engine(config)
    # return

    session = trtlite.InferenceSession(config['engine_file'])
    import pickle
    with open('./output/models/preprocess_output.pkl', 'rb') as f:
        image, images_shape = pickle.load(f)

    images = np.transpose(image, [0, 3, 1, 2])
    output = session.run({'image': images, 'image_shape': images_shape})
    print('output', output)

if __name__ == '__main__':
    test()
