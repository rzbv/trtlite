import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import logging

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

        self.forward = self._forward1

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
                 norm_layer=None):
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

        out += identity
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

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
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
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
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
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer))
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

    def __init__(self, is_resnet_vd, hidden_dim):
        super(ProjectLayer, self).__init__()
        self.backbone = resnet34(input_channels=1)
        num_filters = 2048
        if is_resnet_vd:
            num_filters = 512
        self.post_conv = nn.Conv2d(num_filters, hidden_dim, 1, 1)
        self.post_bn = nn.BatchNorm2d(hidden_dim, eps=1e-3)
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.pe = PositionalEncoding(hidden_dim)
        # self.relu = nn.ReLU()

    def forward(self, x):
        # x shape order is (b, c, h, w)
        x = self.backbone(x)
        x = self.post_conv(x)
        x = self.post_bn(x)
        x = nn.ReLU(inplace=True)(x)
        x = x.permute(0, 3, 2, 1)
        b, w, h, c = x.shape
        x = x.reshape(b, w, h * c)
        x = self.dense(x)
        x = nn.ReLU()(x)
        x = self.pe(x)
        return x


class Embeddings(nn.Module):

    def __init__(self, word_vec_size, word_vocab_size, word_padding_idx, sparse=True):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(word_vocab_size,
                                      word_vec_size,
                                      padding_idx=word_padding_idx,
                                      sparse=sparse)

    def forward(self, source):
        return self.embedding(source)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dim (int): embedding size
    """

    def __init__(self, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))

        position = torch.arange(0, max_len)
        num_timescales = dim // 2
        log_timescale_increment = math.log(10000) / (num_timescales - 1)
        inv_timescales = torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment)
        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

        self.register_buffer('pe', signal)

    def forward(self, emb, step=None):
        """Embed inputs.

        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """
        if step is None:
            emb = emb + self.pe[:emb.size(1)]
        else:
            emb = emb + self.pe[step]
        return emb


################# for encoder #####################
def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device).type_as(lengths).repeat(
        batch_size, 1).lt(lengths.unsqueeze(1)))


def relative_matmul(x, z, transpose):
    """Helper function for relative positions attention."""
    batch_size = x.shape[0]
    heads = x.shape[1]
    length = x.shape[2]
    x_t = x.permute(2, 0, 1, 3)
    x_t_r = x_t.reshape(length, heads * batch_size, -1)
    if transpose:
        z_t = z.transpose(1, 2)
        x_tz_matmul = torch.matmul(x_t_r, z_t)
    else:
        x_tz_matmul = torch.matmul(x_t_r, z)
    x_tz_matmul_r = x_tz_matmul.reshape(length, batch_size, heads, -1)
    x_tz_matmul_r_t = x_tz_matmul_r.permute(1, 2, 0, 3)
    return x_tz_matmul_r_t


def generate_relative_positions_matrix(length, max_relative_positions, cache=False):
    """Generate the clipped relative positions matrix
       for a given length and maximum relative positions"""
    if cache:
        distance_mat = torch.arange(-length + 1, 1, 1).unsqueeze(0)
    else:
        range_vec = torch.arange(length)
        range_mat = range_vec.unsqueeze(-1).expand(-1, length).transpose(0, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
    distance_mat_clipped = torch.clamp(distance_mat,
                                       min=-max_relative_positions,
                                       max=max_relative_positions)
    # Shift values to be >= 0
    final_mat = distance_mat_clipped + max_relative_positions
    return final_mat


class EmptyLayer(nn.Module):

    def forward(self, x):
        return x


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
    """

    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x, ffn_mask=None):
        """Layer definition.

        Args:
            x: ``(batch_size, input_len, model_dim)``

        Returns:
            (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
        """
        short = x
        inter = self.relu(self.w_1(self.layer_norm(x)))
        output = self.w_2(inter)
        if ffn_mask is not None:
            output = torch.mul(output, ffn_mask)
        output = output + short
        return output


class MultiHeadedAttention(nn.Module):

    def __init__(self, head_count, model_dim, max_relative_positions=0):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head,
                                      bias=False)
        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head,
                                     bias=False)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head,
                                       bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.final_linear = nn.Linear(model_dim, model_dim, bias=False)

        self.max_relative_positions = max_relative_positions
        self.emp_node = EmptyLayer()

    def forward(self, key, value, query, mask=None, layer_cache=None, attn_type=None):
        """
        attn_type is self, query, key, value: (bb, 1, 512)
        attn_type is context, key, value is (bb, step_len, 512), query is (bb, 1, 512)
        """
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if attn_type == "self":
                query, key, value = self.linear_query(query),\
                                    self.linear_keys(query),\
                                    self.linear_values(query)
                key = shape(
                    key
                )  # (bb, head_count, -1, per_head_size), which is (60, 8, -1, 64)
                value = shape(value)
                if layer_cache["self_keys"] is not None:
                    key = torch.cat((layer_cache["self_keys"], key), dim=2)
                if layer_cache["self_values"] is not None:
                    value = torch.cat((layer_cache["self_values"], value), dim=2)
                layer_cache["self_keys"] = key
                layer_cache["self_values"] = value
            elif attn_type == "context":
                query = self.linear_query(query)
                if layer_cache["memory_keys"] is None:
                    key, value = self.linear_keys(key),\
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
                else:
                    key, value = layer_cache["memory_keys"],\
                               layer_cache["memory_values"]
                layer_cache["memory_keys"] = key
                layer_cache["memory_values"] = value
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)
        # query (bb, 1, 512)
        query = shape(query)
        # query (bb, head_count, 1, 64)
        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        # batch x num_heads x query_len x key_len, (bb, head_count, 1, ken_len)
        query_key = torch.matmul(query, key.transpose(2, 3))

        scores = query_key
        scores = scores.float()

        if mask is not None:
            mask = ~mask.unsqueeze(1)  # [B, 1, 1, T_values]
            mask = self.emp_node(mask)
            scores = scores.masked_fill(mask, -1e9)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)
        drop_attn = attn

        context_original = torch.matmul(drop_attn, value)

        context = unshape(context_original)  # (bb, 1, 512)

        output = self.final_linear(context)
        # Return multi-head attn
        attns = attn.view(batch_size, head_count, query_len, key_len)
        # output = self.emp_node(output)
        return output, attns


def print_first_k(tensor, name="tensor", k=50):
    print(f"{name} shape: {tensor.shape}")
    arr = tensor.cpu().numpy().flatten()

    print(f"{name} first {k} elements: {arr.tolist()[:k]}")
    print(f'abs mean: {np.abs(arr).mean()}')
    print(f'abs sum: {np.abs(arr).sum()}')
    print('<<<<<<<<<<<<\n')


class Conv1dShortcut(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            #padding='same',
            padding=1,
            dilation=1):
        super(Conv1dShortcut, self).__init__()
        self.conv1 = nn.Conv1d(in_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation)
        self.norm1 = nn.LayerNorm(out_channels, eps=1e-6)
        self.conv2 = nn.Conv1d(in_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation)
        self.norm2 = nn.LayerNorm(out_channels, eps=1e-6)

    def forward(self, input):
        shortcut = input
        out = self.conv1(input.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.norm1(nn.ReLU()(out))
        out = self.conv2(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.norm2(nn.ReLU()(out))

        out += shortcut
        return out


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
    """

    def __init__(self,
                 d_model,
                 heads,
                 d_ff,
                 max_relative_positions=0,
                 use_lookaround_conv=True):
        super(TransformerEncoderLayer, self).__init__()
        self.use_lookaround_conv = use_lookaround_conv
        if self.use_lookaround_conv:
            self.conv1d_shortcut = Conv1dShortcut(d_model, d_model)
        else:
            self.conv1d_shortcut = None

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.self_attn = MultiHeadedAttention(
            heads, d_model, max_relative_positions=max_relative_positions)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

    def forward(self, inputs, attn_mask, ffn_mask):
        if self.use_lookaround_conv:
            inputs = self.conv1d_shortcut(inputs)

        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm,
                                    input_norm,
                                    input_norm,
                                    mask=attn_mask,
                                    attn_type="self")
        out = context + inputs
        return self.feed_forward(out, ffn_mask)


class TransformerEncoder(nn.Module):
    """
    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self,
                 num_layers,
                 d_model,
                 heads,
                 d_ff,
                 max_relative_positions=0,
                 use_lookaround_conv=True):
        super(TransformerEncoder, self).__init__()

        self.transformer = nn.ModuleList([
            TransformerEncoderLayer(d_model,
                                    heads,
                                    d_ff,
                                    max_relative_positions=max_relative_positions,
                                    use_lookaround_conv=use_lookaround_conv)
            for i in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    @classmethod
    def from_opt(cls, opt):
        """Alternate constructor."""
        return cls(opt.encode_num_layer, opt.encode_hidden_dim, opt.encode_head_num,
                   opt.intermediate_size, 0, opt.use_lookaround_conv)

    def forward(self, src, attn_mask, ffn_mask):
        out = src.contiguous()
        # print_first_k(out, 'encode_input')
        # Run the forward pass of every layer of the tranformer.
        for layer in self.transformer:
            out = layer(out, attn_mask, ffn_mask)
        out = self.layer_norm(out)
        return out.transpose(0, 1).contiguous()


################# end encoder #####################

################# for encoder #####################
USE_CACHE_BATCH_MAJOR_ATTENTION = False


def get_op_cache_config(size_per_head, is_fp16):
    x = 8 if is_fp16 else 4
    use_batch_major_op_cache = False
    if USE_CACHE_BATCH_MAJOR_ATTENTION and size_per_head % x == 0:
        use_batch_major_op_cache = True
    
    x = x if use_batch_major_op_cache else 1
    return use_batch_major_op_cache, x


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, heads, d_ff):
        super(TransformerDecoderLayer, self).__init__()

        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.self_attn = MultiHeadedAttention(heads, d_model)

        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.context_attn = MultiHeadedAttention(heads, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.emp_node1 = EmptyLayer()

    def forward(self, *args, **kwargs):
        output, attns = self._forward(*args, **kwargs)
        top_attn = attns[:, 0, :, :].contiguous()

        return output, top_attn

    def _forward(self,
                 inputs,
                 memory_bank,
                 src_pad_mask=None,
                 tgt_pad_mask=None,
                 layer_cache=None,
                 step=None,
                 future=False):
        """ A naive forward pass for transformer decoder.

        # T: could be 1 in the case of stepwise decoding or tgt_len

        Args:
            inputs (FloatTensor): ``(batch_size, T, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (LongTensor): ``(batch_size, 1, src_len)``
            tgt_pad_mask (LongTensor): ``(batch_size, 1, T)``
            layer_cache (dict or None): cached layer info when stepwise decode
            step (int or None): stepwise decoding counter
            future (bool): If set True, do not apply future_mask.

        Returns:
            (FloatTensor, FloatTensor):

            * output ``(batch_size, T, model_dim)``
            * attns ``(batch_size, head, T, src_len)``

        """
        dec_mask = None

        if step is None:
            raise ValueError('step cant be none')

        input_norm = self.layer_norm_1(inputs)

        query, _ = self.self_attn(input_norm,
                                  input_norm,
                                  input_norm,
                                  mask=dec_mask,
                                  layer_cache=layer_cache,
                                  attn_type="self")

        query = query + inputs
        short = query
        query_norm = self.layer_norm_2(query)
        mid, attns = self.context_attn(memory_bank,
                                       memory_bank,
                                       query_norm,
                                       mask=src_pad_mask,
                                       layer_cache=layer_cache,
                                       attn_type="context")
        query = short + mid
        output = self.feed_forward(query)
        return output, attns


class TransformerDecoder(nn.Module):
    """The Transformer decoder from "Attention is All You Need".
    Args:
        num_layers (int): number of encoder layers.
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        copy_attn (bool): if using a separate copy attention
        self_attn_type (str): type of self-attention scaled-dot, average
        embeddings (onmt.modules.Embeddings):
            embeddings to use, should have positional encodings
        max_relative_positions (int):
            Max distance between inputs in relative positions representations
        aan_useffn (bool): Turn on the FFN layer in the AAN decoder
        full_context_alignment (bool):
            whether enable an extra full context decoder forward for alignment
        alignment_layer (int): N° Layer to supervise with for alignment guiding
        alignment_heads (int):
            N. of cross attention heads to use for alignment guiding
    """

    def __init__(self,
                 num_layers,
                 d_model,
                 heads,
                 head_size,
                 d_ff,
                 word_vocab_size,
                 word_vec_size,
                 word_padding_idx,
                 data_type='fp32'):
        super(TransformerDecoder, self).__init__()

        self.embeddings = Embeddings(word_vec_size, word_vocab_size, word_padding_idx)
        self.pe = PositionalEncoding(word_vec_size)
        self.data_type = data_type
        self.is_fp16 = True if self.data_type == 'fp16' else False
        self.use_batch_major_op_cache, self.op_cache_dim_x = get_op_cache_config(
            head_size, self.is_fp16)
        self.d_model = d_model
        self.head_num = heads
        self.size_per_head = head_size

        # Decoder State
        self.state = {}

        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff) for i in range(num_layers)])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def init_state(self, src, memory_bank, enc_hidden):
        """Initialize decoder state."""
        self.state["src"] = src
        self.state["cache"] = None

    def map_state(self, fn):

        def _recursive_map(struct, batch_dim=0, use_batch_major_op_cache=False):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v, batch_dim, use_batch_major_op_cache)
                    else:
                        if isinstance(v, list):
                            batch_dim_ = 0 if use_batch_major_op_cache else 1
                            struct[k] = [fn(vv, batch_dim_) for vv in struct[k]]
                        else:
                            struct[k] = fn(v, batch_dim)

        self.state["src"] = fn(self.state["src"], 1)

        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"], 0)

    def detach_state(self):
        self.state["src"] = self.state["src"].detach()

    def forward(self, tgt, memory_bank, step=None, **kwargs):
        decoding_max_seq_len = kwargs["decoding_max_seq_len"]
        if step == 0:
            self._init_cache(memory_bank, decoding_max_seq_len)

        mask = torch.not_equal(tgt, 0)
        emb = self.embeddings(tgt.squeeze(2))
        assert emb.dim() == 3  # len x batch x embedding_dim

        emb = emb * mask
        emb = emb * (self.d_model**0.5)
        emb = self.pe(emb, step)
        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        src_lens = kwargs["memory_lengths"]
        src_max_len = self.state["src"].shape[0]
        src_pad_mask = sequence_mask(src_lens, src_max_len).unsqueeze(1)

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = self.state["cache"]["layer_{}".format(i)] \
                if step is not None else None
            output, attn = layer(output,
                                 src_memory_bank,
                                 src_pad_mask=src_pad_mask,
                                 tgt_pad_mask=None,
                                 layer_cache=layer_cache,
                                 step=step)

        dec_outs = self.layer_norm(output)
        return dec_outs

    def _init_cache(self, memory_bank, decoding_max_seq_len):
        self.state["cache"] = {}
        for i, layer in enumerate(self.transformer_layers):
            layer_cache = {"memory_keys": None, "memory_values": None}
            layer_cache["self_keys"] = None
            layer_cache["self_values"] = None
            self.state["cache"]["layer_{}".format(i)] = layer_cache


################# end encoder #####################


def gather_nd(params, indices):
    indices = indices.t().long()
    ndim = indices.size(0)
    idx = torch.zeros_like(indices[0]).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)

    params = params.reshape((-1, *tuple(torch.tensor(params.size()[ndim:]))))
    return params[idx]


def gather_tree(step_ids, parent_ids, max_sequence_lengths, end_token):
    beams = torch.empty_like(step_ids)
    beams.fill_(end_token)
    max_len = step_ids.size(0)
    batch_size = step_ids.size(1)
    beam_size = step_ids.size(-1)
    batch_beam = batch_size * beam_size
    for i in range(batch_beam):
        batch = i // beam_size
        beam = i % beam_size
        max_seq_len_b = min(max_len, max_sequence_lengths[batch])
        if max_seq_len_b <= 0:
            continue
        beams[max_seq_len_b - 1, batch, beam] = step_ids[max_seq_len_b - 1, batch, beam]
        parent = parent_ids[max_seq_len_b - 1, batch, beam]
        for level in range(max_seq_len_b - 2, -1, -1):
            if parent < 0 or parent > beam_size:
                raise ValueError("wrong parent id")
            beams[level, batch, beam] = step_ids[level, batch, parent]
            parent = parent_ids[level, batch, parent]
        finished = False
        for time in range(max_seq_len_b):
            if finished:
                beams[time, batch, beam] = end_token
            elif beams[time, batch, beam] == end_token:
                finished = True
    return beams


def finalize_v2(beam_size, output_ids, parent_ids, max_seq_len, end_id):
    max_lens = max_seq_len * torch.ones(
        [output_ids.shape[0]], dtype=torch.int32, device=output_ids.device)
    shape = (max_seq_len, -1, beam_size)
    parent_ids = parent_ids[:max_seq_len, :]
    output_ids = output_ids[:max_seq_len, :]

    output_ids = torch.reshape(output_ids, shape)
    parent_ids = torch.reshape(parent_ids, shape)

    ids = gather_tree(output_ids, parent_ids, max_lens, end_id)
    ids = torch.einsum('ijk->jki', ids)
    lengths = torch.eq(ids, end_id)
    lengths = 1 - lengths.to(output_ids.dtype)
    lengths = torch.sum(lengths, -1)
    return ids, lengths


def finalize(beam_size, output_ids, parent_ids, out_seq_lens, end_id, max_seq_len=None):
    out_seq_lens = torch.reshape(out_seq_lens, (-1, beam_size))
    max_lens = torch.max(out_seq_lens, 1)[0]
    if max_seq_len:
        shape = (max_seq_len.item(), -1, beam_size)
    else:
        shape = (torch.max(max_lens).item(), -1, beam_size)
    output_ids = torch.reshape(output_ids, shape)
    parent_ids = torch.reshape(parent_ids, shape)

    ids = gather_tree(output_ids, parent_ids, max_lens, end_id)
    ids = torch.einsum('ijk->jki', ids)  # batch_size, beam_size, max_seq_len
    lengths = torch.eq(ids, end_id)
    lengths = 1 - lengths.to(output_ids.dtype)
    lengths = torch.sum(lengths, -1)
    return ids, lengths


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


class TransformerDecoding(nn.Module):

    def __init__(
        self,
        layer_num,
        head_num,
        head_size,
        vocab_size,
        start_id,
        end_id,
        pad_id,
        batch_size,
        beam_size,
        extra_decode_length,
        decoder_emb_size,
        is_resnet_vd,
        beam_search_diversity_rate=0.0,
    ):
        super().__init__()
        self.layer_num = layer_num
        self.hidden_dim = head_num * head_size
        self.start_id = start_id
        self.end_id = end_id
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.extra_decode_length = extra_decode_length
        self.diversity_rate = beam_search_diversity_rate
        self.decoder = TransformerDecoder(layer_num, self.hidden_dim, head_num,
                                          head_size, 2 * self.hidden_dim, vocab_size,
                                          decoder_emb_size, pad_id)
        self.generator = nn.Linear(self.hidden_dim, vocab_size, bias=False)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.is_resnet_vd = is_resnet_vd

    @classmethod
    def from_opt(cls, opt, vocab):
        return cls(
            opt.decoder_num_layer,
            opt.decoder_head_num,
            opt.decoder_size_per_head,
            vocab.num_vocab,
            opt.start_of_sentence_id,
            opt.end_of_sentence_id,
            vocab.stoi[vocab.special_tokens['pad_token']],
            opt.batch_size,
            opt.beam_size,
            opt.extra_decode_length,
            opt.decoder_emb_size,
            opt.is_resnet_vd,
        )

    def search(self, step, word_ids, extended_memory, extended_memory_seq_lens,
               max_seq_len):
        dec_out = self.decoder(word_ids,
                               extended_memory,
                               memory_lengths=extended_memory_seq_lens,
                               step=step,
                               decoding_max_seq_len=max_seq_len)

        logits = self.generator(dec_out)

        log_probs = self.logsoftmax(logits.to(torch.float32))
        scores, word_ids = torch.topk(log_probs.view(-1, 1, self.vocab_size), 1)
        return word_ids, scores

    def forward(self, memory, memory_seq_lens, batch_size):
        max_seq_len = memory.shape[1] + self.extra_decode_length

        extended_memory = memory.transpose(0, 1).contiguous()

        # 使用torch.zeros和torch.full替代new_full，确保TorchScript兼容性
        start_ids = torch.full((batch_size, ),
                               self.start_id,
                               dtype=torch.int64,
                               device=memory.device)
        finished = torch.zeros((batch_size, ), dtype=torch.bool, device=memory.device)

        output_scores = torch.zeros([max_seq_len, batch_size],
                                    dtype=torch.float32,
                                    device=memory.device)
        output_ids = torch.zeros([max_seq_len, batch_size],
                                 dtype=torch.int64,
                                 device=memory.device)

        self.decoder.init_state(extended_memory, extended_memory, None)

        # 贪婪解码
        word_ids = start_ids.view(1, -1, 1)
        seq_len = -1
        for step in range(max_seq_len):
            if not torch.bitwise_not(finished).any():
                break

            # 执行一步推理
            word_ids, scores = self.search(step, word_ids.view(1, -1,
                                                               1), extended_memory,
                                           memory_seq_lens, max_seq_len)

            # 对已结束序列强制输出 end_id
            finished = torch.bitwise_or(finished,
                                        torch.eq(word_ids.view(-1), self.end_id))
            output_ids[step, :] = word_ids.view(-1)
            output_scores[step, :] = scores.view(-1)
            seq_len += 1

        output_ids = output_ids[:seq_len + 1, :]
        output_scores = output_scores[:seq_len + 1, :]

        beams = torch.transpose(output_ids, 0, 1).view(batch_size, 1, -1)
        scores = torch.transpose(output_scores, 0, 1).view(batch_size, 1, -1)
        return beams, scores


class CtcDecoding(nn.Module):

    def __init__(self, decoder_emb_size, vocab_size):
        super().__init__()
        self.decoder_emb_size = decoder_emb_size
        self.vocab_size = vocab_size
        self.generator = nn.Linear(self.decoder_emb_size, vocab_size, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    @classmethod
    def from_opt(cls, opt, vocab):
        return cls(
            opt.decoder_emb_size,
            vocab.num_vocab,
        )

    def forward(self, memory, memory_seq_lens, batch_size):
        logits = self.generator(memory)
        log_probs = self.softmax(logits.to(torch.float32))
        m = torch.max(log_probs, dim=-1)
        inds = m.indices
        probs = m.values
        return inds, probs


class VisionTransformer(nn.Module):

    def __init__(self, is_resnet_vd, backbone, encoder, decoding):
        super(VisionTransformer, self).__init__()
        self.downsample = 4 if is_resnet_vd else 8
        self.backbone = backbone
        self.encoder = encoder
        self.decoding = decoding

    def _get_padding(self, feats, inputs_length):
        device = feats.device
        feat_shape = feats.shape
        r = torch.arange(feat_shape[1], dtype=torch.int32, device=device)
        indexs = r.repeat(1, feat_shape[0])
        indexs = torch.reshape(indexs, feat_shape[:2])
        inputs_length = inputs_length.repeat(1, feat_shape[1])
        cond = indexs < inputs_length
        return cond

    def _get_mask(self, feats, inputs, inputs_shape):
        b, w, hidden_dim = feats.shape
        inputs_length = (inputs_shape[:, 1:2] / self.downsample).to(torch.int32)
        memory_sequence_length = (inputs_shape[:, 1] / self.downsample).to(torch.int32)
        mask = self._get_padding(feats, inputs_length)
        mask = torch.reshape(mask, [b, w, 1])
        #ffn_mask = torch.tile(mask, [1, 1, hidden_dim])
        ffn_mask = mask.repeat(1, 1, hidden_dim)
        #attention_mask = torch.tile(mask, [1, 1, w])
        attention_mask = mask.repeat(1, 1, w)
        attention_mask = torch.transpose(attention_mask, 2, 1)
        return memory_sequence_length, attention_mask, ffn_mask

    def forward(self, image, image_shape):
        x = self.backbone(image)
        logging.info(f'backbone out {x.shape}, {x.sum(dim=(1,2))}')

        b, w, hidden_dim = x.shape
        # attn_mask: (b, w, w), ffn_mask: (b, w, h)
        mem_seq_len, attn_mask, ffn_mask = self._get_mask(x, image, image_shape)
        encoder_out = self.encoder(x, attn_mask, ffn_mask)
        encoder_out = encoder_out.transpose(0, 1).contiguous()
        outputs = self.decoding(encoder_out, mem_seq_len, b)

        return outputs


class Tokenizer(object):

    def __init__(self, token_list, special_tokens):
        if isinstance(token_list, str):
            assert os.path.exists(token_list)
            self.vocabs = load_charset(token_list)
        else:
            self.vocabs = token_list

        self.num_vocab = len(self.vocabs)
        self.special_tokens = special_tokens
        self.stoi = dict((s, i) for i, s in enumerate(self.vocabs))

    def decode(self, token_ids):
        return ''.join([self.vocabs[token_id] for token_id in token_ids])

    def __size__(self):
        return self.num_vocab


class TransformerArgument:

    def __init__(self, opt, args):
        self.args = args

        for key, value in opt.items():
            setattr(self, key, value)

        if hasattr(self, 'encode_hidden_dim') and self.encode_hidden_dim is None:
            self.encode_hidden_dim = self.encode_head_num * self.encode_size_per_head

        if hasattr(self, 'intermediate_size') and self.intermediate_size is None:
            self.intermediate_size = self.encode_hidden_dim * 2


def load_charset(charsetpath):
    with open(charsetpath) as f:
        lines = f.readlines()
        lines = list(map(lambda x: x.replace('\n', ''), lines))
        chars = [xx for xx in lines if len(xx) > 0]
    
    if len(chars) == 1:
        chars = list(chars[0])

    chars = ['[PAD]', '[EOS]'] + chars
    return chars

def load_np(src_dir, name):
    bin_name = src_dir + '/inputs/bin/' + name
    shape_name = src_dir + '/inputs/shape/' + name
    s = np.fromfile(shape_name, dtype=np.int32)
    ims = np.fromfile(bin_name, dtype=np.float32).astype(np.float32).reshape(s)

    imshape_name = src_dir + '/inputs_shape/bin/' + name
    shape_name = src_dir + '/inputs_shape/shape/' + name
    s = np.fromfile(shape_name, dtype=np.int32)
    ims_shape = np.fromfile(imshape_name, dtype=np.int32).reshape(s)

    return ims, ims_shape


def test_infer():
    pt_path = './output/models/model.pth'
    charset_path = './output/models/charsets_v1.txt'
    is_resnet_vd = True

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    params = {}
    params["encode_hidden_dim"] = 512
    params["encode_num_layer"] = 3
    params["encode_head_num"] = 8
    params["decoder_num_layer"] = 3
    params["decoder_head_num"] = 8
    params["decoder_size_per_head"] = 64
    params["start_of_sentence_id"] = 0
    params["end_of_sentence_id"] = 1
    params["batch_size"] = 32
    params["beam_size"] = 1
    params["extra_decode_length"] = 10
    params["decoder_emb_size"] = 512
    params["intermediate_size"] = None
    params["is_resnet_vd"] = is_resnet_vd
    params['backbone'] = "resnet34_vd"
    params["use_lookaround_conv"] = True

    transargs = TransformerArgument(params, {})

    chars = load_charset(charset_path)
    vocab = Tokenizer(chars,
                      special_tokens={
                          'eos_token': '[EOS]',
                          'pad_token': '[PAD]'
                      })
    
    # build CNN + Project
    backbone = ProjectLayer(True, transargs.encode_hidden_dim)
    # build encoder
    encoder = TransformerEncoder.from_opt(transargs)
    decoding = TransformerDecoding.from_opt(transargs, vocab)
    model = VisionTransformer(is_resnet_vd, backbone, encoder, decoding)
    model.load_state_dict(torch.load(pt_path), strict=False)
    model.eval()


if __name__ == '__main__':
    test_infer()

