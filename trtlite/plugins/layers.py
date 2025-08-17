from typing import List

import numpy as np
import tensorrt as trt
from trtlite.nn.modules.context import get_default_trt_context
from trtlite.nn.modules.module import Module
from trtlite.plugins.common import PluginCreatorContext, create_plugin_field

ITensor = trt.ITensor

PluginCreatorContext.global_init()


class TransformerEncoderLayerV1(Module):
    def __init__(self, d_model, heads, d_ff, index, last_layer, **kwargs):
        super().__init__()
        self.layer_name = 'TransformerEncodePluginDynamic'
        self.index = index
        self.name_scope = 'encoder.transformer' + '.' + str(index)

        type_id = trt.DataType.FLOAT
        type_id = create_plugin_field('type_id', [type_id], np.int32)
        hidden_size = create_plugin_field('hidden_size', [d_model], np.int32)
        num_heads = create_plugin_field('num_heads', [heads], np.int32)
        is_lastLayer = create_plugin_field('isLastLayer', [last_layer], np.int32)

        pfc = trt.PluginFieldCollection([
            type_id, hidden_size, num_heads, is_lastLayer
        ])

        creator = PluginCreatorContext.get_creator(self.layer_name)
        assert creator, f'failed to create a creator [{self.layer_name}]'
        self.fn = creator.create_plugin(self.layer_name, pfc)

    def get_weights(self):
        ctx = get_default_trt_context()
        weight_names = [
            "conv1d_shortcut.conv1.weight",
            'conv1d_shortcut.conv1.bias',
            'conv1d_shortcut.norm1.weight',
            'conv1d_shortcut.norm1.bias',
            'conv1d_shortcut.conv2.weight',
            'conv1d_shortcut.conv2.bias',
            'conv1d_shortcut.norm2.weight',
            'conv1d_shortcut.norm2.bias',
            "layer_norm.weight",
            "layer_norm.bias",
            "self_attn.linear_query.weight",
            "self_attn.linear_keys.weight",
            "self_attn.linear_values.weight",
            "self_attn.final_linear.weight",
            "feed_forward.layer_norm.weight",
            "feed_forward.layer_norm.bias",
            "feed_forward.w_1.weight",
            "feed_forward.w_1.bias",
            "feed_forward.w_2.weight",
            "feed_forward.w_2.bias",
        ]

        weights = []
        for name in weight_names:
            weights.append(ctx.get_weight_by_name(self.name_scope + '.' + name))
        
        weights.append(ctx.get_weight_by_name('encoder.layer_norm.weight'))
        weights.append(ctx.get_weight_by_name('encoder.layer_norm.bias'))

        tensors = []
        for i, w in enumerate(weights):
            name = f"{self.name_scope}.weight.{i}"
            tensors.append(ctx.network.add_constant(w.shape, w).get_output(0))

        return tensors
    
    def forward(self, xs: List[ITensor]):
        ctx = get_default_trt_context()
        weights = self.get_weights()
        xs = xs + weights
        assert len(xs) == 25, len(xs)
        layer = ctx.network.add_plugin_v2(xs, self.fn)
        return layer.get_output(0)


class TransformerDecoderPlugin(Module):
    def __init__(self, layer_num, head_num, head_size, 
                 vocab_size, start_id, end_id, beam_size):
        super().__init__()
        type_id = trt.DataType.FLOAT
        # type_id = trt.DataType.HALF

        self.beam_size = beam_size
        self.num_layer = layer_num
        self.layer_name = 'TransformerDecodePluginDynamic'
        type_id = create_plugin_field('type_id', [type_id], np.int32)
        hidden_size = create_plugin_field(
            'hidden_size', [head_num * head_size], np.int32)
        
        num_heads = create_plugin_field('num_heads', [head_num], np.int32)
        beam_width = create_plugin_field('beam_width', [beam_size], np.int32)
        vocab_size = create_plugin_field('vocab_size', [vocab_size], np.int32)
        start_id = create_plugin_field('start_id', [start_id], np.int32)
        end_id = create_plugin_field('end_id', [end_id], np.int32)
        num_layer = create_plugin_field('num_layer', [layer_num], np.int32)

        pfc = trt.PluginFieldCollection([
            type_id, hidden_size, num_heads, beam_width,
            vocab_size, start_id, end_id, num_layer,
        ])

        creator = PluginCreatorContext.get_creator(self.layer_name)
        assert creator, f'failed to create a creator [{self.layer_name}]'
        self.fn = creator.create_plugin(self.layer_name, pfc)

    def get_weights(self):
        ctx = get_default_trt_context()
        weights = []

        common_names = [
            'decoding.decoder.layer_norm.weight',
            'decoding.decoder.layer_norm.bias',
            'decoding.decoder.embeddings.embedding.weight'
        ]

        weight_names = [
            "layer_norm_1.weight",
            "layer_norm_1.bias",
            "self_attn.linear_query.weight",
            "self_attn.linear_keys.weight",
            "self_attn.linear_values.weight",
            "self_attn.final_linear.weight",
            "layer_norm_2.weight",
            "layer_norm_2.bias",
            "context_attn.linear_query.weight",
            "context_attn.linear_keys.weight",
            "context_attn.linear_values.weight",
            "context_attn.final_linear.weight",
            "feed_forward.layer_norm.weight",
            "feed_forward.layer_norm.bias",
            "feed_forward.w_1.weight",
            "feed_forward.w_1.bias",
            "feed_forward.w_2.weight",
            "feed_forward.w_2.bias",
        ]

        weights = []
        for name in weight_names:
            tmps = []
            for i in range(self.num_layer):
                name_scope = f'decoding.decoder.transformer_layers.{i}'
                tmps.append(ctx.get_weight_by_name(name_scope + '.' + name))

            weights.append(np.concatenate(tmps, axis=0))
        
        for name in common_names:
            weights.append(ctx.get_weight_by_name(name))
        
        assert len(weights) == 18 + 3, len(weights)

        tensors = []
        for _, w in enumerate(weights):
            # if w.dtype == np.float32:
            #     w = w.astype(np.float16)
            tensors.append(ctx.network.add_constant(w.shape, w).get_output(0))

        return tensors
    
    def forward(self, xs: List[ITensor]):
        weights = self.get_weights()

        xs = xs + weights
        ctx = get_default_trt_context()
        layer = ctx.network.add_plugin_v2(xs, self.fn)

        output_ids = layer.get_output(0)
        parent_ids = layer.get_output(1)
        seq_len = layer.get_output(2)

        return output_ids, parent_ids, seq_len
