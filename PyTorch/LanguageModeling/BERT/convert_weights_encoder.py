# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import nnabla as nn
import nnabla.parametric_functions as PF
import numpy
import argparse

parser = argparse.ArgumentParser(description='esrgan')
parser.add_argument('--pretrained_model', default='./RRDB_ESRGAN_x4.pth',
                    help='path to pytorch pretrained model')
parser.add_argument('--save_path', default='./ESRGAN_NNabla_model.h5',
                    help='Path to save h5 file')
args = parser.parse_args()


def pytorch_to_nn_param_map():
    '''map from tensor name to Nnabla default parameter names
    '''
    return {
        'weight': 'W',
        'bias': 'b',
        '.': '/',
        'position_embeddings': 'position_embeddings/embed/',
        'word_embeddings': 'word_embeddings/embed/W',
        'LayerNorm': 'layer_normalization',
        'token_type_embeddings': 'token_type_embeddings/embed',
        'pooler.dense.bias': 'pooler/affine/b',
        'pooler/dense/kernel': 'pooler/affine/W',
        'seq_relationship.output_bias': 'affine_seq_class/affine/b'
    }


def rename_params(key: str, v):
    idx = key.split('.')[0]
    layer_id = int(idx) if idx.isnumeric() else 0
    if 'query.bias' in key:
        key = (f"encoder{layer_id:02d}/transformer_encode/"
               "src_self_attn/multi_head_attention/q_bias")
    elif 'query.weight' in key:
        key = (f"encoder{layer_id:02d}/transformer_encode/"
               "src_self_attn/multi_head_attention/q_weight")
        v = numpy.transpose(v)
    elif 'key.bias' in key:
        key = (f"encoder{layer_id:02d}/transformer_encode/"
               "src_self_attn/multi_head_attention/k_bias")
    elif 'key.weight' in key:
        key = (f"encoder{layer_id:02d}/transformer_encode/"
               "src_self_attn/multi_head_attention/k_weight")
        v = numpy.transpose(v)
    elif 'value.bias' in key:
        key = (f"encoder{layer_id:02d}/transformer_encode/"
               "src_self_attn/multi_head_attention/v_bias")
    elif 'value.weight' in key:
        key = (f"encoder{layer_id:02d}/transformer_encode/"
               "src_self_attn/multi_head_attention/v_weight")
        v = numpy.transpose(v)
    elif 'attention.output.LayerNorm.bias' in key:
        key = (f"encoder{layer_id:02d}/transformer_encode/"
               "enc_layer_norm1/layer_normalization/beta")
        v = numpy.reshape(v, (1, 1, v.shape[-1]))
    elif 'output.LayerNorm.bias' in key:
        key = (f"encoder{layer_id:02d}/transformer_encode/"
               "enc_layer_norm2/layer_normalization/beta")
        v = numpy.reshape(v, (1, 1, v.shape[-1]))
    elif 'attention.output.LayerNorm.weight' in key:
        key = (f"encoder{layer_id:02d}/transformer_encode/"
               "enc_layer_norm1/layer_normalization/gamma")
        v = numpy.reshape(v, (1, 1, v.shape[-1]))
    elif 'output.LayerNorm.weight' in key:
        key = (f"encoder{layer_id:02d}/transformer_encode/"
               "enc_layer_norm2/layer_normalization/gamma")
        v = numpy.reshape(v, (1, 1, v.shape[-1]))
    elif 'attention.output.dense.bias' in key:
        key = (f"encoder{layer_id:02d}/transformer_encode/"
               "src_self_attn/multi_head_attention/out_bias")
    elif 'output.dense.bias' in key:
        key = (f"encoder{layer_id:02d}/transformer_encode/"
               "enc_affine2/affine/b")
    elif 'intermediate.dense_act.bias' in key:
        key = (f"encoder{layer_id:02d}/transformer_encode/"
               "enc_affine1/affine/b")
    elif 'attention.output.dense.weight' in key:
        key = (f"encoder{layer_id:02d}/transformer_encode/"
               "src_self_attn/multi_head_attention/out_weight")
        v = numpy.transpose(v)
    elif 'output.dense.weight' in key:
        key = (f"encoder{layer_id:02d}/transformer_encode/"
               "enc_affine2/affine/W")
        v = numpy.transpose(v)
    elif 'intermediate.dense_act.weight' in key:
        key = (f"encoder{layer_id:02d}/transformer_encode/"
               "enc_affine1/affine/W")
        v = numpy.transpose(v)
    elif 'embeddings.LayerNorm' in key:
        if 'weight' in key:
            key = key.replace('bert.embeddings.LayerNorm.weight',
                              'embed/layer_normalization/gamma')
            v = numpy.reshape(v, (1, 1, v.shape[-1]))
        elif 'bias' in key:
            key = key.replace('bert.embeddings.LayerNorm.bias',
                              'embed/layer_normalization/beta')
            v = numpy.reshape(v, (1, 1, v.shape[-1]))
    elif 'word_embeddings' in key:
        key = 'word_embeddings/embed/W'
    elif 'token_type_embeddings' in key:
        key = 'token_type_embeddings/embed/W'
    elif 'position_embeddings' in key:
        key = 'position_embeddings/embed/W'
    elif 'pooler.dense_act.bias' in key:
        key = 'pooler/affine/b'
    elif 'pooler.dense_act.weight' in key:
        key = 'pooler/affine/W'
        v = numpy.transpose(v)

    elif 'cls.predictions.transform.LayerNorm.gamma' in key:
        key = 'cls/predictions/transform/layer_normalization/gamma'
    elif 'cls.predictions.transform.LayerNorm.beta' in key:
        key = 'cls/predictions/transform/layer_normalization/beta'
    elif 'classifier.weight' in key:
        key = 'affine_seq_class/affine/W'
        v = numpy.transpose(v)
    elif 'classifier.bias' in key:
        key = 'affine_seq_class/affine/b'

    # perfect matches
    elif key.lower() == "cls.predictions.transform.dense_act.weight".lower():
        key = "mlm/linear/affine/W"
        v = numpy.transpose(v)
    elif key.lower() == "cls.predictions.transform.dense_act.bias".lower():
        key = "mlm/linear/affine/b"
    elif key.lower() == "cls.predictions.bias".lower():
        key = "mlm/affine/b"
    elif key.lower() == "cls.predictions.decoder.weight".lower():
        key = "mlm/affine/W"
        v = numpy.transpose(v)
    elif key.lower() == "cls.predictions.transform.LayerNorm.bias".lower():
        key = "layer_norm/layer_normalization/beta"
        v = numpy.reshape(v, [1, 1, v.shape[-1]])
    elif key.lower() == "cls.predictions.transform.LayerNorm.weight".lower():
        key = "layer_norm/layer_normalization/gamma"
        v = numpy.reshape(v, [1, 1, v.shape[-1]])
    elif key.lower() == "cls.seq_relationship.weight".lower():
        key = "nsp/affine/W"
        v = numpy.transpose(v)
    elif key.lower() == "cls.seq_relationship.bias".lower():
        key = "nsp/affine/b"
        v = numpy.transpose(v)
    return key, v


def pytorch_to_nnabla(input_file, h5_file):
    read = torch.load(input_file)
    for k, v in read.items():
        print('before:  ', k, v.shape)
        v = v.cpu().numpy()
        key, v = rename_params(k, v)
        print('after:  ', key, v.shape)
        params = PF.get_parameter_or_create(key, shape=v.shape)
        params.d = v
    nn.parameter.save_parameters(h5_file)


def main():
    pytorch_to_nnabla(args.pretrained_model, args.save_path)


if __name__ == "__main__":
    main()
