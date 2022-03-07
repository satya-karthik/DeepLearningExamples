# Copyright 2021 Sony Group Corporation
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


import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla
import nnabla.initializer as Initializer
from nnabla.initializer import ConstantInitializer


def bert_embed(input_ids, token_type_ids=None,
               position_ids=None, vocab_size=30522, embed_dim=768,
               num_pos_ids=512, dropout_prob=0.1, test=True):
    """Construct the embeddings from word, position and token type."""

    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    if position_ids is None:
        position_ids = F.arange(0, seq_len)
        position_ids = F.broadcast(F.reshape(
            position_ids, (1,)+position_ids.shape),
            (batch_size,) + position_ids.shape)
    if token_type_ids is None:
        token_type_ids = F.constant(val=0, shape=(batch_size, seq_len))

    embeddings = PF.embed(input_ids, vocab_size,
                          embed_dim, name='word_embeddings')
    position_embeddings = PF.embed(
        position_ids, num_pos_ids, embed_dim, name='position_embeddings')
    token_type_embeddings = PF.embed(
        token_type_ids, 2, embed_dim, name='token_type_embeddings')

    embeddings += position_embeddings
    embeddings += token_type_embeddings
    embeddings = PF.layer_normalization(
        embeddings, batch_axis=(0, 1), eps=1e-12, name='embed')

    if dropout_prob > 0.0 and not test:
        embeddings = F.dropout(embeddings, dropout_prob)

    return embeddings


def bert_layer(hs, num_layers=12, embed_dim=768, num_heads=12,
               dim_feedforward=3072, activation=None, attention_mask=None,
               head_mask=None, encoder_hidden_states=None,
               encoder_attention_mask=None, dropout_prob=0.1, test=True):
    """ Generate Transformer Layers"""
    # Transpose the input to the shape (L,B,E) accepted by parameter
    # function transformer_encode
    hs = F.transpose(hs, (1, 0, 2))
    for i in range(num_layers):
        if test:
            hs = PF.transformer_encode(hs, embed_dim, num_heads,
                                       dim_feedforward=dim_feedforward,
                                       dropout=0.0, activation=activation,
                                       name='encoder{:02d}'.format(i))
        else:
            hs = PF.transformer_encode(hs, embed_dim, num_heads,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout_prob,
                                       activation=activation,
                                       name='encoder{:02d}'.format(i))
    # Transpose back to (B,L,E)
    self_outputs = F.transpose(hs, (1, 0, 2))

    return self_outputs


def bert_encode(hs, attention_mask=None, head_mask=None,
                num_attention_layers=12,
                num_attention_embed_dim=768, num_attention_heads=12,
                num_attention_dim_feedforward=3072, attention_activation=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                dropout_prob=0.1, test=True):
    layer_outputs = bert_layer(hs, num_layers=num_attention_layers,
                               embed_dim=num_attention_embed_dim,
                               num_heads=num_attention_heads,
                               dim_feedforward=num_attention_dim_feedforward,
                               activation=attention_activation,
                               attention_mask=attention_mask,
                               head_mask=head_mask,
                               encoder_hidden_states=encoder_hidden_states,
                               encoder_attention_mask=encoder_attention_mask,
                               dropout_prob=dropout_prob, test=test)

    return layer_outputs


def bert_pool(hs, out_dim=768):
    '''
    BERT Pooler, Pool the model by taking hidden state corresponding
    to the first token
    hs: Hidden State (B, L, E)
    '''

    first_token_tensor = hs[:, 0]
    pooled_output = F.tanh(
        PF.affine(first_token_tensor, out_dim, name="pooler"))

    return pooled_output


def linear_activation(x: nnabla.Variable,
                      layer_name: str = 'linear_activation'):
    """Linear activation layer. A simple affine layer with
    He initialization for weights,Uniform initialization for bias.
    input and output shapes are consider to be the same.

    Args:
        x (nnabla.Variable): input vairable of shape:
        [batch_size, sequence_len, embed_size]

        layer_name (str, optional): name for the layer.
        Defaults to 'linear_activation'.

    Raises:
        ValueError: if input and output shapes don't match,
        will raise ValueError.

    Returns:
        y (nnabla.Variable): output of the nnabla.affine layer.
    """
    # x input shape is of [batch_size, sequence_len, embed_size]
    # example: [-1, 512, 768]

    # creating He initializer for our weights
    w_k = Initializer.calc_normal_std_he_forward(x.shape[-1], x.shape[-1])
    w = Initializer.NormalInitializer(w_k)

    # NOTE: this is as per the nvidia-code
    #  we use uniform Initializer for our bias
    bound = 1/(x.shape[-1])**0.5
    b = Initializer.UniformInitializer(lim=(-bound, bound))

    with nnabla.parameter_scope(f"{layer_name}"):
        y = F.gelu(PF.affine(x, x.shape[-1], w_init=w, b_init=b,
                             base_axis=len(x.shape)-1))

    # if not will raise error.
    if x.shape != y.shape:
        raise ValueError(
            f"input shape {x.shape} doesn't match with output shape {y.shape}")

    return y


class BertModel():
    def __init__(self):
        pass

    def __call__(self, args, input_ids, attention_mask=None,
                 token_type_ids=None, position_ids=None,
                 head_mask=None, vocab_size=30522,
                 num_embed_dim=768, num_pos_ids=512,
                 num_attention_layers=12, num_attention_embed_dim=768,
                 num_attention_heads=12, num_attention_dim_feedforward=3072,
                 attention_activation=None, pool_outmap=768,
                 embed_dropout_prob=0.1, attention_dropout_prob=0.1,
                 encoder_hidden_states=None, encoder_attention_mask=None,
                 test=True):

        input_shape = input_ids.shape

        if attention_mask is None:
            attention_mask = F.constant(val=1, shape=(128, 128))
        if token_type_ids is None:
            token_type_ids = F.constant(val=0, shape=input_shape)

        if len(attention_mask.shape) == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif len(attention_mask.shape) == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError("Wrong shape for input_ids " +
                             f"(shape {input_shape})" +
                             " or attention_mask " +
                             f"(shape {attention_mask.shape})")

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoder_extended_attention_mask = None
        head_mask = None

        embedding_output = bert_embed(input_ids, position_ids=position_ids,
                                      token_type_ids=token_type_ids,
                                      vocab_size=vocab_size,
                                      embed_dim=num_embed_dim,
                                      num_pos_ids=num_pos_ids,
                                      dropout_prob=embed_dropout_prob,
                                      test=test)

        encoder_output = bert_encode(
            embedding_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            num_attention_layers=num_attention_layers,
            num_attention_embed_dim=num_attention_embed_dim,
            num_attention_heads=num_attention_heads,
            num_attention_dim_feedforward=num_attention_dim_feedforward,
            attention_activation=attention_activation,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            dropout_prob=args.attention_dropout, test=test)

        pooled_output = bert_pool(encoder_output, out_dim=pool_outmap)
        return encoder_output, pooled_output


def bert_for_pre_training(encoder_output: nnabla.Variable,
                          pooled_output: nnabla.Variable):

    # encoder_output [batch-size, seq_len, embeding_dim] ~: [-1, 128, 768]
    # pooled_output [batch_size, embeding_dim] ~: [-1, 768]

    # encoder output ==> Linear_activation ==> layer_norm
    # ==> affine layer

    encoder_output = linear_activation(encoder_output, "mlm/linear")
    encoder_output = PF.layer_normalization(
        encoder_output, batch_axis=(0, 1), eps=1e-12,
        param_init={'gamma': ConstantInitializer(1),
                    'beta': ConstantInitializer(0)},
        name='layer_norm')

    # this affine layer uses word embedding weights.
    # get the embedding layer weights
    # transpose it so that we get [ embeding_dim, vocab_size] ~: [ 768, 30528]
    weight_dict = nnabla.get_parameters()
    embed_weight = weight_dict['word_embeddings/embed/W']
    embed_weight = F.transpose(embed_weight, (1, 0)).d
    vocab_size = embed_weight.shape[-1]

    # this is the output for MLM task
    mlm_pred = PF.affine(encoder_output, vocab_size,
                         base_axis=len(encoder_output.shape)-1,
                         w_init=embed_weight, b_init=ConstantInitializer(0),
                         name='mlm')

    # pooled_output ==> affine layer
    # this is the output for NSP task
    nsp_pred = PF.affine(pooled_output, 2, name="nsp")

    return mlm_pred, nsp_pred
