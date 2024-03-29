# common layers for deep learning
# Almost all layers listed are already implemented efficiently in Keras.
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf


tf.keras.Model.save_weights()
tf.keras.layers.Attention

class Dense(tf.keras.layers.Layer):
    def __init__(self, out_features, name=None):
        super().__init__(name=name)
        self.out_features = out_features

    def build(self, input_shape):
        self.w = tf.Variable(
            tf.random.normal([input_shape[-1], self.out_features]), name='w')
        self.b = tf.Variable(tf.zeros([self.out_features]), name='b')

    def call(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)


# TODO
class Convo(tf.keras.layers.Layer):

    def __init__(self):
        pass

    def call(self, inputs, **kwargs):
        raise NotImplementedError("implement this first")


class RNNCell(tf.keras.layers.Layer):
    # to be used in RNNLayer
    def __init__(self, hidden_dimension, num_classes=2, name=None):
        super(RNNCell, self).__init__(name=name)
        if hidden_dimension is None:
            raise ValueError("hidden_dimension cannot be None")
        self.hidden_dimension = hidden_dimension
        self.num_classes = num_classes

    def build(self, input_shape):
        self.w_one = tf.Variable(
            tf.random.normal([input_shape[-1], self.hidden_dimension]), name='w_one')
        self.w_two = tf.Variable(
            tf.random.normal([self.hidden_dimension, self.hidden_dimension]), name='w_two')

        self.b = tf.Variable(tf.zeros([self.hidden_dimension]), name='b')

        self.v = tf.Variable(
            tf.random.uniform([self.hidden_dimension, self.num_classes]), name='v')
        self.c = tf.Variable(tf.zeros([self.num_classes]), name='c')

    def call(self, inputs, states):
        # input shape --> (batch size, embed dimension)

        at = tf.linalg.matmul(inputs, self.w_one) + tf.linalg.matmul(states, self.w_two) + self.b
        state = tf.nn.tanh(at)
        ot = self.c + tf.linalg.matmul(state, self.v)
        yt = tf.nn.softmax(ot)
        return yt, state


class RNNLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_dimension, num_classes=2, name=None):
        super(RNNLayer, self).__init__(name=name)
        self.hidden_dimension = hidden_dimension
        self.num_classes = num_classes
        self.rnn_cell = RNNCell(
            hidden_dimension=self.hidden_dimension,
            num_classes=self.num_classes
        )

    def build(self, input_shape):
        pass

    def call(self, inputs):
        time_first_input = tf.transpose(inputs, perm=[1, 0, 2])  # steps * batch * embedding

        current_state = tf.zeros(shape=(time_first_input.shape[1], self.hidden_dimension))

        states = []
        outputs = []
        # TODO void doing lists
        for i in range(time_first_input.shape[0]):
            yt, current_state = self.rnn_cell(inputs=time_first_input[i], states=current_state)
            states.append(current_state)
            outputs.append(yt)
        outputs = tf.stack(outputs)  # stack array
        states = tf.stack(states)
        outputs = tf.transpose(outputs, perm=[1, 0, 2])
        states = tf.transpose(states, perm=[1, 0, 2])
        return outputs, states


# Note: keras does offer the implementation of additive and dot product attention


class AttentionLayer(tf.keras.layers.Layer):
    """
    Args:
        use_scale: bool. if true then scale the scale the attention weights. not used currently
        attention_type: The attention type. Options {"additive", "dot_product"}
                        default is "additive"
    """

    def __init__(self, use_scale=False, attention_type="additive"):
        super(AttentionLayer, self).__init__()
        self.use_scale = use_scale
        self.attention_type = attention_type

    @staticmethod
    def additive_attention_score(query, key):
        addition = tf.expand_dims(query, axis=-2) + tf.expand_dims(key, axis=-3)  # (batch size, seq, seq, dim)
        e_values = tf.reduce_sum(tf.math.tanh(addition), axis=-1)                 # (batch size, seq, seq)
        weights = tf.math.softmax(e_values, axis=-1)                              # (batch size, seq, seq)
        return weights

    @staticmethod
    def dot_product_attention_score(query, key):
        e_values = tf.matmul(query, key, transpose_b=True)                         # (batch size, seq, dim)
        weights = tf.math.softmax(e_values, axis=-1)                               # (batch size, seq, seq)
        return weights

    @property
    def attention_function(self):
        func_map = {"additive": self.additive_attention_score,
                    "dot_product": self.dot_product_attention_score}
        return func_map[self.attention_type]

    def compute_mask(self, inputs, mask=None):
        pass

    def call(self, inputs, **kwargs):
        if len(inputs) < 2:
            raise ValueError("Expected 2 or 3 inputs "
                             "as [query, value] or [query, value, key]")
        q = inputs[0]
        v = inputs[1]
        k = inputs[2] if len(inputs) > 2 else v

        weights = self.attention_function(query=q, key=k)
        out = tf.matmul(weights, v)                    # (batch size, seq, dim)
        return out


class AdditiveSelfAttentionLayer(tf.keras.layers.Layer):
    """Additive self attention or Bahdanau Self Attention

    input -- > (batch size, query, dim)
    output --> (batch size, query, dim)

    This is the same as tf.keras.layers.AdditiveAttention with use_scale=False. You can verify below
    Note this is self attention and hence query and key are both
    the same(you pass query and key separately in keras attention layers)


    >> inputs = tf.random.uniform((10,3,4))
    >> at = tf.keras.layers.AdditiveAttention(use_scale=False)
    >> att = AdditiveSelfAttentionLayer()
    >> tf.equal(at([inputs, inputs]), att(inputs))
    <tf.Tensor: shape=(10, 3, 4), dtype=bool, numpy=
    array([[[ True,  True,  True,  True],
        [ True,  True,  True,  True],
        [ True,  True,  True,  True]],

       [[ True,  True,  True,  True],
        [ True,  True,  True,  True],
        [ True,  True,  True,  True]],

       [[ True,  True,  True,  True],
        [ True,  True,  True,  True],
        [ True,  True,  True,  True]],
        .
        .
        .
        .
                                    ]]
    """

    def __int__(self):
        super(AdditiveSelfAttentionLayer, self).__int__()

    def compute_mask(self, inputs, mask=None):
        pass

    def call(self, inputs, **kwargs):
        # inputs - (batch size, seq, dim)
        addition = tf.expand_dims(inputs, axis=-2) + tf.expand_dims(inputs, axis=-3)  # (batch size, seq, seq, dim)
        e_values = tf.reduce_sum(tf.math.tanh(addition), axis=-1)  # (batch size, seq, seq)
        weights = tf.math.softmax(e_values, axis=-1)  # (batch size, seq, seq)
        out = tf.matmul(weights, inputs)  # (batch size, seq, dim)
        return out


class DotProductSelfAttentionLayer(tf.keras.layers.Layer):
    """Dot product self attention

    input -- > (batch size, query, dim)
    output --> (batch size, query, dim)

    This is same as tf.keras.layers.Attention with use_scale=False.
    Note this is self attention and hence query and key are both
    the same(you pass query and key separately in keras attention layers)

    """

    def __int__(self):
        super(DotProductSelfAttentionLayer, self).__int__()

    def call(self, inputs, **kwargs):
        # inputs - (batch size, seq, dim)
        e_values = tf.matmul(inputs, inputs, transpose_b=True)  # (batch size, seq, dim)
        weights = tf.math.softmax(e_values, axis=-1)  # (batch size, seq, seq)
        out = tf.matmul(weights, inputs)  # (batch size, seq, dim)
        return out


class HANLayer(tf.keras.layers.Layer):
    def __init__(self):
        pass

    def call(self, inputs, **kwargs):
        raise NotImplementedError("implement this first")


class EmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self,
                 input_dim,
                 output_dim,
                 trainable=True,
                 weights=None,
                 mask_zero=False
                 ):
        super(EmbeddingLayer, self).__init__()
        if weights is None:
            if trainable is False:
                raise ValueError("If trainable is False then pretrained "
                                 "weights are expected")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mask_zero = mask_zero

        if input_dim is None:
            raise ValueError("Expected input_dim as an integer but got {}".format(input_dim))

        if output_dim is None:
            raise ValueError("Expected output_dim as an integer but got {}".format(output_dim))

        if trainable is True:
            if weights is None:
                self.embedding_weights = tf.Variable(
                    tf.random.uniform([self.input_dim, self.output_dim], dtype=tf.float32), name='embedding')
            else:
                self._verify_weight_dimension(weights)
                self.embedding_weights = tf.constant(weights, dtype=tf.float32)

    def compute_mask(self, inputs, mask=None):
        pass

    def _verify_weight_dimension(self, weights):
        pass

    def call(self, inputs):
        # inputs - convert last axis numbers to embeddings
        return tf.nn.embedding_lookup(params=self.embedding_weights, ids=inputs)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self):
        pass

    def call(self, inputs, **kwargs):
        raise NotImplementedError("implement this first")


class Transformer(tf.keras.layers.Layer):
    def __init__(self):
        pass

    def call(self, inputs, **kwargs):
        raise NotImplementedError("implement this first")


class LayerNormalization(tf.keras.layers.Layer):
    pass


class BatchNormalization(tf.keras.layers.Layer):
    pass


class ResNet(tf.keras.layers.Layer):
    pass


class VGG(tf.keras.layers.Layer):
    pass

