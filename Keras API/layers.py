# common layers for deep learning
# Almost all layers listed are already implemented efficiently in Keras.
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf


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
        outputs = tf.stack(outputs)                  # stack array
        states = tf.stack(states)
        outputs = tf.transpose(outputs, perm=[1, 0, 2])
        states = tf.transpose(states, perm=[1, 0, 2])
        return outputs, states


class AttentionLayer(tf.keras.layers.Layer):
    pass

    def __init__(self):
        pass

    def call(self, inputs, **kwargs):
        raise NotImplementedError("implement this first")


class SelfAttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        pass

    def call(self, inputs, **kwargs):
        raise NotImplementedError("implement this first")


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
