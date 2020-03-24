#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import sonnet as snt

__date__ = '17/6月/2019'


class Model(snt.AbstractModule):
    def __init__(self, hidden_size, output_size, drop_rate, num_highways=2,
                 use_batch_norm=False, use_dropout=False, use_layer_norm=False,
                 activation='linear'):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.drop_rate = drop_rate

        self.num_highways = num_highways
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.use_layer_norm = use_layer_norm
        self.activation = activation

    def _build(self, inputs, is_training, k=-1, r=0):
        if not isinstance(k, int):
            inputs = tf.cond(tf.equal(k, 0),
                             true_fn=lambda: mixup_process(inputs=inputs, r=r),
                             false_fn=lambda: inputs)

        h = snt.Linear(output_size=self.hidden_size)(inputs)
        h = tf.layers.Dropout(rate=self.drop_rate)(h, is_training)

        if not isinstance(k, int):
            h = tf.cond(tf.equal(k, 1),
                        true_fn=lambda: mixup_process(inputs=h, r=r),
                        false_fn=lambda: h)

        for i in range(self.num_highways):
            h = Highway()(h)

            if self.use_batch_norm:
                h = snt.BatchNormV2(data_format='NC')(h, is_training)
            elif self.use_layer_norm:
                h = snt.LayerNorm(axis=1)(h)

            if self.activation != 'linear':
                h = Activation(activation=self.activation)(h)

            if self.use_dropout:
                h = tf.layers.Dropout(rate=self.drop_rate)(h, is_training)

            if not isinstance(k, int):
                h = tf.cond(tf.equal(k, i + 2),
                            true_fn=lambda: mixup_process(inputs=h, r=r),
                            false_fn=lambda: h)

        outputs = snt.Linear(output_size=self.output_size)(h)
        return outputs


def mixup_process(inputs, r):
    # dividing in half
    inputs1, inputs2 = tf.split(inputs, num_or_size_splits=2, axis=0)
    outputs = r * inputs1 + (1 - r) * inputs2
    return outputs


class Activation(snt.AbstractModule):
    def __init__(self, activation):
        super().__init__()
        self.f_activation = {
            'relu': tf.nn.relu, 'sigmoid': tf.nn.sigmoid,
            'tanh': tf.nn.tanh, 'elu': tf.nn.elu,
            'linear': lambda x: x
        }[activation]

    def _build(self, inputs):
        return self.f_activation(inputs)


class Highway(snt.AbstractModule):
    def __init__(self):
        super().__init__()

    def _build(self, inputs):
        size = inputs.get_shape()[1].value

        transformed = snt.Sequential([
            snt.Linear(output_size=size),
            tf.nn.sigmoid
        ])(inputs)

        gate = snt.Sequential([
            snt.Linear(output_size=size),
            tf.nn.sigmoid
        ])(inputs)

        outputs = gate * transformed + (1 - gate) * inputs
        return outputs
