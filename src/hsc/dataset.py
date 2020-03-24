#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

from collections import namedtuple

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from joblib import Parallel, delayed

from data.cosmology import Cosmology

__date__ = '17/6月/2019'


class Data(object):
    def __init__(self, x, y=None, weight=None, index=None):
        if index is not None:
            self.x = x[index]
            self.y = y if y is None else y[index]
            self.weight = weight if weight is None else weight[index]
        else:
            self.x = x
            self.y = y
            self.weight = weight


class InputSetting(object):
    def __init__(self, batch_size, mixup='none',
                 mixup_alpha=2, mixup_beta=2, max_batch_size=10000,
                 balance=False):
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size

        self.balance = balance
        self.mixup = mixup
        self.mixup_alpha = mixup_alpha
        self.mixup_beta = mixup_beta

    @property
    def test_batch_size(self):
        return min(self.batch_size * 10, 10000)

    @property
    def parameters(self):
        tmp = {'mixup': self.mixup, 'balance': self.balance,
               'batch_size': self.batch_size,
               'max_batch_size': self.max_batch_size}
        if self.mixup != 'none':
            tmp.update({'mixup_alpha': self.mixup_alpha,
                        'mixup_beta': self.mixup_beta})

        return tmp


class InputData(object):
    def __init__(self, training_data, validation_data, test_data, mean, std,
                 input1, input2, remove_y, is_hsc,
                 n_classes, input_setting):
        self.training_data = training_data
        self.validation_data = validation_data
        self.test_data = test_data

        self.mean = mean
        self.std = std

        self.input1 = input1
        self.input2 = input2
        self.remove_y = remove_y

        self.is_hsc = is_hsc
        self.n_classes = n_classes

        self.input_setting = input_setting

    @property
    def parameters(self):
        tmp = {'input1': self.input1, 'input2': self.input2,
               'remove_y': self.remove_y, 'is_hsc': self.is_hsc}
        tmp.update(self.input_setting.parameters)

        return tmp


class DatasetOperators(object):
    def __init__(self, training, validation, test):
        self.training_iterator, self.training_element = training
        self.validation_iterator, self.validation_element = validation
        self.test_iterator, self.test_element = test


class MixupDatasetOperators(DatasetOperators):
    def __init__(self, training1, training2, validation, test, n_classes,
                 mixup='mixup'):
        dev_iterator1, dev_element1 = training1
        dev_iterator2, dev_element2 = training2

        beta = tf.distributions.Beta(2.0, 2.0)
        r = beta.sample(sample_shape=(tf.shape(dev_element1.x)[0],))
        r = tf.reshape(r, [-1, 1])

        if mixup == 'mixup':
            x = r * dev_element1.x + (1 - r) * dev_element2.x
        elif mixup == 'manifold':
            # two data to mix up in a intermediate layer
            x = tf.concat([dev_element1.x, dev_element2.x], axis=0)
        else:
            raise ValueError(mixup)
        y = (r * tf.one_hot(dev_element1.y, n_classes) +
             (1 - r) * tf.one_hot(dev_element2.y, n_classes))

        # 名前は片方だけを使う
        Dataset = namedtuple('Dataset', ['x', 'y', 'name', 'ratio'])
        dev_element = Dataset(x=x, y=y, name=dev_element1.name, ratio=r)

        class Iterator(object):
            def __init__(self, iterator1, iterator2):
                self.iterator1 = iterator1
                self.iterator2 = iterator2

            @property
            def initializer(self):
                return tf.group(self.iterator1.initializer,
                                self.iterator2.initializer)

        super().__init__(
            training=(Iterator(dev_iterator1, dev_iterator2), dev_element),
            validation=validation, test=test
        )

        # for checking the status of learning
        self.training_iterator2, self.training_element2 = training1


def compute_distmod(x, absolute_magnitude, threads):
    if absolute_magnitude:
        # calculating the distance modulus of each sample
        lcdm = Cosmology()
        # distmod = np.array([lcdm.DistMod(z) for z in x['redshift']])
        distmod = np.asarray(Parallel(n_jobs=threads)(
            delayed(lcdm.DistMod)(z) for z in x['redshift']
        ))
    else:
        # not used, something needed for processing
        distmod = x['redshift']

    return distmod


def make_plasticc_training_dataset(x, y, weights, input_data, threads):
    unique_class = np.unique(y)
    n_classes = len(unique_class)
    # The class IDs are given to be continuous from 0.
    # Otherwise, the calculation of cross entropy is troublesome.
    assert unique_class[0] == 0
    assert n_classes == unique_class[-1] + 1

    absolute_magnitude = 'absolute' in input_data.input1
    distmod = compute_distmod(x=x, absolute_magnitude=absolute_magnitude,
                              threads=threads)

    ohe = OneHotEncoder(n_values=n_classes, categories='auto', sparse=False)
    ohe.fit(np.arange(n_classes).reshape([-1, 1]))

    def make_batch(indices):
        tmp = x[indices]
        flux = tmp['flux']
        flux_err = tmp['flux_err']
        dm = np.reshape(distmod[indices], [-1, 1])
        label = y[indices]

        def make_input(input_mode):
            noise = np.random.randn(*flux.shape) * flux_err

            return transform_input_np(
                flux=flux + noise, input_mode=input_mode, distmod=dm,
                is_hsc=input_data.is_hsc
            )

        batch_x = concat_input_np(
            input1=input_data.input1, input2=input_data.input2,
            f_make=make_input, redshift=None, use_redshift=False
        )
        batch_y = ohe.transform(np.reshape(label, [-1, 1]))

        return batch_x, batch_y, tmp['name']

    Dataset = namedtuple('Dataset', ['x', 'y', 'name'])

    def generator():
        # the number of data
        data_size = len(x)
        batch_size = input_data.input_setting.batch_size
        n = (data_size + batch_size - 1) // batch_size
        for i in range(n):
            index1 = np.random.choice(data_size, batch_size, p=weights)
            batch_x1, batch_y1, name = make_batch(indices=index1)

            index2 = np.random.choice(data_size, batch_size, p=weights)
            batch_x2, batch_y2, _ = make_batch(indices=index2)

            r = np.random.beta(2, 2, size=[len(batch_x1), 1])

            batch_x = r * batch_x1 + (1 - r) * batch_x2
            batch_y = r * batch_y1 + (1 - r) * batch_y2

            batch_x = (batch_x - input_data.mean) / input_data.std

            batch_x = batch_x.astype(np.float32)
            batch_y = batch_y.astype(np.float32)

            yield Dataset(x=batch_x, y=batch_y, name=name)

    input_dim = x['flux'].shape[1]
    if input_data.input2 != 'none':
        input_dim *= 2

    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_types=Dataset(x=tf.float32, y=tf.float32, name=tf.string),
        output_shapes=Dataset(
            x=tf.TensorShape([None, input_dim]),
            y=tf.TensorShape([None, n_classes]),
            name=tf.TensorShape([None])
        )
    )
    dataset = dataset.prefetch(4)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    return iterator, next_element


def make_training_dataset_balance(data, input_data):
    # It is simple to keep the number of samples per class equal in the batch
    # two batches for mixup

    unique_class, counts = np.unique(data.y, return_counts=True)
    n_classes = len(unique_class)
    # the labels are given to be continuous from 0.
    # Otherwise, the calculation of cross entropy is troublesome.
    assert unique_class[0] == 0
    assert n_classes == unique_class[-1] + 1

    # setting sampling ratio for each sample
    w = 1 / counts / n_classes
    weights = np.array([w[i] for i in data.y])

    absolute_magnitude = 'absolute' in input_data.input1
    distmod = compute_distmod(x=data.x, absolute_magnitude=absolute_magnitude,
                              threads=1)

    ohe = OneHotEncoder(n_values=n_classes, categories='auto', sparse=False)
    ohe.fit(np.arange(n_classes).reshape([-1, 1]))

    def make_batch(indices):
        tmp = data.x[indices]
        flux = tmp['flux']
        flux_err = tmp['flux_err']
        dm = np.reshape(distmod[indices], [-1, 1])
        label = data.y[indices]

        def make_input(input_mode):
            noise = np.random.randn(*flux.shape) * flux_err

            return transform_input_np(
                flux=flux + noise, input_mode=input_mode, distmod=dm,
                is_hsc=input_data.is_hsc
            )

        batch_x = concat_input_np(
            input1=input_data.input1, input2=input_data.input2,
            f_make=make_input, redshift=None, use_redshift=False
        )
        batch_y = ohe.transform(np.reshape(label, [-1, 1]))

        return batch_x, batch_y, tmp['name']

    data_size = len(data.x)
    batch_size = input_data.input_setting.batch_size
    n = (data_size + batch_size - 1) // batch_size

    Dataset = namedtuple('Dataset', ['x', 'y', 'name'])

    def generator():
        for _ in range(n):
            index1 = np.random.choice(data_size, batch_size, p=weights)
            batch_x1, batch_y1, name = make_batch(indices=index1)

            index2 = np.random.choice(data_size, batch_size, p=weights)
            batch_x2, batch_y2, _ = make_batch(indices=index2)

            r = np.random.beta(2, 2, size=[len(batch_x1), 1])

            batch_x = r * batch_x1 + (1 - r) * batch_x2
            batch_y = r * batch_y1 + (1 - r) * batch_y2

            batch_x = (batch_x - input_data.mean) / input_data.std

            batch_x = batch_x.astype(np.float32)
            batch_y = batch_y.astype(np.float32)

            yield Dataset(x=batch_x, y=batch_y, name=name)

    dims = len(data.x[0]['flux'])
    if input_data.input2 != 'none':
        dims *= 2

    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_types=Dataset(x=tf.float32, y=tf.float32, name=tf.string),
        output_shapes=Dataset(
            x=tf.TensorShape((None, dims)),
            y=tf.TensorShape((None, n_classes)),
            name=tf.TensorShape((None,))
        )
    )
    dataset = dataset.prefetch(1)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    return iterator, next_element


def make_input_np(flux, flux_err, input_mode, distmod, is_hsc):
    noise = flux_err * np.random.randn(*flux.shape)
    flux = flux + noise
    return transform_input_np(flux=flux, input_mode=input_mode,
                              distmod=distmod, is_hsc=is_hsc)


def make_dataset(x, y, is_training, mean, std, batch_size, input1, input2,
                 use_redshift, is_hsc, threads, seed_offset=0, weights=None):
    absolute_magnitude = 'absolute' in input1
    distmod = compute_distmod(x=x, absolute_magnitude=absolute_magnitude,
                              threads=threads).astype(np.float32)

    if weights is not None:
        Dataset = namedtuple('Dataset', ['x', 'y', 'name', 'weight'])
        dataset = tf.data.Dataset.from_tensor_slices((
            x['flux'], x['flux_err'], distmod, y, x['name'], weights
        ))
    else:
        Dataset = namedtuple('Dataset', ['x', 'y', 'name'])
        dataset = tf.data.Dataset.from_tensor_slices((
            x['flux'], x['flux_err'], distmod, y, x['name']
        ))
    dataset = dataset.repeat(1)
    if is_training:
        global_step = tf.train.get_or_create_global_step()
        dataset = dataset.shuffle(100000, seed=global_step + seed_offset)

        def map_func(flux, flux_err, d, *args):
            inputs = map_func_training(
                flux=flux, flux_err=flux_err, redshift=None,
                mean=mean, std=std, input1=input1, input2=input2,
                use_redshift=use_redshift, distmod=d, is_hsc=is_hsc
            )
            # noinspection PyProtectedMember
            return Dataset._make((inputs,) + args)
    else:
        def map_func(flux, flux_err, d, *args):
            _ = flux_err
            inputs = map_func_test(
                flux=flux, redshift=None, mean=mean, std=std,
                input1=input1, input2=input2,
                use_redshift=use_redshift, distmod=d, is_hsc=is_hsc
            )
            # noinspection PyProtectedMember
            return Dataset._make((inputs,) + args)
    dataset = dataset.map(map_func=map_func, num_parallel_calls=threads)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=2)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    return iterator, next_element


def map_func_training(flux, flux_err, redshift, mean, std, input1, input2,
                      use_redshift, distmod, is_hsc):
    def make_input(input_mode):
        noise = tf.random_normal(tf.shape(flux)) * flux_err

        return transform_input_tf(
            flux=flux + noise, input_mode=input_mode, distmod=distmod,
            is_hsc=is_hsc
        )

    x = concat_input_tf(input1=input1, input2=input2, f_make=make_input,
                        redshift=redshift, use_redshift=use_redshift)
    return (x - mean) / std


def map_func_test(flux, redshift, mean, std, input1, input2, use_redshift,
                  distmod, is_hsc):
    x = concat_input_tf(
        input1=input1, input2=input2,
        f_make=lambda v: transform_input_tf(
            flux=flux, input_mode=v, distmod=distmod, is_hsc=is_hsc
        ),
        redshift=redshift, use_redshift=use_redshift
    )
    return (x - mean) / std


def transform_input_tf(flux, input_mode, distmod, is_hsc):
    base = 27.0 if is_hsc else 27.5

    tmp = flux

    if 'scaled' in input_mode:
        m = tf.maximum(tf.reduce_max(tmp), 1.0)
        tmp = tmp / m

    if 'magnitude' in input_mode:
        v = tf.asinh(tmp * 0.5)
        if 'absolute' in input_mode:
            # c = tf.constant(2.5 * np.log10(np.e), dtype=tf.float32)
            # tmp = 27.5 - v * c - distmod
            tmp = base - 2.5 * v / tf.math.log(10.0) - distmod
        else:
            tmp = base - 2.5 * v / tf.math.log(10.0)
    return tmp


def transform_input_np(flux, input_mode, distmod, is_hsc):
    base = 27.0 if is_hsc else 27.5

    if 'scaled' in input_mode:
        m = np.max(flux, axis=1, keepdims=True)
        m[m < 1] = 1
        v = flux / m
    else:
        v = flux
    if 'magnitude' in input_mode:
        tmp = np.arcsinh(v * 0.5)
        if 'absolute' in input_mode:
            distmod = np.reshape(distmod, [-1, 1])
            # 1.0 / np.log(10)はnp.log10(np.e)に等しい
            v = base - 2.5 * tmp * np.log10(np.e) - distmod
        else:
            v = base - 2.5 * tmp * np.log10(np.e)
    return v


def concat_input_tf(input1, input2, f_make, redshift=None, use_redshift=False):
    x1 = f_make(input1)
    if input2 == 'none':
        if use_redshift:
            x = tf.concat((x1, tf.reshape(redshift, [1])), axis=-1)
        else:
            x = x1
    else:
        x2 = f_make(input2)
        if use_redshift:
            x = tf.concat((x1, x2, tf.reshape(redshift, [1])), axis=-1)
        else:
            x = tf.concat((x1, x2), axis=-1)
    return x


def concat_input_np(input1, input2, f_make, redshift=None, use_redshift=False):
    x1 = f_make(input1)
    if input2 == 'none':
        if use_redshift:
            x = np.hstack((x1, redshift))
        else:
            x = x1
    else:
        x2 = f_make(input2)
        if use_redshift:
            x = np.hstack((x1, x2, redshift))
        else:
            x = np.hstack((x1, x2))

    return x


def compute_moments(train_data, input1, input2, norm, use_redshift,
                    is_hsc, threads):
    if norm:
        flux = train_data['flux']
        absolute_magnitude = 'absolute' in input1
        distmod = compute_distmod(
            x=train_data, absolute_magnitude=absolute_magnitude,
            threads=threads
        )

        def _compute_moments(input_mode):
            v = transform_input_np(
                flux=flux, input_mode=input_mode, distmod=distmod,
                is_hsc=is_hsc
            )
            input_mean = np.mean(v, axis=0)
            input_std = np.std(v, axis=0)

            # putting together to handle in one process
            moments = np.vstack((input_mean, input_std))
            return moments

        if use_redshift:
            redshift = train_data['redshift']
            redshift_moments = np.array([[np.mean(redshift)],
                                         [np.std(redshift)]])
        else:
            # not used, None is ok.
            redshift_moments = None

        m = concat_input_np(
            input1=input1, input2=input2, f_make=_compute_moments,
            redshift=redshift_moments, use_redshift=use_redshift
        )

        mean = m[0]
        std = m[1]
    else:
        n = train_data['flux'].shape[1]
        if input2 != 'none':
            n *= 2
        if use_redshift:
            n += 1
        mean = np.zeros(n, dtype=np.float32)
        std = np.ones_like(mean)

    return mean, std


def build_plasticc_dataset(input_data, sampling_ratio, threads):
    test_iterator, test_element = make_dataset2(
        data=input_data.test_data, input_data=input_data, is_training=False,
        threads=threads
    )
    val_iterator, val_element = make_dataset2(
        data=input_data.validation_data, input_data=input_data,
        is_training=False, threads=threads,
    )

    if input_data.input_setting.mixup != 'none':
        training_data = input_data.training_data
        if input_data.input_setting.balance:
            dev_iterator, dev_element = make_training_dataset_balance(
                data=input_data.training_data, input_data=input_data
            )
        else:
            dev_iterator, dev_element = make_plasticc_training_dataset(
                x=training_data.x, y=training_data.y,
                weights=training_data.weight,
                input_data=input_data, threads=threads
            )
            training_data = Data(x=np.hstack(training_data.x),
                                 y=np.hstack(training_data.y))

        ops = DatasetOperators(training=(dev_iterator, dev_element),
                               validation=(val_iterator, val_element),
                               test=(test_iterator, test_element))
        # for checking learning status
        dev_iterator1, dev_element1 = make_dataset2(
            data=training_data, input_data=input_data,
            is_training=False, threads=threads
        )
        ops.training_iterator2 = dev_iterator1
        ops.training_element2 = dev_element1
    else:
        # no mixup
        dev_iterator, dev_element = make_dataset2(
            data=input_data.training_data, input_data=input_data,
            is_training=True, threads=threads
        )

        ops = DatasetOperators(training=(dev_iterator, dev_element),
                               validation=(val_iterator, val_element),
                               test=(test_iterator, test_element))
    return ops


def build_hsc_dataset(input_data, threads):
    test_iterator, test_element = make_dataset2(
        data=input_data.test_data, input_data=input_data, is_training=False,
        threads=threads
    )
    val_iterator, val_element = make_dataset2(
        data=input_data.validation_data, input_data=input_data,
        is_training=False, threads=threads
    )

    if input_data.input_setting.mixup == 'none':
        dev_iterator, dev_element = make_dataset2(
            data=input_data.training_data, input_data=input_data,
            is_training=True, threads=threads
        )

        ops = DatasetOperators(training=(dev_iterator, dev_element),
                               validation=(val_iterator, val_element),
                               test=(test_iterator, test_element))
        # for prediction
        dev_iterator1, dev_element1 = make_dataset2(
            data=input_data.training_data, input_data=input_data,
            is_training=False, threads=threads
        )
        ops.training_iterator2 = dev_iterator1
        ops.training_element2 = dev_element1
    else:
        if input_data.input_setting.balance:
            dev_iterator, dev_element = make_training_dataset_balance(
                data=input_data.training_data, input_data=input_data
            )

            ops = DatasetOperators(training=(dev_iterator, dev_element),
                                   validation=(val_iterator, val_element),
                                   test=(test_iterator, test_element))
            # for checking learning status
            dev_iterator1, dev_element1 = make_dataset2(
                data=input_data.training_data, input_data=input_data,
                is_training=False, threads=threads
            )
            ops.training_iterator2 = dev_iterator1
            ops.training_element2 = dev_element1
        else:
            dev_iterator1, dev_element1 = make_dataset2(
                data=input_data.training_data, input_data=input_data,
                is_training=True, threads=threads
            )
            dev_iterator2, dev_element2 = make_dataset2(
                data=input_data.training_data, input_data=input_data,
                is_training=True, threads=threads, seed_offset=1
            )

            ops = MixupDatasetOperators(
                training1=(dev_iterator1, dev_element1),
                training2=(dev_iterator2, dev_element2),
                validation=(val_iterator, val_element),
                test=(test_iterator, test_element),
                n_classes=input_data.n_classes,
                mixup=input_data.input_setting.mixup
            )

    return ops


def make_dataset2(data, input_data, is_training, threads, seed_offset=0):
    if data.y is None:
        y = np.empty(len(data.x), dtype=np.int32)
    else:
        y = data.y
    if is_training:
        batch_size = input_data.input_setting.batch_size
    else:
        batch_size = input_data.input_setting.test_batch_size

    return make_dataset(
        x=data.x, y=y, is_training=is_training,
        mean=input_data.mean, std=input_data.std, batch_size=batch_size,
        input1=input_data.input1, input2=input_data.input2,
        use_redshift=False, is_hsc=input_data.is_hsc, threads=threads,
        seed_offset=seed_offset, weights=data.weight
    )
