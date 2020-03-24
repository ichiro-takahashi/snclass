#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
regression of the redshift value
"""

import os
import platform
from collections import namedtuple
from itertools import product
from pathlib import Path
from typing import Optional, Callable, Any

import click
import mlflow
import numpy as np
import optuna
import pandas as pd
import sklearn.utils
import sonnet as snt
import tensorflow as tf
from joblib import Parallel, delayed
# noinspection PyProtectedMember
from mlflow import log_param, log_params
from optuna.distributions import BaseDistribution
from optuna.samplers import TPESampler
from optuna.samplers.tpe.sampler import default_gamma, default_weights
from optuna.structs import FrozenTrial
from optuna.study import InTrialStudy
from sklearn.model_selection import KFold, train_test_split

try:
    import matplotlib as mpl
    mpl.use('Agg')
finally:
    import matplotlib.pyplot as plt
    import seaborn as sns

try:
    import sys
    if platform.system() == "Windows":
        sys.path.append(str(Path(__file__).parents[1] / 'references' /
                            'AdaBound-Tensorflow'))
    else:
        sys.path.append(str(Path(__file__).absolute().parents[1] /
                            'references' / 'AdaBound-Tensorflow'))
finally:
    # noinspection PyUnresolvedReferences
    from AdaBound import AdaBoundOptimizer

from data.cosmology import Cosmology
from hsc_sn_type import (load_hsc_data, compute_moments, LoopSetting,
                         OptimizerSetting, Data, get_parameter,
                         load_hsc_test_data)

from hsc.dataset import (transform_input_tf, concat_input_tf,
                         InputData, InputSetting,
                         transform_input_np, concat_input_np,
                         build_hsc_dataset, build_plasticc_dataset)
from hsc.model import Activation, Highway
from hsc.loader import load_sim_sn_data


__date__ = '13/6月/2019'


class MyTPESampler(TPESampler):
    def __init__(
            self,
            consider_prior=True,  # type: bool
            prior_weight=1.0,  # type: float
            consider_magic_clip=True,  # type: bool
            consider_endpoints=False,  # type: bool
            n_startup_trials=10,  # type: int
            n_ei_candidates=24,  # type: int
            gamma=default_gamma,  # type: Callable[[int], int]
            weights=default_weights,  # type: Callable[[int], np.ndarray]
            seed=None  # type: Optional[int]
     ):
        super().__init__(
            consider_prior=consider_prior,
            prior_weight=prior_weight,
            consider_magic_clip=consider_magic_clip,
            consider_endpoints=consider_endpoints,
            n_startup_trials=n_startup_trials,
            n_ei_candidates=n_ei_candidates,
            gamma=gamma,
            weights=weights,
            seed=seed
        )

        candidate = [
            {'drop_rate': 0.001, 'num_highways': n_highways,
             'hidden_size': hidden_size, 'use_batch_norm': batch_norm,
             'activation': activation}
            for hidden_size, n_highways, batch_norm, activation in product(
                [750, 1500], [3, 5, 7], [True, False],
                ['relu', 'tanh', 'sigmoid', 'linear']
            )
        ]
        self.my_candidate = candidate

    def sample_independent(self, study, trial, param_name, param_distribution):
        # type: (InTrialStudy, FrozenTrial, str, BaseDistribution) -> Any
        if trial.trial_id <= len(self.my_candidate):
            return self.my_candidate[trial.trial_id - 1][param_name]
        else:
            return super().sample_independent(study, trial, param_name,
                                              param_distribution)


def make_dataset(x, y, is_training, mean, std, batch_size, input1, input2,
                 threads, is_hsc, seed_offset=0):
    dataset = tf.data.Dataset.from_tensor_slices((
        x['flux'], x['flux_err'], y, x['name']
    ))
    dataset = dataset.repeat(1)

    Dataset = namedtuple('Dataset', ['x', 'y', 'name'])

    if is_training:
        global_step = tf.train.get_or_create_global_step()
        dataset = dataset.shuffle(100000, seed=global_step + seed_offset)

        def map_func(flux, flux_err, t, name):
            inputs = map_func_training(
                flux=flux, flux_err=flux_err, mean=mean, std=std,
                input1=input1, input2=input2, is_hsc=is_hsc
            )
            return Dataset(x=inputs, y=t, name=name)
    else:
        def map_func(flux, flux_err, t, name):
            inputs = map_func_test(
                flux=flux, flux_err=flux_err, mean=mean, std=std,
                input1=input1, input2=input2, is_hsc=is_hsc
            )
            return Dataset(x=inputs, y=t, name=name)

    dataset = dataset.map(map_func=map_func, num_parallel_calls=threads)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=2)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    return iterator, next_element


def make_dataset_mixup(x, y, mean, std, batch_size, input1, input2, is_hsc):
    def make_batch(indices):
        tmp = x[indices]
        flux = tmp['flux']
        flux_err = tmp['flux_err']
        target = y[indices]

        def make_input(input_mode):
            noise = np.random.randn(*flux.shape) * flux_err

            return transform_input_np(
                flux=flux + noise, input_mode=input_mode, distmod=None,
                is_hsc=is_hsc
            )

        batch_x = concat_input_np(
            input1=input1, input2=input2, f_make=make_input, redshift=None,
            use_redshift=None
        )

        return batch_x, target, tmp['name']

    Dataset = namedtuple('Dataset', ['x', 'y', 'name'])

    def generator():
        data_size = len(x)
        n = (len(x) + batch_size - 1) // batch_size
        for i in range(n):
            index1 = np.random.choice(data_size, batch_size)
            batch_x1, batch_y1, name = make_batch(indices=index1)

            index2 = np.random.choice(data_size, batch_size)
            batch_x2, batch_y2, _ = make_batch(indices=index2)

            r = np.random.beta(2, 2, size=[len(batch_x1), 1])

            batch_x = r * batch_x1 + (1 - r) * batch_x2
            r = np.reshape(r, [-1])
            batch_y = r + batch_y1 + (1 - r) * batch_y2

            batch_x = (batch_x - mean) / std

            batch_x = batch_x.astype(np.float32)
            batch_y = batch_y.astype(np.float32)

            yield Dataset(x=batch_x, y=batch_y, name=name)

    input_dim = x['flux'].shape[1]
    if input2 != 'none':
        input_dim *= 2

    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_types=Dataset(x=tf.float32, y=tf.float32, name=tf.string),
        output_shapes=Dataset(
            x=tf.TensorShape([None, input_dim]),
            y=tf.TensorShape([None]),
            name=tf.TensorShape([None])
        )
    )
    dataset = dataset.prefetch(1)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    return iterator, next_element


def map_func_training(flux, flux_err, mean, std, input1, input2, is_hsc):
    def make_input(input_mode):
        noise = tf.random_normal(tf.shape(flux)) * flux_err

        return transform_input_tf(
            flux=flux + noise, input_mode=input_mode,
            distmod=None, is_hsc=is_hsc
        )

    x = concat_input_tf(input1=input1, input2=input2, f_make=make_input,
                        redshift=None, use_redshift=False)
    return (x - mean) / std


def map_func_test(flux, flux_err, mean, std, input1, input2, is_hsc):
    _ = flux_err
    x = concat_input_tf(
        input1=input1, input2=input2,
        f_make=lambda v: transform_input_tf(
            flux=flux, input_mode=v, distmod=None, is_hsc=is_hsc
        ),
        redshift=None, use_redshift=False
    )
    return (x - mean) / std


DataBunch = namedtuple('DataBunch',
                       ['training', 'validation', 'test', 'training2'])
DataOperator = namedtuple('DataOperator', ['iterator', 'element'])


def build_dataset(input_data, threads):
    def _make_dataset(data, is_training):
        if input_data.input_setting.mixup != 'none' and is_training:
            iterator, next_element = make_dataset_mixup(
                x=data.x, y=data.y, mean=input_data.mean, std=input_data.std,
                batch_size=input_data.input_setting.batch_size,
                input1=input_data.input1, input2=input_data.input2,
                is_hsc=input_data.is_hsc
            )
        else:
            iterator, next_element = make_dataset(
                x=data.x, y=data.y, is_training=is_training,
                mean=input_data.mean, std=input_data.std,
                batch_size=input_data.input_setting.batch_size,
                input1=input_data.input1, input2=input_data.input2,
                threads=threads, is_hsc=input_data.is_hsc
            )
        return DataOperator(iterator=iterator, element=next_element)

    bunch = DataBunch(
        training=_make_dataset(data=input_data.training_data,
                               is_training=True),
        validation=_make_dataset(data=input_data.validation_data,
                                 is_training=False),
        test=_make_dataset(data=input_data.test_data, is_training=False),
        training2=_make_dataset(data=input_data.training_data,
                                is_training=False)
    )

    return bunch


class Model(snt.AbstractModule):
    def __init__(self, hidden_size, drop_rate, use_batch_norm, num_highways=2,
                 activation='linear', num_outputs=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.drop_rate = drop_rate

        self.num_highways = num_highways
        self.activation = activation
        self.use_batch_norm = use_batch_norm

        # 1ならredshiftの値のみ、2ならuncertaintyも推定
        self.num_outputs = num_outputs

    def _build(self, inputs, is_training):
        h = snt.Linear(output_size=self.hidden_size)(inputs)
        h = tf.layers.Dropout(rate=self.drop_rate)(h, is_training)

        for i in range(self.num_highways):
            h = Highway()(h)

            if self.use_batch_norm:
                h = snt.BatchNormV2(data_format='NC')(h, is_training)
            if self.activation != 'linear':
                h = Activation(activation=self.activation)(h)

            # h = tf.layers.Dropout(rate=self.drop_rate)(h, is_training)

        outputs = snt.Linear(output_size=self.num_outputs)(h)
        return outputs


def make_metrics(name, outputs, targets):
    with tf.variable_scope('{}_metrics'.format(name)) as vs:
        mean_se = tf.metrics.mean_squared_error(labels=targets,
                                                predictions=outputs)
        mean_log_cosh = tf.metrics.mean(tf.log(tf.cosh(outputs - targets)))

        # R2 scoreのための平均
        mean1 = tf.metrics.mean(targets)
        mean2 = tf.metrics.mean(tf.square(targets))

        local = tf.contrib.framework.get_variables(
            vs, collection=tf.GraphKeys.LOCAL_VARIABLES
        )
        reset_op = tf.variables_initializer(local)
    r2_score = 1 - (mean_se[0] / (mean2[0] - tf.square(mean1[0]) + 1e-9))

    summary_op = tf.summary.merge([
        tf.summary.scalar('{}/squared_loss'.format(name), mean_se[0]),
        tf.summary.scalar('{}/log_cosh_loss'.format(name), mean_log_cosh[0]),
        tf.summary.scalar('{}/r2_score'.format(name), r2_score)
    ])

    update_op = tf.group(mean_se[1], mean_log_cosh[1], mean1[1], mean2[1])

    return update_op, summary_op, reset_op, r2_score


Operators = namedtuple(
    'Operators',
    ['update', 'summary', 'reset', 'initialize', 'accuracy']
)
TrainingOperators = namedtuple(
    'TrainingOperators',
    ['optimize', 'update', 'summary', 'reset', 'initialize', 'accuracy']
)


def make_operators(name, data_pair, model, optimizer=None):
    is_training = optimizer is not None
    output = tf.reshape(model(data_pair.element.x, is_training), [-1])

    update_op, summary_op, reset_op, r2_score = make_metrics(
        name=name, outputs=output, targets=data_pair.element.y
    )

    if is_training:
        # loss = tf.reduce_mean(tf.log(tf.cosh(output - element.y)))
        loss = tf.reduce_mean(tf.squared_difference(
            output, data_pair.element.y
        ))
        optimize_op = optimizer.minimize(loss)

        return TrainingOperators(
            optimize=optimize_op, update=update_op, summary=summary_op,
            reset=reset_op, initialize=data_pair.iterator.initializer,
            accuracy=r2_score
        )
    else:
        return Operators(
            update=update_op, summary=summary_op, reset=reset_op,
            initialize=data_pair.iterator.initializer, accuracy=r2_score
        )


def update_metrics(sess, ops, writer, step):
    if hasattr(ops, 'optimize'):
        update_op = (ops.optimize, ops.update)
    else:
        update_op = ops.update

    sess.run(ops.initialize)
    while True:
        try:
            sess.run(update_op)
        except tf.errors.OutOfRangeError:
            break
    if writer is not None:
        summary = sess.run(ops.summary)
        writer.add_summary(summary=summary, global_step=step)

    r2_score = sess.run(ops.accuracy)

    sess.run(ops.reset)

    return r2_score


PredictionDataset = namedtuple('PredictionDataset',
                               ['prediction', 'target', 'name', 'initialize'])


def make_prediction(model, data_pair, output_mean, output_std):
    prediction = model(data_pair.element.x, False) * output_std + output_mean
    target = data_pair.element.y * output_std + output_mean

    dataset = PredictionDataset(
        prediction=prediction, target=target, name=data_pair.element.name,
        initialize=data_pair.iterator.initializer
    )
    return dataset


def update_model(graph, update_ops, output_ops, eval_frequency,
                 patience, log_dir):
    global_step = tf.train.get_or_create_global_step()
    count_up = tf.assign_add(global_step, 1)

    writer = tf.summary.FileWriter(str(log_dir))
    saver = tf.train.Saver()

    accuracy_list = []
    previous_accuracy = -100  # 十分に小さい数

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True)
    )
    with tf.Session(graph=graph, config=config) as sess:
        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))

        while True:
            step = sess.run(count_up)

            update_metrics(sess=sess, ops=update_ops.training,
                           writer=writer, step=step)
            if step % eval_frequency == 0:
                val_score = update_metrics(
                    sess=sess, ops=update_ops.validation,
                    writer=writer, step=step
                )
                update_metrics(
                    sess=sess, ops=update_ops.test, writer=writer, step=step
                )

                accuracy_list.append(val_score)
                if len(accuracy_list) * eval_frequency >= patience:
                    saver.save(
                        sess=sess, save_path=str(log_dir / 'model'),
                        global_step=global_step, write_meta_graph=False
                    )

                    current_accuracy = np.mean(accuracy_list)
                    print(current_accuracy, previous_accuracy, end=' ')
                    if current_accuracy <= previous_accuracy:
                        # 精度が改善していない
                        print('stop')
                        break
                    else:
                        print('update')
                        accuracy_list.clear()
                        previous_accuracy = current_accuracy

        save_prediction(
            sess=sess, output_ops=output_ops.training,
            file_path=log_dir / 'predictions_training.csv'
        )
        save_prediction(
            sess=sess, output_ops=output_ops.validation,
            file_path=log_dir / 'predictions_validation.csv'
        )
        save_prediction(
            sess=sess, output_ops=output_ops.test,
            file_path=log_dir / 'predictions_test.csv'
        )

    writer.flush()


def save_prediction(sess, output_ops, file_path):
    sess.run(output_ops.initialize)
    predictions = []
    targets = []
    name_list = []
    while True:
        try:
            p, t, data_name = sess.run([
                output_ops.prediction, output_ops.target, output_ops.name
            ])
            predictions.append(p)
            targets.append(t)
            name_list.extend([n.decode() for n in data_name])
        except tf.errors.OutOfRangeError:
            break
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    df = pd.DataFrame(
        predictions, columns=['prediction'], index=name_list
    )
    df['target'] = targets
    df.to_csv(file_path)


@click.group()
def cmd():
    pass


@cmd.command()
@click.option('--sim-sn-path', type=click.Path(exists=True))
@click.option('--hsc-path', type=click.Path(exists=True))
@click.option('--model-dir', type=click.Path())
@click.option('--cv', type=int, default=5)
@click.option('--fold', type=int, default=-1)
@click.option('--batch-size', type=int, default=10000)
@click.option('--lr', type=float, default=1e-3)
@click.option('--optimizer',
              type=click.Choice(['adam', 'momentum', 'adabound', 'amsbound']),
              default='adam')
@click.option('--adabound-gamma', type=float, default=1e-3)
@click.option('--adabound-final-lr', type=float, default=0.1)
@click.option('--seed', type=int, default=0)
@click.option('--epochs', type=int, default=500)
@click.option('--patience', type=int, default=20)
@click.option('--n-highways', type=int, default=2)
@click.option('--hidden-size', type=int, default=100)
@click.option('--drop-rate', type=float, default=0.1)
@click.option('--use-batch-norm', is_flag=True)
@click.option('--activation',
              type=click.Choice(['linear', 'relu', 'sigmoid', 'tanh', 'elu']),
              default='linear')
@click.option('--norm/--no-norm', is_flag=True, default=True)
@click.option('--input1',
              type=click.Choice(['flux', 'magnitude',
                                 'scaled-flux', 'scaled-magnitude']))
@click.option('--input2',
              type=click.Choice(['none', 'flux', 'magnitude',
                                 'scaled-flux', 'scaled-magnitude']),
              default='none')
@click.option('--threads', type=int, default=4)
@click.option('--eval-frequency', type=int, default=10)
@click.option('--task-name', type=str)
@click.option('--remove-y', is_flag=True)
@click.option('--target-distmod/--target-redshift', is_flag=True)
def learn(sim_sn_path, hsc_path, model_dir, cv, fold, batch_size, optimizer,
          adabound_gamma, adabound_final_lr, lr, seed, epochs, patience,
          n_highways, hidden_size, drop_rate, use_batch_norm, activation, norm,
          input1, input2, threads, eval_frequency, task_name,
          remove_y, target_distmod):
    #
    # record arguments
    #

    if platform.system() == 'Windows':
        tmp = (Path(__file__).parents[1] / 'mlruns' /
               'hsc-regression' / 'mlruns')
        # it works well on windows10
        uri = str(tmp.absolute().as_uri())
        # uri = 'file://' + str(tmp.absolute())
    else:
        tmp = (Path(__file__).absolute().parents[1] / 'mlruns' /
               'hsc-regression' / 'mlruns')
        uri = str(tmp.absolute().as_uri())
    mlflow.set_tracking_uri(uri)

    name = '{target}-{task_name}-{input1}-{input2}'.format(
        target='distmod' if target_distmod else 'redshift',
        input1=input1, input2=input2, task_name=task_name
    )
    mlflow.set_experiment(name)

    log_param('sim_sn', sim_sn_path)
    log_param('hsc_path', hsc_path)
    log_param('cv', cv)
    log_param('fold', fold)
    log_param('seed', seed)
    log_param('epochs', epochs)
    log_param('patience', patience)
    log_param('n_highways', n_highways)
    log_param('hidden_size', hidden_size)
    log_param('drop_rate', drop_rate)
    log_param('use_batch_norm', use_batch_norm)
    log_param('normalization', norm)
    log_param('activation', activation)
    log_param('model_dir', str(model_dir))
    log_param('target_distmod', target_distmod)

    input_setting = InputSetting(
        batch_size=batch_size, mixup='none',
        mixup_alpha=2, mixup_beta=2, max_batch_size=batch_size - 1
    )
    # the training data, etc. are not prepared.
    # set the data after preparing.
    input_data = InputData(
        training_data=None, validation_data=None, test_data=None,
        mean=None, std=None, input1=input1, input2=input2,
        remove_y=remove_y, is_hsc=True,
        n_classes=2, input_setting=input_setting
    )

    optimizer_setting = OptimizerSetting(
        name=optimizer, lr=lr, gamma=adabound_gamma,
        final_lr=adabound_final_lr
    )

    loop_setting = LoopSetting(epochs=epochs, patience=patience,
                               eval_frequency=eval_frequency,
                               end_by_epochs=False)

    log_params(input_data.parameters)
    log_params(optimizer_setting.parameters)
    log_params(loop_setting.parameters)

    model_dir = Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir(parents=True)

    # output the path of the recorded arguments to `model_dir`
    if fold < 0:
        # all folds of cv(cross validation)
        with (model_dir / 'uri.txt').open('w') as f:
            f.write(mlflow.get_artifact_uri())
    else:
        # a single fold of cv
        if not (model_dir / str(fold)).exists():
            (model_dir / str(fold)).mkdir(parents=True)
        with (model_dir / str(fold) / 'uri.txt').open('w') as f:
            f.write(mlflow.get_artifact_uri())

    #
    # load data
    #
    sim_sn, hsc_data = load_hsc_data(
        sim_sn_path=sim_sn_path, hsc_path=hsc_path, remove_y=remove_y,
        raw_redshift=True
    )

    # fill nan and inf
    test_flag = np.logical_or(np.isnan(hsc_data['redshift']),
                              np.isinf(hsc_data['redshift']))
    hsc_data['redshift'][test_flag] = 0.1
    test_x = hsc_data

    sim_sn = sklearn.utils.shuffle(sim_sn, random_state=seed)

    skf = KFold(n_splits=cv, random_state=seed)
    # skf = StratifiedKFold(n_splits=cv, random_state=seed)
    split = skf.split(sim_sn['sn_type'], sim_sn['sn_type'])
    #
    # training
    #
    for i, (dev_index, val_index) in enumerate(split):
        if 0 <= fold != i:
            continue
        log_dir = model_dir / str(i)

        dev_x = sim_sn[dev_index]
        val_x = sim_sn[val_index]

        if target_distmod:
            lcdm = Cosmology()
            dev_y = np.asarray(Parallel(n_jobs=threads)(
                delayed(lcdm.DistMod)(z) for z in dev_x['redshift']
            ))
            val_y = np.asarray(Parallel(n_jobs=threads)(
                delayed(lcdm.DistMod)(z) for z in val_x['redshift']
            ))
            test_y = np.asarray(Parallel(n_jobs=threads)(
                delayed(lcdm.DistMod)(z) for z in test_x['redshift']
            ))
        else:
            dev_y = dev_x['redshift']
            val_y = val_x['redshift']
            test_y = test_x['redshift']

        mean, std = compute_moments(
            train_data=dev_x, input1=input1, input2=input2, norm=norm,
            use_redshift=False, is_hsc=True, threads=threads
        )

        if norm:
            output_mean, output_std = np.mean(dev_y), np.std(dev_y)
            dev_y = ((dev_y - output_mean) / output_std).astype(np.float32)
            val_y = ((val_y - output_mean) / output_std).astype(np.float32)
            test_y = ((test_y - output_mean) / output_std).astype(np.float32)

            if not log_dir.exists():
                log_dir.mkdir(parents=True)
            np.savez_compressed(str(log_dir / 'moments.npz'),
                                input_mean=mean, input_std=std,
                                output_mean=output_mean, output_std=output_std)
        else:
            output_mean, output_std = 0, 1

            dev_y = dev_y.astype(np.float32)
            val_y = val_y.astype(np.float32)
            test_y = test_y.astype(np.float32)

        input_data.mean, input_data.std = mean, std
        input_data.training_data = Data(x=dev_x, y=dev_y)
        input_data.validation_data = Data(x=val_x, y=val_y)
        input_data.test_data = Data(x=test_x, y=test_y)

        with tf.Graph().as_default() as graph:
            dataset_ops = build_dataset(input_data=input_data, threads=threads)

            model = Model(hidden_size=hidden_size, drop_rate=drop_rate,
                          num_highways=n_highways, activation=activation,
                          num_outputs=1, use_batch_norm=use_batch_norm)

            optimizer = optimizer_setting.get_optimizer()

            operators = DataBunch(
                training=make_operators(
                    name='training', data_pair=dataset_ops.training,
                    model=model, optimizer=optimizer
                ),
                validation=make_operators(
                    name='validation', data_pair=dataset_ops.validation,
                    model=model
                ),
                test=make_operators(
                    name='test', data_pair=dataset_ops.test, model=model
                ),
                training2=None
            )
            output_ops = DataBunch(
                training=make_prediction(
                    model=model, data_pair=dataset_ops.training2,
                    output_mean=output_mean, output_std=output_std
                ),
                validation=make_prediction(
                    model=model, data_pair=dataset_ops.validation,
                    output_mean=output_mean, output_std=output_std
                ),
                test=make_prediction(
                    model=model, data_pair=dataset_ops.test,
                    output_mean=output_mean, output_std=output_std
                ),
                training2=None
            )

            update_model(graph=graph, update_ops=operators,
                         output_ops=output_ops, eval_frequency=eval_frequency,
                         patience=patience, log_dir=log_dir)


@cmd.command()
@click.option('--model-dir', type=click.Path(exists=True))
@click.option('--data-path', type=click.Path(exists=True))
@click.option('--batch-size', type=int, default=10000)
@click.option('--data-type', type=click.Choice(['SimSN', 'PLAsTiCC', 'HSC']))
@click.option('--output-name', type=str)
@click.option('--threads', type=int, default=4)
def predict(model_dir, data_path, batch_size, data_type, output_name, threads):
    assert output_name is not None

    model_dir = Path(model_dir)

    def parameter(p_name):
        return get_parameter(model_dir=model_dir, name=p_name)

    # load data
    if data_type == 'HSC':
        data = load_hsc_test_data(
            hsc_path=data_path, remove_y=parameter('remove_y') == 'True'
        )
    else:
        data = load_sim_sn_data(sim_sn_path=data_path, use_flux_err2=False,
                                remove_y=parameter('remove_y') == 'True')

    #
    # load normalization factor
    #
    if (model_dir / 'moments.npz').exists():
        tmp = np.load(str(model_dir / 'moments.npz'))
        mean = tmp['input_mean']
        std = tmp['input_std']

        output_mean = tmp['output_mean']
        output_std = tmp['output_std']
    else:
        n = data['flux'][0].shape[0]
        if parameter('input2') != 'none':
            n *= 2
        mean = np.zeros(n)
        std = np.ones_like(mean)

        output_mean, output_std = 0, 1

    #
    # make dataset
    #
    if data_type == 'HSC':
        test_flag = np.logical_or(np.isnan(data['redshift']),
                                  np.isinf(data['redshift']))
        data['redshift'][test_flag] = 0.1
        iterator, next_element = make_dataset(
            x=data, y=data['redshift'], mean=mean, std=std,
            batch_size=batch_size, input1=parameter('input1'),
            input2=parameter('input2'),
            is_hsc=True, threads=threads, is_training=False
        )
    else:
        iterator, next_element = make_dataset(
            x=data, y=data['redshift'], is_training=False, mean=mean, std=std,
            batch_size=batch_size, input1=parameter('input1'),
            input2=parameter('input2'),
            is_hsc=False, threads=threads
        )

    parameters = dict(
        hidden_size=int(parameter('hidden_size')),
        num_highways=int(parameter('n_highways')),
        output_size=1,
        drop_rate=float(parameter('drop_rate')),
        activation=parameter('activation'),
        use_batch_norm=parameter('use_batch_norm') == 'True',
        inpt1=parameter('input1'), input2=parameter('input2'),
        remove_y=parameter('remove_y') == 'True'
    )
    print(parameters)

    model = Model(hidden_size=int(parameter('hidden_size')),
                  drop_rate=float(parameter('drop_rate')),
                  num_highways=int(parameter('n_highways')),
                  activation=parameter('activation'),
                  num_outputs=1,
                  use_batch_norm=parameter('use_batch_norm') == 'True')
    with tf.device('/gpu:0'):
        logits = model(next_element.x, False)
        logits = logits * output_std + output_mean

    saver = tf.train.Saver()

    #
    # run prediction
    #
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True)
    )
    with tf.Session(config=config) as sess:
        checkpoint = tf.train.get_checkpoint_state(model_dir)
        saver.restore(sess, checkpoint.model_checkpoint_path)

        sess.run(iterator.initializer)

        predictions = []
        labels = []
        name_list = []
        while True:
            try:
                if data_type == 'HSC':
                    p, data_name = sess.run([logits, next_element.name])
                else:
                    p, y, data_name = sess.run([
                        logits, next_element[1], next_element[2]
                    ])
                    labels.append(y)
                predictions.append(p)
                name_list.extend([n.decode() for n in data_name])
            except tf.errors.OutOfRangeError:
                break
    #
    # save result
    #
    predictions = np.concatenate(predictions, axis=0)
    df = pd.DataFrame(
        predictions, columns=['prediction'], index=name_list
    )
    if data_type != 'HSC':
        labels = np.concatenate(labels, axis=0)
        df['target'] = labels

    df.to_csv(model_dir / output_name)


@cmd.command()
@click.option('--sim-sn-path', type=click.Path(exists=True))
@click.option('--hsc-path', type=click.Path(exists=True))
@click.option('--model-dir', type=click.Path())
@click.option('--batch-size', type=int, default=1000)
@click.option('--lr', type=float, default=1e-3)
@click.option('--optimizer',
              type=click.Choice(['adam', 'momentum', 'adabound', 'amsbound']),
              default='adam')
@click.option('--adabound-gamma', type=float, default=1e-3)
@click.option('--adabound-final-lr', type=float, default=0.1)
@click.option('--seed', type=int, default=0)
@click.option('--epochs', type=int, default=500)
@click.option('--patience', type=int, default=100)
@click.option('--n-trials', type=int, default=15)
@click.option('--norm/--no-norm', is_flag=True, default=True)
@click.option('--input1',
              type=click.Choice(['flux', 'magnitude',
                                 'scaled-flux', 'scaled-magnitude']))
@click.option('--input2',
              type=click.Choice(['none', 'flux', 'magnitude',
                                 'scaled-flux', 'scaled-magnitude']),
              default='none')
@click.option('--threads', type=int, default=4)
@click.option('--eval-frequency', type=int, default=10)
@click.option('--task-name', type=str)
@click.option('--remove-y', is_flag=True)
@click.option('--target-distmod/--target-redshift', is_flag=True)
def search(sim_sn_path, hsc_path, model_dir, batch_size, optimizer,
           adabound_gamma, adabound_final_lr, lr, seed, epochs, patience,
           n_trials, norm, input1, input2, threads, eval_frequency, task_name,
           remove_y, target_distmod):
    storage = 'sqlite:///{}/example.db'.format(model_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if platform.system() == 'Windows':
        tmp = (Path(__file__).parents[1] / 'mlruns' /
               'search-hsc-redshift' / 'mlruns')
        uri = str(tmp.absolute().as_uri())
        # uri = 'file://' + str(tmp.absolute())
    else:
        tmp = (Path(__file__).parents[1] / 'mlruns' /
               'search-hsc-redshift' / 'mlruns')
        uri = str(tmp.absolute().as_uri())
    mlflow.set_tracking_uri(uri)

    print(model_dir)

    name = '{flag}-{input1}-{input2}'.format(flag=task_name, input1=input1,
                                             input2=input2)
    if remove_y:
        name += '-remove-y'
    name += '-{}'.format('distmod' if target_distmod else 'redshift')
    mlflow.set_experiment(name)

    db_path = os.path.join(model_dir, 'example.db')
    sampler = MyTPESampler()
    if os.path.exists(db_path):
        study = optuna.Study(study_name='study190513', storage=storage,
                             sampler=sampler)
    else:
        study = optuna.create_study(study_name='study190513', storage=storage,
                                    sampler=sampler)

    input_setting = InputSetting(
        batch_size=batch_size, mixup='none', mixup_alpha=2, mixup_beta=2
    )
    input_data = InputData(
        training_data=None, validation_data=None, test_data=None,
        mean=None, std=None, input1=input1, input2=input2,
        remove_y=remove_y, is_hsc=True, n_classes=1,
        input_setting=input_setting
    )

    optimizer_setting = OptimizerSetting(
        name=optimizer, lr=lr, gamma=adabound_gamma,
        final_lr=adabound_final_lr
    )
    loop_setting = LoopSetting(epochs=epochs, patience=patience,
                               eval_frequency=eval_frequency,
                               end_by_epochs=False)
    print('loading data')
    sim_sn, _ = load_hsc_data(
        sim_sn_path=sim_sn_path, hsc_path=hsc_path,
        remove_y=input_data.remove_y
    )
    sim_sn = sklearn.utils.shuffle(sim_sn, random_state=seed)
    if target_distmod:
        lcdm = Cosmology()
        target = np.asarray(Parallel(n_jobs=threads)(
            delayed(lcdm.DistMod)(z) for z in sim_sn['redshift']
        ))
    else:
        target = sim_sn['redshift']

    tmp_x, test_x, tmp_y, test_y = train_test_split(
        sim_sn, target, test_size=0.3, random_state=42
    )
    dev_x, val_x, dev_y, val_y = train_test_split(
        tmp_x, tmp_y, test_size=0.3, random_state=44
    )

    if norm:
        output_mean, output_std = np.mean(dev_y), np.std(dev_y)
        dev_y = ((dev_y - output_mean) / output_std).astype(np.float32)
        val_y = ((val_y - output_mean) / output_std).astype(np.float32)
        test_y = ((test_y - output_mean) / output_std).astype(np.float32)
        # ハイパーパラメータ探索の最中なので平均と分散は保存しない
    else:
        # output_mean, output_std = 0, 1

        dev_y = dev_y.astype(np.float32)
        val_y = val_y.astype(np.float32)
        test_y = test_y.astype(np.float32)
    mean, std = compute_moments(
        train_data=dev_x, input1=input1, input2=input2, norm=norm,
        use_redshift=False, is_hsc=True, threads=threads
    )
    input_data.mean, input_data.std = mean, std

    training_data = Data(x=dev_x, y=dev_y)
    validation_data = Data(x=val_x, y=val_y)
    test_data = Data(x=test_x, y=test_y)
    input_data.training_data = training_data
    input_data.validation_data = validation_data
    input_data.test_data = test_data

    for i in range(n_trials):
        study.optimize(
            lambda trial: objective_hsc(
                trial=trial, sim_sn_path=sim_sn_path, hsc_path=hsc_path,
                optimizer_setting=optimizer_setting, seed=seed,
                loop_setting=loop_setting, normalization=norm, threads=threads,
                input_data=input_data
            ),
            n_trials=1
        )

        df = study.trials_dataframe()
        df.to_csv(os.path.join(model_dir, 'result.csv'))


def objective_hsc(trial, sim_sn_path, hsc_path, optimizer_setting, seed,
                  loop_setting, normalization, threads, input_data):
    with mlflow.start_run():
        num_highways = trial.suggest_int('num_highways', 3, 7)
        use_batch_norm = trial.suggest_categorical(
            'use_batch_norm', [False, True]
        )
        use_drop_out = False
        activation = trial.suggest_categorical(
            'activation', ['linear', 'relu', 'sigmoid', 'tanh']
        )
        drop_rate = trial.suggest_loguniform('drop_rate', 5e-4, 0.25)
        hidden_size = trial.suggest_int('hidden_size', 100, 3000)

        log_param('sim_sn', sim_sn_path)
        log_param('hsc_path', hsc_path)
        log_param('seed', seed)
        log_param('hidden_size', hidden_size)
        log_param('drop_rate', drop_rate)
        log_param('num_highways', num_highways)
        log_param('use_batch_norm', use_batch_norm)
        log_param('use_drop_out', use_drop_out)
        log_param('activation', activation)
        log_param('normalization', normalization)

        log_params(input_data.parameters)
        log_params(optimizer_setting.parameters)
        log_params(loop_setting.parameters)

        best_score = run(
            trial=trial, num_highways=num_highways,
            use_batch_norm=use_batch_norm, use_drop_out=use_drop_out,
            activation=activation, drop_rate=drop_rate,
            hidden_size=hidden_size, loop_setting=loop_setting,
            optimizer_setting=optimizer_setting, input_data=input_data,
            threads=threads
        )
    return best_score


def run(trial, num_highways, use_batch_norm, use_drop_out, activation,
        drop_rate, hidden_size, loop_setting, optimizer_setting, input_data,
        threads):
    _ = use_drop_out
    with tf.Graph().as_default() as graph:
        if input_data.is_hsc:
            dataset_ops = build_hsc_dataset(input_data=input_data,
                                            threads=threads)
        else:
            dataset_ops = build_plasticc_dataset(
                input_data=input_data, sampling_ratio=0.1, threads=threads
            )

        model = Model(hidden_size=hidden_size, drop_rate=drop_rate,
                      use_batch_norm=use_batch_norm, num_highways=num_highways,
                      activation=activation, num_outputs=1)

        dev_iterator = dataset_ops.training_iterator
        dev_element = dataset_ops.training_element
        val_iterator = dataset_ops.validation_iterator
        val_element = dataset_ops.validation_element
        test_iterator = dataset_ops.test_iterator
        test_element = dataset_ops.test_element

        optimizer = optimizer_setting.get_optimizer()

        dev_ops = make_operators(
            name='training',
            data_pair=DataOperator(iterator=dev_iterator, element=dev_element),
            model=model, optimizer=optimizer
        )
        val_ops = make_operators(
            name='validation',
            data_pair=DataOperator(iterator=val_iterator, element=val_element),
            model=model
        )
        test_ops = make_operators(
            name='test',
            data_pair=DataOperator(iterator=test_iterator,
                                   element=test_element),
            model=model
        )

        best_val_accuracy = 0
        best_test_accuracy = 0
        val_accuracy_list = []

        global_step = tf.train.get_or_create_global_step()
        count_up = tf.assign_add(global_step, 1)

        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True)
        )
        with tf.Session(graph=graph, config=config) as sess:
            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))

            for _ in range(loop_setting.epochs):
                # データのシャッフルが同じになってしまうので、更新する必要がある
                step = sess.run(count_up)

                update_metrics(
                    sess=sess, ops=dev_ops, step=None, writer=None
                )

                if step % loop_setting.eval_frequency == 0:
                    val_accuracy = update_metrics(
                        sess=sess, ops=val_ops, writer=None, step=None
                    )
                    val_accuracy_list.append(val_accuracy)

                    span = len(val_accuracy_list) * loop_setting.eval_frequency
                    if span >= loop_setting.patience:
                        test_accuracy = update_metrics(
                            sess=sess, ops=test_ops, writer=None,
                            step=None
                        )
                        # 最小化
                        trial.report(-test_accuracy, step=step)
                        if test_accuracy > best_test_accuracy:
                            best_test_accuracy = test_accuracy

                        tmp = np.mean(val_accuracy_list)
                        if tmp > best_val_accuracy:
                            best_val_accuracy = tmp
                            val_accuracy_list = []
                        else:
                            break

                        if trial.should_prune(step=step):
                            raise optuna.structs.TrialPruned()
    # 最小化
    return -best_test_accuracy


def main():
    cmd()


if __name__ == '__main__':
    main()
