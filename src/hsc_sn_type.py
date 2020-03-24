#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
SN type classifier
(binary classification and 3-class classification)

This script contains the codes to train and to predict.
The codes for hyper parameter search is hsc_search.py.
"""

import os
import platform
import re
from collections import namedtuple
from pathlib import Path
from urllib.parse import urlparse

import click
import mlflow
import numpy as np
import pandas as pd
import sklearn.utils
import tensorflow as tf
# noinspection PyProtectedMember
from mlflow import log_param, log_metric, log_params
from scipy.special import logsumexp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

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

from hsc.loader import (load_plasticc_data, load_hsc_data,
                        load_hsc_test_data, load_hsc_data_n_observations,
                        load_sim_sn_data)
from hsc.model import Model
from hsc.dataset import (build_plasticc_dataset, build_hsc_dataset,
                         compute_moments, make_dataset,
                         Data, InputSetting, InputData)

__date__ = '15/2/2019'


class OptimizerSetting(object):
    def __init__(self, name, lr, gamma, final_lr):
        self.name = name
        self.lr = lr
        self.gamma = gamma
        self.final_lr = final_lr

    @property
    def parameters(self):
        tmp = {'optimizer': self.name, 'lr': self.lr}
        if 'bound' in self.name:
            tmp.update({'adabound_gamma': self.gamma,
                        'adabound_final_lr': self.final_lr})
        return tmp

    def get_optimizer(self):
        if self.name == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        elif self.name == 'momentum':
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=self.lr, momentum=0.9
            )
        elif self.name == 'adabound':
            optimizer = AdaBoundOptimizer(
                learning_rate=self.lr, gamma=self.gamma,
                final_lr=self.final_lr
            )
        elif self.name == 'amsbound':
            optimizer = AdaBoundOptimizer(
                learning_rate=self.lr, gamma=self.gamma,
                final_lr=self.final_lr, amsbound=True
            )
        else:
            raise ValueError(self.name)

        return optimizer


def get_label_map(binary):
    if binary:
        label_map = {b'Ia': 0,
                     b'Ibc': 1, b'Ib': 1, b'Ic': 1,
                     b'II': 1, b'IIL': 1, b'IIN': 1, b'IIP': 1}
    else:
        label_map = {b'Ia': 0,
                     b'Ibc': 1, b'Ib': 1, b'Ic': 1,
                     b'II': 2, b'IIL': 2, b'IIN': 2, b'IIP': 2}
    return label_map


class LoopSetting(object):
    def __init__(self, epochs, patience, eval_frequency, end_by_epochs):
        self.epochs = epochs
        self.end_by_epochs = end_by_epochs

        self.patience = patience
        self.eval_frequency = eval_frequency

    @property
    def parameters(self):
        tmp = {'epochs': self.epochs, 'end_by_epochs': self.end_by_epochs,
               'patience': self.patience,
               'eval_frequency': self.eval_frequency}
        return tmp


@click.group()
def cmd():
    pass


@cmd.command()
@click.option('--sim-sn-path', type=click.Path(exists=True))
@click.option('--training-cosmos-path', type=click.Path(exists=True))
@click.option('--test-cosmos-path', type=click.Path(exists=True))
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
@click.option('--eval-frequency', type=int, default=10)
@click.option('--end-by-epochs', is_flag=True)
@click.option('--n-highways', type=int, default=2)
@click.option('--hidden-size', type=int, default=100)
@click.option('--drop-rate', type=float, default=0.1)
@click.option('--activation',
              type=click.Choice(['linear', 'relu', 'sigmoid', 'tanh', 'elu']))
@click.option('--norm/--no-norm', is_flag=True, default=True)
@click.option('--use-batch-norm', is_flag=True)
@click.option('--use-dropout', is_flag=True)
@click.option('--flux-err', type=int, default=2)
@click.option('--input1',
              type=click.Choice(['flux', 'magnitude', 'absolute-magnitude',
                                 'scaled-flux', 'scaled-magnitude']))
@click.option('--input2',
              type=click.Choice(['none', 'flux', 'magnitude',
                                 'absolute-magnitude',
                                 'scaled-flux', 'scaled-magnitude']),
              default='none')
@click.option('--sampling-ratio', type=float, default=0.1)
@click.option('--balance', is_flag=True)
@click.option('--binary', is_flag=True)
@click.option('--mixup', type=click.Choice(['none', 'mixup', 'manifold']),
              default='none')
@click.option('--threads', type=int, default=4)
def fit_plasticc(sim_sn_path, training_cosmos_path, test_cosmos_path,
                 model_dir, cv, fold, batch_size, optimizer, adabound_gamma,
                 adabound_final_lr, lr, seed, epochs, patience,
                 eval_frequency, end_by_epochs, n_highways,
                 hidden_size, drop_rate, activation, norm, use_batch_norm,
                 use_dropout, flux_err, input1, input2, sampling_ratio,
                 balance, binary, mixup, threads):
    if platform.system() == 'Windows':
        tmp = (Path(__file__).parents[1] / 'mlruns' /
               'plasticc-classification' / 'mlruns')
        # it works well on windows10
        uri = str(tmp.absolute().as_uri())
        # uri = 'file://' + str(tmp.absolute())
    else:
        tmp = (Path(__file__).absolute().parents[1] / 'mlruns' /
               'platicc-classification' / 'mlruns')
        uri = str(tmp.absolute().as_uri())
    mlflow.set_tracking_uri(uri)

    n_classes = 2 if binary else 3
    name = 'plasticc{n_classes}-{input1}-{input2}'.format(
        n_classes=n_classes, input1=input1, input2=input2
    )
    mlflow.set_experiment(name)

    log_param('sim_sn', sim_sn_path)
    log_param('training_cosmos', training_cosmos_path)
    log_param('test_cosmos', test_cosmos_path)
    log_param('cv', cv)
    log_param('fold', fold)
    log_param('seed', seed)
    log_param('n_highways', n_highways)
    log_param('hidden_size', hidden_size)
    log_param('drop_rate', drop_rate)
    log_param('normalization', norm)
    log_param('activation', activation)
    log_param('flux_err', flux_err)
    log_param('use_batch_norm', use_batch_norm)
    log_param('use_dropout', use_dropout)
    log_param('model_dir', str(model_dir))
    log_param('sampling_ratio', sampling_ratio)
    log_param('binary', binary)
    log_param('mixup', mixup)

    input_setting = InputSetting(
        batch_size=batch_size, mixup=mixup, balance=balance,
        mixup_alpha=2, mixup_beta=2,
        max_batch_size=batch_size - 1
    )
    input_data = InputData(
        training_data=None, validation_data=None, test_data=None,
        mean=None, std=None, input1=input1, input2=input2,
        remove_y=False, is_hsc=False,
        n_classes=n_classes, input_setting=input_setting
    )

    optimizer_setting = OptimizerSetting(
        name=optimizer, lr=lr, gamma=adabound_gamma,
        final_lr=adabound_final_lr
    )

    loop_setting = LoopSetting(epochs=epochs, patience=patience,
                               eval_frequency=eval_frequency,
                               end_by_epochs=end_by_epochs)

    log_params(input_data.parameters)
    log_params(optimizer_setting.parameters)
    log_params(loop_setting.parameters)

    model_dir = Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir(parents=True)

    if fold < 0:
        with (model_dir / 'uri.txt').open('w') as f:
            f.write(mlflow.get_artifact_uri())
    else:
        if not (model_dir / str(fold)).exists():
            (model_dir / str(fold)).mkdir(parents=True)
        with (model_dir / str(fold) / 'uri.txt').open('w') as f:
            f.write(mlflow.get_artifact_uri())

    print('loading data')
    sim_sn, training_cosmos, test_cosmos = load_plasticc_data(
        sim_sn_path=sim_sn_path, training_cosmos_path=training_cosmos_path,
        test_cosmos_path=test_cosmos_path, use_flux_err2=flux_err == 2
    )
    sim_sn = sklearn.utils.shuffle(sim_sn, random_state=seed)
    training_cosmos = sklearn.utils.shuffle(training_cosmos,
                                            random_state=seed + 1)

    # converting sn type into class ID
    label_map = get_label_map(binary=binary)
    sim_sn_y = np.array([label_map[c] for c in sim_sn['sn_type']])
    training_cosmos_y = np.array([label_map[c]
                                  for c in training_cosmos['sn_type']])
    test_cosmos_y = np.array([label_map[c] for c in test_cosmos['sn_type']])

    # splitter for sim_sn dataset
    skf = StratifiedKFold(n_splits=cv, random_state=seed)
    split1 = skf.split(sim_sn['sn_type'], sim_sn['sn_type'])
    # splitter for training_cosmos dataset
    skf = StratifiedKFold(n_splits=cv, random_state=seed + 1)
    split2 = skf.split(training_cosmos['sn_type'], training_cosmos['sn_type'])

    print('training')
    # auc values of each fold
    auc_score_list_dev, auc_score_list_val = [], []
    auc_score_list_test = []
    for i, tmp in tqdm(enumerate(zip(split1, split2)), total=cv):
        if 0 <= fold != i:
            continue
        out_dir = model_dir / str(i)

        (dev_index1, val_index1), (dev_index2, val_index2) = tmp
        dev_x1 = sim_sn[dev_index1]
        dev_y1 = sim_sn_y[dev_index1]
        dev_x2 = training_cosmos[dev_index2]
        dev_y2 = training_cosmos_y[dev_index2]
        weight = np.asarray([0.9 / len(dev_x1)] * len(dev_x1) +
                            [0.1 / len(dev_x2)] * len(dev_x2))

        mean, std = compute_moments(
            train_data=dev_x1, input1=input1, input2=input2,
            norm=norm, use_redshift=False, is_hsc=False,
            threads=threads
        )
        if norm:
            if not out_dir.exists():
                out_dir.mkdir(parents=True)
            np.savez_compressed(str(out_dir / 'moments.npz'),
                                mean=mean, std=std)

        val_x = np.hstack((sim_sn[val_index1], training_cosmos[val_index2]))
        val_y = np.hstack((
            sim_sn_y[val_index1], training_cosmos_y[val_index2]
        ))

        input_data.mean, input_data.std = mean, std

        training_data = Data(x=np.hstack((dev_x1, dev_x2)),
                             y=np.hstack((dev_y1, dev_y2)),
                             weight=weight)
        validation_data = Data(x=val_x, y=val_y)
        test_data = Data(x=test_cosmos, y=test_cosmos_y)

        input_data.training_data = training_data
        input_data.validation_data = validation_data
        input_data.test_data = test_data

        with tf.Graph().as_default() as graph:
            dataset_ops = build_plasticc_dataset(
                input_data=input_data, sampling_ratio=sampling_ratio,
                threads=threads
            )

            model = Model(
                hidden_size=hidden_size, output_size=n_classes,
                drop_rate=drop_rate, activation=activation,
                num_highways=n_highways, use_batch_norm=use_batch_norm,
                use_dropout=use_dropout
            )
            optimizer = optimizer_setting.get_optimizer()

            dev_auc_score, val_auc_score, test_auc_score = fit_cv(
                graph=graph, dataset_ops=dataset_ops, model=model, mixup=mixup,
                optimizer=optimizer,
                n_highways=n_highways, n_classes=n_classes, hsc=False,
                model_dir=model_dir, i=i, patience=patience,
                eval_frequency=eval_frequency
            )

            auc_score_list_dev.append(dev_auc_score)
            auc_score_list_val.append(val_auc_score)
            auc_score_list_test.append(test_auc_score)
    df_dev = pd.DataFrame(auc_score_list_dev)
    df_dev.to_csv(model_dir / 'auc-training.csv', index=False)

    df_val = pd.DataFrame(auc_score_list_val)
    df_val.to_csv(model_dir / 'auc-validation.csv', index=False)

    df_test = pd.DataFrame(auc_score_list_test)
    df_test.to_csv(model_dir / 'auc-test.csv', index=False)

    # log_artifacts(model_dir)


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
@click.option('--eval-frequency', type=int, default=10)
@click.option('--end-by-epochs', is_flag=True)
@click.option('--n-highways', type=int, default=2)
@click.option('--hidden-size', type=int, default=100)
@click.option('--drop-rate', type=float, default=0.1)
@click.option('--activation',
              type=click.Choice(['linear', 'relu', 'sigmoid', 'tanh', 'elu']))
@click.option('--norm/--no-norm', is_flag=True, default=True)
@click.option('--input1',
              type=click.Choice(['flux', 'magnitude', 'absolute-magnitude',
                                 'scaled-flux', 'scaled-magnitude']))
@click.option('--input2',
              type=click.Choice(['none', 'flux', 'magnitude',
                                 'absolute-magnitude',
                                 'scaled-flux', 'scaled-magnitude']),
              default='none')
@click.option('--mixup',
              type=click.Choice(['none', 'mixup', 'manifold', 'weighted']),
              default='none')
@click.option('--threads', type=int, default=4)
@click.option('--task-name', type=str)
@click.option('--binary/--multi', is_flag=True, default=True)
@click.option('--remove-y', is_flag=True)
@click.option('--use-batch-norm', is_flag=True)
@click.option('--use-dropout', is_flag=True)
@click.option('--use-layer-norm', is_flag=True)
@click.option('--mixup-alpha', type=float, default=2)
@click.option('--mixup-beta', type=float, default=2)
@click.option('--max-batch-size', type=int, default=1000)
@click.option('--n-observations', type=int, default=-1)
def fit_hsc(sim_sn_path, hsc_path, model_dir, cv, fold, batch_size, optimizer,
            adabound_gamma, adabound_final_lr, lr, seed, epochs, patience,
            eval_frequency, end_by_epochs,
            n_highways, hidden_size, drop_rate, activation, norm,
            input1, input2, mixup, threads, task_name, binary, remove_y,
            use_batch_norm, use_dropout, use_layer_norm,
            mixup_alpha, mixup_beta,
            max_batch_size, n_observations):
    """
    training a classifier for hsc data

    :param sim_sn_path:
    :param hsc_path:
    :param model_dir:
    :param cv:
    :param fold:
    :param batch_size:
    :param optimizer:
    :param adabound_gamma:
    :param adabound_final_lr:
    :param lr:
    :param seed:
    :param epochs:
    :param patience:
    :param eval_frequency:
    :param end_by_epochs:
    :param n_highways:
    :param hidden_size:
    :param drop_rate:
    :param activation:
    :param norm:
    :param input1:
    :param input2:
    :param mixup:
    :param threads:
    :param task_name:
    :param binary:
    :param remove_y:
    :param use_batch_norm:
    :param use_dropout:
    :param use_layer_norm:
    :param mixup_alpha:
    :param mixup_beta:
    :param max_batch_size:
    :param n_observations:
    :return:
    """

    #
    # record the arguments with mlflow
    #

    if platform.system() == 'Windows':
        tmp = (Path(__file__).parents[1] / 'mlruns' /
               'hsc-classification' / 'mlruns')
        # it works well on windows10
        uri = str(tmp.absolute().as_uri())
        # uri = 'file://' + str(tmp.absolute())
    else:
        tmp = (Path(__file__).absolute().parents[1] / 'mlruns' /
               'hsc-classification' / 'mlruns')
        uri = str(tmp.absolute().as_uri())
    mlflow.set_tracking_uri(uri)

    n_classes = 2 if binary else 3
    name = '{n_classes}-{task_name}-{input1}-{input2}'.format(
        n_classes=n_classes, input1=input1, input2=input2, task_name=task_name
    )
    if n_observations > 0:
        name += '-n-observations'
    mlflow.set_experiment(name)

    log_param('sim_sn', sim_sn_path)
    log_param('hsc_path', hsc_path)
    log_param('cv', cv)
    log_param('fold', fold)
    log_param('seed', seed)
    log_param('n_highways', n_highways)
    log_param('hidden_size', hidden_size)
    log_param('drop_rate', drop_rate)
    log_param('normalization', norm)
    log_param('activation', activation)
    log_param('model_dir', str(model_dir))
    log_param('binary', binary)
    log_param('use_batch_norm', use_batch_norm)
    log_param('use_dropout', use_dropout)
    log_param('use_layer_norm', use_layer_norm)
    log_param('n_observations', n_observations)

    input_setting = InputSetting(
        batch_size=batch_size, mixup=mixup,
        mixup_alpha=mixup_alpha, mixup_beta=mixup_beta,
        max_batch_size=max_batch_size
    )
    # the training data, etc. are not prepared.
    # set the data after preparing.
    input_data = InputData(
        training_data=None, validation_data=None, test_data=None,
        mean=None, std=None, input1=input1, input2=input2,
        remove_y=remove_y, is_hsc=True,
        n_classes=n_classes, input_setting=input_setting
    )

    optimizer_setting = OptimizerSetting(
        name=optimizer, lr=lr, gamma=adabound_gamma,
        final_lr=adabound_final_lr
    )

    loop_setting = LoopSetting(epochs=epochs, patience=patience,
                               eval_frequency=eval_frequency,
                               end_by_epochs=end_by_epochs)

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
        out_dir = model_dir / str(fold)
        if not out_dir.exists():
            out_dir.mkdir(parents=True)
        with (out_dir / 'uri.txt').open('w') as f:
            f.write(mlflow.get_artifact_uri())

    #
    # load data
    #
    print('loading data')
    if n_observations > 0:
        sim_sn, hsc_data = load_hsc_data_n_observations(
            sim_sn_path=sim_sn_path, hsc_path=hsc_path,
            n_observations=n_observations, remove_y=remove_y
        )

        threshold = sim_sn['flux_err'] * 3
        selected = np.any(sim_sn['flux'] > threshold, axis=1)
        sim_sn = sim_sn[selected]
    else:
        sim_sn, hsc_data = load_hsc_data(
            sim_sn_path=sim_sn_path, hsc_path=hsc_path, remove_y=remove_y
        )
    print('flux shape:', sim_sn['flux'].shape)

    sim_sn = sklearn.utils.shuffle(sim_sn, random_state=seed)

    # convert sn type to class ID
    label_map = get_label_map(binary=binary)
    sim_sn_y = np.array([label_map[c] for c in sim_sn['sn_type']])

    skf = StratifiedKFold(n_splits=cv, random_state=seed)
    split = skf.split(sim_sn['sn_type'], sim_sn['sn_type'])

    #
    # training
    #
    print('training')
    auc_score_list_dev, auc_score_list_val = [], []
    for i, (dev_index, val_index) in enumerate(split):
        if 0 <= fold != i:
            continue

        dev_x = sim_sn[dev_index]
        dev_y = sim_sn_y[dev_index]

        mean, std = compute_moments(
            train_data=dev_x, input1=input1, input2=input2, norm=norm,
            use_redshift=False, is_hsc=True, threads=threads
        )
        if norm:
            log_dir = model_dir / str(i)
            if not log_dir.exists():
                log_dir.mkdir(parents=True)
            np.savez_compressed(str(log_dir / 'moments.npz'),
                                mean=mean, std=std)

        val_x = sim_sn[val_index]
        val_y = sim_sn_y[val_index]

        input_data.mean, input_data.std = mean, std

        training_data = Data(x=dev_x, y=dev_y)
        validation_data = Data(x=val_x, y=val_y)
        test_data = Data(x=hsc_data)

        input_data.training_data = training_data
        input_data.validation_data = validation_data
        input_data.test_data = test_data

        with tf.Graph().as_default() as graph:
            dataset_ops = build_hsc_dataset(input_data=input_data,
                                            threads=threads)

            model = Model(
                hidden_size=hidden_size, output_size=n_classes,
                drop_rate=drop_rate, activation=activation,
                num_highways=n_highways, use_batch_norm=use_batch_norm,
                use_dropout=use_dropout
            )
            optimizer = optimizer_setting.get_optimizer()

            dev_auc_score, val_auc_score = fit_cv(
                graph=graph, dataset_ops=dataset_ops, model=model,
                mixup=mixup, optimizer=optimizer,
                n_highways=n_highways, n_classes=n_classes, hsc=True,
                model_dir=model_dir, i=i, patience=patience,
                eval_frequency=eval_frequency
            )

            auc_score_list_dev.append(dev_auc_score)
            auc_score_list_val.append(val_auc_score)
    #
    # output AUC values
    #
    df_dev = pd.DataFrame(auc_score_list_dev)
    df_dev.to_csv(model_dir / 'auc-training.csv', index=False)

    df_val = pd.DataFrame(auc_score_list_val)
    df_val.to_csv(model_dir / 'auc-validation.csv', index=False)

    # log_artifacts(model_dir)


def fit_cv(graph, dataset_ops, model, mixup, optimizer,
           n_highways, n_classes, hsc, model_dir, i, patience, eval_frequency):
    """
    implementations to train with cv

    :param graph:
    :param dataset_ops:
    :param model:
    :param mixup:
    :param optimizer:
    :param n_highways:
    :param n_classes:
    :param hsc:
    :param model_dir:
    :param i:
    :param patience:
    :param eval_frequency:
    :return:
    """
    with tf.device('/gpu:0'):
        if mixup == 'manifold':
            # noinspection PyUnboundLocalVariable
            k = tf.random_uniform(
                shape=[],
                minval=0, maxval=n_highways + 2, dtype=tf.int32
            )
            dev_output = model(dataset_ops.training_element.x, True,
                               k=k, r=dataset_ops.training_element.ratio)
        else:
            dev_output = model(dataset_ops.training_element.x, True)
        if mixup != 'none':
            # dataset to predict
            # noinspection PyUnboundLocalVariable
            dev_prediction_output = model(dataset_ops.training_element2.x,
                                          False)

        val_output = model(dataset_ops.validation_element.x, False)
        test_output = model(dataset_ops.test_element.x, False)

    #
    # operators to train/predict, update metrics and so on
    #
    dev_ops = make_operators(
        name='training', outputs=dev_output,
        next_element=dataset_ops.training_element,
        iterator=dataset_ops.training_iterator, optimizer=optimizer,
        is_training=True, mixup=mixup
    )
    val_ops = make_operators(
        name='validation', outputs=val_output,
        next_element=dataset_ops.validation_element,
        iterator=dataset_ops.validation_iterator,
        is_training=False, mixup=mixup
    )
    test_ops = make_operators(
        name='test', outputs=test_output,
        next_element=dataset_ops.test_element,
        iterator=dataset_ops.test_iterator,
        is_training=False, mixup=mixup
    )

    global_step = tf.train.get_or_create_global_step()
    count_up = tf.assign_add(global_step, 1)

    log_dir = model_dir / str(i)

    writer = tf.summary.FileWriter(str(log_dir))
    saver = tf.train.Saver()

    #
    # training
    #
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True)
    )
    with tf.Session(graph=graph, config=config) as sess:
        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))

        def _save_prediction(epoch):
            fmt = 'predictions_{0}{1:04d}.csv'
            name_list = ('training', 'validation', 'test')
            if mixup != 'none':
                iterator_list = (dataset_ops.training_iterator2,
                                 dataset_ops.validation_iterator,
                                 dataset_ops.test_iterator)
                next_element_list = (dataset_ops.training_element2,
                                     dataset_ops.validation_element,
                                     dataset_ops.test_element)
                output_list = (dev_prediction_output, val_output, test_output)
            else:
                iterator_list = (dataset_ops.training_iterator,
                                 dataset_ops.validation_iterator,
                                 dataset_ops.test_iterator)
                next_element_list = (dataset_ops.training_element,
                                     dataset_ops.validation_element,
                                     dataset_ops.test_element)
                output_list = (dev_output, val_output, test_output)
            for name, iterator, next_element, output in zip(
                    name_list, iterator_list, next_element_list, output_list):
                # set a flag controlling whether to output the target labels
                if hsc and name == 'test':
                    # not to output the target labels
                    data_type = name
                else:
                    # the target labels is output if it is not 'test'
                    data_type = 'hoge'
                save_prediction(
                    sess=sess, initializer=iterator.initializer,
                    output_op=output, next_element=next_element,
                    file_path=log_dir / fmt.format(name, epoch),
                    data_type=data_type
                )

        _save_prediction(epoch=0)

        print('start: cv {}'.format(i))

        # early stopping
        # watching the validation accuracy to stop training
        previous_accuracy = 0
        accuracy_list = []

        while True:
            step = sess.run(count_up)

            update_metrics(sess=sess, ops=dev_ops, writer=writer,
                           step=step, i=i)
            if step % 100 == 0:
                _save_prediction(epoch=step)

            if step % eval_frequency == 0:
                _, val = update_metrics(sess=sess, ops=val_ops,
                                        writer=writer, step=step, i=i)

                if not hsc:
                    # noinspection PyUnboundLocalVariable
                    update_metrics(sess=sess, ops=test_ops,
                                   writer=writer, step=step, i=i)

                accuracy_list.append(val)
                if len(accuracy_list) * eval_frequency >= patience:
                    # saver.save(
                    #     sess=sess,
                    #     save_path=str(log_dir / 'model'),
                    #     global_step=global_step, write_meta_graph=False
                    # )

                    current_accuracy = np.mean(accuracy_list)
                    print(current_accuracy, previous_accuracy, end=' ')
                    if current_accuracy <= previous_accuracy:
                        print('stop')
                        break
                    else:
                        print('updated')
                        accuracy_list.clear()
                        previous_accuracy = current_accuracy

                        saver.save(
                            sess=sess,
                            save_path=str(log_dir / 'model'),
                            global_step=global_step, write_meta_graph=False
                        )

        # training is finished
        # restoring the last saved model
        checkpoint = tf.train.get_checkpoint_state(str(log_dir))
        saver.restore(sess, checkpoint.model_checkpoint_path)

        #
        # draw ROC curve
        #

        if mixup != 'none':
            # noinspection PyUnboundLocalVariable
            dev_auc_score = compute_auc(
                sess=sess,
                initializer=dataset_ops.training_iterator2.initializer,
                logits_op=dev_prediction_output,
                next_element=dataset_ops.training_element2,
                model_dir=log_dir, n_classes=n_classes,
                name='training'
            )
        else:
            dev_auc_score = compute_auc(
                sess=sess,
                initializer=dataset_ops.training_iterator.initializer,
                logits_op=dev_output,
                next_element=dataset_ops.training_element,
                model_dir=log_dir, n_classes=n_classes,
                name='training'
            )

        val_auc_score = compute_auc(
            sess=sess, initializer=dataset_ops.validation_iterator.initializer,
            logits_op=val_output, next_element=dataset_ops.validation_element,
            model_dir=log_dir, n_classes=n_classes,
            name='validation'
        )

        if hsc:
            save_prediction(
                sess=sess, initializer=dataset_ops.test_iterator.initializer,
                output_op=test_output, next_element=dataset_ops.test_element,
                file_path=log_dir / 'predictions_test.csv', data_type='test'
            )
        else:
            test_auc_score = compute_auc(
                sess=sess, initializer=dataset_ops.test_iterator.initializer,
                logits_op=test_output, next_element=dataset_ops.test_element,
                model_dir=log_dir, n_classes=n_classes,
                name='test'
            )
    # it seams programs will finish before outputting the buffer
    writer.flush()

    if hsc:
        return dev_auc_score, val_auc_score
    else:
        # noinspection PyUnboundLocalVariable
        return dev_auc_score, val_auc_score, test_auc_score


Operators = namedtuple('Operators', ['initialize', 'update', 'summary',
                                     'reset', 'loss', 'accuracy', 'name'])


def make_operators(name, outputs, next_element, iterator,
                   is_training, optimizer=None, mixup='none'):
    (update_op, summary_op, reset_op,
     mean_loss, accuracy, losses) = make_metrics(
        name=name, outputs=outputs, labels=next_element.y,
        is_training=is_training, mixup=mixup
    )

    if name == 'training':
        opt_op = optimizer.minimize(tf.reduce_mean(losses))
        update_op = tf.group(update_op, opt_op)

    ops = Operators(initialize=iterator.initializer, update=update_op,
                    summary=summary_op, reset=reset_op,
                    loss=mean_loss, accuracy=accuracy, name=name)
    return ops


def make_metrics(name, outputs, labels, is_training, mixup='none'):
    if mixup != 'none' and is_training:
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.stop_gradient(labels), logits=outputs
        )
    else:
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=outputs
        )

    total_loss = losses

    with tf.variable_scope('{}_metrics'.format(name)) as vs:
        mean_loss = tf.metrics.mean(losses)
        if mixup != 'none' and is_training:
            accuracy = tf.metrics.accuracy(
                labels=tf.argmax(labels, axis=1),
                predictions=tf.argmax(outputs, axis=1)
            )
        else:
            accuracy = tf.metrics.accuracy(
                labels=labels,
                predictions=tf.argmax(outputs, axis=1)
            )

        local = tf.contrib.framework.get_variables(
            vs, collection=tf.GraphKeys.LOCAL_VARIABLES
        )
        reset_op = tf.variables_initializer(local)
    summary_op = tf.summary.merge([
        tf.summary.scalar('{}/loss'.format(name), mean_loss[0]),
        tf.summary.scalar('{}/accuracy'.format(name), accuracy[0])
    ])
    update_op = tf.group(mean_loss[1], accuracy[1])

    return (update_op, summary_op, reset_op, mean_loss[0], accuracy[0],
            total_loss)


def update_metrics(sess, ops, writer, step, i, ph=None, batch_size=0):
    if ph is None:
        sess.run(ops.initialize)
    else:
        sess.run(ops.initialize, feed_dict={ph: batch_size})
    while True:
        try:
            sess.run(ops.update)
        except tf.errors.OutOfRangeError:
            break

    if writer is not None:
        summary = sess.run(ops.summary)
        writer.add_summary(summary=summary, global_step=step)
    loss, accuracy = sess.run([ops.loss, ops.accuracy])
    sess.run(ops.reset)

    log_metric('{}_loss{}'.format(ops.name, i), loss)
    log_metric('{}_accuracy{}'.format(ops.name, i), accuracy)

    return loss, accuracy


def save_prediction(sess, initializer, output_op, next_element, file_path,
                    data_type):
    sess.run(initializer)
    predictions = []
    labels = []
    name_list = []
    while True:
        try:
            p, y, data_name = sess.run([
                output_op, next_element.y, next_element.name
            ])
            predictions.append(p)
            labels.append(y)
            name_list.extend([n.decode() for n in data_name])
        except tf.errors.OutOfRangeError:
            break
    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)

    # just output class id instead of type name
    df = pd.DataFrame(
        predictions, columns=list(range(predictions.shape[1])), index=name_list
    )
    if data_type != 'test':
        # output the target labels
        df['label'] = labels
    df.to_csv(file_path)


def compute_auc(sess, initializer, logits_op, next_element, model_dir,
                n_classes, name):
    sess.run(initializer)
    predictions = []
    labels = []
    name_list = []
    while True:
        try:
            p, y, data_name = sess.run([
                logits_op, next_element[1], next_element[2]
            ])
            predictions.append(p)
            labels.append(y)
            name_list.extend([n.decode() for n in data_name])
        except tf.errors.OutOfRangeError:
            break
    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Columns are numbers because class names are not easy to prepare.
    df = pd.DataFrame(
        predictions, columns=list(range(n_classes)), index=name_list
    )
    df['label'] = labels
    df.to_csv(model_dir / 'predictions_{}.csv'.format(name))

    if n_classes == 2:
        # set Ia as positive class
        y_true = 1 - labels
        y_score = predictions[:, 0] - predictions[:, 1]
        fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_score)
        auc_score = draw_roc_curve(
            fpr=fpr, tpr=tpr, model_dir=model_dir, i='', name=name
        )
    else:
        # Calculated by one v.s. others for each class
        auc_score = np.empty(n_classes)

        # normalization
        p = predictions - logsumexp(predictions, axis=1, keepdims=True)
        for i in range(n_classes):
            y_true = np.where(labels == i, 1, 0)
            y_score = p[:, i]
            fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_score)
            score = draw_roc_curve(
                fpr=fpr, tpr=tpr, model_dir=model_dir, i=i, name=name
            )

            auc_score[i] = score
    plt.close()

    return auc_score


def draw_roc_curve(fpr, tpr, model_dir, i, name):
    auc_score = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label='AUC:{0:.3f}'.format(auc_score))
    ax.grid()
    ax.legend(loc='best')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    fig.savefig(str(model_dir / '{}_roc_curve{}.png'.format(name, i)))

    return auc_score


@cmd.command()
@click.option('--model-dir', type=click.Path(exists=True))
@click.option('--data-path', type=click.Path(exists=True))
@click.option('--batch-size', type=int, default=10000)
@click.option('--data-type', type=click.Choice(['SimSN', 'PLAsTiCC', 'HSC']))
@click.option('--output-name', type=str)
@click.option('--threads', type=int, default=4)
def predict(model_dir, data_path, batch_size, data_type, output_name, threads):
    """
    predicting with the trained model

    :param model_dir:
    :param data_path:
    :param batch_size:
    :param data_type:
    :param output_name:
    :param threads:
    :return:
    """
    assert output_name is not None

    model_dir = Path(model_dir)

    def parameter(p_name):
        return get_parameter(model_dir=model_dir, name=p_name)

    #
    # load data
    #
    if data_type == 'HSC':
        data = load_hsc_test_data(
            hsc_path=data_path, remove_y=parameter('remove_y') == 'True'
        )
    else:
        data = load_sim_sn_data(sim_sn_path=data_path, use_flux_err2=False,
                                remove_y=parameter('remove_y') == 'True')

    #
    # import model
    #
    if (model_dir / 'moments.npz').exists():
        tmp = np.load(str(model_dir / 'moments.npz'))
        mean = tmp['mean']
        std = tmp['std']
    else:
        n = data['flux'][0].shape[0]
        if parameter('input2') != 'none':
            n *= 2
        mean = np.zeros(n)
        std = np.ones_like(mean)

    #
    # make dataset
    #
    if data_type == 'HSC':
        dummy = np.empty(len(data), dtype=np.int32)
        iterator, next_element = make_dataset(
            x=data, y=dummy, mean=mean, std=std,
            batch_size=batch_size, input1=parameter('input1'),
            input2=parameter('input2'),
            is_hsc=True, threads=threads, is_training=False, use_redshift=False
        )
    else:
        label_map = get_label_map(binary=parameter('binary') == 'True')
        y = np.array([label_map[c] for c in data['sn_type']])
        iterator, next_element = make_dataset(
            x=data, y=y, is_training=False, mean=mean, std=std,
            batch_size=batch_size, input1=parameter('input1'),
            input2=parameter('input2'),
            use_redshift=False, is_hsc=False, threads=threads
        )

    #
    # load model parameters
    #

    parameters = dict(
        binary=parameter('binary') == 'True',
        hidden_size=int(parameter('hidden_size')),
        num_highways=int(parameter('n_highways')),
        output_size=2 if parameter('binary') == 'True' else 3,
        drop_rate=float(parameter('drop_rate')),
        activation=parameter('activation'),
        use_batch_norm=parameter('use_batch_norm') == 'True',
        use_dropout=parameter('use_dropout') == 'True',
        inpt1=parameter('input1'), input2=parameter('input2'),
        remove_y=parameter('remove_y') == 'True'
    )
    print(parameters)

    n_classes = 2 if parameter('binary') == 'True' else 3
    model = Model(
        hidden_size=int(parameter('hidden_size')),
        num_highways=int(parameter('n_highways')),
        output_size=n_classes,
        drop_rate=float(parameter('drop_rate')),
        activation=parameter('activation'),
        use_batch_norm=parameter('use_batch_norm') == 'True',
        use_dropout=parameter('use_dropout') == 'True'
    )
    with tf.device('/gpu:0'):
        logits = model(next_element.x, False)

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
    predictions = np.concatenate(predictions, axis=0)
    df = pd.DataFrame(
        predictions, columns=list(range(n_classes)), index=name_list
    )
    if data_type != 'HSC':
        labels = np.concatenate(labels, axis=0)
        df['label'] = labels

    df.to_csv(model_dir / output_name)


def get_parameter(model_dir, name):
    uri_path = model_dir / 'uri.txt'
    if not uri_path.exists():
        uri_path = model_dir.parent / 'uri.txt'

    with uri_path.open('r') as f:
        uri = f.read()
    if platform.system() == 'Windows':
        path = os.path.join(os.path.dirname(uri), 'params', name)
        path = re.sub(r'\\', '/', path)
        p = urlparse(path)
        with open(p.path[1:], 'r') as f:
            parameter = f.read()
    else:
        path = os.path.join(os.path.dirname(uri), 'params', name)
        with open(path[len('file://'):], 'r') as f:
            parameter = f.read()
    return parameter


def main():
    cmd()


if __name__ == '__main__':
    main()
