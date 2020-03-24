#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import os
import platform
from itertools import product
from pathlib import Path
# noinspection PyUnresolvedReferences
from typing import Optional, Callable, Any

import click
import mlflow
import numpy as np
import optuna
import sklearn.utils
import tensorflow as tf
# noinspection PyProtectedMember
from mlflow import log_param, log_params
from optuna.distributions import BaseDistribution
from optuna.samplers import TPESampler
from optuna.samplers.tpe.sampler import default_gamma, default_weights
from optuna.structs import FrozenTrial
from optuna.study import InTrialStudy
from sklearn.model_selection import train_test_split

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

from hsc.loader import load_hsc_data, load_plasticc_data
from hsc.model import Model
from hsc.dataset import (build_hsc_dataset, build_plasticc_dataset,
                         compute_moments, Data, InputSetting, InputData)
from hsc_sn_type import (OptimizerSetting, LoopSetting, make_operators,
                         update_metrics, get_label_map)

__date__ = '01/8月/2019'


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
            {'drop_rate': drop_rate, 'num_highways': n_highways,
             'hidden_size': hidden_size, 'use_batch_norm': True,
             'activation': activation}
            for drop_rate, hidden_size, n_highways, activation in product(
                [0.005, 0.035], [100, 300], [1, 3, 5],
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


@click.group()
def cmd():
    pass


@cmd.command()
@click.option('--sim-sn-path', type=click.Path(exists=True))
@click.option('--training-cosmos-path', type=click.Path(exists=True))
@click.option('--test-cosmos-path', type=click.Path(exists=True))
@click.option('--model-dir', type=click.Path())
@click.option('--batch-size', type=int, default=10000)
@click.option('--lr', type=float, default=1e-3)
@click.option('--optimizer',
              type=click.Choice(['adam', 'momentum', 'adabound', 'amsbound']),
              default='adam')
@click.option('--adabound-gamma', type=float, default=1e-3)
@click.option('--adabound-final-lr', type=float, default=0.1)
@click.option('--seed', type=int, default=0)
@click.option('--epochs', type=int, default=500)
@click.option('--patience', type=int, default=50)
@click.option('--n-trials', type=int, default=15)
@click.option('--norm/--no-norm', is_flag=True, default=True)
@click.option('--flux-err', type=int, default=2)
@click.option('--input1',
              type=click.Choice(['flux', 'magnitude', 'absolute-magnitude',
                                 'scaled-flux', 'scaled-magnitude']))
@click.option('--input2',
              type=click.Choice(['none', 'flux', 'magnitude',
                                 'absolute-magnitude',
                                 'scaled-flux', 'scaled-magnitude']),
              default='none')
@click.option('--mixup', type=click.Choice(['none', 'mixup', 'manifold']),
              default='none')
@click.option('--threads', type=int, default=4)
@click.option('--eval-frequency', type=int, default=10)
@click.option('--binary/--multi', is_flag=True, default=True)
@click.option('--mixup-alpha', type=float, default=2)
@click.option('--mixup-beta', type=float, default=2)
def search_plasticc(sim_sn_path, training_cosmos_path, test_cosmos_path,
                    model_dir, batch_size, optimizer, adabound_gamma,
                    adabound_final_lr, lr, seed, epochs, patience, n_trials,
                    norm, flux_err, input1, input2, mixup, threads,
                    eval_frequency, binary,
                    mixup_alpha, mixup_beta):
    storage = 'sqlite:///{}/example.db'.format(model_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if platform.system() == 'Windows':
        tmp = (Path(__file__).parents[1] / 'mlruns' /
               'search-plasticc-classification' / 'mlruns')
        uri = str(tmp.absolute().as_uri())
        # uri = 'file://' + str(tmp.absolute())
    else:
        tmp = (Path(__file__).parents[1] / 'mlruns' /
               'search-plasticc-classification' / 'mlruns')
        uri = str(tmp.absolute().as_uri())
    mlflow.set_tracking_uri(uri)

    n_classes = 2 if binary else 3
    name = '{n_classes}-{input1}-{input2}'.format(
        n_classes=n_classes, input1=input1, input2=input2
    )
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
        batch_size=batch_size, mixup=mixup, mixup_alpha=mixup_alpha,
        mixup_beta=mixup_beta, balance=False
    )
    input_data = InputData(
        training_data=None, validation_data=None, test_data=None,
        mean=None, std=None, input1=input1, input2=input2, remove_y=False,
        is_hsc=False, n_classes=n_classes, input_setting=input_setting
    )

    optimizer_setting = OptimizerSetting(
        name=optimizer, lr=lr, gamma=adabound_gamma,
        final_lr=adabound_final_lr
    )
    loop_setting = LoopSetting(epochs=epochs, patience=patience,
                               eval_frequency=eval_frequency,
                               end_by_epochs=False)

    print('loading data')
    # 今までのflux_errなら1, 新しいflux_errなら2
    sim_sn, training_cosmos, _ = load_plasticc_data(
        sim_sn_path=sim_sn_path, training_cosmos_path=training_cosmos_path,
        test_cosmos_path=test_cosmos_path, use_flux_err2=flux_err == 2
    )
    sim_sn = sklearn.utils.shuffle(sim_sn, random_state=seed)
    training_cosmos = sklearn.utils.shuffle(training_cosmos,
                                            random_state=seed + 1)
    for data in (sim_sn, training_cosmos):
        for key in ('flux', 'flux_err'):
            tmp = data[key]
            data[key][np.isnan(tmp)] = 0

    # クラスラベルを数字にする
    label_map = get_label_map(binary=binary)
    sim_sn_y = np.array([label_map[c] for c in sim_sn['sn_type']])
    training_cosmos_y = np.array([label_map[c]
                                  for c in training_cosmos['sn_type']])

    sim_x1, sim_x2, sim_y1, sim_y2 = train_test_split(
        sim_sn, sim_sn_y, test_size=0.3, random_state=42, stratify=sim_sn_y
    )
    cosmos_x1, cosmos_x2, cosmos_y1, cosmos_y2 = train_test_split(
        training_cosmos, training_cosmos_y, test_size=0.3, random_state=43,
        stratify=training_cosmos_y
    )

    sim_dev_x, sim_val_x, sim_dev_y, sim_val_y = train_test_split(
        sim_x1, sim_y1, test_size=0.3, random_state=44, stratify=sim_y1
    )
    cosmos_dev_x, cosmos_val_x, cosmos_dev_y, cosmos_val_y = train_test_split(
        cosmos_x1, cosmos_y1, test_size=0.3, random_state=45,
        stratify=cosmos_y1
    )

    weight = np.asarray([0.9 / len(sim_dev_y)] * len(sim_dev_y) +
                        [0.1 / len(cosmos_dev_y)] * len(cosmos_dev_y))
    training_data = Data(x=np.hstack([sim_dev_x, cosmos_dev_x]),
                         y=np.hstack([sim_dev_y, cosmos_dev_y]),
                         weight=weight)
    validation_data = Data(x=np.hstack([sim_val_x, cosmos_val_x]),
                           y=np.hstack([sim_val_y, cosmos_val_y]))
    test_data = Data(x=np.hstack([sim_x2, cosmos_x2]),
                     y=np.hstack([sim_y2, cosmos_y2]))
    input_data.training_data = training_data
    input_data.validation_data = validation_data
    input_data.test_data = test_data

    mean, std = compute_moments(
        train_data=training_data.x, input1=input1, input2=input2, norm=norm,
        use_redshift=False, is_hsc=False, threads=threads
    )
    input_data.mean, input_data.std = mean, std

    for i in range(n_trials):
        study.optimize(
            lambda trial: objective_plasticc(
                trial=trial, input_data=input_data,
                optimizer_setting=optimizer_setting, seed=seed,
                loop_setting=loop_setting, normalization=norm,
                threads=threads, binary=binary, sim_sn_path=sim_sn_path,
                training_cosmos_path=training_cosmos_path, flux_err=flux_err
            ),
            n_trials=1
        )

        df = study.trials_dataframe()
        df.to_csv(os.path.join(model_dir, 'result.csv'))


def objective_plasticc(trial, input_data, optimizer_setting,
                       seed, loop_setting, normalization, threads, binary,
                       sim_sn_path, training_cosmos_path, flux_err):
    with mlflow.start_run():
        num_highways = trial.suggest_int('num_highways', 1, 5)
        use_batch_norm = trial.suggest_categorical(
            'use_batch_norm', [False, True]
        )
        use_drop_out = False
        activation = trial.suggest_categorical(
            'activation', ['linear', 'relu', 'sigmoid', 'tanh']
        )
        drop_rate = trial.suggest_loguniform('drop_rate', 5e-4, 0.25)
        hidden_size = trial.suggest_int('hidden_size', 50, 1000)

        log_param('sim_sn', sim_sn_path)
        log_param('training_cosmos', training_cosmos_path)
        log_param('seed', seed)
        log_param('hidden_size', hidden_size)
        log_param('drop_rate', drop_rate)
        log_param('num_highways', num_highways)
        log_param('use_batch_norm', use_batch_norm)
        log_param('use_drop_out', use_drop_out)
        log_param('activation', activation)
        log_param('binary', binary)
        log_param('normalization', normalization)

        log_param('flux_err', flux_err)

        log_params(input_data.parameters)
        log_params(optimizer_setting.parameters)
        log_params(loop_setting.parameters)

        best_validation_score = objective_single(
            input_data=input_data, optimizer_setting=optimizer_setting,
            loop_setting=loop_setting, threads=threads, trial=trial,
            num_highways=num_highways,
            use_batch_norm=use_batch_norm, use_drop_out=use_drop_out,
            activation=activation, drop_rate=drop_rate,
            hidden_size=hidden_size
        )
    return best_validation_score


@cmd.command()
@click.option('--sim-sn-path', type=click.Path(exists=True))
@click.option('--hsc-path', type=click.Path(exists=True))
@click.option('--model-dir', type=click.Path())
@click.option('--batch-size', type=int, default=10000)
@click.option('--lr', type=float, default=1e-3)
@click.option('--optimizer',
              type=click.Choice(['adam', 'momentum', 'adabound', 'amsbound']),
              default='adam')
@click.option('--adabound-gamma', type=float, default=1e-3)
@click.option('--adabound-final-lr', type=float, default=0.1)
@click.option('--seed', type=int, default=0)
@click.option('--epochs', type=int, default=500)
@click.option('--patience', type=int, default=50)
@click.option('--n-trials', type=int, default=15)
@click.option('--norm/--no-norm', is_flag=True, default=True)
@click.option('--input1',
              type=click.Choice(['flux', 'magnitude', 'absolute-magnitude',
                                 'scaled-flux', 'scaled-magnitude']))
@click.option('--input2',
              type=click.Choice(['none', 'flux', 'magnitude',
                                 'absolute-magnitude',
                                 'scaled-flux', 'scaled-magnitude']),
              default='none')
@click.option('--mixup', type=click.Choice(['none', 'mixup', 'manifold']),
              default='none')
@click.option('--threads', type=int, default=4)
@click.option('--eval-frequency', type=int, default=10)
@click.option('--binary/--multi', is_flag=True, default=True)
@click.option('--task-name', type=str)
@click.option('--remove-y', is_flag=True)
@click.option('--mixup-alpha', type=float, default=2)
@click.option('--mixup-beta', type=float, default=2)
def search_hsc(sim_sn_path, hsc_path, model_dir, batch_size, optimizer,
               adabound_gamma, adabound_final_lr, lr, seed, epochs, patience,
               n_trials, norm, input1, input2,
               mixup, threads, eval_frequency, binary, task_name, remove_y,
               mixup_alpha, mixup_beta):
    storage = 'sqlite:///{}/example.db'.format(model_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if platform.system() == 'Windows':
        tmp = (Path(__file__).parents[1] / 'mlruns' /
               'search-hsc-classification' / 'mlruns')
        uri = str(tmp.absolute().as_uri())
        # uri = 'file://' + str(tmp.absolute())
    else:
        tmp = (Path(__file__).absolute().parents[1] / 'mlruns' /
               'search-hsc-classification' / 'mlruns')
        uri = str(tmp.absolute().as_uri())
    mlflow.set_tracking_uri(uri)
    mlflow.set_tracking_uri(uri)

    n_classes = 2 if binary else 3
    name = '{n_classes}-{task_name}-{input1}-{input2}'.format(
        n_classes=n_classes, task_name=task_name, input1=input1, input2=input2
    )
    if remove_y:
        name += '-remove-y'
    mlflow.set_experiment(name)

    print(model_dir)
    db_path = os.path.join(model_dir, 'example.db')
    sampler = MyTPESampler()
    if os.path.exists(db_path):
        study = optuna.Study(study_name='study190513', storage=storage,
                             sampler=sampler)
    else:
        study = optuna.create_study(study_name='study190513', storage=storage,
                                    sampler=sampler)

    input_setting = InputSetting(
        batch_size=batch_size, mixup=mixup,
        mixup_alpha=mixup_alpha, mixup_beta=mixup_beta
    )
    input_data = InputData(
        training_data=None, validation_data=None, test_data=None,
        mean=None, std=None, input1=input1, input2=input2,
        remove_y=remove_y, is_hsc=True, n_classes=n_classes,
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

    # クラスラベルを数字にする
    label_map = get_label_map(binary=binary)
    sim_sn_y = np.array([label_map[c] for c in sim_sn['sn_type']])

    sim_x1, sim_x2, sim_y1, sim_y2 = train_test_split(
        sim_sn, sim_sn_y, test_size=0.3, random_state=42, stratify=sim_sn_y
    )
    sim_dev_x, sim_val_x, sim_dev_y, sim_val_y = train_test_split(
        sim_x1, sim_y1, test_size=0.3, random_state=44, stratify=sim_y1
    )

    training_data = Data(x=sim_dev_x, y=sim_dev_y)
    validation_data = Data(x=sim_val_x, y=sim_val_y)
    test_data = Data(x=sim_x2, y=sim_y2)
    input_data.training_data = training_data
    input_data.validation_data = validation_data
    input_data.test_data = test_data

    mean, std = compute_moments(
        train_data=training_data.x, input1=input1, input2=input2, norm=norm,
        use_redshift=False, is_hsc=True, threads=threads
    )
    input_data.mean, input_data.std = mean, std

    for i in range(n_trials):
        study.optimize(
            lambda trial: objective_hsc(
                trial=trial, sim_sn_path=sim_sn_path, hsc_path=hsc_path,
                optimizer_setting=optimizer_setting, seed=seed,
                loop_setting=loop_setting, normalization=norm,
                threads=threads, binary=binary, input_data=input_data
            ),
            n_trials=1
        )

        df = study.trials_dataframe()
        df.to_csv(os.path.join(model_dir, 'result.csv'))


def objective_hsc(trial, sim_sn_path, hsc_path, optimizer_setting,
                  seed, loop_setting, normalization, threads, binary,
                  input_data):
    with mlflow.start_run():
        num_highways = trial.suggest_int('num_highways', 1, 5)
        use_batch_norm = trial.suggest_categorical(
            'use_batch_norm', [False, True]
        )
        use_drop_out = False
        activation = trial.suggest_categorical(
            'activation', ['linear', 'relu', 'sigmoid', 'tanh']
        )
        drop_rate = trial.suggest_loguniform('drop_rate', 5e-4, 0.25)
        hidden_size = trial.suggest_int('hidden_size', 50, 1000)

        log_param('sim_sn', sim_sn_path)
        log_param('hsc_path', hsc_path)
        log_param('seed', seed)
        log_param('hidden_size', hidden_size)
        log_param('drop_rate', drop_rate)
        log_param('num_highways', num_highways)
        log_param('use_batch_norm', use_batch_norm)
        log_param('use_drop_out', use_drop_out)
        log_param('activation', activation)
        log_param('binary', binary)
        log_param('normalization', normalization)

        log_params(input_data.parameters)
        log_params(optimizer_setting.parameters)
        log_params(loop_setting.parameters)

        best_validation_score = objective_single(
            input_data=input_data, optimizer_setting=optimizer_setting,
            loop_setting=loop_setting, threads=threads, trial=trial,
            num_highways=num_highways,
            use_batch_norm=use_batch_norm, use_drop_out=use_drop_out,
            activation=activation, drop_rate=drop_rate,
            hidden_size=hidden_size
        )
    return best_validation_score


def objective_single(input_data,  optimizer_setting, loop_setting, threads,
                     trial, num_highways, use_batch_norm, use_drop_out,
                     activation, drop_rate, hidden_size):
    best_validation_score = run(
        trial=trial, num_highways=num_highways,
        use_batch_norm=use_batch_norm,
        use_drop_out=use_drop_out, activation=activation,
        drop_rate=drop_rate, hidden_size=hidden_size,
        loop_setting=loop_setting,
        optimizer_setting=optimizer_setting,
        threads=threads, input_data=input_data
    )
    return best_validation_score


def run(trial, num_highways, use_batch_norm, use_drop_out, activation,
        drop_rate, hidden_size, loop_setting, optimizer_setting,
        input_data, threads):
    with tf.Graph().as_default() as graph:
        if input_data.is_hsc:
            dataset_ops = build_hsc_dataset(input_data=input_data,
                                            threads=threads)
        else:
            dataset_ops = build_plasticc_dataset(
                input_data=input_data, sampling_ratio=0.1, threads=threads
            )

        model = Model(
            hidden_size=hidden_size, output_size=input_data.n_classes,
            drop_rate=drop_rate, num_highways=num_highways,
            use_batch_norm=use_batch_norm, use_dropout=use_drop_out,
            activation=activation
        )

        dev_iterator = dataset_ops.training_iterator
        dev_element = dataset_ops.training_element
        val_iterator = dataset_ops.validation_iterator
        val_element = dataset_ops.validation_element
        test_iterator = dataset_ops.test_iterator
        test_element = dataset_ops.test_element

        with tf.device('/gpu:0'):
            dev_output = model(dev_element.x, True)
            val_output = model(val_element.x, False)
            test_output = model(test_element.x, False)

        optimizer = optimizer_setting.get_optimizer()

        dev_ops = make_operators(
            name='training', outputs=dev_output, next_element=dev_element,
            iterator=dev_iterator, optimizer=optimizer, is_training=True,
            mixup=input_data.input_setting.mixup
        )
        val_ops = make_operators(
            name='validation', outputs=val_output,
            next_element=val_element, iterator=val_iterator,
            is_training=False, mixup='none'
        )
        test_ops = make_operators(
            name='test', outputs=test_output,
            next_element=test_element, iterator=test_iterator,
            is_training=False, mixup='none'
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
                    sess=sess, ops=dev_ops, writer=None, step=None, i=''
                )

                if step % loop_setting.eval_frequency == 0:
                    val_loss, val_accuracy = update_metrics(
                        sess=sess, ops=val_ops, writer=None, step=None, i=''
                    )
                    val_accuracy_list.append(val_accuracy)

                    span = len(val_accuracy_list) * loop_setting.eval_frequency
                    if span >= loop_setting.patience:
                        test_loss, test_accuracy = update_metrics(
                            sess=sess, ops=test_ops, writer=None,
                            step=None, i=''
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
