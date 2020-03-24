#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from subprocess import call
from time import sleep
from itertools import product

import click
import pandas as pd

__date__ = '21/2/2019'


def write_job(index, hidden_size, drop_rate, norm, drop_na, fill_with_mean):
    job = """#!/usr/bin/env bash
#SBATCH -p gpu-1080ti
#SBATCH --gres=gpu:1
#SBATCH -o "{slurm}"

source activate tf1.10
echo Gpu device: $CUDA_VISIBLE_DEVICES

ROOT=/home/imoto/crest_auto
export PYTHONPATH=/home/imoto/crest_auto/src
DATA_DIR=${{ROOT}}/data/processed

MODEL_DIR=${{ROOT}}/models/180119

SRC_DIR=${{ROOT}}/src
python ${{SRC_DIR}}/hsc_dnn2.py fit \
    --sim-sn-path=${{DATA_DIR}}/train/dataset.h5 \
    --training-cosmos-path=${{DATA_DIR}}/190206/training_cosmos.h5 \
    --test-cosmos-path=${{DATA_DIR}}/190206/test_cosmos.h5 \
    --model-dir=${{MODEL_DIR}}/test{index} --seed=0 \
    --hidden-size={hidden_size} --patience=100 \
    --batch-size=10000 --drop-rate={drop_rate} --debug --{norm} --{drop_na} \
    --fill-with-{mean}
"""

    dir_name = 'sim180119'
    name = dir_name
    job_dir = Path('bash') / dir_name
    if not job_dir.exists():
        job_dir.mkdir(parents=True)
    slurm_dir = Path('slurm') / dir_name
    if not slurm_dir.exists():
        slurm_dir.mkdir(parents=True)

    tmp = '{0}-{1}-{2}'.format(name, index, drop_rate)
    file_path = job_dir / '{}.sh'.format(tmp)
    slurm_path = slurm_dir / '%j-{0}.out'.format(tmp)

    job = job.format(slurm=slurm_path, index=index, drop_rate=drop_rate,
                     hidden_size=hidden_size,
                     norm='norm' if norm else 'no-norm',
                     drop_na='drop-na' if drop_na else 'no-drop-na',
                     mean='mean' if fill_with_mean else 'zero')
    with file_path.open(mode='w') as f:
        f.write(job)

    return file_path


def write_job2_plasticc(index, n_highways, hidden_size, activation, drop_rate,
                        use_batch_norm, use_dropout, flux_err, norm, epochs,
                        input1, input2, sampling_ratio, binary, mixup, threads,
                        batch_size, patience, eval_frequency, end_by_epochs,
                        lr, final_lr, fold, balance):
    job = """#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1 -c{job_threads}
#SBATCH -o "{slurm}"

source activate tf1.13
echo Gpu device: $CUDA_VISIBLE_DEVICES
echo Host: `hostname`

ROOT=/home/imoto/crest_auto2
export PYTHONPATH=${{ROOT}}/src
DATA_DIR=${{ROOT}}/data/processed/190206
TRAIN_DIR=${{DATA_DIR}}/../Simdataset_190410_fillnan
# TRAIN_DIR=${{DATA_DIR}}/../Simdataset_190325_adjustnum

MODEL_DIR=${{ROOT}}/models/searched/plasticc2-bn/cls{n_classes}

SRC_DIR=${{ROOT}}/src
python ${{SRC_DIR}}/hsc_dnn2.py fit-plasticc \
    --sim-sn-path=${{TRAIN_DIR}}/simsn.h5 \
    --training-cosmos-path=${{DATA_DIR}}/training_cosmos2.h5 \
    --test-cosmos-path=${{DATA_DIR}}/test_cosmos2.h5 \
    --model-dir=${{MODEL_DIR}}/{index} --seed=0 \
    --n-highways={n_highways} {use_batch_norm} {use_dropout} \
    --hidden-size={hidden_size} --flux-err={flux_err} \
    --drop-rate={drop_rate} --{norm} \
    --activation={activation} --epochs={epochs} --batch-size={batch_size} \
    --patience={patience} \
    --eval-frequency={eval_frequency} {end_by_epochs} \
    --input1={input1} --input2={input2} \
    --optimizer=adam --adabound-final-lr={final_lr} --lr={lr} \
    --fold={fold} --threads={threads} \
    --sampling-ratio={sampling_ratio} {binary} \
    --mixup={mixup} {balance}
"""

    n_classes = 2 if binary else 3
    dir_name = 'sim190206-searched{}'.format(n_classes)
    job_dir = Path('bash') / dir_name
    if not job_dir.exists():
        job_dir.mkdir(parents=True)
    slurm_dir = Path('slurm') / dir_name
    if not slurm_dir.exists():
        slurm_dir.mkdir(parents=True)

    tmp = 'plasticc{0}-{1}-{2}'.format(n_classes, fold, index)
    file_path = job_dir / '{}.sh'.format(tmp)
    slurm_path = slurm_dir / '%j-{0}.out'.format(tmp)

    job = job.format(
        slurm=slurm_path, job_threads=max(threads // 2, 1), index=index,
        n_highways=n_highways, hidden_size=hidden_size, flux_err=flux_err,
        use_batch_norm='--use-batch-norm' if use_batch_norm else '',
        use_dropout='--use-dropout' if use_dropout else '',
        drop_rate=drop_rate, norm='norm' if norm else 'no-norm',
        activation=activation,
        epochs=epochs, patience=patience, eval_frequency=eval_frequency,
        batch_size=batch_size,
        end_by_epochs='--end-by-epochs' if end_by_epochs else '',
        input1=input1, input2=input2,
        lr=lr, final_lr=final_lr,
        sampling_ratio=sampling_ratio,
        binary='--binary' if binary else '',
        mixup=mixup, threads=threads, fold=fold, n_classes=n_classes,
        balance='--balance' if balance else ''
    )
    with file_path.open(mode='w') as f:
        f.write(job)

    return file_path


def write_job2_hsc(index, n_highways, hidden_size, activation, drop_rate,
                   norm, epochs, patience, eval_frequency, end_by_epochs,
                   input1, input2, mixup, fold, lr, final_lr,
                   threads, flag_index, binary, remove_y, use_batch_norm,
                   use_dropout, mixup_alpha, mixup_beta, warm_start,
                   batch_size, max_batch_size, increase_rate, n_classifiers):
    # HSCの観測データを対象に実験
    job = """#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1 -c{job_threads}
#SBATCH --exclude=ks304,ks305,ks210
#SBATCH -o "{slurm}"

source activate tf1.13
echo Gpu device: $CUDA_VISIBLE_DEVICES
echo Host: `hostname`

ROOT=/home/imoto/crest_auto2
export PYTHONPATH=/home/imoto/crest_auto2/src
DATA_DIR=${{ROOT}}/data/processed
TRAIN_DIR=${{DATA_DIR}}/dataset_190513

MODEL_DIR=${{ROOT}}/models
# OUT_DIR=${{MODEL_DIR}}/adaboost-lr3-lb{n_classifiers}/hsc190513-{flag_index}
OUT_DIR=${{MODEL_DIR}}/searched/hsc2/cls{n_classes}/flag{flag_index}
# OUT_DIR=${{MODEL_DIR}}/grid/hsc/cls{n_classes}/flag{flag_index}

SRC_DIR=${{ROOT}}/src
cd ${{SRC_DIR}}

python ${{SRC_DIR}}/hsc_dnn2.py fit-hsc \
    --sim-sn-path=${{TRAIN_DIR}}/simsn{flag_index}.h5 \
    --hsc-path=${{TRAIN_DIR}}/sndata{flag_index}.h5 \
    --model-dir=${{OUT_DIR}}/{index} --seed=0 \
    --n-highways={n_highways} \
    --hidden-size={hidden_size} --patience={patience} \
    --batch-size={batch_size} --drop-rate={drop_rate} --{norm} \
    --activation={activation} --epochs={epochs} \
    --input1={input1} --input2={input2} \
    --optimizer=adam --adabound-final-lr={final_lr} --lr={lr} \
    --eval-frequency={eval_frequency} {end_by_epochs} \
    --mixup={mixup} --fold={fold} --threads={threads} \
    --{binary} {remove_y} --hsc-index={flag_index} {use_batch_norm} \
    {use_dropout} --n-classifiers={n_classifiers} \
    --mixup-alpha={mixup_alpha} --mixup-beta={mixup_beta} {warm_start} \
    --max-batch-size={max_batch_size} --increase-rate={increase_rate}
"""

    dir_name = 'hsc190513-{}'.format(flag_index)
    name = dir_name
    job_dir = Path('bash') / dir_name
    if not job_dir.exists():
        job_dir.mkdir(parents=True)
    slurm_dir = Path('slurm') / dir_name
    if not slurm_dir.exists():
        slurm_dir.mkdir(parents=True)

    tmp = '{0}-{1}-{2}'.format(name, index, fold)
    file_path = job_dir / '{}.sh'.format(tmp)
    slurm_path = slurm_dir / '%j-{0}.out'.format(tmp)

    job = job.format(
        slurm=slurm_path, index=index, drop_rate=drop_rate,
        n_highways=n_highways, hidden_size=hidden_size,
        norm='norm' if norm else 'no-norm',
        activation=activation, fold=fold,
        epochs=epochs, patience=patience, eval_frequency=eval_frequency,
        end_by_epochs='--end-by-epochs' if end_by_epochs else '',
        input1=input1, input2=input2,
        mixup=mixup, lr=lr, final_lr=final_lr, threads=threads,
        job_threads=max(threads // 2, 1), flag_index=flag_index,
        binary='binary' if binary else 'multi',
        remove_y='--remove-y' if remove_y else '',
        use_batch_norm='--use-batch-norm' if use_batch_norm else '',
        use_dropout='--use-dropout' if use_dropout else '',
        mixup_alpha=mixup_alpha, mixup_beta=mixup_beta,
        warm_start='--warm-start' if warm_start else '',
        max_batch_size=max_batch_size, increase_rate=increase_rate,
        n_classifiers=n_classifiers, batch_size=batch_size,
        n_classes=2 if binary else 3
    )
    with file_path.open(mode='w') as f:
        f.write(job)

    return file_path


def write_job_pauc(index, n_highways, hidden_size, activation, drop_rate,
                   norm, epochs, input1, input2,
                   absolute_magnitude, fold, final_lr,
                   threads, flag_index, binary, remove_y, use_batch_norm,
                   use_dropout, beta):
    # HSCの観測データを対象に実験
    job = """#!/usr/bin/env bash
#SBATCH -p gpu-k80
#SBATCH --gres=gpu:1 -c{job_threads}
#SBATCH -o "{slurm}"

source activate tf1.13
echo Gpu device: $CUDA_VISIBLE_DEVICES
echo Host: `hostname`

ROOT=/home/imoto/crest_auto2
export PYTHONPATH=/home/imoto/crest_auto2/src
DATA_DIR=${{ROOT}}/data/processed
TRAIN_DIR=${{DATA_DIR}}/dataset_190513

MODEL_DIR=${{ROOT}}/models/pauc/hsc190513-{flag_index}

SRC_DIR=${{ROOT}}/src
cd ${{SRC_DIR}}

python ${{SRC_DIR}}/hsc_pauc.py \
    --sim-sn-path=${{TRAIN_DIR}}/simsn{flag_index}.h5 \
    --hsc-path=${{TRAIN_DIR}}/sndata{flag_index}.h5 \
    --model-dir=${{MODEL_DIR}}/{index} --seed=0 \
    --n-highways={n_highways} \
    --hidden-size={hidden_size} --patience=200 \
    --batch-size=100 --drop-rate={drop_rate} --{norm} \
    --activation={activation} --epochs={epochs} \
    --input1={input1} --input2={input2} \
    --optimizer=adam --adabound-final-lr={final_lr} --lr=1e-3 \
    --eval-frequency=10 \
    {abs_mag} --fold={fold} --threads={threads} \
    --{binary} {remove_y} --hsc-index={flag_index} {use_batch_norm} \
    {use_dropout} --beta={beta}
"""

    dir_name = 'pauc-hsc190513-{}'.format(flag_index)
    name = dir_name
    job_dir = Path('bash') / dir_name
    if not job_dir.exists():
        job_dir.mkdir(parents=True)
    slurm_dir = Path('slurm') / dir_name
    if not slurm_dir.exists():
        slurm_dir.mkdir(parents=True)

    tmp = '{0}-{1}-{2}'.format(name, index, fold)
    file_path = job_dir / '{}.sh'.format(tmp)
    slurm_path = slurm_dir / '%j-{0}.out'.format(tmp)

    job = job.format(
        slurm=slurm_path, index=index, drop_rate=drop_rate,
        n_highways=n_highways, hidden_size=hidden_size,
        norm='norm' if norm else 'no-norm',
        activation=activation,
        epochs=epochs, input1=input1, input2=input2,
        abs_mag='--absolute-magnitude' if absolute_magnitude else '',
        fold=fold, final_lr=final_lr, threads=threads,
        job_threads=max(threads // 2, 1), flag_index=flag_index,
        binary='binary' if binary else 'multi',
        remove_y='--remove-y' if remove_y else '',
        use_batch_norm='--use-batch-norm' if use_batch_norm else '',
        use_dropout='--use-dropout' if use_dropout else '',
        beta=beta
    )
    with file_path.open(mode='w') as f:
        f.write(job)

    return file_path


def write_job3(index, hidden_size, drop_rate, epochs, batch_size, outlier_rate,
               blackout_rate, n_highways):
    job = """#!/usr/bin/env bash
#SBATCH -p gpu-1080ti
#SBATCH --gres=gpu:1
#SBATCH -o "{slurm}"

source activate pytorch
echo Gpu device: $CUDA_VISIBLE_DEVICES

ROOT=/home/imoto/crest_auto
export PYTHONPATH=/home/imoto/crest_auto/src
DATA_DIR=${{ROOT}}/data/processed/190206

MODEL_DIR=${{ROOT}}/models/190206

SRC_DIR=${{ROOT}}/src
python ${{SRC_DIR}}/hsc_dnn3.py \
    --sim-sn-path=${{DATA_DIR}}/simsn2.h5 \
    --training-cosmos-path=${{DATA_DIR}}/training_cosmos2.h5 \
    --test-cosmos-path=${{DATA_DIR}}/test_cosmos2.h5 \
    --model-dir=${{MODEL_DIR}}/test{index} --seed=0 \
    --hidden-size={hidden_size}  --dropout-rate={drop_rate} \
    --batch-size={batch_size} --epochs={epochs} --outlier-rate={outlier_rate} \
    --blackout-rate={blackout_rate} --n-highways={n_highways}
"""
    dir_name = 'sim190206'
    name = dir_name
    job_dir = Path('bash') / dir_name
    if not job_dir.exists():
        job_dir.mkdir(parents=True)
    slurm_dir = Path('slurm') / dir_name
    if not slurm_dir.exists():
        slurm_dir.mkdir(parents=True)

    tmp = '{0}-{1}-{2}'.format(name, index, drop_rate)
    file_path = job_dir / '{}.sh'.format(tmp)
    slurm_path = slurm_dir / '%j-{0}.out'.format(tmp)

    job = job.format(slurm=slurm_path, index=index, drop_rate=drop_rate,
                     hidden_size=hidden_size, epochs=epochs,
                     batch_size=batch_size, outlier_rate=outlier_rate,
                     blackout_rate=blackout_rate, n_highways=n_highways)
    with file_path.open(mode='w') as f:
        f.write(job)

    return file_path


def run(new, index, hidden_size, drop_rate, norm, drop_na, fill_with_mean,
        activation, binary, epochs):
    if new:
        file_path = write_job2_plasticc(
            index=index, n_highways=2, hidden_size=hidden_size,
            drop_rate=drop_rate, norm=norm,
            activation=activation, epochs=epochs, flux_err=1,
            input1='magnitude', input2='none',
            sampling_ratio=0.2,
            binary=binary, mixup=False, batch_size=1000, patience=100,
            eval_frequency=10, end_by_epochs=False, lr=1e-2, final_lr=1e0,
            fold=0, threads=1, use_batch_norm=False, use_dropout=False,
            balance=False
        )
    else:
        file_path = write_job(
            index=index, hidden_size=hidden_size, drop_rate=drop_rate,
            norm=norm, drop_na=drop_na, fill_with_mean=fill_with_mean
        )
    call(['sbatch', str(file_path)])
    sleep(1)


def write_search_plasticc(batch_size, lr, epochs, patience, n_trials,
                          input1, input2, threads, eval_frequency, binary,
                          worker_id):
    job = """#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1 -c{job_threads}
#SBATCH --exclude=ks304,ks305,ks210,ks307
#SBATCH -o "{slurm}"

sleep {sleep_time}

source activate tf1.13
echo Gpu device: $CUDA_VISIBLE_DEVICES
echo Host: `hostname`

ROOT=/home/imoto/crest_auto2
export PYTHONPATH=/home/imoto/crest_auto2/src
DATA_DIR=${{ROOT}}/data/processed
TRAIN_DIR=${{DATA_DIR}}/190206

MODEL_DIR=${{ROOT}}/models/hp-search/plasticc3/cls{n_classes}

SRC_DIR=${{ROOT}}/src
cd ${{SRC_DIR}}

python ${{SRC_DIR}}/hsc_search.py search-plasticc \
    --sim-sn-path=${{TRAIN_DIR}}/simsn2.h5 \
    --training-cosmos-path=${{TRAIN_DIR}}/training_cosmos2.h5 \
    --test-cosmos-path=${{TRAIN_DIR}}/test_cosmos2.h5 \
    --model-dir=${{MODEL_DIR}}/{model_name} --seed=0 \
    --batch-size={batch_size} --norm --optimizer=adam --lr={lr} \
    --patience={patience} --n-trials={n_trials} \
    --input1={input1} --input2={input2} --mixup=mixup \
    --eval-frequency={eval_frequency} --epochs={epochs} \
    --threads={threads} --{binary} 
"""

    n_classes = 2 if binary else 3
    dir_name = 'hp-search-plasticc{}'.format(n_classes)
    name = dir_name
    job_dir = Path('bash') / dir_name
    if not job_dir.exists():
        job_dir.mkdir(parents=True)
    slurm_dir = Path('slurm') / dir_name
    if not slurm_dir.exists():
        slurm_dir.mkdir(parents=True)

    model_name = '{input1}-{input2}'.format(input1=input1, input2=input2)
    tmp = '{0}-{2}-{1}'.format(name, model_name, worker_id)
    file_path = job_dir / '{}.sh'.format(tmp)
    slurm_path = slurm_dir / '%j-{0}.out'.format(tmp)

    job = job.format(
        slurm=slurm_path, job_threads=max(threads // 2, 1),
        n_classes=n_classes, model_name=model_name,
        batch_size=batch_size, lr=lr, patience=patience,
        n_trials=n_trials, input1=input1, input2=input2,
        eval_frequency=eval_frequency, epochs=epochs,
        threads=threads, binary='binary' if binary else 'multi',
        sleep_time=max(10 * (worker_id - 1), 1)
    )
    with file_path.open(mode='w') as f:
        f.write(job)

    return file_path


def write_search_hsc(epochs, input1, input2, threads, flag_index, binary,
                     n_trials, remove_y, batch_size, lr, patience,
                     eval_frequency, worker_id=None):
    job = """#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1 -c{job_threads}
#SBATCH --exclude=ks304,ks305,ks210
#SBATCH -o "{slurm}"

sleep {sleep_time}

source activate tf1.13
echo Gpu device: $CUDA_VISIBLE_DEVICES
echo Host: `hostname`

ROOT=/home/imoto/crest_auto2
export PYTHONPATH=/home/imoto/crest_auto2/src
DATA_DIR=${{ROOT}}/data/processed
TRAIN_DIR=${{DATA_DIR}}/dataset_190513

MODEL_DIR=${{ROOT}}/models/hp-search/hsc4/cls{n_classes}/flag{flag_index}

SRC_DIR=${{ROOT}}/src
cd ${{SRC_DIR}}

python ${{SRC_DIR}}/hsc_search.py search-hsc \
    --sim-sn-path=${{TRAIN_DIR}}/simsn{flag_index}.h5 \
    --hsc-path=${{TRAIN_DIR}}/sndata{flag_index}_v2.h5 \
    --model-dir=${{MODEL_DIR}}/{model_name} --seed=0 \
    --norm --epochs={epochs} \
    --batch-size={batch_size} --optimizer=adam --lr={lr} \
    --patience={patience} --n-trials={n_trials} \
    --input1={input1} --input2={input2} --mixup=mixup \
    --eval-frequency={eval_frequency} --epochs={epochs} \
    --threads={threads} --{binary} --flag-index={flag_index} {remove_y}
"""

    n_classes = 2 if binary else 3
    dir_name = 'hp-search-hsc{}'.format(n_classes)
    name = dir_name
    job_dir = Path('bash') / dir_name
    if not job_dir.exists():
        job_dir.mkdir(parents=True)
    slurm_dir = Path('slurm') / dir_name
    if not slurm_dir.exists():
        slurm_dir.mkdir(parents=True)

    model_name = '{input1}-{input2}'.format(input1=input1, input2=input2)
    if remove_y:
        model_name = model_name + '-remove-y'
    if worker_id is None:
        tmp = '{0}-{2}-{1}'.format(name, model_name, flag_index)
    else:
        tmp = '{0}-{3}-{2}-{1}'.format(name, model_name, flag_index, worker_id)
    file_path = job_dir / '{}.sh'.format(tmp)
    slurm_path = slurm_dir / '%j-{0}.out'.format(tmp)

    job = job.format(
        slurm=slurm_path, job_threads=max(threads // 2, 1),
        n_classes=n_classes, model_name=model_name,
        epochs=epochs, input1=input1, input2=input2,
        batch_size=batch_size, lr=lr, patience=patience,
        eval_frequency=eval_frequency,
        threads=threads, flag_index=flag_index,
        binary='binary' if binary else 'multi', n_trials=n_trials,
        remove_y='--remove-y' if remove_y else '',
        sleep_time=max(10 * (worker_id - 1), 1)
    )
    with file_path.open(mode='w') as f:
        f.write(job)

    return file_path


@click.group()
def cmd():
    pass


@cmd.command()
@click.option('--new', is_flag=True)
@click.option('--index', type=str)
@click.option('--hidden-size', type=int, default=300)
@click.option('--drop-rate', type=float, default=0.01)
@click.option('--norm/--no-norm', is_flag=True, default=True)
@click.option('--drop-na/--no-drop-na', is_flag=True, default=False)
@click.option('--fill-with-mean/--fill-with-zero', is_flag=True, default=False)
@click.option('--activation',
              type=click.Choice(['linear', 'relu', 'sigmoid', 'tanh', 'elu']))
@click.option('--binary', is_flag=True)
@click.option('--epochs', type=int, default=500)
def fit(new, index, hidden_size, drop_rate, norm, drop_na, fill_with_mean,
        activation, binary, epochs):
    run(new=new, index=index, hidden_size=hidden_size, drop_rate=drop_rate,
        norm=norm, drop_na=drop_na, fill_with_mean=fill_with_mean,
        activation=activation, binary=binary, epochs=epochs)


@cmd.command()
@click.option('--index', type=str)
@click.option('--n-highways', type=int, default=2)
@click.option('--hidden-size', type=int, default=974)
@click.option('--drop-rate', type=float, default=0.1456)
@click.option('--epochs', type=int, default=500)
@click.option('--patience', type=int, default=20)
@click.option('--eval-frequency', type=int, default=10)
@click.option('--end-by-epochs', is_flag=True)
@click.option('--norm/--no-norm', is_flag=True, default=True)
@click.option('--activation',
              type=click.Choice(['linear', 'relu', 'sigmoid', 'tanh', 'elu']))
@click.option('--flux-err', type=int, default=2)
@click.option('--input1',
              type=click.Choice(['flux', 'magnitude', 'absolute-magnitude',
                                 'scaled-flux', 'scaled-magnitude']))
@click.option('--input2',
              type=click.Choice(['none', 'flux', 'magnitude',
                                 'absolute-magnitude',
                                 'scaled-flux', 'scaled-magnitude']),
              default='none')
@click.option('--mixup', type=click.Choice(['none', 'mixup', 'weighted']),
              default='none')
@click.option('--fold', type=int, default=-1)
@click.option('--lr', type=float, default=1e-3)
@click.option('--final-lr', type=float, default=1e2)
@click.option('--threads', type=int, default=10)
@click.option('--binary/--multi', is_flag=True, default=True)
@click.option('--balance', is_flag=True)
@click.option('--use-batch-norm', is_flag=True)
@click.option('--use-dropout', is_flag=True)
@click.option('--batch-size', type=int, default=1000)
@click.option('--sampling-ratio', type=float, default=0.1)
def dnn2_plasticc(index, n_highways, hidden_size, drop_rate, epochs, patience,
                  eval_frequency, end_by_epochs, norm, activation, flux_err,
                  input1, input2, mixup, fold, lr, final_lr, threads,
                  binary, balance, use_batch_norm, use_dropout,
                  batch_size, sampling_ratio):
    path = write_job2_plasticc(
        index=index, n_highways=n_highways, hidden_size=hidden_size,
        drop_rate=drop_rate, epochs=epochs, flux_err=flux_err, norm=norm,
        activation=activation, input1=input1, input2=input2,
        sampling_ratio=sampling_ratio, binary=binary, mixup=mixup,
        batch_size=batch_size, patience=patience,
        eval_frequency=eval_frequency, end_by_epochs=end_by_epochs,
        lr=lr, final_lr=final_lr, fold=fold, threads=threads,
        use_batch_norm=use_batch_norm, use_dropout=use_dropout,
        balance=balance
    )
    call(['sbatch', str(path)])


@cmd.command()
@click.option('--index', type=str)
@click.option('--n-highways', type=int, default=2)
@click.option('--hidden-size', type=int, default=974)
@click.option('--drop-rate', type=float, default=0.1456)
@click.option('--epochs', type=int, default=500)
@click.option('--patience', type=int, default=20)
@click.option('--eval-frequency', type=int, default=10)
@click.option('--end-by-epochs', is_flag=True)
@click.option('--norm/--no-norm', is_flag=True, default=True)
@click.option('--activation',
              type=click.Choice(['linear', 'relu', 'sigmoid', 'tanh', 'elu']))
@click.option('--input1',
              type=click.Choice(['flux', 'magnitude', 'absolute-magnitude',
                                 'scaled-flux', 'scaled-magnitude']))
@click.option('--input2',
              type=click.Choice(['none', 'flux', 'magnitude', 
                                 'absolute-magnitude',
                                 'scaled-flux', 'scaled-magnitude']),
              default='none')
@click.option('--mixup', type=click.Choice(['none', 'mixup', 'weighted']),
              default='none')
@click.option('--fold', type=int, default=-1)
@click.option('--lr', type=float, default=1e-3)
@click.option('--final-lr', type=float, default=1e2)
@click.option('--threads', type=int, default=10)
@click.option('--flag-index', type=int)
@click.option('--binary/--multi', is_flag=True, default=True)
@click.option('--remove-y', is_flag=True)
@click.option('--mixup-alpha', type=float, default=2)
@click.option('--mixup-beta', type=float, default=2)
@click.option('--use-batch-norm', is_flag=True)
@click.option('--warm-start', is_flag=True)
@click.option('--batch-size', type=int, default=10000)
@click.option('--max-batch-size', type=int, default=10000)
@click.option('--increase-rate', type=int, default=2)
@click.option('--n-classifiers', type=int, default=2)
def dnn2_hsc(index, n_highways, hidden_size, drop_rate, epochs, patience,
             eval_frequency, end_by_epochs, norm, activation, input1, input2, 
             mixup, fold, lr, final_lr, threads,
             flag_index, binary, remove_y, mixup_alpha, mixup_beta,
             use_batch_norm, warm_start, batch_size, max_batch_size,
             increase_rate, n_classifiers):
    # result = Path(
    #     '/home/imoto/crest_auto2/models/hp-search/'
    #     'hsc190513-{}'.format(flag_index)
    # ) / index / 'result.csv'
    # df = pd.read_csv(result, header=[0, 1], index_col=0)
    # df = df[df[('state', 'Unnamed: 2_level_1')] == 'TrialState.COMPLETE']
    # best_index = df[('value', 'Unnamed: 3_level_1')].idxmin()
    # params = df.loc[best_index]['params']

    # n_highways = params['num_highways']
    # hidden_size = params['hidden_size']
    # drop_rate = params['drop_rate']
    # activation = params['activation']
    # use_batch_norm = params['use_batch_norm']
    # use_dropout = params['use_drop_out']
    # print(params)

    if fold < 0:
        for i in range(5):
            path = write_job2_hsc(
                index=index, n_highways=n_highways,
                hidden_size=hidden_size,
                drop_rate=drop_rate, epochs=epochs, norm=norm,
                activation=activation, input1=input1, input2=input2,
                mixup=mixup, fold=i, lr=lr, final_lr=final_lr, threads=threads,
                flag_index=flag_index, binary=binary, remove_y=remove_y,
                use_batch_norm=use_batch_norm, use_dropout=False,
                mixup_alpha=mixup_alpha, mixup_beta=mixup_beta,
                warm_start=warm_start, max_batch_size=max_batch_size,
                increase_rate=increase_rate, n_classifiers=n_classifiers,
                batch_size=batch_size, patience=patience,
                eval_frequency=eval_frequency, end_by_epochs=end_by_epochs
            )
            call(['sbatch', str(path)])
            sleep(20)
    else:
        path = write_job2_hsc(
            index=index, n_highways=n_highways, hidden_size=hidden_size,
            drop_rate=drop_rate, epochs=epochs, norm=norm,
            activation=activation, input1=input1, input2=input2, mixup=mixup,
            fold=fold, lr=lr, final_lr=final_lr, threads=threads,
            flag_index=flag_index, binary=binary, remove_y=remove_y,
            use_batch_norm=use_batch_norm, use_dropout=False,
            mixup_alpha=mixup_alpha, mixup_beta=mixup_beta,
            warm_start=warm_start, max_batch_size=max_batch_size,
            increase_rate=increase_rate, n_classifiers=n_classifiers,
            batch_size=batch_size, patience=patience,
            eval_frequency=eval_frequency, end_by_epochs=end_by_epochs
        )
        call(['sbatch', str(path)])
        sleep(15)


@cmd.command()
@click.option('--epochs', type=int, default=500)
@click.option('--patience', type=int, default=100)
@click.option('--eval-frequency', type=int, default=10)
@click.option('--end-by-epochs', is_flag=True)
@click.option('--norm/--no-norm', is_flag=True, default=True)
@click.option('--mixup', type=click.Choice(['none', 'mixup', 'weighted']),
              default='mixup')
@click.option('--fold', type=int, default=-1)
@click.option('--lr', type=float, default=1e-3)
@click.option('--final-lr', type=float, default=1e2)
@click.option('--threads', type=int, default=10)
@click.option('--flag-index', type=int)
@click.option('--binary/--multi', is_flag=True, default=True)
@click.option('--remove-y', is_flag=True)
@click.option('--batch-size', type=int, default=10000)
@click.option('--max-batch-size', type=int, default=10000)
def hsc_grid(epochs, patience, eval_frequency, end_by_epochs, norm,
             mixup, fold, lr, final_lr, threads, flag_index, binary, remove_y,
             batch_size, max_batch_size):
    input1 = 'absolute-magnitude'
    input2 = 'scaled-flux'

    drops = [0.015, 0.035, 0.055]
    n_highs = [1, 2, 3, 4, 5]
    n_hiddens = [100, 300, 500]
    batch_norms = [True, False]
    acts = ['relu', 'sigmoid']
    for (drop_rate, n_highways, hidden_size,
         use_batch_norm, activation) in product(drops, n_highs, n_hiddens,
                                                batch_norms, acts):
        name = ('{n_highways}-{hidden_size}-{activation}-{drop_rate}-'
                '{use_batch_norm}').format(
            n_highways=n_highways, hidden_size=hidden_size,
            activation=activation, drop_rate=drop_rate,
            use_batch_norm=use_batch_norm
        )
        path = write_job2_hsc(
            index=name, n_highways=n_highways, hidden_size=hidden_size,
            drop_rate=drop_rate, epochs=epochs, norm=norm,
            activation=activation, input1=input1, input2=input2, mixup=mixup,
            fold=fold, lr=lr, final_lr=final_lr, threads=threads,
            flag_index=flag_index, binary=binary, remove_y=remove_y,
            use_batch_norm=use_batch_norm, use_dropout=False,
            mixup_alpha=2, mixup_beta=2,
            warm_start=False, max_batch_size=max_batch_size,
            increase_rate=2, n_classifiers=1,
            batch_size=batch_size, patience=patience,
            eval_frequency=eval_frequency, end_by_epochs=end_by_epochs
        )
        call(['sbatch', str(path)])
        sleep(1)


def get_plasticc_searched_parameters(result_dir, n_classes, model_name,
                                     return_score=False):
    result_path = (Path(result_dir) / 'cls{}'.format(n_classes) /
                   model_name / 'result.csv')
    return _get_searched_parameters(result_path=result_path,
                                    return_score=return_score)


def get_hsc_searched_parameters(result_dir, n_classes, flag_index, model_name,
                                return_score=False):
    result_path = (Path(result_dir) /
                   'cls{0}/flag{1}'.format(n_classes, flag_index) /
                   model_name / 'result.csv')
    return _get_searched_parameters(result_path=result_path,
                                    return_score=return_score)


def _get_searched_parameters(result_path, return_score=False):
    df = pd.read_csv(result_path, header=[0, 1], index_col=0)
    df = df[df[('state', 'Unnamed: 2_level_1')] == 'TrialState.COMPLETE']
    best_index = df[('value', 'Unnamed: 3_level_1')].idxmin()
    params = df.loc[best_index]['params']

    n_highways = params['num_highways']
    hidden_size = params['hidden_size']
    drop_rate = params['drop_rate']
    activation = params['activation']
    use_batch_norm = params['use_batch_norm']
    use_dropout = False
    # print(params)

    d = dict(
        n_highways=n_highways, hidden_size=hidden_size, drop_rate=drop_rate,
        activation=activation, use_batch_norm=use_batch_norm,
        use_dropout=use_dropout
    )
    if return_score:
        score = df[('value', 'Unnamed: 3_level_1')].min()
        return d, score, best_index
    return d


@cmd.command()
@click.option('--epochs', type=int, default=500)
@click.option('--patience', type=int, default=100)
@click.option('--eval-frequency', type=int, default=10)
@click.option('--end-by-epochs', is_flag=True)
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
@click.option('--fold', type=int, default=-1)
@click.option('--lr', type=float, default=1e-3)
@click.option('--final-lr', type=float, default=1e2)
@click.option('--threads', type=int, default=1)
@click.option('--binary/--multi', is_flag=True, default=True)
@click.option('--batch-size', type=int, default=1000)
@click.option('--sampling-ratio', type=float, default=0.1)
def plasticc_searched(epochs, patience, eval_frequency, end_by_epochs,
                      flux_err, norm,
                      input1, input2, fold, lr, final_lr, threads,
                      binary, batch_size, sampling_ratio):
    model_name = '{}-{}'.format(input1, input2)
    n_classes = 2 if binary else 3
    model_dir = '/home/imoto/crest_auto2/models'
    parameters, score, best_index = get_plasticc_searched_parameters(
        result_dir=Path(model_dir) / 'hp-search/plasticc3',
        n_classes=n_classes, model_name=model_name, return_score=True
    )
    print('score:', score)
    print('parameters:', parameters)

    # model_name += '3'
    # 探索の途中で最適化を行うので、best parametersであるかを確認する
    out_dir = (Path(model_dir) / 'searched/plasticc2-bn' /
               'cls{}'.format(n_classes))
    out_dir = out_dir / model_name / str(fold)
    score_data = out_dir / 'score_data.json'
    if score_data.exists():
        with score_data.open() as f:
            tmp = json.load(f)
        if tmp['best_index'] == best_index:
            print('already optimized with best parameters({}'.format(tmp))
            return
    # ファイルがなかったかbest parametersが更新された
    d = {'score': float(score), 'best_index': int(best_index),
         'n_highways': int(parameters['n_highways']),
         'hidden_size': int(parameters['hidden_size']),
         'drop_rate': float(parameters['drop_rate']),
         'activation': str(parameters['activation']),
         'use_batch_norm': bool(parameters['use_batch_norm']),
         'use_dropout': bool(parameters['use_dropout'])}
    if not score_data.parent.exists():
        score_data.parent.mkdir(parents=True)
    with score_data.open('w') as f:
        json.dump(d, f, indent=4)

    path = write_job2_plasticc(
        index=model_name, flux_err=flux_err, norm=norm, epochs=epochs,
        input1=input1, input2=input2, sampling_ratio=sampling_ratio,
        binary=binary, mixup='mixup', threads=threads, batch_size=batch_size,
        patience=patience, eval_frequency=eval_frequency,
        end_by_epochs=end_by_epochs, lr=lr, final_lr=final_lr, fold=fold,
        balance=False, **parameters
    )
    call(['sbatch', str(path)])
    sleep(1)


@cmd.command()
@click.option('--dependency', type=str)
@click.option('--input1',
              type=click.Choice(['flux', 'magnitude', 'absolute-magnitude',
                                 'scaled-flux', 'scaled-magnitude']))
@click.option('--input2',
              type=click.Choice(['none', 'flux', 'magnitude',
                                 'absolute-magnitude',
                                 'scaled-flux', 'scaled-magnitude']),
              default='none')
@click.option('--binary/--multi', is_flag=True)
def delay_plasticc_searched(dependency, input1, input2, binary):
    dir_name = 'plasticc-{}'.format(2 if binary else 3)
    name = dir_name
    job_dir = Path('bash') / dir_name
    if not job_dir.exists():
        job_dir.mkdir(parents=True)
    slurm_dir = Path('slurm') / dir_name
    if not slurm_dir.exists():
        slurm_dir.mkdir(parents=True)

    model_name = '{}-{}'.format(input1, input2)
    tmp = 'delay-{0}-{1}'.format(name, model_name)
    file_path = job_dir / '{}.sh'.format(tmp)
    slurm_path = slurm_dir / '%j-{0}.out'.format(tmp)

    job = """#!/usr/bin/env bash
#SBATCH -p cpu
#SBATCH --dependency=afterany:{dependency}
#SBATCH -o "{slurm}"

source activate py3.6
cd /home/imoto/crest_auto2/src/job

python job_hsc_dnn.py plasticc-searched --{binary} --threads=8 --patience=100 \
    --input1={input1} --input2={input2} --fold=-1 
""".format(dependency=dependency, input1=input1, input2=input2,
           binary='binary' if binary else 'multi', slurm=slurm_path)

    with file_path.open(mode='w') as f:
        f.write(job)

    call(['sbatch', str(file_path)])


@cmd.command()
@click.option('--epochs', type=int, default=500)
@click.option('--patience', type=int, default=100)
@click.option('--eval-frequency', type=int, default=10)
@click.option('--end-by-epochs', is_flag=True)
@click.option('--input1',
              type=click.Choice(['flux', 'magnitude', 'absolute-magnitude',
                                 'scaled-flux', 'scaled-magnitude']))
@click.option('--input2',
              type=click.Choice(['none', 'flux', 'magnitude',
                                 'absolute-magnitude',
                                 'scaled-flux', 'scaled-magnitude']),
              default='none')
@click.option('--fold', type=int, default=-1)
@click.option('--lr', type=float, default=1e-3)
@click.option('--final-lr', type=float, default=1e2)
@click.option('--threads', type=int, default=1)
@click.option('--flag-index', type=int)
@click.option('--binary/--multi', is_flag=True, default=True)
@click.option('--remove-y', is_flag=True)
@click.option('--batch-size', type=int, default=1000)
def hsc_searched(epochs, patience, eval_frequency, end_by_epochs,
                 input1, input2, fold, lr, final_lr, threads, flag_index,
                 binary, remove_y, batch_size):
    model_name = '{}-{}'.format(input1, input2)
    if remove_y:
        model_name += '-remove-y'
    n_classes = 2 if binary else 3
    model_dir = '/home/imoto/crest_auto2/models'
    parameters, score, best_index = get_hsc_searched_parameters(
        result_dir=Path(model_dir) / 'hp-search/hsc4', n_classes=n_classes,
        flag_index=flag_index, model_name=model_name, return_score=True
    )
    print('score:', score)
    print('parameters:', parameters)

    # model_name += '2'
    # 探索の途中で最適化を行うので、best parametersであるかを確認する
    out_dir = (Path(model_dir) / 'searched/hsc2' /
               'cls{}'.format(n_classes))
    out_dir = out_dir / 'flag{}'.format(flag_index) / model_name / str(fold)
    score_data = out_dir / 'score_data.json'
    if score_data.exists():
        with score_data.open() as f:
            tmp = json.load(f)
        if tmp['best_index'] == best_index:
            print('already optimized with best parameters({}'.format(tmp))
            return
    # ファイルがなかったかbest parametersが更新された
    d = {'score': float(score), 'best_index': int(best_index),
         'n_highways': int(parameters['n_highways']),
         'hidden_size': int(parameters['hidden_size']),
         'drop_rate': float(parameters['drop_rate']),
         'activation': str(parameters['activation']),
         'use_batch_norm': bool(parameters['use_batch_norm']),
         'use_dropout': bool(parameters['use_dropout'])}
    if not score_data.parent.exists():
        score_data.parent.mkdir(parents=True)
    with score_data.open('w') as f:
        json.dump(d, f, indent=4)

    path = write_job2_hsc(
        index=model_name, norm=True, epochs=epochs, patience=patience,
        eval_frequency=eval_frequency, end_by_epochs=end_by_epochs,
        input1=input1, input2=input2, mixup='mixup', fold=fold,
        lr=lr, final_lr=final_lr, threads=threads, flag_index=flag_index,
        binary=binary, remove_y=remove_y, mixup_alpha=2, mixup_beta=2,
        warm_start=False, batch_size=batch_size, max_batch_size=batch_size - 1,
        increase_rate=2, n_classifiers=1, **parameters
    )
    call(['sbatch', str(path)])
    sleep(1)


@cmd.command()
@click.option('--dependency', type=str)
@click.option('--input1',
              type=click.Choice(['flux', 'magnitude', 'absolute-magnitude',
                                 'scaled-flux', 'scaled-magnitude']))
@click.option('--input2',
              type=click.Choice(['none', 'flux', 'magnitude',
                                 'absolute-magnitude',
                                 'scaled-flux', 'scaled-magnitude']),
              default='none')
@click.option('--remove-y', is_flag=True)
@click.option('--flag-index', type=int)
def delay_hsc_searched(dependency, input1, input2, remove_y, flag_index):
    dir_name = 'hsc190513-{}'.format(flag_index)
    name = dir_name
    job_dir = Path('bash') / dir_name
    if not job_dir.exists():
        job_dir.mkdir(parents=True)
    slurm_dir = Path('slurm') / dir_name
    if not slurm_dir.exists():
        slurm_dir.mkdir(parents=True)

    model_name = '{}-{}'.format(input1, input2)
    if remove_y:
        model_name += '-remove-y'
    tmp = 'delay-{0}-{1}'.format(name, model_name)
    file_path = job_dir / '{}.sh'.format(tmp)
    slurm_path = slurm_dir / '%j-{0}.out'.format(tmp)

    job = """#!/usr/bin/env bash
#SBATCH -p cpu
#SBATCH --dependency=afterany:{dependency}
#SBATCH -o "{slurm}"

source activate py3.6
cd /home/imoto/crest_auto2/src/job

python job_hsc_dnn.py hsc-searched --binary --threads=8 --patience=100 \
    --input1={input1} --input2={input2} {remove_y} \
    --flag-index={flag_index} --fold=-1 
""".format(dependency=dependency, input1=input1, input2=input2,
           flag_index=flag_index, remove_y='--remove-y' if remove_y else '',
           slurm=slurm_path)

    with file_path.open(mode='w') as f:
        f.write(job)

    call(['sbatch', str(file_path)])


@cmd.command()
@click.option('--epochs', type=int, default=500)
@click.option('--patience', type=int, default=100)
@click.option('--eval-frequency', type=int, default=10)
@click.option('--end-by-epochs', is_flag=True)
@click.option('--input1',
              type=click.Choice(['flux', 'magnitude', 'absolute-magnitude',
                                 'scaled-flux', 'scaled-magnitude']))
@click.option('--input2',
              type=click.Choice(['none', 'flux', 'magnitude',
                                 'absolute-magnitude',
                                 'scaled-flux', 'scaled-magnitude']),
              default='none')
@click.option('--fold', type=int, default=-1)
@click.option('--lr', type=float, default=1e-3)
@click.option('--final-lr', type=float, default=1e2)
@click.option('--threads', type=int, default=8)
@click.option('--binary/--multi', is_flag=True, default=True)
@click.option('--remove-y', is_flag=True)
@click.option('--batch-size', type=int, default=1000)
@click.option('--n-observations', type=int)
def hsc_n_observed(epochs, patience, eval_frequency, end_by_epochs,
                   input1, input2, fold, lr, final_lr, threads,
                   binary, remove_y, batch_size, n_observations):
    model_name = '{}-{}'.format(input1, input2)
    if remove_y:
        model_name += '-remove-y'
    n_classes = 2 if binary else 3
    model_dir = '/home/imoto/crest_auto2/models'
    parameters, score, best_index = get_hsc_searched_parameters(
        result_dir=Path(model_dir) / 'hp-search/hsc4', n_classes=n_classes,
        flag_index=0, model_name=model_name, return_score=True
    )
    print('score:', score)
    print('parameters:', parameters)

    model_name += '-{}'.format(n_observations)
    # 探索の途中で最適化を行うので、best parametersであるかを確認する
    out_dir = (Path(model_dir) / 'n_observations/hsc' /
               'cls{}'.format(n_classes))
    out_dir = out_dir / model_name / str(fold)
    score_data = out_dir / 'score_data.json'
    if score_data.exists():
        with score_data.open() as f:
            tmp = json.load(f)
        if tmp['best_index'] == best_index:
            print('already optimized with best parameters({}'.format(tmp))
            return
    # ファイルがなかったかbest parametersが更新された
    d = {'score': float(score), 'best_index': int(best_index),
         'n_highways': int(parameters['n_highways']),
         'hidden_size': int(parameters['hidden_size']),
         'drop_rate': float(parameters['drop_rate']),
         'activation': str(parameters['activation']),
         'use_batch_norm': bool(parameters['use_batch_norm']),
         'use_dropout': bool(parameters['use_dropout'])}
    if not score_data.parent.exists():
        score_data.parent.mkdir(parents=True)
    with score_data.open('w') as f:
        json.dump(d, f, indent=4)

    dir_name = 'hsc190513-n-observations'
    name = dir_name
    job_dir = Path('bash') / dir_name
    if not job_dir.exists():
        job_dir.mkdir(parents=True)
    slurm_dir = Path('slurm') / dir_name
    if not slurm_dir.exists():
        slurm_dir.mkdir(parents=True)

    tmp = '{0}-{1}'.format(name, fold)
    file_path = job_dir / '{}.sh'.format(tmp)
    slurm_path = slurm_dir / '%j-{0}.out'.format(tmp)

    job = """#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1 -c{job_threads}
#SBATCH --exclude=ks304,ks305,ks210
#SBATCH -o "{slurm}"

source activate tf1.13
echo Gpu device: $CUDA_VISIBLE_DEVICES
echo Host: `hostname`

ROOT=/home/imoto/crest_auto2
export PYTHONPATH=/home/imoto/crest_auto2/src
DATA_DIR=${{ROOT}}/data/processed
TRAIN_DIR=${{DATA_DIR}}/dataset_190513

MODEL_DIR=${{ROOT}}/models
OUT_DIR=${{MODEL_DIR}}/n_observations/hsc/cls{n_classes}

SRC_DIR=${{ROOT}}/src
cd ${{SRC_DIR}}

python ${{SRC_DIR}}/hsc_dnn2.py fit-hsc \
    --sim-sn-path=${{TRAIN_DIR}}/simsn0.h5 \
    --hsc-path=${{TRAIN_DIR}}/sndata0.h5 \
    --model-dir=${{OUT_DIR}}/{model_name} --seed=0 \
    --n-highways={n_highways} \
    --hidden-size={hidden_size} --patience={patience} \
    --batch-size={batch_size} --drop-rate={drop_rate} --norm \
    --activation={activation} --epochs={epochs} \
    --input1={input1} --input2={input2} \
    --optimizer=adam --adabound-final-lr={final_lr} --lr={lr} \
    --eval-frequency={eval_frequency} {end_by_epochs} \
    --mixup={mixup} --fold={fold} --threads={threads} \
    --{binary} {remove_y} --hsc-index={flag_index} {use_batch_norm} \
    {use_dropout} --n-observations={n_observations}
""".format(
        slurm=slurm_path, model_name=model_name, drop_rate=d['drop_rate'],
        n_highways=d['n_highways'], hidden_size=d['hidden_size'],
        activation=d['activation'], fold=fold,
        epochs=epochs, patience=patience, eval_frequency=eval_frequency,
        end_by_epochs='--end-by-epochs' if end_by_epochs else '',
        input1=input1, input2=input2,
        mixup='mixup', lr=lr, final_lr=final_lr, threads=threads,
        job_threads=max(threads // 2, 1), flag_index=0,
        binary='binary' if binary else 'multi',
        remove_y='--remove-y' if remove_y else '',
        use_batch_norm='--use-batch-norm' if d['use_batch_norm'] else '',
        use_dropout='--use-dropout' if d['use_dropout'] else '',
        batch_size=batch_size, n_classes=2 if binary else 3,
        n_observations=n_observations
    )
    with file_path.open(mode='w') as f:
        f.write(job)
    call(['sbatch', str(file_path)])
    sleep(1)


@cmd.command()
@click.option('--index', type=str)
@click.option('--n-highways', type=int, default=2)
@click.option('--hidden-size', type=int, default=974)
@click.option('--drop-rate', type=float, default=0.1456)
@click.option('--epochs', type=int, default=500)
@click.option('--norm/--no-norm', is_flag=True, default=True)
@click.option('--activation',
              type=click.Choice(['linear', 'relu', 'sigmoid', 'tanh', 'elu']))
@click.option('--input1',
              type=click.Choice(['flux', 'magnitude',
                                 'scaled-flux', 'scaled-magnitude',
                                 'cumulative']))
@click.option('--input2',
              type=click.Choice(['none', 'flux', 'magnitude',
                                 'scaled-flux', 'scaled-magnitude',
                                 'cumulative']),
              default='none')
@click.option('--absolute-magnitude', is_flag=True)
@click.option('--fold', type=int, default=-1)
@click.option('--final-lr', type=float, default=1e2)
@click.option('--threads', type=int, default=10)
@click.option('--flag-index', type=int)
@click.option('--binary/--multi', is_flag=True, default=True)
@click.option('--remove-y', is_flag=True)
@click.option('--beta', type=float)
def pauc_hsc(index, n_highways, hidden_size, drop_rate, epochs,
             norm, activation, input1, input2,
             absolute_magnitude, fold, final_lr, threads,
             flag_index, binary, remove_y, beta):
    if fold < 0:
        for i in range(5):
            path = write_job_pauc(
                index=index, n_highways=n_highways,
                hidden_size=hidden_size,
                drop_rate=drop_rate, epochs=epochs, norm=norm,
                activation=activation, input1=input1, input2=input2,
                absolute_magnitude=absolute_magnitude,
                fold=i, final_lr=final_lr, threads=threads,
                flag_index=flag_index, binary=binary, remove_y=remove_y,
                use_batch_norm=False, use_dropout=False, beta=beta
            )
            call(['sbatch', str(path)])
            sleep(20)
    else:
        path = write_job_pauc(
            index=index, n_highways=n_highways,
            hidden_size=hidden_size,
            drop_rate=drop_rate, epochs=epochs, norm=norm,
            activation=activation, input1=input1, input2=input2,
            absolute_magnitude=absolute_magnitude,
            fold=fold, final_lr=final_lr, threads=threads,
            flag_index=flag_index, binary=binary, remove_y=remove_y,
            use_batch_norm=False, use_dropout=False, beta=beta
        )
        call(['sbatch', str(path)])
        sleep(1)


@cmd.command()
@click.option('--batch-size', type=int, default=1000)
@click.option('--lr', type=float, default=1e-3)
@click.option('--epochs', type=int, default=500)
@click.option('--patience', type=int, default=200)
@click.option('--n-trials', type=int, default=50)
@click.option('--input1',
              type=click.Choice(['flux', 'magnitude', 'absolute-magnitude',
                                 'scaled-flux', 'scaled-magnitude']))
@click.option('--input2',
              type=click.Choice(['none', 'flux', 'magnitude',
                                 'absolute-magnitude',
                                 'scaled-flux', 'scaled-magnitude']),
              default='none')
@click.option('--threads', type=int, default=1)
@click.option('--eval-frequency', type=int, default=10)
@click.option('--flag-index', type=int, default=-1)
@click.option('--binary/--multi', is_flag=True, default=True)
@click.option('--remove-y', is_flag=True)
@click.option('--worker-id', type=int, default=None)
def search_hsc(batch_size, lr, epochs, patience, n_trials, input1, input2,
               threads, eval_frequency, flag_index, binary, remove_y,
               worker_id):
    if flag_index < 0:
        for flag_index in range(5):
            path = write_search_hsc(
                batch_size=batch_size, lr=lr, epochs=epochs, patience=patience,
                input1=input1, input2=input2, threads=threads,
                flag_index=flag_index, binary=binary, n_trials=n_trials,
                remove_y=remove_y, eval_frequency=eval_frequency,
                worker_id=worker_id
            )
            call(['sbatch', str(path)])
            sleep(1)
    else:
        path = write_search_hsc(
            batch_size=batch_size, lr=lr, epochs=epochs, patience=patience,
            input1=input1, input2=input2, threads=threads,
            flag_index=flag_index, binary=binary, n_trials=n_trials,
            remove_y=remove_y, eval_frequency=eval_frequency,
            worker_id=worker_id
        )
        call(['sbatch', str(path)])
        sleep(1)


@cmd.command()
@click.option('--batch-size', type=int, default=1000)
@click.option('--lr', type=float, default=1e-3)
@click.option('--epochs', type=int, default=500)
@click.option('--patience', type=int, default=200)
@click.option('--n-trials', type=int, default=50)
@click.option('--input1',
              type=click.Choice(['flux', 'magnitude', 'absolute-magnitude',
                                 'scaled-flux', 'scaled-magnitude']))
@click.option('--input2',
              type=click.Choice(['none', 'flux', 'magnitude',
                                 'absolute-magnitude',
                                 'scaled-flux', 'scaled-magnitude']),
              default='none')
@click.option('--threads', type=int, default=1)
@click.option('--eval-frequency', type=int, default=10)
@click.option('--binary/--multi', is_flag=True, default=True)
@click.option('--worker-id', type=int, default=None)
def search_plasticc(batch_size, lr, epochs, patience, n_trials, input1,
                    input2, threads, eval_frequency, binary, worker_id):
    path = write_search_plasticc(
        batch_size=batch_size, lr=lr, epochs=epochs, patience=patience,
        input1=input1, input2=input2, threads=threads,
        binary=binary, n_trials=n_trials, eval_frequency=eval_frequency,
        worker_id=worker_id
    )
    call(['sbatch', str(path)])
    sleep(1)


@cmd.command()
@click.option('--batch-size', type=int, default=1000)
@click.option('--lr', type=float, default=1e-2)
@click.option('--epochs', type=int, default=500)
@click.option('--patience', type=int, default=200)
@click.option('--n-trials', type=int, default=50)
@click.option('--input1',
              type=click.Choice(['flux', 'magnitude', 'absolute-magnitude',
                                 'scaled-flux', 'scaled-magnitude']))
@click.option('--input2',
              type=click.Choice(['none', 'flux', 'magnitude',
                                 'absolute-magnitude',
                                 'scaled-flux', 'scaled-magnitude']),
              default='none')
@click.option('--threads', type=int, default=1)
@click.option('--eval-frequency', type=int, default=10)
@click.option('--flag-index', type=int, default=-1)
@click.option('--binary/--multi', is_flag=True, default=True)
@click.option('--remove-y', is_flag=True)
def search_hsc_optimize(batch_size, lr, epochs, patience, n_trials,
                        input1, input2, threads, eval_frequency, flag_index,
                        binary, remove_y):
    args = ('--sim-sn-path=${{SIM_PATH}} --hsc-path=${{HSC_PATH}} --seed=0 '
            '--norm --epochs={epochs} --batch-size={batch_size} '
            '--optimizer=adam --lr={lr} --patience={patience} '
            '--mixup=mixup '
            '--eval-frequency={eval_frequency} --threads={threads} --multi '
            '--flag-index={flag_index}').format(
        epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
        eval_frequency=eval_frequency, threads=threads, flag_index=flag_index
    )

    n_classes = 2 if binary else 3
    dir_name = 'hp-search-hsc{}'.format(n_classes)
    name = dir_name
    job_dir = Path('bash') / dir_name
    if not job_dir.exists():
        job_dir.mkdir(parents=True)
    slurm_dir = Path('slurm') / dir_name
    if not slurm_dir.exists():
        slurm_dir.mkdir(parents=True)

    model_name = '{input1}-{input2}'.format(input1=input1, input2=input2)
    if remove_y:
        model_name = model_name + '-remove-y'
    tmp = '{0}-{2}-{1}'.format(name, model_name, flag_index)
    file_path = job_dir / '{}.sh'.format(tmp)
    slurm_path = slurm_dir / '%j-{0}.out'.format(tmp)

    job = """#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1 -c{job_threads}
#SBATCH -o "{slurm}"

source activate tf1.13
echo Gpu device: $CUDA_VISIBLE_DEVICES
echo Host: `hostname`

ROOT=/home/imoto/crest_auto2
export PYTHONPATH=/home/imoto/crest_auto2/src
DATA_DIR=${{ROOT}}/data/processed
TRAIN_DIR=${{DATA_DIR}}/dataset_190513

MODEL_DIR=${{ROOT}}/models/hp-search/hsc/cls3/flag{flag_index}

SRC_DIR=${{ROOT}}/src
cd ${{SRC_DIR}}

SIM_PATH=${{TRAIN_DIR}}/simsn{flag_index}.h5
HSC_PATH=${{TRAIN_DIR}}/sndata{flag_index}.h5

python ${{SRC_DIR}}/hsc_search.py search-hsc \
    --model-dir=${{MODEL_DIR}}/{model_name} \
    --input1={input1} --input2={input2}  {remove_y} {args} \
    --n-trials={n_trials} 
    
OUT_DIR=${{ROOT}}/models/searched/hsc/cls{n_classes}/flag{flag_index}

python ${{SRC_DIR}}/hsc_dnn2.py fit-hsc-optimize \
    --model-dir=${{OUT_DIR}}/{model_name}2 \
    --input1={input1} --input2={input2}  {remove_y} {args} \
    --boosting=none --n-classifiers=1 \
    --mixup-alpha=2 --mixup-beta=2 \
    --max-batch-size={max_batch_size} --increase-rate=2
""".format(flag_index=flag_index, job_threads=max(threads // 2, 1),
           args=args, slurm=slurm_path, input1=input1, input2=input2,
           model_name=model_name, remove_y='--remove-y' if remove_y else '',
           n_classes=n_classes, n_trials=n_trials,
           max_batch_size=batch_size - 1)

    with file_path.open(mode='w') as f:
        f.write(job)

    call(['sbatch', str(file_path)])
    sleep(1)


def main():
    cmd()


if __name__ == '__main__':
    main()
