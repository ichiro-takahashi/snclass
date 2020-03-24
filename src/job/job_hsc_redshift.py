#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

from pathlib import Path
from subprocess import call
from time import sleep

import click

__date__ = '22/8月/2019'


def write_hsc(model_name, n_highways, hidden_size, activation, drop_rate,
              norm, epochs, patience, eval_frequency, end_by_epochs,
              input1, input2, fold, lr, final_lr,
              threads, flag_index, remove_y, use_batch_norm, batch_size,
              target_type):
    dir_name = 'hsc190513-{}-redshift'.format(flag_index)
    name = dir_name
    job_dir = Path('bash') / dir_name
    if not job_dir.exists():
        job_dir.mkdir(parents=True)
    slurm_dir = Path('slurm') / dir_name
    if not slurm_dir.exists():
        slurm_dir.mkdir(parents=True)

    tmp = '{0}-{1}-{2}'.format(name, fold, model_name)
    file_path = job_dir / '{}.sh'.format(tmp)
    slurm_path = slurm_dir / '%j-{0}.out'.format(tmp)

    # HSCの観測データを対象に実験
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

MODEL_DIR=${{ROOT}}/models
OUT_DIR=${{MODEL_DIR}}/redshift/hsc/flag{flag_index}

SRC_DIR=${{ROOT}}/src
cd ${{SRC_DIR}}

python ${{SRC_DIR}}/hsc_redshift.py learn \
    --sim-sn-path=${{TRAIN_DIR}}/simsn{flag_index}.h5 \
    --hsc-path=${{TRAIN_DIR}}/sndata{flag_index}_v2.h5 \
    --model-dir=${{OUT_DIR}}/{model_name} --seed=0 \
    --n-highways={n_highways} \
    --hidden-size={hidden_size} --patience={patience} \
    --batch-size={batch_size} --drop-rate={drop_rate} --{norm} \
    --activation={activation} --epochs={epochs} \
    --input1={input1} --input2={input2} \
    --optimizer=adam --adabound-final-lr={final_lr} --lr={lr} \
    --eval-frequency={eval_frequency} {end_by_epochs} \
    --fold={fold} --threads={threads} \
    {remove_y} --hsc-index={flag_index} {use_batch_norm} \
    --target-{target_type}
""".format(
        slurm=slurm_path, model_name=model_name, drop_rate=drop_rate,
        n_highways=n_highways, hidden_size=hidden_size,
        norm='norm' if norm else 'no-norm',
        activation=activation, fold=fold,
        epochs=epochs, patience=patience, eval_frequency=eval_frequency,
        end_by_epochs='--end-by-epochs' if end_by_epochs else '',
        input1=input1, input2=input2,
        lr=lr, final_lr=final_lr, threads=threads,
        job_threads=max(threads // 2, 1), flag_index=flag_index,
        remove_y='--remove-y' if remove_y else '',
        use_batch_norm='--use-batch-norm' if use_batch_norm else '',
        batch_size=batch_size, target_type=target_type
    )
    with file_path.open(mode='w') as f:
        f.write(job)

    return file_path


@click.group()
def cmd():
    pass


@cmd.command()
@click.option('--model-name', type=str)
@click.option('--n-highways', type=int, default=2)
@click.option('--hidden-size', type=int, default=974)
@click.option('--drop-rate', type=float, default=0.1456)
@click.option('--epochs', type=int, default=500)
@click.option('--patience', type=int, default=100)
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
@click.option('--fold', type=int, default=-1)
@click.option('--lr', type=float, default=1e-3)
@click.option('--final-lr', type=float, default=1e2)
@click.option('--threads', type=int, default=10)
@click.option('--flag-index', type=int)
@click.option('--remove-y', is_flag=True)
@click.option('--use-batch-norm', is_flag=True)
@click.option('--batch-size', type=int, default=10000)
@click.option('--target-distmod/--target-redshift', is_flag=True)
def redshift_hsc(model_name, n_highways, hidden_size, drop_rate, epochs,
                 patience, eval_frequency, end_by_epochs, norm, activation,
                 input1, input2, fold, lr, final_lr, threads,
                 flag_index, remove_y, use_batch_norm, batch_size,
                 target_distmod):
    target_type = 'distmod' if target_distmod else 'redshift'

    if fold < 0:
        for i in range(5):
            path = write_hsc(
                model_name=model_name, n_highways=n_highways,
                hidden_size=hidden_size,
                drop_rate=drop_rate, epochs=epochs, norm=norm,
                activation=activation, input1=input1, input2=input2,
                fold=i, lr=lr, final_lr=final_lr, threads=threads,
                flag_index=flag_index, remove_y=remove_y,
                use_batch_norm=use_batch_norm,
                batch_size=batch_size, patience=patience,
                eval_frequency=eval_frequency, end_by_epochs=end_by_epochs,
                target_type=target_type
            )
            call(['sbatch', str(path)])
            sleep(20)
    else:
        path = write_hsc(
            model_name=model_name, n_highways=n_highways,
            hidden_size=hidden_size,
            drop_rate=drop_rate, epochs=epochs, norm=norm,
            activation=activation, input1=input1, input2=input2,
            fold=fold, lr=lr, final_lr=final_lr, threads=threads,
            flag_index=flag_index, remove_y=remove_y,
            use_batch_norm=use_batch_norm,
            batch_size=batch_size, patience=patience,
            eval_frequency=eval_frequency, end_by_epochs=end_by_epochs,
            target_type=target_type
        )
        call(['sbatch', str(path)])
        sleep(15)


@cmd.command()
@click.option('--patience', type=int, default=100)
@click.option('--eval-frequency', type=int, default=10)
@click.option('--norm/--no-norm', is_flag=True, default=True)
@click.option('--input1',
              type=click.Choice(['flux', 'magnitude', 'absolute-magnitude',
                                 'scaled-flux', 'scaled-magnitude']))
@click.option('--input2',
              type=click.Choice(['none', 'flux', 'magnitude',
                                 'absolute-magnitude',
                                 'scaled-flux', 'scaled-magnitude']),
              default='none')
@click.option('--lr', type=float, default=1e-3)
@click.option('--final-lr', type=float, default=1e2)
@click.option('--threads', type=int, default=8)
@click.option('--flag-index', type=int)
@click.option('--remove-y', is_flag=True)
@click.option('--batch-size', type=int, default=1000)
@click.option('--target-distmod/--target-redshift', is_flag=True)
@click.option('--n-trials', type=int, default=50)
@click.option('--worker-id', type=int, default=-1)
def redshift_hsc_search(patience, eval_frequency, norm,
                        input1, input2, lr, final_lr, threads,
                        flag_index, remove_y, batch_size, target_distmod,
                        n_trials, worker_id):
    dir_name = 'hsc190513-{}-redshift-search'.format(flag_index)
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
    target = 'distmod' if target_distmod else 'redshift'
    model_name += '-{}'.format(target)

    tmp = '{0}-{2}-{1}'.format(name, model_name, worker_id)
    file_path = job_dir / '{}.sh'.format(tmp)
    slurm_path = slurm_dir / '%j-{0}.out'.format(tmp)

    job = """#!/usr/bin/env bash
#SBATCH -p gpu-1080ti
#SBATCH --gres=gpu:1 -c{job_threads}
#SBATCH --exclude=ks210,ks213
#SBATCH -o "{slurm}"

source activate tf1.13
echo Gpu device: $CUDA_VISIBLE_DEVICES
echo Host: `hostname`

sleep {sleep_time}

ROOT=/home/imoto/crest_auto2
export PYTHONPATH=/home/imoto/crest_auto2/src
DATA_DIR=${{ROOT}}/data/processed
TRAIN_DIR=${{DATA_DIR}}/dataset_190513

MODEL_DIR=${{ROOT}}/models
OUT_DIR=${{MODEL_DIR}}/redshift-hp-search3/hsc/flag{flag_index}

SRC_DIR=${{ROOT}}/src
cd ${{SRC_DIR}}

python ${{SRC_DIR}}/hsc_redshift.py search \
    --sim-sn-path=${{TRAIN_DIR}}/simsn{flag_index}.h5 \
    --hsc-path=${{TRAIN_DIR}}/sndata{flag_index}_v2.h5 \
    --model-dir=${{OUT_DIR}}/{model_name} --seed=0 \
    --patience={patience} --batch-size={batch_size} --{norm} \
    --input1={input1} --input2={input2} \
    --optimizer=adam --adabound-final-lr={final_lr} --lr={lr} \
    --eval-frequency={eval_frequency} --threads={threads} \
    {remove_y} --hsc-index={flag_index} --target-{target_type} \
    --n-trials={n_trials}
""".format(
        slurm=slurm_path, model_name=model_name,
        norm='norm' if norm else 'no-norm',
        patience=patience, eval_frequency=eval_frequency,
        input1=input1, input2=input2,
        lr=lr, final_lr=final_lr, threads=threads,
        job_threads=max(threads // 2, 1), flag_index=flag_index,
        remove_y='--remove-y' if remove_y else '',
        batch_size=batch_size, target_type=target, n_trials=n_trials,
        sleep_time=max((worker_id - 1) * 10, 1)
    )
    with file_path.open(mode='w') as f:
        f.write(job)

    call(['sbatch', str(file_path)])
    sleep(1)


def main():
    cmd()


if __name__ == '__main__':
    main()
