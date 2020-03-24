#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
訓練とテストのデータ比をいろいろ変更して実行する
"""

from pathlib import Path
from subprocess import call
from time import sleep
from itertools import product

import click

__date__ = '13/2/2019'


def write_job(test_ratio, seed):
    job = """#!/usr/bin/env bash
#SBATCH -p cpu -c16
#SBATCH --nodelist=ks100
#SBATCH -o "{slurm}"

hostname
ulimit -a
export OMP_NUM_THREADS=16

source activate py3.6

ROOT=/home/imoto/crest_auto
export PYTHONPATH=/home/imoto/crest_auto/src

cd ${{ROOT}}/src
python plasticc_lgbm_data_ratio.py \
    --model-dir=../models/plasticc/lgbm/data_ratio_selected{test_ratio}_{seed} \
    --feature-dir=../data/processed/4th \
    --use-meta-prediction --test-ratio={test_ratio} --seed={seed} \
    --exgal-feature-size1=90 --exgal-feature-size2=80
"""

    dir_name = 'data_ratio'
    name = dir_name
    job_dir = Path('script') / dir_name
    if not job_dir.exists():
        job_dir.mkdir(parents=True)
    slurm_dir = Path('slurm') / dir_name
    if not slurm_dir.exists():
        slurm_dir.mkdir(parents=True)

    tmp = '{0}-{1}-{2}'.format(name, test_ratio, seed)
    file_path = job_dir / '{}.sh'.format(tmp)
    slurm_path = slurm_dir / '%j-{}.out'.format(tmp)

    job = job.format(slurm=slurm_path, test_ratio=test_ratio, seed=seed)
    with file_path.open('w') as f:
        f.write(job)

    return file_path


def run(test_ratio, seed):
    file_path = write_job(test_ratio=test_ratio, seed=seed)
    call(['sbatch', str(file_path)])
    sleep(1)


@click.command()
@click.option('--seed', type=int, default=0)
def cmd(seed):
    ratio_list = (0.1, )
    for test_ratio in ratio_list:
        run(test_ratio=test_ratio, seed=seed)


def main():
    cmd()


if __name__ == '__main__':
    main()
