#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

from pathlib import Path
from subprocess import call
from time import sleep
from itertools import product

import click

__date__ = '04/1/2019'


def write_job(alpha, feature_size):
    job = """#!/usr/bin/env bash
#SBATCH -p gpu-v100
#SBATCH --gres=gpu:1
#SBATCH -o "{slurm}"

source activate tf1.10
echo Gpu device: $CUDA_VISIBLE_DEVICES

ROOT=/home/imoto/crest_auto
export PYTHONPATH=/home/imoto/crest_auto/src
DATA_DIR=${{ROOT}}/data/processed/4th

# MODEL_DIR=${{ROOT}}/models/plasticc-static-feature
MODEL_DIR=${{ROOT}}/models/vae
# MODEL_DIR=${{ROOT}}/models/plasticc-combined-feature

SRC_DIR=${{ROOT}}/src/features
python ${{SRC_DIR}}/vae_feature.py fit --feature-dir=${{DATA_DIR}} \
    --model-dir=${{MODEL_DIR}}/alpha{alpha}-{feature_size} \
    --meta-dir=${{ROOT}}/data/raw --batch-size=500 --epochs=1000 \
    --alpha={alpha} --feature-size={feature_size}
"""

    dir_name = 'vae_feature'
    name = dir_name
    job_dir = Path('slurm') / dir_name
    if not job_dir.exists():
        job_dir.mkdir(parents=True)
    slurm_dir = Path('slurm') / dir_name
    if not slurm_dir.exists():
        slurm_dir.mkdir(parents=True)

    tmp = '{0}-{1}-{2}'.format(name, alpha, feature_size)
    file_path = job_dir / '{}.sh'.format(tmp)
    slurm_path = slurm_dir / '%j-{0}.out'.format(tmp)

    job = job.format(slurm=slurm_path, alpha=alpha, feature_size=feature_size)
    with file_path.open(mode='w') as f:
        f.write(job)

    return file_path


def run(alpha, feature_size):
    file_path = write_job(alpha=alpha, feature_size=feature_size)
    call(['sbatch', str(file_path)])
    sleep(1)


@click.command()
def cmd():
    for alpha, feature_size in product([100], [25, 50]):
        run(alpha=alpha, feature_size=feature_size)


def main():
    cmd()


if __name__ == '__main__':
    main()