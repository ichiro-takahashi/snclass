#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

from pathlib import Path
from subprocess import call
from time import sleep

import click

__date__ = '04/12/2018'


def write_job(index, cv, hidden_size):
    job = """#!/usr/bin/env bash
#SBATCH -p gpu-1080ti
#SBATCH --gres=gpu:1
#SBATCH -o "{slurm}"

source activate tf1.10
echo Gpu device: $CUDA_VISIBLE_DEVICES

ROOT=/home/imoto/crest_auto
export PYTHONPATH=/home/imoto/crest_auto/src
# DATA_DIR=${{ROOT}}/data/raw
# DATA_DIR=${{ROOT}}/data/interim/training_set_181212_1
DATA_DIR=${{ROOT}}/data/interim/dataset181217_2
DATA_DIR2=${{ROOT}}/data/raw

# MODEL_DIR=${{ROOT}}/models/plasticc-static-feature
MODEL_DIR=${{ROOT}}/models/plasticc-astronomy-feature
# MODEL_DIR=${{ROOT}}/models/plasticc-combined-feature

SRC_DIR=${{ROOT}}/src
python ${{SRC_DIR}}/plasticc_simple1.py learn --data-dir=${{DATA_DIR}} \
    --model-dir=${{MODEL_DIR}}/extra-dataset{cv:02d}_v2-{hidden_size}-{index:02d} \
    --n-jobs=1 --hidden-size={hidden_size} \
    --astronomy-feature --shuffle-only --epochs=3000 \
     --use-magnitude --only-extra --dataset-index={cv} 
"""

    dir_name = 'simple7classes'
    name = dir_name
    job_dir = Path('slurm') / dir_name
    if not job_dir.exists():
        job_dir.mkdir(parents=True)
    slurm_dir = Path('slurm') / dir_name
    if not slurm_dir.exists():
        slurm_dir.mkdir(parents=True)

    tmp = '{0}-{1:02d}-{2}'.format(name, index, cv)
    file_path = job_dir / '{}.sh'.format(tmp)
    slurm_path = slurm_dir / '%j-{0}.out'.format(tmp)

    job = job.format(slurm=slurm_path, index=index, cv=cv,
                     hidden_size=hidden_size)
    with file_path.open(mode='w') as f:
        f.write(job)

    return file_path


def write_job_static(index, cv, hidden_size):
    job = """#!/usr/bin/env bash
#SBATCH -p gpu-1080ti
#SBATCH --gres=gpu:1
#SBATCH -o "{slurm}"

source activate tf1.10
echo Gpu device: $CUDA_VISIBLE_DEVICES

ROOT=/home/imoto/crest_auto
export PYTHONPATH=/home/imoto/crest_auto/src
# DATA_DIR=${{ROOT}}/data/raw
# DATA_DIR=${{ROOT}}/data/interim/training_set_181212_1
DATA_DIR=${{ROOT}}/data/interim/dataset181217_2
# DATA_DIR2=${{ROOT}}/data/raw

MODEL_DIR=${{ROOT}}/models/plasticc-static-feature
# MODEL_DIR=${{ROOT}}/models/plasticc-astronomy-feature
# MODEL_DIR=${{ROOT}}/models/plasticc-combined-feature

SRC_DIR=${{ROOT}}/src
python ${{SRC_DIR}}/plasticc_simple1.py learn --data-dir=${{DATA_DIR}} \
    --model-dir=${{MODEL_DIR}}/extra-magnitude-{hidden_size}-{index:02d} \
    --n-jobs=1 --hidden-size={hidden_size} \
    --static-feature --shuffle-only --epochs=3000 \
     --use-magnitude --only-extra
"""

    dir_name = 'simple7static'
    name = dir_name
    job_dir = Path('slurm') / dir_name
    if not job_dir.exists():
        job_dir.mkdir(parents=True)
    slurm_dir = Path('slurm') / dir_name
    if not slurm_dir.exists():
        slurm_dir.mkdir(parents=True)

    tmp = '{0}-{1:02d}'.format(name, index)
    file_path = job_dir / '{}.sh'.format(tmp)
    slurm_path = slurm_dir / '%j-{0}.out'.format(tmp)

    job = job.format(slurm=slurm_path, index=index, hidden_size=hidden_size)
    with file_path.open(mode='w') as f:
        f.write(job)

    return file_path


def run(index, cv, hidden_size):
    if cv is None:
        file_path = write_job_static(index=index, cv=cv,
                                     hidden_size=hidden_size)
    else:
        file_path = write_job(index=index, cv=cv, hidden_size=hidden_size)
    call(['sbatch', str(file_path)])
    sleep(1)


@click.command()
@click.option('--index', type=int)
@click.option('--feature',
              type=click.Choice(['static', 'astronomy', 'combined']))
@click.option('--hidden-size', type=int, default=96)
def cmd(index, feature, hidden_size):
    if feature == 'static':
        run(index=index, cv=None, hidden_size=hidden_size)
    else:
        for cv in range(2, 11):
            run(index=index, cv=cv, hidden_size=hidden_size)


def main():
    cmd()


if __name__ == '__main__':
    main()
