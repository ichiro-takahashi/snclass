#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

from pathlib import Path
from subprocess import call
from time import sleep

import click

__date__ = '14/12/2018'


def write_job(index, n_splits):
    job = """#!/usr/bin/env bash
#SBATCH -p cpu
#SBATCH -o "{slurm}"

source activate tf1.10

ROOT=/home/imoto/crest_auto
export PYTHONPATH=/home/imoto/crest_auto/src/data

SRC_DIR=${{ROOT}}/src
python ${{SRC_DIR}}/data/convert_test_dataset.py --index={index} --n-splits={n_splits}
"""

    dir_name = 'converter'
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

    job = job.format(slurm=slurm_path, index=index, n_splits=n_splits)
    with file_path.open(mode='w') as f:
        f.write(job)

    return file_path


def run(index, n_splits):
    file_path = write_job(index=index, n_splits=n_splits)
    call(['sbatch', str(file_path)])
    sleep(1)


@click.command()
@click.option('--n-splits', type=int, default=50)
def cmd(n_splits):
    for i in range(n_splits):
        run(index=i, n_splits=n_splits)


def main():
    cmd()


if __name__ == '__main__':
    main()
