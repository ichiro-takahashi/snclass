#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

from pathlib import Path
from subprocess import call
from time import sleep
from itertools import product

import click

__date__ = '21/12/2018'


def write_job(index, source):
    job = """#!/usr/bin/env bash
#SBATCH -p cpu
#SBATCH -o "{slurm}"

export OMP_NUM_THREADS=1

source activate py3.6

ROOT=/home/imoto/crest_auto
export PYTHONPATH=/home/imoto/crest_auto/src

cd ${{ROOT}}/src/features
python light_curve_fitting.py Fit --index={index} --n-split=512 --source={source} --local-scheduler
"""

    dir_name = 'template'
    name = dir_name
    job_dir = Path('slurm') / dir_name
    if not job_dir.exists():
        job_dir.mkdir(parents=True)
    slurm_dir = Path('slurm') / dir_name
    if not slurm_dir.exists():
        slurm_dir.mkdir(parents=True)

    tmp = '{0}-{1:02d}-{2}'.format(name, index, source)
    file_path = job_dir / '{}.sh'.format(tmp)
    slurm_path = slurm_dir / '%j-{0}.out'.format(tmp)

    job = job.format(slurm=slurm_path, index=index, source=source)
    with file_path.open(mode='w') as f:
        f.write(job)

    return file_path


def run(index, source):
    file_path = write_job(index=index, source=source)
    call(['sbatch', str(file_path)])
    sleep(1)


def check_file(source, index):
    path = (Path('/home/imoto/crest_auto/data/interim/features/test') /
            source / '{0:03d}.pickle'.format(index))
    return path.exists()


@click.command()
def cmd():
    source_list = [
        'salt2-extended', 'nugent-sn1bc', 's11-2005hl', 's11-2006fo',
        'nugent-sn2p', 'nugent-sn2l', 'nugent-sn2n', 's11-2004hx',
        'snana-2007ms', 'whalen-z15b',
        'salt2', 'nugent-hyper', 's11-2006jo', 'snana-2004fe',
        'snana-2007pg', 'snana-2006ix', 'whalen-z40g'
    ]
    # sleep(3600 * 4)
    for index in range(30, 100):
        run(index=index, source=source_list[0])

    # for previous, source in zip(source_list[:-1], source_list[1:]):
    #     while True:
    #         if (check_file(source=previous, index=450) or
    #                 check_file(source=previous, index=460) or
    #                 check_file(source=previous, index=470) or
    #                 check_file(source=previous, index=480)):
    #             break
    #         sleep(10)
    #
    #     for index in range(512):
    #         run(index=index, source=source)


def main():
    cmd()


if __name__ == '__main__':
    main()
