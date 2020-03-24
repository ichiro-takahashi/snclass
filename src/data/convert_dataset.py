#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

from pathlib import Path

import click
import numpy as np
import pandas as pd
import tables as tb
from tqdm import tqdm

__date__ = '26/12月/2019'


BAND_NAMES = 'ugrizY'

class_map = {
    90: 'Ia', 67: 'Ia-91bg', 52: 'Iax', 42: 'II', 62: 'Ibc', 95: 'SLSN',
    15: 'TDE', 64: 'KN', 88: 'AGN', 92: 'RRL', 65: 'M-dwarf', 16: 'EB',
    53: 'Mira', 6: 'muLens-Single', 991: 'muLens-Binary', 992: 'ILOT',
    993: 'CaRT', 994: 'PISN', 995: 'muLens-String'
}


@click.group()
def cmd():
    pass


@cmd.command()
@click.option('--data', type=click.Path(exists=True, dir_okay=False))
@click.option('--metadata', type=click.Path(exists=True, dir_okay=False))
@click.option('--output-path', type=click.Path())
@click.option('--data-type', type=click.Choice(['train', 'test']))
def plasticc(data, metadata, output_path, data_type):
    df_data = pd.read_csv(data, header=0)
    df_meta = pd.read_csv(metadata, header=0)

    # it gets the number of data per sample
    # getting a name from the top row
    name = df_data['object_id'].iloc[0]
    size = np.count_nonzero(df_data['object_id'] == name)

    class Data(tb.IsDescription):
        flux = tb.Float32Col(shape=size)
        flux_err = tb.Float32Col(shape=size)
        redshift = tb.Float32Col()
        sn_type = tb.StringCol(len('unlabeled'))
        name = tb.StringCol(64)

    class Meta(tb.IsDescription):
        observed_day = tb.Int32Col()  # MJD
        band = tb.Int8Col()

    df_meta.set_index('object_id', inplace=True)

    output_path = Path(output_path)
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    label_column = 'target' if data_type == 'train' else 'true_target'

    band_map = {i: b for i, b in enumerate(BAND_NAMES)}
    df_data['passband'] = df_data['passband'].map(band_map)

    meta_flag = False
    with tb.open_file(str(output_path), 'w') as f:
        group = f.create_group('/', 'PLAsTiCC')

        table = f.create_table(group, 'data', Data)
        sample = table.row
        for name, g in tqdm(df_data.groupby('object_id')):
            g = g.sort_values(['passband', 'mjd'])

            sample['flux'] = g['flux'].values
            sample['flux_err'] = g['flux_err'].values
            sample['name'] = name

            tmp = df_meta.loc[name]
            sample['sn_type'] = class_map[tmp[label_column]]
            sample['redshift'] = tmp['hostgal_photoz']

            sample.append()

            if meta_flag:
                continue

            meta_flag = True

            meta_table = f.create_table(group, 'meta', Meta)
            meta_sample = meta_table.row
            for i in range(size):
                tmp = g.iloc[i]
                meta_sample['observed_day'] = tmp['mjd']
                meta_sample['band'] = tmp['passband']

                meta_sample.append()


@cmd.command()
@click.option('--data', type=click.Path(exists=True, dir_okay=False))
@click.option('--metadata', type=click.Path(exists=True, dir_okay=False))
@click.option('--output-path', type=click.Path())
@click.option('--data-type', type=click.Choice(['train', 'test']))
def hsc(data, metadata, output_path, data_type):
    df_data = pd.read_csv(data, header=0)
    df_meta = pd.read_csv(metadata, header=0)

    if data_type == 'test':
        if 'sn_type' not in df_meta.columns:
            df_meta['sn_type'] = 'unlabeled'
        if 'redshift' not in df_meta.columns:
            df_meta['redshift'] = 0.1

    # it gets the number of data per sample
    # getting a name from the top row
    name = df_data['object_id'].iloc[0]
    size = np.count_nonzero(df_data['object_id'] == name)

    class Data(tb.IsDescription):
        flux = tb.Float32Col(shape=size)
        flux_err = tb.Float32Col(shape=size)
        flux_err2 = tb.Float32Col(shape=size)
        redshift = tb.Float32Col()
        sn_epoch = tb.Float32Col(shape=size)
        sn_type = tb.StringCol(len('unlabeled'))
        name = tb.StringCol(64)
        x1 = tb.Float32Col()
        color = tb.Float32Col()
        offset = tb.Int32Col()

    class Meta(tb.IsDescription):
        elapsed_day = tb.Int32Col()
        observed_day = tb.Int32Col()  # MJD
        band = tb.Int8Col()
        index = tb.Int8Col()

    # converting the band names to class id(integer)
    band_map = {name: i for i, name in enumerate(BAND_NAMES)}
    df_data['passband'] = df_data['passband'].map(band_map)

    df_meta.set_index('object_id', inplace=True)

    output_path = Path(output_path)
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    meta_flag = False
    with tb.open_file(str(output_path), 'w') as f:
        group = f.create_group('/', 'SimSN' if data_type == 'train' else 'HSC')

        table = f.create_table(group, 'data', Data)
        sample = table.row
        for name, g in tqdm(df_data.groupby('object_id')):
            g = g.sort_values(['passband', 'index'])

            sample['flux'] = g['flux'].values
            sample['flux_err'] = g['flux_err'].values
            sample['name'] = name

            tmp = df_meta.loc[name]
            sample['sn_type'] = tmp['sn_type']
            sample['redshift'] = tmp['redshift']

            # These columns are not used,
            # but the columns are supposed to exist in the loading process.
            sample['flux_err2'] = 0
            sample['sn_epoch'] = 0
            sample['x1'] = 0
            sample['color'] = 0
            sample['offset'] = 0

            sample.append()

            if meta_flag:
                continue

            meta_flag = True

            meta_table = f.create_table(group, 'meta', Meta)
            meta_sample = meta_table.row
            for i in range(size):
                tmp = g.iloc[i]
                meta_sample['observed_day'] = tmp['mjd']
                meta_sample['band'] = tmp['passband']
                meta_sample['index'] = tmp['index']
                # This is not used, but needs to exists.
                meta_sample['elapsed_day'] = 0

                meta_sample.append()


def main():
    cmd()


if __name__ == '__main__':
    main()
