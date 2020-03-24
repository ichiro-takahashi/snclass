#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import numpy as np
import tables as tb
import pandas as pd

__date__ = '17/6月/2019'


def load_plasticc_data(sim_sn_path, training_cosmos_path, test_cosmos_path,
                       use_flux_err2):
    with tb.open_file(training_cosmos_path, 'r') as f:
        training_cosmos = f.root['PLAsTiCC'].data[:]
    with tb.open_file(test_cosmos_path, 'r') as f:
        test_cosmos = f.root['PLAsTiCC'].data[:]

    with tb.open_file(sim_sn_path, 'r') as f:
        # remove the extra column (sn_epoch)
        sim_sn = np.empty(len(f.root['SimSN'].data),
                          dtype=training_cosmos.dtype)
        for key in ('flux', 'flux_err', 'redshift', 'sn_type', 'name'):
            # if use_flux_err2 and key == 'flux_er':
            #     sim_sn[key] = f.root['SimSN'].data.col('flux_err2')
            sim_sn[key] = f.root['SimSN'].data.col(key)

    # train and validation
    # data = np.concatenate((sim_sn, training_cosmos))
    #
    # return data, test_cosmos

    return sim_sn, training_cosmos, test_cosmos


def load_hsc_data(sim_sn_path, hsc_path, use_flux_err2=True, remove_y=False,
                  raw_redshift=False):
    sim_sn = load_sim_sn_data(
        sim_sn_path=sim_sn_path, use_flux_err2=use_flux_err2,
        remove_y=remove_y
    )
    hsc_data = load_hsc_test_data(hsc_path=hsc_path, remove_y=remove_y,
                                  raw_redshift=raw_redshift)
    return sim_sn, hsc_data


def load_hsc_data_n_observations(sim_sn_path, hsc_path, n_observations,
                                 use_flux_err2=True, remove_y=False,
                                 raw_redshift=False):
    sim_sn = load_hsc_simsn_n_observations(
        sim_sn_path=sim_sn_path, n_observations=n_observations,
        use_flux_err2=use_flux_err2, remove_y=remove_y
    )
    hsc_data = load_hsc_test_n_observations(
        hsc_path=hsc_path, n_observations=n_observations,
        remove_y=remove_y, raw_redshift=raw_redshift
    )
    return sim_sn, hsc_data


hsc_band_name = 'ugrizY'


def load_sim_sn_data(sim_sn_path, use_flux_err2=True, remove_y=False):
    with tb.open_file(sim_sn_path, 'r') as f:
        sim_sn = f.root['SimSN'].data[:]
        if use_flux_err2:
            sim_sn['flux_err'] = sim_sn['flux_err2']

        if remove_y:
            meta = f.root['SimSN'].meta[:]
            assert hsc_band_name[5] == 'Y'
            flag = meta['band'] != 5
            n = np.count_nonzero(flag)

            dtype = np.dtype([
                ('color', '<f4'), ('flux', '<f4', (n,)),
                ('flux_err', '<f4', (n,)), ('flux_err2', '<f4', (n,)),
                ('name', 'S64'), ('offset', '<i4'), ('redshift', '<f4'),
                ('sn_epoch', '<f4', (n,)), ('sn_type', 'S9'), ('x1', '<f4')
            ])
            name_list = ('color', 'flux', 'flux_err', 'flux_err', 'name',
                         'offset', 'redshift', 'sn_epoch', 'sn_type', 'x1')
            tmp = np.empty(len(sim_sn), dtype=dtype)
            for name in name_list:
                if name in ('flux', 'flux_err', 'flux_err2', 'sn_epoch'):
                    tmp[name] = sim_sn[name][:, flag]
                else:
                    tmp[name] = sim_sn[name]
            sim_sn = tmp
    return sim_sn


def load_hsc_test_data(hsc_path, remove_y=False, raw_redshift=False):
    with tb.open_file(hsc_path, 'r') as f:
        hsc_data = f.root['HSC'].data[:]

        if remove_y:
            meta = f.root['HSC'].meta[:]
            assert hsc_band_name[5] == 'Y'
            flag = meta['band'] != 5
            n = np.count_nonzero(flag)

            dtype = np.dtype([
                ('flux', '<f4', (n,)), ('flux_err', '<f4', (n,)),
                ('name', 'S64'), ('redshift', '<f4'), ('sn_type', 'S9')
            ])
            name_list = ('flux', 'flux_err', 'name', 'redshift', 'sn_type')
            tmp = np.empty(len(hsc_data), dtype=dtype)
            for name in name_list:
                if name in ('flux', 'flux_err'):
                    tmp[name] = hsc_data[name][:, flag]
                else:
                    tmp[name] = hsc_data[name]
            hsc_data = tmp

    if not raw_redshift:
        tmp = np.nan_to_num(hsc_data['redshift'])
        tmp = np.where(tmp < 0.1, 0.1, tmp)
        hsc_data['redshift'] = tmp

    return hsc_data


def load_hsc_simsn_n_observations(sim_sn_path, n_observations,
                                  use_flux_err2=True, remove_y=False):
    with tb.open_file(sim_sn_path, 'r') as f:
        meta = f.root['SimSN'].meta[:]
        meta_df = pd.DataFrame(meta)
        if remove_y:
            meta_df = meta_df[meta_df['band'] != 5]
        meta_df2 = meta_df.sort_values(['observed_day', 'band'])
        index = meta_df2.iloc[:n_observations].index

        sim_sn = f.root['SimSN'].data[:]
        if use_flux_err2:
            sim_sn['flux_err'] = sim_sn['flux_err2']

        if remove_y:
            meta = f.root['SimSN'].meta[:]
            assert hsc_band_name[5] == 'Y'
            flag = meta['band'] != 5
            n = np.count_nonzero(flag)

            dtype = np.dtype([
                ('color', '<f4'), ('flux', '<f4', (n,)),
                ('flux_err', '<f4', (n,)), ('flux_err2', '<f4', (n,)),
                ('name', 'S64'), ('offset', '<i4'), ('redshift', '<f4'),
                ('sn_epoch', '<f4', (n,)), ('sn_type', 'S9'), ('x1', '<f4')
            ])
            name_list = ('color', 'flux', 'flux_err', 'name',
                         'offset', 'redshift', 'sn_epoch', 'sn_type', 'x1')
            tmp = np.empty(len(sim_sn), dtype=dtype)
            for name in name_list:
                if name in ('flux', 'flux_err', 'flux_err2', 'sn_epoch'):
                    tmp[name] = sim_sn[name][:, flag]
                else:
                    tmp[name] = sim_sn[name]
            sim_sn = tmp

        dtype = np.dtype([
            ('color', '<f4'), ('flux', '<f4', (n_observations,)),
            ('flux_err', '<f4', (n_observations,)),
            ('name', 'S64'), ('offset', '<i4'), ('redshift', '<f4'),
            ('sn_epoch', '<f4', (n_observations,)), ('sn_type', 'S9'),
            ('x1', '<f4')
        ])
        name_list = ('color', 'flux', 'flux_err', 'name',
                     'offset', 'redshift', 'sn_epoch', 'sn_type', 'x1')
        tmp = np.empty(len(sim_sn), dtype=dtype)
        for name in name_list:
            if name in ('flux', 'flux_err', 'sn_epoch'):
                tmp[name] = sim_sn[name][:, index]
            else:
                tmp[name] = sim_sn[name]
        sim_sn = tmp

    return sim_sn


def load_hsc_test_n_observations(hsc_path, n_observations,
                                 remove_y=False, raw_redshift=False):
    with tb.open_file(hsc_path, 'r') as f:
        meta = f.root['HSC'].meta[:]
        meta_df = pd.DataFrame(meta)
        if remove_y:
            meta_df = meta_df[meta_df['band'] != 5]
        meta_df2 = meta_df.sort_values(['observed_day', 'band'])
        index = meta_df2.iloc[:n_observations].index

        hsc_data = f.root['HSC'].data[:]

        if remove_y:
            meta = f.root['HSC'].meta[:]
            assert hsc_band_name[5] == 'Y'
            flag = meta['band'] != 5
            n = np.count_nonzero(flag)

            dtype = np.dtype([
                ('flux', '<f4', (n,)), ('flux_err', '<f4', (n,)),
                ('name', 'S64'), ('redshift', '<f4'), ('sn_type', 'S9')
            ])
            name_list = ('flux', 'flux_err', 'name', 'redshift', 'sn_type')
            tmp = np.empty(len(hsc_data), dtype=dtype)
            for name in name_list:
                if name in ('flux', 'flux_err'):
                    tmp[name] = hsc_data[name][:, flag]
                else:
                    tmp[name] = hsc_data[name]
            hsc_data = tmp

    if not raw_redshift:
        tmp = np.nan_to_num(hsc_data['redshift'])
        tmp = np.where(tmp < 0.1, 0.1, tmp)
        hsc_data['redshift'] = tmp

    dtype = np.dtype([
        ('flux', '<f4', (n_observations,)),
        ('flux_err', '<f4', (n_observations,)),
        ('name', 'S64'), ('redshift', '<f4'), ('sn_type', 'S9')
    ])
    name_list = ('flux', 'flux_err', 'name', 'redshift', 'sn_type')
    tmp = np.empty(len(hsc_data), dtype=dtype)
    for name in name_list:
        if name in ('flux', 'flux_err'):
            tmp[name] = hsc_data[name][:, index]
        else:
            tmp[name] = hsc_data[name]
    hsc_data = tmp

    return hsc_data
