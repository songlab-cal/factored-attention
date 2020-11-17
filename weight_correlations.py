import argparse
from functools import partial
from multiprocessing import Pool
from mogwai.metrics import precisions_in_range
import pickle as pkl
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from tqdm import tqdm
from mogwai.utils.functional import apc
from mogwai import models
import torch
import os
import boto3
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from compare_wandb import load_full_df, get_test_pdbs, load_run_dict

import wandb
api = wandb.Api()
entity = 'proteins'
project = 'iclr2021-rebuttal'

s3 = boto3.client("s3")
s3_bucket = "proteindata"


model_name = 'factored_attention'
fatt_model = models.get(model_name)
gremlin_model = models.get('gremlin')



def download_statedict(run_id, dest='fatt.h5'):
    run = api.run(f"{entity}/{project}/{run_id}")
    key = os.path.join("iclr-2021-factored-attention",
                       *run.path, "model_state_dict.h5")
    with open(dest, 'wb') as f:
        s3.download_fileobj(s3_bucket, key, f)
    return dest


def get_fatt_run_id(pdb, attention_head_size, num_attention_heads, df):
    run_id = df[(df['pdb'] == pdb) & (df['num_attention_heads']
                                      == num_attention_heads)]['run_id'].values
    return run_id[0]


def get_gremlin_run_id(pdb, df):
    run_id = df[df['pdb'] == pdb]['run_id'].values
    return run_id[0]


def get_gremlin_statedict(pdb, gremlin_df, dest='gremlin.h5'):
    run_id = get_gremlin_run_id(pdb, gremlin_df)
    f_statedict = download_statedict(run_id, dest=dest)
    statedict = torch.load(f_statedict)
    return statedict


def get_fatt_statedict(pdb, num_attention_heads, fatt_df, attention_head_size=32, dest='fatt.h5'):
    run_id = get_fatt_run_id(pdb, attention_head_size,
                             num_attention_heads, fatt_df)
    f_statedict = download_statedict(
        run_id, dest=dest)
    statedict = torch.load(f_statedict)
    return statedict


def get_msa_hparams(pdb, df):
    pdb = df[df['pdb'] == pdb]
    hparam_dict = pdb.to_dict()
    msa_length = int(list(hparam_dict['msa_length'].values())[0])
    num_seqs = int(list(hparam_dict['num_seqs'].values())[0])
    return {'msa_length': msa_length, 'num_seqs': num_seqs}


def get_correlations(w, w_fatt, L, make_plot=False, pdb=None):
    idx = np.triu_indices(L, 1)

    fatt_w_no_diag = w_fatt.detach()[idx[0], :, idx[1], :]
    gremlin_w_no_diag = w[idx[0], :, idx[1], :]

    fatt_w_compare_idx = torch.flatten(fatt_w_no_diag)
    gremlin_w_compare_idx = torch.flatten(gremlin_w_no_diag)
    w_spearman = spearmanr(fatt_w_compare_idx, gremlin_w_compare_idx)[0]
    w_pearson = pearsonr(fatt_w_compare_idx, gremlin_w_compare_idx)[0]

    if make_plot:
        # plotting correlation
        # plotting the whole thing takes time
        subset = np.random.choice(len(gremlin_w_compare_idx), size=100000)
        plt.title(
            f'Spearman: {w_spearman:.2f} Pearson: {w_pearson:.2f} PDB: {pdb}')
        plt.xlabel('gremlin w')
        plt.ylabel('fatt w')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.scatter(gremlin_w_compare_idx[subset],
                    fatt_w_compare_idx[subset], s=1)
        plt.show()
    return w_spearman, w_pearson


def get_info(num_attention_heads,
             pdb,
             attention_head_size,
             fatt_df,
             fatt_dest,
             gremlin_df,
             gremlin_dest):
    fatt_statedict = get_fatt_statedict(
        pdb, num_attention_heads, fatt_df, attention_head_size=attention_head_size, dest=fatt_dest)
    gremlin_statedict = get_gremlin_statedict(
        pdb, gremlin_df, dest=gremlin_dest)

    hparams = get_msa_hparams(pdb, df=fatt_df)
    msa_length = hparams['msa_length']
    num_seqs = hparams['num_seqs']
    hparams['attention_head_size'] = attention_head_size
    hparams['num_attention_heads'] = num_attention_heads
    # initialize a matrix, which will be overriden when we load later.
    hparams['true_contacts'] = torch.ones(
        [hparams['msa_length'], hparams['msa_length']])
    model = fatt_model(**hparams)
    model.load_state_dict(fatt_statedict)

    w_fatt = model.compute_mrf_weight()
    w = gremlin_statedict['weight']

    metrics = {}
    metrics['pdb'] = pdb
    metrics['msa_length'] = msa_length
    metrics['num_seqs'] = num_seqs
    metrics['attention_head_size'] = attention_head_size
    metrics['num_attention_heads'] = num_attention_heads

    predictions = apc(model.get_contacts())
    targets = model._true_contacts

    short = precisions_in_range(predictions, targets, minsep=6, maxsep=13)
    for k, v in short.items():
        metrics[f'short_{k}'] = float(v.squeeze())
    medium = precisions_in_range(predictions, targets, minsep=13, maxsep=25)
    for k, v in medium.items():
        metrics[f'medium_{k}'] = float(v.squeeze())
    long = precisions_in_range(predictions, targets, minsep=13, maxsep=25)
    for k, v in long.items():
        metrics[f'long_{k}'] = float(v.squeeze())
    wspearman, wpearson = get_correlations(w, w_fatt, msa_length, pdb=pdb)
    metrics['w_spearman'] = wspearman
    metrics['w_pearson'] = wpearson
#     metrics['predicted_contacts_apc'] = predictions
#     metrics['true_contacts'] = targets
    return metrics



def main(pdb_filename, shard):

    # load metatest pdbs
    with open(pdb_filename, 'r') as f:
        lines = f.readlines()
        metatest_pdbs = [l.strip() for l in lines]

    head_sweep_runs = {
        'fatt-metatest-head-sweep-512': 'bxnkt0uq',
        'fatt-metatest-head-256': 'xuofwjtc',
        'fatt-metatest-head-sweep-64-kqe7or39': 'kqe7or39',
        'fatt-metatest-head-sweep-128': '32emd6ri',
        'fatt-metatest-head-sweep-8-32': '8yi6a4w5',
    }

    dict_of_dfs = load_run_dict(head_sweep_runs)
    fatt_df = pd.concat(list(dict_of_dfs.values()))

    gremlin_runs = {'gremlin': 'dbuvl02g'}
    gremlin_df_dict = load_run_dict(gremlin_runs)
    gremlin_df = pd.concat(list(gremlin_df_dict.values()))

    num_heads_sweep = [8, 16, 32, 64, 128, 256, 512]
    filenames = ['gremlin.h5'] + [f'fatt_{n}.h5' for n in num_heads_sweep]
    info = []

    try:
        for pdb in tqdm(metatest_pdbs):
            for num_attention_heads in num_heads_sweep:
                info.append(get_info(num_attention_heads,
                                     attention_head_size=32,
                                     pdb=pdb,
                                     fatt_df=fatt_df,
                                     fatt_dest=f'fatt_s{shard}.h5',
                                     gremlin_df = gremlin_df,
                                     gremlin_dest=f'gremlin_s{shard}.h5'))
    finally:
        print(pdb)
        sweep_df = pd.DataFrame.from_records(info)
        with open(f'num_head_sweep_df_s_{shard}.pkl', 'wb') as f:
            pkl.dump(sweep_df, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_filename")
    parser.add_argument("--shard", type=int)
    args = parser.parse_args()
    main(args.pdb_filename, args.shard)
