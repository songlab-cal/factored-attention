from mogwai.utils.functional import apc
import argparse
import pickle as pkl

import pandas as pd
from compare_wandb import load_attention_msa_runs
import numpy as np
from tqdm import tqdm
from pathlib import Path
import wandb

from mogwai.parsing import read_contacts
from mogwai.metrics import precisions_in_range
import torch
import seaborn as sns

from mogwai import models
from weight_correlations import get_gremlin_statedict, get_msa_hparams
from compare_wandb import load_full_df, get_test_pdbs, load_run_dict


api = wandb.Api()

def get_gremlin_stratified_metrics(df, shard):
    Gremlin = models.get('gremlin')
    metric_list = []
    for index, row in tqdm(df.iterrows()):
        pdb = row['pdb']
        hparams = get_msa_hparams(pdb, df=df)
        gremlin_statedict = get_gremlin_statedict(
            pdb, df, dest=f'gremlin_{shard}.h5')
        # initialize a matrix, which will be overriden when we load later.
        hparams['true_contacts'] = torch.ones(
            [hparams['msa_length'], hparams['msa_length']])
        model = Gremlin(**hparams)
        model.load_state_dict(gremlin_statedict)


        targets = model._true_contacts
        predictions = apc(model.get_contacts())
        metrics = {}
        metrics['pdb'] = pdb
        metrics['model'] = 'gremlin'
        short = precisions_in_range(predictions, targets, minsep=6, maxsep=13)
        for k, v in short.items():
            metrics[f'short_{k}'] = float(v.squeeze())
        medium = precisions_in_range(predictions, targets, minsep=13, maxsep=25)
        for k, v in medium.items():
            metrics[f'medium_{k}'] = float(v.squeeze())
        long = precisions_in_range(predictions, targets, minsep=25)
        for k, v in long.items():
            metrics[f'long_{k}'] = float(v.squeeze())
        metric_list.append(metrics)
    return metric_list


def main(shard, total_shards):
    gremlin_runs = {'gremlin': 'dbuvl02g'}
    gremlin_df_dict = load_run_dict(gremlin_runs)
    gremlin_df = pd.concat(list(gremlin_df_dict.values()))

    n = gremlin_df.shape[0]
    step = n//total_shards
    start = shard*step
    end = (shard + 1) * step
    metric_list = get_gremlin_stratified_metrics(gremlin_df[start:end], shard=shard)
    df = pd.DataFrame.from_records(metric_list)
    with open(f'gremlin_metric_df_{shard}.pkl',  'wb') as f:
        pkl.dump(df, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int)
    parser.add_argument("--total_shards", type=int)
    args = parser.parse_args()
    main(args.shard, args.total_shards)
