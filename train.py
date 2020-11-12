from argparse import ArgumentParser
import os
from pathlib import Path
import random
import string

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import wandb
import boto3

from mogwai.data_loading import MSADataModule
from mogwai.parsing import read_contacts
from mogwai import models
from mogwai.utils.functional import apc
from mogwai.metrics import contact_auc
from mogwai.plotting import (
    plot_colored_preds_on_trues,
    plot_precision_vs_length,
)
from mogwai.vocab import FastaVocab

from loggers import WandbLoggerFrozenVal

s3_client = boto3.client("s3")
s3_bucket = "proteindata"


def train():
    # Initialize parser
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        default="gremlin",
        choices=models.MODELS.keys(),
        help="Which model to train.",
    )
    model_name = parser.parse_known_args()[0].model
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Whether to save the model state dict.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="iclr2021-rebuttal",
        help="W&B project used for logging.",
    )
    parser.add_argument(
        "--pdb",
        type=str,
        help="PDB id for training",
    )

    parser = MSADataModule.add_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=1,
        min_steps=50,
        max_steps=1000,
        log_every_n_steps=10,
    )
    model_type = models.get(model_name)
    model_type.add_args(parser)
    args = parser.parse_args()

    # Modify name
    pdb = args.pdb
    args.data = "data/npz/" + pdb + ".npz"

    # Load msa
    msa_dm = MSADataModule.from_args(args)
    msa_dm.setup()

    # Load contacts
    true_contacts = torch.from_numpy(read_contacts(args.data))

    # Initialize model
    num_seqs, msa_length, msa_counts = msa_dm.get_stats()
    model = model_type.from_args(
        args,
        num_seqs=num_seqs,
        msa_length=msa_length,
        msa_counts=msa_counts,
        vocab_size=len(FastaVocab),
        pad_idx=FastaVocab.pad_idx,
        true_contacts=true_contacts,
    )

    kwargs = {}
    randstring = "".join(random.choice(string.ascii_lowercase) for i in range(6))
    run_name = "_".join([args.model, pdb, randstring])
    logger = WandbLoggerFrozenVal(project=args.wandb_project, name=run_name)
    logger.log_hyperparams(args)
    logger.log_hyperparams(
        {
            "pdb": pdb,
            "num_seqs": num_seqs,
            "msa_length": msa_length,
        }
    )
    kwargs["logger"] = logger

    # Initialize Trainer
    trainer = pl.Trainer.from_argparse_args(args, **kwargs)

    trainer.fit(model, msa_dm)

    # Log and print some metrics after training.
    contacts = model.get_contacts()
    apc_contacts = apc(contacts)

    auc = contact_auc(contacts, true_contacts).item()
    auc_apc = contact_auc(apc_contacts, true_contacts).item()
    print(f"AUC: {auc:0.3f}, AUC_APC: {auc_apc:0.3f}")

    filename = "top_L_contacts.png"
    plot_colored_preds_on_trues(contacts, true_contacts, point_size=5, cutoff=1)
    plt.title(f"Top L no APC {model.get_precision(do_apc=False)}")
    logger.log_metrics({filename: wandb.Image(plt)})
    plt.close()

    filename = "top_L_contacts_apc.png"
    plot_colored_preds_on_trues(apc_contacts, true_contacts, point_size=5, cutoff=1)
    plt.title(f"Top L APC {model.get_precision(do_apc=True)}")
    logger.log_metrics({filename: wandb.Image(plt)})
    plt.close()

    filename = "top_L_5_contacts.png"
    plot_colored_preds_on_trues(contacts, true_contacts, point_size=5, cutoff=5)
    plt.title(f"Top L/5 no APC {model.get_precision(do_apc=False, cutoff=5)}")
    logger.log_metrics({filename: wandb.Image(plt)})
    plt.close()

    filename = "top_L_5_contacts_apc.png"
    plot_colored_preds_on_trues(apc_contacts, true_contacts, point_size=5, cutoff=5)
    plt.title(f"Top L/5 APC {model.get_precision(do_apc=True, cutoff=5)}")
    logger.log_metrics({filename: wandb.Image(plt)})
    plt.close()

    filename = "precision_vs_L.png"
    plot_precision_vs_length(apc_contacts, true_contacts)
    logger.log_metrics({filename: wandb.Image(plt)})
    plt.close()

    if args.save_model:
        modelfile = os.path.join(wandb.run.dir, "model_state_dict.h5")
        torch.save(model.state_dict(), modelfile)
        s3_target = os.path.join(
            "iclr-2021-factored-attention", wandb.run.path, "model_state_dict.h5"
        )
        response = s3_client.upload_file(
            modelfile, s3_bucket, s3_target, ExtraArgs={"ACL": "public-read"}
        )


if __name__ == "__main__":
    train()
