from argparse import ArgumentParser
import os
from pathlib import Path
import random
import string
import io

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import wandb

from mogwai.data_loading import MSADataModule
from mogwai.metrics import contact_auc
from mogwai.models import FactoredAttention
from mogwai.parsing import read_contacts
from mogwai.plotting import (
    plot_colored_preds_on_trues,
    plot_precision_vs_length,
)
from mogwai.utils.functional import apc
from mogwai.vocab import FastaVocab

from loggers import WandbLoggerFrozenVal


def train():
    # Initialize parser
    parser = ArgumentParser()
    parser.add_argument(
        "--freeze_random_value",
        action="store_true",
        help="Whether to freeze values at init.",
    )
    parser.add_argument(
        "--weight_save_path", type=str, default=None, help="Where to store state dict."
    )

    parser.add_argument(
        "--load_values_from",
        type=str,
        default=None,
        help="Path to state dict for saved values. Freezes values.",
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
    parser = FactoredAttention.add_args(parser)
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
    model = FactoredAttention.from_args(
        args,
        num_seqs=num_seqs,
        msa_length=msa_length,
        msa_counts=msa_counts,
        vocab_size=len(FastaVocab),
        pad_idx=FastaVocab.pad_idx,
        true_contacts=true_contacts,
    )

    cached_val = None

    if args.freeze_random_value:
        model.value.requires_grad = False
        cached_val = model.value.data.clone().cpu()

    if args.load_values_from:
        saved_state = torch.load(args.load_values_from)
        model.value.requires_grad = False
        model.value.data = saved_state["value"]
        cached_val = model.value.data.clone().cpu()

    kwargs = {}
    randstring = "".join(random.choice(string.ascii_lowercase) for i in range(6))
    run_name = "_".join(["fatt", pdb, randstring])
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

    if args.load_values_from or args.freeze_random_value:
        if torch.all(cached_val.eq(model.value.data.clone().cpu())):
            print("Values unchanged!")
        else:
            raise ValueError(
                "Value matrix changed during training but was supposed to be frozen."
            )

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

    if args.weight_save_path:
        weight_save_path = Path(args.weight_save_path)
        torch.save(model.state_dict(), weight_save_path)


if __name__ == "__main__":
    train()
