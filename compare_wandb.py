from typing import Dict

import wandb
import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

api = wandb.Api()

entity = "proteins"


def add_apc_default(df: pd.DataFrame, sweep_name: str) -> pd.DataFrame:
    # Adds modified sweep whose default metrics are apc'd
    d = df[df.sweep_name == sweep_name]
    d.loc[:, "sweep_name"] = sweep_name + "-apc"
    d.loc[:, "pr_at_L"] = d.loc[:, "pr_at_L_apc"]
    d.loc[:, "pr_at_L_5"] = d.loc[:, "pr_at_L_5_apc"]
    d.loc[:, "auc"] = d.loc[:, "auc_apc"]
    return df.append(d)


def load_attention_msa_df(sweep_id, sweep_name, model_name, pdb_map):
    # Loads sweep df for runs from old repo
    project = "gremlin-contacts"
    runs = api.runs(f"{entity}/{project}", {"sweep": f"{sweep_id}"}, per_page=1000)
    print(f"{sweep_id} has {len(runs)} runs")
    id_list = []
    summary_list = []
    config_list = []
    name_list = []
    model_list = []
    state_list = []
    tags_list = []

    num_contacts_list = []
    for run in tqdm(runs):
        tags_list.append(run.tags)
        state_list.append(run.state)
        id_list.append(run.id)
        # run.summary are the output key/values like accuracy.  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)
        # run.config is the input metrics.  We remove special values that start with _.

        config_list.append(
            {str(k): v for k, v in run.config.items() if not k.startswith("_")}
        )
        # run.name is the name of the run.
        name_list.append(run.name)
        model_list.append(model_name)
        # currently unused, very slow to download true contact files
        # num_contacts_list.append(get_num_true_contacts(run))

    # num_contacts_df = pd.DataFrame({'num_true_contacts': num_contacts_list})
    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_dict(config_list)
    pdb_id_df = pd.DataFrame({"pdb_idx": list(config_df.pdb.map(pdb_map))})
    state_df = pd.DataFrame({"run_state": state_list})
    name_df = pd.DataFrame({"name": name_list})
    run_id_df = pd.DataFrame({"run_id": id_list})
    # currently unused
    tags_df = pd.DataFrame({"tags": tags_list})
    sweep_name_df = pd.DataFrame({"sweep_name": [sweep_name] * len(summary_df)})

    # We now log model in the refactor.
    model_df = pd.DataFrame({"model": model_list})
    df = pd.concat(
        [
            name_df,
            run_id_df,
            model_df,
            config_df,
            pdb_id_df,
            summary_df,
            state_df,
            sweep_name_df,
        ],
        axis=1,
    )
    df.pdb = df.pdb.astype(str)
    return df


def load_attention_msa_results() -> Dict[str, pd.DataFrame]:
    # Loads static set of runs from attention_msa with fixed sweep names.
    old_models_to_id = {
        "gremlin": "9ppr5f9y",
        "old-fatt": "wqzai5ya",
        "transformer": "zd8rc6j7",
        "protbert-bfd": "l37wrnsa",
    }
    sweep_ids_to_name = {v: k for k, v in old_models_to_id.items()}
    pdb_id_map = get_test_pdbs()
    sweep_dfs = {
        sweep_name: load_attention_msa_df(
            sweep_id,
            sweep_name,
            sweep_name,
            pdb_id_map,
        )
        for sweep_id, sweep_name in sweep_ids_to_name.items()
    }

    sweep_dfs = {k: s[s.run_state == "finished"] for k, s in sweep_dfs.items()}
    return sweep_dfs


def get_sweep_df(sweep_id, sweep_name, model_name, pdb_map):
    # Loads sweep df from new repo

    project = "iclr2021-rebuttal"
    runs = api.runs(f"{entity}/{project}", {"sweep": f"{sweep_id}"}, per_page=1000)
    print(f"{sweep_id} has {len(runs)} runs")
    id_list = []
    summary_list = []
    config_list = []
    name_list = []
    model_list = []
    state_list = []
    tags_list = []

    num_contacts_list = []
    for run in tqdm(runs):
        tags_list.append(run.tags)
        state_list.append(run.state)
        id_list.append(run.id)
        # run.summary are the output key/values like accuracy.  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)
        # run.config is the input metrics.  We remove special values that start with _.

        config_list.append(
            {str(k): v for k, v in run.config.items() if not k.startswith("_")}
        )
        # run.name is the name of the run.
        name_list.append(run.name)
        model_list.append(model_name)
        # currently unused, very slow to download true contact files
        # num_contacts_list.append(get_num_true_contacts(run))

    # num_contacts_df = pd.DataFrame({'num_true_contacts': num_contacts_list})
    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_dict(config_list)
    pdb_id_df = pd.DataFrame({"pdb_idx": list(config_df.pdb.map(pdb_map))})
    state_df = pd.DataFrame({"run_state": state_list})
    name_df = pd.DataFrame({"name": name_list})
    run_id_df = pd.DataFrame({"run_id": id_list})
    sweep_name_df = pd.DataFrame({"sweep_name": [sweep_name] * len(summary_df)})
    # currently unused
    tags_df = pd.DataFrame({"tags": tags_list})

    df = pd.concat(
        [name_df, run_id_df, config_df, pdb_id_df, summary_df, state_df, sweep_name_df],
        axis=1,
    )

    df.pdb = df.pdb.astype(str)
    return df


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# -- NOT SURE IF ANYTHING BELOW HERE USED OR WORKING -------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


def get_run_file(run_id, filename, dest="."):
    run = api.run(f"{entity}/{project}/{run_id}")
    run.file(filename).download(root=dest)


def get_model(sweep_name):
    for model in models:
        if model in sweep_name:
            return model


def get_sweep_ids():
    with open("sweep_names.txt", "r") as f:
        lines = [l.strip() for l in f.readlines()]

    id_to_name = {}
    for l in lines:
        description, sweep_id = l.rsplit("-", maxsplit=1)
        id_to_name[sweep_id] = description
    return id_to_name


def get_test_pdbs():
    with open("sampled_pdbs.txt", "r") as f:
        lines = f.readlines()
        sampled_pdbs = [line.strip() for line in lines]
    return {pdb: i for i, pdb in enumerate(sampled_pdbs)}


def get_run(sweep_id, pdb):
    filters = {"$and": [{"sweep": f"{sweep_id}"}, {"config.run": f"{pdb}"}]}
    runs = api.runs(f"{entity}/{project}", filters, per_page=10)
    if len(runs) > 0:
        return runs[0]
    else:
        return None


# TODO filter crashed runs ch
def get_contact_maps(run):
    predicted_filename = "predicted_contacts"
    predicted_ref_filename = "predicted_contacts_with_reference"
    true_filename = "true_contacts"
    run.file(true_filename).download(replace=True)
    run.file(predicted_filename).download(replace=True)
    run.file(predicted_ref_filename).download(replace=True)
    true_cons = np.load(true_filename)
    pred_cons = np.load(predicted_filename)
    pred_cons_with_ref = np.load(predicted_ref_filename)
    return {"true": true_cons, "pred": pred_cons, "pred_with_ref": pred_cons_with_ref}


def get_num_true_contacts(run):
    # warning, will fail with any parallelism
    true_filename = "true_contacts.npy"
    run.file(true_filename).download(replace=True)
    true_cons = np.load(true_filename)
    eval_idx = np.triu_indices_from(true_cons, 6)
    true_cons_ = true_cons[eval_idx]
    num_contacts = len(np.nonzero(true_cons_))
    return num_contacts


def plot_performance_vs_length(sweep_df, sweep_name):
    x, z, y = sweep_df.num_seqs, sweep_df.len_ref, sweep_df.Train_Auc
    cmap = sns.cubehelix_palette(as_cmap=True)

    f, ax = plt.subplots()
    points = ax.scatter(x, y, c=z, s=1, cmap=cmap)
    cbar = f.colorbar(points)
    cbar.set_label("Length")
    plt.title(f"Performance vs length {sweep_name}")
    plt.xlabel("Number of MSA seqs")
    plt.ylabel("Train AUC")


def plot_vs_performance(x_df, y_df, x_name, y_name):
    x = x_df.Train_Auc
    y = y_df.Train_Auc
    cmap = sns.cubehelix_palette(as_cmap=True)

    f, ax = plt.subplots()
    points = ax.scatter(x, y, c=z, s=1, cmap=cmap)
    cbar = f.colorbar(points)
    cbar.set_label("Length")

    plt.title(f"{y_name} vs {x_name} AUC")
    plt.xlabel(f"{x_name}")
    plt.ylabel(f"{y_name}")


if __name__ == "__main__":
    sweep_ids_to_name = get_sweep_ids()
    sweep_ids = list(sweep_ids_to_name.keys())
    pdb_id_map = get_test_pdbs()
    sweep_name = list(sweep_ids_to_name.values())[0]
    sweep_id = list(sweep_ids_to_name.keys())[0]
    get_sweep_df(sweep_id, sweep_name, get_model(sweep_name), pdb_id_map)
    # sweep_df_x.join(sweep_df_y, on='pdb', how='inner')