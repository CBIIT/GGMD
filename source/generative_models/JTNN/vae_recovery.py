"""
Functions to assess reconstruction accuracy of JT-VAE models on various test compound sets
"""

import os
import sys
import numpy as np
import pdb
import pandas as pd
import random
import argparse
from datetime import datetime
from tqdm import tqdm
import torch.multiprocessing as mp
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.distance import jaccard
from atomsci.glo.generative_networks import VAEUtils as vaeu


def read_smiles(smiles_path):
    """
    :param smiles_path: a path to a file with newline-delimited smiles strings
    :return: list of smiles strings
    """

    with open(smiles_path, "r") as file_handle:
        smiles = file_handle.readlines()
        smiles_list = [smi.rstrip("\n") for smi in smiles]

    return smiles_list


# --------------------------------------------------------------------------------------------


def tani_sim(orig, decoded):
    """
    Compute Tanimoto similarity between original and decoded SMILES strings
    """
    orig_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(orig), 2, 1024)
    decoded_fp = AllChem.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(decoded), 2, 1024
    )
    return 1.0 - jaccard(orig_fp, decoded_fp)


# --------------------------------------------------------------------------------------------


def score_model(
    model_path,
    smiles_path,
    latent_path,
    workers=mp.cpu_count(),
    verbose=True,
    prob_decode=0,
    timeout=60,
):
    """
        Takes a model, does a reconsturction of the smiles contained in smiles_path, writes latent vectors and decoded
        smiles to a dataframe for later analysis. A stripped down version of what Kevin M. already did :)
    :param model_path:
    :param smiles_path:
    :param latent_path:
    :param workers:
    :param prob_decode: this should be a bool in the future, passing as an int for now to avoid breakign code 
    :param timeout:
    :param verbose:
    :return:
    """

    mp.set_sharing_strategy("file_system")

    smiles = read_smiles(smiles_path)

    # assuming that model_path points to a dictionary
    evaluator = vaeu.DistributedEvaluator(
        vae_path=model_path, usecuda=False, workers=workers
    )

    latent, valid_smiles_dataset = evaluator.encode_smiles(smiles)

    valid_smiles = valid_smiles_dataset.get_smiles()

    latent_tree_size = int(latent.shape[1] / 2)
    latent_mol_size = int(latent.shape[1] - latent_tree_size)

    tree_cols = ["t%.02d" % i for i in range(latent_tree_size)]
    mol_cols = ["m%.02d" % i for i in range(latent_mol_size)]
    latent_cols = tree_cols + mol_cols

    latent_df = pd.DataFrame.from_records(latent, columns=latent_cols)

    decoded_smiles = evaluator.decode_smiles(
        evaluator.vae,
        latent,
        processes=workers,
        verbose=verbose,
        prob_decode=prob_decode,
        timeout=timeout,
    )

    latent_df["original"] = valid_smiles
    latent_df["decoded"] = decoded_smiles
    latent_df["matches"] = [int(x == y) for x, y in zip(valid_smiles, decoded_smiles)]

    # NOTE: Make sure that the decoded result is a string instance that is not a failed result (i.e. 'None') to avoid crashing the code
    latent_df["tanimoto_sim"] = [
        tani_sim(x, y) if isinstance(y, str) and y != "None" else 0
        for x, y in zip(valid_smiles, decoded_smiles)
    ]

    # check to see if the directory containing latent_path exists of not, if not then create it
    output_dir = os.path.dirname(latent_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    latent_df.to_csv(latent_path, index=False)


# --------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--command",
        default="test_vae_recovery",
        choices=["test_vae_recovery", "recovery_vs_iters", "score_model"],
    )
    """
    parser.add_argument('--test_set', default='aurk_base', help='Key in file_paths dict for input and output paths')
    parser.add_argument('--num_smiles', type=int, default=1000)
    parser.add_argument('--randomize', default=False, action='store_true',
                        help='Whether to select SMILES strings randomly from test file')
    # Following is for test_vae_recovery only
    parser.add_argument('--iter', type=int, default=10000, required=False)
    # Following are for recovery_vs_iters only
    parser.add_argument('--max_iter', type=int, default=40000)
    parser.add_argument('--iter_step', type=int, default=1000)
    """
    # following is for score model only
    parser.add_argument(
        "--model-path", default=None, help="path to specific model to load"
    )
    parser.add_argument(
        "--smiles-path", default=None, help="path to specific set of smiles to load"
    )
    parser.add_argument(
        "--latent-path", default=None, help="path to output set of latent codes"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of workers to use in 'distributed' evaluator",
    )
    parser.add_argument(
        "--prob-decode",
        type=int,
        default=0,
        choices=[0, 1],
        help="used to indicate deterministic vs non-determinstic sampling",
    )
    args = parser.parse_args()
    print(args)

    # TODO(derek): cleaning house, removing non-essential functions
    """
    if args.command == 'test_vae_recovery':
        _ = test_vae_recovery(args.test_set, args.iter, args.num_smiles, args.randomize)
    elif args.command == 'recovery_vs_iters':
        _ = recovery_vs_iters(args.test_set, args.max_iter, args.iter_step, args.num_smiles)
    #elif args.command == 'score_model':
    """
    if args.command == "score_model":
        _ = score_model(
            model_path=args.model_path,
            smiles_path=args.smiles_path,
            latent_path=args.latent_path,
            workers=args.workers,
            prob_decode=args.prob_decode,
        )


# --------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
    sys.exit(0)
