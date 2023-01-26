#!/usr/bin/env python
# Script to preprocess and split a dataset for training and testing JT-VAE models

import sys
import os
import numpy as np
import pandas as pd
import rdkit
from sklearn.utils import shuffle
from sklearn.utils.random import sample_without_replacement
from multiprocessing import pool
from argparse import ArgumentParser

from atomsci.ddm.utils.struct_utils import base_smiles_from_smiles, mols_from_smiles
from atomsci.glo.generative_networks.icml18_jtnn.fast_jtnn import mol_tree
from atomsci.glo.generative_networks.icml18_jtnn.preprocess import tensorize, _write_files


dset_data = dict(
    enamine_kinase = dict(
        raw_file = '/usr/workspace/atom/enamine/Enamine_Kinase.txt'
    ),
    neurocrine = dict(
        raw_file = '/usr/workspace/atom/neurocrine/vae_training_data/neurocrine_vae_smiles.txt'
    ),
    parp1_pred_inhib = dict(
        raw_file = '/usr/workspace/atom/vae_training/parp1_pred_inhib/PARP1_predicted_inhib_ge_8.2_vae_set.csv'
    ),
    parp1_pred_inhib_81 = dict(
        raw_file = '/usr/workspace/atom/vae_training/parp1_pred_inhib/PARP1_predicted_inhib_ge_8.1_vae_set.csv'
    ),
    parp_inhib = dict(
        raw_file = '/usr/workspace/atom/vae_training/parp_inhib/parp_inhib_vae_set.csv'
    ),
    test = dict(
        raw_file = '/usr/workspace/atom/enamine/test.txt'
    )
)
default_data_root = '/usr/workspace/atom/vae_training/icml18_jtnn/data'

def prep_data(dset_name, test_size=1000, data_root=default_data_root, force_update=False, 
              ntrain_splits=64, nworkers=16, smiles_col=None):
    """
    Prepare the named dataset for VAE training. Looks up the raw dataset path in the dset_data dictionary
    using dset_name as a key. Generates a vocabulary file. Splits out a test set of the given size, then
    runs the preprocess.py script on the remainder to create a directory of tensorized training data.
    """
    raw_path = dset_data[dset_name]['raw_file']
    data_dir = os.path.join(data_root, dset_name)
    os.makedirs(data_dir, exist_ok=True)

    # Read the combined SMILES strings for training and testing and standardize them
    if raw_path.endswith('.csv'):
        if smiles_col is None:
            raise Exception(f"Need to specify smiles_col in order to process {raw_path}")
        raw_df = pd.read_csv(raw_path)
        raw_smiles = raw_df[smiles_col].values.tolist()
    else:
        # If input isn't CSV, it's assumed to be a list of SMILES strings, one per line
        with open(raw_path, 'r') as fp:
            raw_smiles = fp.readlines()
    print("Standardizing %d SMILES strings..." % len(raw_smiles))
    base_smiles = shuffle(np.array(list(set(base_smiles_from_smiles(raw_smiles, workers=nworkers)))))
    nsmiles = len(base_smiles)
    print("Got %d unique base SMILES strings." % nsmiles)

    # Generate the vocabulary file
    vocab_path = os.path.join(data_dir, 'vocab.txt')
    if not os.path.exists(vocab_path) or force_update:
        print("Extracting vocabulary...")
        vocab = mol_tree.generate_vocab(base_smiles, workers=nworkers, outfile=vocab_path)
        print("Vocabulary has %d substructures." % len(vocab))
        
    # Split the SMILES strings into training and test sets
    if nsmiles < test_size:
        raise ValueError('test_size is larger than the total number of SMILES strings')
    print("Splitting into train and test subsets...")
    test_ind = sample_without_replacement(nsmiles, test_size)
    train_ind = sorted(set(range(nsmiles)) - set(test_ind))
    train_smiles = base_smiles[train_ind]
    test_smiles = base_smiles[test_ind]
    train_path = os.path.join(data_dir, "%s_train_smiles.txt" % dset_name)
    np.savetxt(train_path, train_smiles, fmt='%s')
    test_path = os.path.join(data_dir, "%s_test_smiles.txt" % dset_name)
    np.savetxt(test_path, test_smiles, fmt='%s')

    # Process the training data into "tensorized" format
    proc_dir = os.path.join(data_dir, 'processed')
    needs_proc = False
    if os.path.exists(proc_dir):
        proc_files = os.listdir(proc_dir)
        if len(proc_files) < 10:
            needs_proc = True
    else:
        needs_proc = True
    if needs_proc or force_update:
        print("Tensorizing training data...")
        os.makedirs(proc_dir, exist_ok=True)
        with pool.Pool(nworkers) as p:
            train_data = p.map(tensorize, train_smiles)
            _write_files(train_data, proc_dir, ntrain_splits)
    print("Done.")

def check_dataset(dset_name, data_root=default_data_root, smiles_col=None, min_atoms=6):
    """
    Check dataset for anomalies such as molecules with no bonds between heavy atoms
    """
    raw_path = dset_data[dset_name]['raw_file']

    # Read the combined SMILES strings for training and testing and standardize them
    if raw_path.endswith('.csv'):
        if smiles_col is None:
            raise Exception(f"Need to specify smiles_col in order to process {raw_path}")
        raw_df = pd.read_csv(raw_path)
        raw_smiles = raw_df[smiles_col].values.tolist()
    else:
        # If input isn't CSV, it's assumed to be a list of SMILES strings, one per line
        with open(raw_path, 'r') as fp:
            raw_smiles = fp.readlines()
    mols = mols_from_smiles(raw_smiles, workers=36)

    for mol, smiles in zip(mols, raw_smiles):
        nbonds = mol.GetNumBonds()
        natoms = mol.GetNumAtoms()
        if nbonds == 0:
            print(f"SMILES {smiles} has no bonds")
        if natoms < min_atoms:
            print(f"SMILES {smiles} has only {natoms} atoms")


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = ArgumentParser()
    parser.add_argument("--dset_name",        dest="dset_name")
    parser.add_argument("--test_size",        dest="test_size", default=1000, type=int)
    parser.add_argument("--nsplits",        dest="nsplits",      default=64, type=int)
    parser.add_argument("--workers",         dest="nworkers",        default=16,  type=int)
    parser.add_argument("--smiles_col",         dest="smiles_col", default=None)
    parser.add_argument("--data_root", dest="data_root", default=default_data_root)
    parser.add_argument("--force_update",  dest="force_update",  action='store_true')
    args = parser.parse_args()
    

    prep_data(dset_name=args.dset_name, test_size=args.test_size, data_root=args.data_root, force_update=args.force_update, 
              ntrain_splits=args.nsplits, nworkers=args.nworkers, smiles_col=args.smiles_col)
