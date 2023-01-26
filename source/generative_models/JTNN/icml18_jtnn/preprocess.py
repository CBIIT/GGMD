#!/usr/bin/env python

import torch
import torch.nn as nn
import pandas as pd
import warnings
from multiprocessing import pool

import math, random, sys, os, pickle, rdkit
from argparse import ArgumentParser
from atomsci.glo.generative_networks.icml18_jtnn.fast_jtnn import *


def tensorize(smiles, assm=True):
    mol_tree = MolTree(smiles)
    mol_tree.recover()
    if assm:
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)

    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol

    return mol_tree


def _write_files(data, directory, num_splits):
    le = int((len(data) + num_splits - 1) / num_splits)

    for split_id in range(num_splits):
        st = int(split_id * le)
        sub_data = data[st : st + le]
        with open(directory + "/tensors-%d.pkl" % split_id, "wb+") as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = ArgumentParser()
    parser.add_argument("--train", dest="train_path")
    parser.add_argument("--test", dest="test_path")
    parser.add_argument("--split", dest="nsplits", default=10, type=int)
    parser.add_argument("--jobs", dest="njobs", default=8, type=int)
    parser.add_argument("--train_output", dest="train_output", default="train_split")
    parser.add_argument("--test_output", dest="test_output", default="test_split")
    args = parser.parse_args()

    if not os.path.exists(args.train_output):
        os.mkdir(args.train_output)
    else:
        warnings.warn("Directory " + args.train_output + " already exists.")

    if not os.path.exists(args.test_output):
        os.mkdir(args.test_output)
    else:
        warnings.warn("Directory " + args.test_output + " already exists.")

    train_smiles = pd.read_csv(args.train_path, header=None)
    test_smiles = pd.read_csv(args.test_path, header=None)
    all_smiles = pd.concat([train_smiles, test_smiles]).values.flatten().tolist()
    vocab = mol_tree.generate_vocab(all_smiles, workers=args.njobs)

    print("Library Size: " + str(len(all_smiles)))
    print("Train Set Size: " + str(len(train_smiles)))
    print("Test Set Size: " + str(len(test_smiles)))
    print("Vocab Size: " + str(len(vocab)))

    train_smiles = train_smiles.values.flatten().tolist()
    with pool.Pool(args.njobs) as p:
        train_data = p.map(tensorize, train_smiles)
        _write_files(train_data, args.train_output, args.nsplits)

        test_data = p.map(tensorize, test_smiles)
        _write_files(test_data, args.test_output, args.nsplits)

    print("Preprocessing complete")
