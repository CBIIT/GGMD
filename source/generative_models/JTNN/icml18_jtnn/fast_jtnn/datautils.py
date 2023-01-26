import torch
from torch.utils.data import Dataset, DataLoader
from mol_tree import MolTree
import numpy as np
from jtnn_enc import JTNNEncoder
from mpn import MPN
from jtmpn import JTMPN
from multiprocessing import pool
from functools import partial
import os, random, pickle


class PairTreeFolder(object):
    def __init__(
        self,
        data_folder,
        vocab,
        batch_size,
        num_workers=4,
        shuffle=True,
        y_assm=True,
        replicate=None,
    ):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.y_assm = y_assm
        self.shuffle = shuffle

        if replicate is not None:  # expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn) as f:
                data = pickle.load(f)

            if self.shuffle:
                random.shuffle(data)  # shuffle data before batch

            batches = [
                data[i : i + self.batch_size]
                for i in range(0, len(data), self.batch_size)
            ]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = PairTreeDataset(batches, self.vocab, self.y_assm)
            dataloader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=lambda x: x[0],
            )

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader


class MolTreeFolder(object):
    def __init__(
        self,
        data_folder,
        vocab,
        batch_size,
        num_workers=4,
        shuffle=True,
        assm=True,
        replicate=None,
    ):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm

        if replicate is not None:  # expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, "rb") as f:
                data = pickle.load(f)

            if self.shuffle:
                random.shuffle(data)  # shuffle data before batch

            batches = [
                data[i : i + self.batch_size]
                for i in range(0, len(data), self.batch_size)
            ]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = MolTreeDataset(batches, self.vocab, self.assm)
            dataloader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=lambda x: x[0],
            )

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader


class PairTreeDataset(Dataset):
    def __init__(self, data, vocab, y_assm):
        self.data = data
        self.vocab = vocab
        self.y_assm = y_assm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch0, batch1 = list(zip(*self.data[idx]))
        return (
            tensorize(batch0, self.vocab, assm=False),
            tensorize(batch1, self.vocab, assm=self.y_assm),
        )


class MolTreeDataset(Dataset):
    def __init__(self, data, vocab, assm=True):
        self.data = data
        self.vocab = vocab
        self.assm = assm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return tensorize(self.data[idx], self.vocab, assm=self.assm)

    def get_smiles(self):
        return [tree[0].smiles for tree in self.data]

    def check_vocab(self, workers=1):
        invalid_nodes, invalid_cmpds = check_vocab_job(
            self.data, self.vocab, workers=workers
        )
        return invalid_nodes, invalid_cmpds

    def remove_invalid_vocab(self, workers=1, verbose=True):
        ncmpd = len(self.data)
        invalid_nodes, invalid_cmpds = check_vocab_job(
            self.data, self.vocab, workers=workers, verbose=verbose
        )
        if len(invalid_cmpds) >= 1:
            smiles = self.get_smiles()
            idx = [smiles.index(cmpd) for cmpd in invalid_cmpds]
            new_data = list(np.delete(np.asarray(self.data), idx))
            self.data = [[data] for data in new_data]

            if verbose:
                print_str = str(invalid_cmpds)
                print_str = print_str[1 : len(print_str) - 1]
                print_str = (
                    "Removed "
                    + str(len(idx))
                    + " out of "
                    + str(ncmpd)
                    + " compounds not compatible with the vocabulary: \n "
                    + print_str
                )
                print(print_str.replace(",", "").replace("'", ""))

        return self, invalid_cmpds


# TODO(derek): this is confusing...
def check_vocab_job(mols, vocab, workers=1, batch_size=4, verbose=True):
    if workers > 1:
        mol_batches = [
            mols[i : i + batch_size] for i in range(0, len(mols), batch_size)
        ]
        with pool.Pool(workers) as p:
            f = partial(check_vocab_job, vocab=vocab, verbose=verbose)
            invalid = p.map(f, mol_batches)
            invalid_nodes = np.concatenate([inv[0] for inv in invalid])
            invalid_cmpds = np.concatenate([inv[1] for inv in invalid])
    else:

        invalid_nodes = []
        invalid_cmpds = []
        for mol in mols:
            try:
                if isinstance(mol, list) & (len(mol) == 1):
                    mol = mol[0]
                for node in mol.nodes:
                    smiles = node.smiles
                    vocab.vmap[node.smiles]
            except Exception as e:
                if verbose:
                    print("Unrecognized vocab: " + str(node.smiles))
                invalid_cmpds.append(mol.smiles)
                invalid_nodes.append(node.smiles)
    return (np.unique(invalid_nodes), np.unique(invalid_cmpds))


def tensorize(tree_batch, vocab, assm=True):
    set_batch_nodeID(tree_batch, vocab)
    smiles_batch = [tree.smiles for tree in tree_batch]
    jtenc_holder, mess_dict = JTNNEncoder.tensorize(tree_batch)
    jtenc_holder = jtenc_holder
    mpn_holder = MPN.tensorize(smiles_batch)

    if assm is False:
        return tree_batch, jtenc_holder, mpn_holder

    cands = []
    batch_idx = []
    for i, mol_tree in enumerate(tree_batch):
        for node in mol_tree.nodes:
            # Leaf node's attachment is determined by neighboring node's attachment
            # if node.is_leaf or len(node.cands) == 1: continue - limiting by cands cause trouble for some molecules
            if node.is_leaf:
                continue
            cands.extend([(cand, mol_tree.nodes, node) for cand in node.cands])
            batch_idx.extend([i] * len(node.cands))

    if len(cands) > 0:
        jtmpn_holder = JTMPN.tensorize(cands, mess_dict)
    else:
        jtmpn_holder = (
            []
        )  # Some molecules have no non-leaf elements, which casues issues

    batch_idx = torch.LongTensor(batch_idx)

    return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder, batch_idx)


def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1


def mol_tensor_to_device(tensor, device):
    tensor = list(tensor)
    tensor[3] = list(tensor[3])
    tensor[3][0] = list(tensor[3][0])
    tensor[1] = tuple(
        [tsr.to(device) if isinstance(tsr, torch.Tensor) else tsr for tsr in tensor[1]]
    )
    tensor[2] = tuple(
        [tsr.to(device) if isinstance(tsr, torch.Tensor) else tsr for tsr in tensor[2]]
    )
    tensor[3][0] = tuple(
        [
            tsr.to(device) if isinstance(tsr, torch.Tensor) else tsr
            for tsr in tensor[3][0]
        ]
    )
    tensor[3][1] = tensor[3][1].to(device)
    tensor[3][0] = tuple(tensor[3][0])
    tensor[3] = tuple(tensor[3])
    tensor = tuple(tensor)

    return tensor
