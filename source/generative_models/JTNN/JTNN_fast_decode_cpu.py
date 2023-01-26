import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os, sys, torch, rdkit

import numpy as np

from icml18_jtnn.fast_jtnn_cpu.mol_tree import Vocab
from icml18_jtnn.fast_jtnn_cpu.jtnn_vae import JTNNVAE

from tqdm import tqdm
from rdkit import RDLogger
import signal

torch.set_num_threads(1)  # required to prevent pytorch from spawning useless threads

lg = RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


def LoadAutoEncoder(
    vae_path=os.path.dirname(os.path.abspath(__file__))
    + "/icml18_jtnn/fast_molvae/moses-h450z56/model.iter-400000",
    vocab_path=os.path.dirname(os.path.abspath(__file__))
    + "/icml18_jtnn/data/moses/vocab.txt",
    hidden_size=450,
    latent_space=56,
    depthT=20,
    depthG=3,
):
    """Loads a previously trained JTNNVAE from vae_path with a starting vocab from vocab_path using the cpu or the gpu

    """
    vocab = [x.strip("\r\n ") for x in open(vocab_path)]
    vocab = Vocab(vocab)

    vae = JTNNVAE(vocab, hidden_size, latent_space, depthT, depthG)

    load_dict = torch.load(vae_path, map_location="cpu")

    missing = {k: v for k, v in vae.state_dict().items() if k not in load_dict}
    load_dict.update(missing)
    vae.load_state_dict(load_dict)

    vae = vae.cpu()

    return vae


def timeout_handler(num, stack):
    """ Handles the timeout exception for the system signal when a smiles decode step takes too long"""
    raise Exception("SMILES_DECODE_TIMEOUT")


def decode(latent_space, vae, vocab, mol_size=28, tree_size=56):
    """
    Loads in the decoder and applies it iteratively to an input latent_space
    Args:
        latent_space (2D tensor): The latent space from the encoder
        mol_size (int): The size of the molecular graph latent space
        tree_size (int): The size of the mol tree latent space
    Returns:
        smiles (list (str)): A list of smiles strings decoded from the latent space

    """
    model_glob = LoadAutoEncoder(
        vae_path=vae, vocab_path=vocab
    )  # loads in the cpu autoencoder
    smiles = []  # empty list for appending the smiles
    # mol_size = 28 #size of the moltree latent space
    # tree_size = 56 #size of the molgraph latent space
    for ind in latent_space:
        mol_trees = ind[0:mol_size].unsqueeze(
            0
        )  # adding in the required dimensionality for mol_tress and graph_trees
        graph_trees = ind[mol_size:tree_size].unsqueeze(0)
        smiles.append(
            model_glob.decode(mol_trees, graph_trees, False)
        )  # decoding a single latent vector

    return smiles


def decode_timer(latent_space, vae, vocab, verbose=False):
    """
    Loads in the decoder and applies it iteratively to an input latent_space, includes timer and verbosity
    Args:
        latent_space (2D tensor): The latent space from the encoder
        verbose (bool): If True, uses tqdm as a status bar and also continues to a new string if the decoder takes longer than time_limit
    Returns:
        smiles (list (str)): A list of smiles strings decoded from the latent space

    """
    model_glob = LoadAutoEncoder(vae_path=vae, vocab_path=vocab)
    time_limit = 10
    smiles = []
    mol_size = 28
    tree_size = 56
    if verbose:
        for ind in tqdm(list(range(len(latent_space))), total=len(latent_space)):
            mol_trees = latent_space[ind, 0:mol_size].unsqueeze(0)
            graph_trees = latent_space[ind, mol_size:tree_size].unsqueeze(0)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(time_limit)
            try:
                smiles.append(model_glob.decode(mol_trees, graph_trees, False))
            except Exception as ex:
                if ex == "SMILES_DECODE_TIMEOUT":
                    continue
                # else:  #unfortunately, this else statement prints something if there is a timeout
                #    print(ex)
    else:
        for ind in latent_space:
            mol_trees = ind[0:mol_size].unsqueeze(0)
            graph_trees = ind[mol_size:tree_size].unsqueeze(0)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(time_limit)
            try:
                smiles.append(model_glob.decode(mol_trees, graph_trees, False))
            except Exception as ex:
                if ex == "SMILES_DECODE_TIMEOUT":
                    continue

    return smiles


if __name__ == "__main__":
    """Main method for command line use, takes in a single argument of a file_name (str) containing a latent space """
    file_name = sys.argv[3]
    vae_path = sys.argv[1]
    vocab_path = sys.argv[2]
    latent = np.loadtxt(file_name, delimiter=",")
    latent_tree_size = int(latent.shape[1] / 2)
    latent = torch.Tensor(latent)

    smiles = decode(
        latent,
        vae_path,
        vocab_path,
        mol_size=latent_tree_size,
        tree_size=latent.shape[1],
    )

    print(*smiles, sep="\n")  # prints each smile string on a new line
