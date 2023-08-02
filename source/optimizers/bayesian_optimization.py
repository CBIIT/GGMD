from optimizers.base_optimizer import Optimizer

import torch, time, rdkit, pickle, gzip, os.path

from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
import scipy.stats as sps
import networkx as nx
import torch.nn as nn
import numpy as np

import sascorer

from sparse_gp import SparseGP
from generative_models.FNL_JTNN.fast_jtnn import create_var, JTNNVAE, Vocab

class BayesianOptimizer(Optimizer):
    def __init__(self, params):
        vocab = [x.strip("\r\n ") for x in open(params.vocab_path)] 
        self.vocab = Vocab(vocab)

        hidden_size = int(params.hidden_size)
        latent_size = int(params.latent_size)
        depthT = 20 #Default, need to update to accept as parameter
        depthG = 3 #Default, need to update to accept as parameter

        model = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG)
        model.load_state_dict(torch.load(params.model_path, map_location='cpu'))
        model = model.cpu()


    def optimize(self):
        pass