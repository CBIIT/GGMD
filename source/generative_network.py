import sys
import re
import pandas as pd
#import torch
#import torch.multiprocessing as mp
import pdb
import numpy as np
import ast
import argparse
import logging
import yaml
from yaml import Loader

import optimizer2

from generative_models.FNL_JTNN.fast_jtnn.gen_latent import encode_smiles
from generative_models.JTNN.VAEUtils import DistributedEvaluator
#from rdkit import Chem
#from rdkit import DataStructs
#from rdkit.Chem import AllChem

Log = logging.getLogger(__name__)

# Hack to keep molvs package from issuing debug message on bad Unicode string
molvs_log = logging.getLogger('molvs')
molvs_log.setLevel(logging.WARNING)

#from moses.models_storage import ModelsStorage
#from atomsci.glo.generative_networks.VAEUtils import DistributedEvaluator
#from atomsci.glo import GLOParamParser
#from atomsci.ddm.utils.struct_utils import base_smiles_from_smiles


def create_generative_model(params):
    """
    Factory function for creating optmizer objects of the correct subclass for params.optmizer_type.
    Args:
        params: parameters to pass
    Returns:
        optmizer (object):  wrapper
    Raises: 
        ValueError: Only params.VAE_type = "JTNN" is supported
    """
    if params.model_type.lower() == "jtnn":
        return JTNN(params)
    elif params.model_type.lower() == "moses_charvae":
        print("loading moses charvae")
        return charVAE(params)
    elif params.model_type.lower() == 'jtnn-fnl':
        return JTNN_FNL(params)
    else:
        raise ValueError("Unknown model_type %s" % params.model_type)


class GenerativeModel(object):
    def __init__(self, params, **kwargs):
        """
        Initialization method for the GenerativeModel class

        Args:
            params (namespace object): contains the parameters used to initialize the class
        MJT Note: do we need the **kwargs argument?
        """
        self.params = params

    def load_txt_to_encode(self, txt_filepath):
        """
        Loads in a text file of SMLIES strings specified in txt_filepath

        Args:
            txt_filepath (str): The full path to the text file containing SMILES strings (expects SMILES strings to be
                separated by a newline)

        Returns: self.SMILES (list (str)): list of the SMILES strings

        """
        with open(txt_filepath, "r") as file_handle:
            SMILES = file_handle.readlines()
            SMILES = [smi.rstrip("\n") for smi in SMILES]
        self.SMILES = SMILES

    def sanitize(self):
        """
        Sanitize the SMILES in self.SMILES. Dependent on self.SMILES attribute

        Returns: self.SMILES (list (str)): list of the sanitized SMILES strings
        """
        # TODO: Add a Kekulize option to base_smiles_from_smiles
        self.SMILES = base_smiles_from_smiles(
            orig_smiles=self.SMILES,
            useIsomericSmiles=True,
            removeCharges=False,
            workers=1,
        )
        return self

    def encode(self):
        """
        encode smiles not implemented in super class
        """
        raise NotImplementedError

    def decode(self):
        """
        decode smiles not implemented in super class
        """
        raise NotImplementedError
    
    def optimize(self):
        """
        optimize function not implemented in super class
        """
        raise NotImplementedError



class JTNN_FNL(GenerativeModel):

    def __init__(self, params):
        self.encoder = encode_smiles(params)
    def encode(self, smiles):
        
        latent = self.encoder.encode(smiles)
        

    def decode(self):
        pass
    def optimize(self):
        pass
    


class JTNN(GenerativeModel):

    def __init__(self, params):
        self.device = params.device
        self.vae_path = params.vae_path
        self.vocab_path = params.vocab_path

        self.vae = DistributedEvaluator(
            #device=self.device,
            timeout=15, #TODO: is this a parameter likely to be tweaked?
            vae=self.vae_path,
            vocab=self.vocab_path,
        )

        self.optimizer = optimizer2.create_optimizer(params)

    def encode(self, population):
        smiles = population['smiles'].tolist()
        latent, keep_compounds, dataset = self.vae.encode_smiles(smiles) #TODO: do we need this returned variable: dataset???
        print("smiles encoded")

        population = population.iloc[keep_compounds]
        population['latent'] = list(latent)
        #TODO: Look into this warning:
        #/mnt/projects/ATOM/blackst/GenGMD/source/generative_network.py:149: SettingWithCopyWarning: 
        #A value is trying to be set on a copy of a slice from a DataFrame.
        #Try using .loc[row_indexer,col_indexer] = value instead
        
        return population
    
    def encode_test(self, smiles):
        # TODO: This is a test function to test the idea of only sending the compounds
        # that have not previously been encoded. This would save time over generations. 
        # Need to continue to explore how to handle removed compounds.

        latent, keep_compounds, dataset = self.vae.encode_smiles(smiles) #TODO: do we need this returned variable: dataset???

        return smiles, latent
        
    def decode(self, latent):

        if len(latent) == 0:
            raise Exception("the latent seems emtpy...")

        if type(latent) is list:
            latent = [np.asarray(l) for l in latent]
        else:
            print(type(latent))

        smiles, _ = self.vae.decode_smiles(latent)
        
        return smiles

    def optimize_test(self, population):
        # TODO: This is a test function to test the idea of only sending the compounds
        # that have not previously been encoded. This would save time over generations. 
        # Need to continue to explore how to handle removed compounds.

        smiles_to_encode = population['smiles'].loc[population['latent'].isna()]

        latent = self.encode_test(smiles_to_encode)

        for s, l in zip(smiles_to_encode, latent):
            population['latent'].loc[population['smiles'] == s] = l

        population = self.optimizer.optimize(population)
    
    def optimize(self, population):
        population = self.encode(population)

        population = self.optimizer.optimize(population)


        latent = list(population['latent'].loc[population['smiles'].isna()])

        print(type(latent))
        smiles = self.decode(latent)
        for s, l in zip(smiles, latent):
            population['smiles'].loc[population['latent'] == l] = s

        #TODO: Write test functions for 

        return population



class CHAR_VAE(GenerativeModel):

    def __init__(self, params):
        pass
    def encode(self):
        pass
    def decode(self):
        pass
    def optimize(self):
        pass


def test_jtvae(args):
    population = pd.read_csv("/mnt/projects/ATOM/blackst/GenGMD/source/evaluated_pop.csv")
    latent = list(population['latent'].loc[population['smiles'].isna()])

    print(type(latent))
    print(type(latent[0]))
    #print(type(np.asarray(latent)))
    #print(len(latent))
    #print(len(latent[0]))

    vae = JTNN(args)

    smiles = vae.decode(latent)

    
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', help="Config file location *.yml", action='append', required=True)
    args = parser.parse_args()

    for conf_fname in args.config:
        with open(conf_fname, 'r') as f:
            parser.set_defaults(**yaml.load(f, Loader=Loader))

    args = parser.parse_args()

    test_jtvae(args)










