import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os, sys, io, shutil, torch, rdkit, pickle, time, traceback, threading #Removing rdkit/torch temporarily to keep using temp conda environment


import torch.multiprocessing as mp 
mp.set_sharing_strategy("file_system") 

from math import isnan

import numpy as np
from numpy import ndarray, ceil, asarray
from pandas import DataFrame, read_csv
from torch.utils.data import DataLoader 


from generative_models.JTNN.icml18_jtnn.fast_jtnn.mol_tree import Vocab
from generative_models.JTNN.icml18_jtnn.fast_jtnn.jtnn_vae import JTNNVAE
from generative_models.JTNN.icml18_jtnn.fast_jtnn.datautils import MolTree, MolTreeDataset, mol_tensor_to_device
#from atomsci.ddm.pipeline.chem_diversity import calc_dist_smiles


from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, TimeoutError
from pebble import concurrent, ProcessPool 
from concurrent.futures import TimeoutError

from datetime import datetime
from tqdm import tqdm

import pandas as pd

from rdkit import rdBase, RDLogger

lg = RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

default_vae_path = os.path.dirname(os.path.abspath(__file__)) + '/icml18_jtnn/fast_molvae/moses-h450z56/model.iter-400000'
def load_vae(vae_path   = default_vae_path,
             vocab_path = None,
             verbose    = True,
             device     = 'cuda'):

    vae_dict = torch.load(vae_path, map_location=device)

    if "vocab" in vae_dict.keys():
        vocab = vae_dict["vocab"]
        hyperparams = vae_dict["hyperparams"]
        hyperparams['epoch'] = vae_dict['epoch']
        hyperparams['exact_epoch'] = vae_dict['exact_epoch']
    elif "hyperparams" in vae_dict.keys():
        try:
            hyperparams = vae_dict["hyperparams"]
            vocab = [x.strip("\r\n ") for x in open(hyperparams["vocab"])]
        except:
            hyperparams = {"hidden_size": 450, "latent_size": 56, "depthT": 20, "depthG": 3}
        vocab = pd.read_csv(vocab_path,header=None).values.squeeze()
    else:
        hyperparams = {"hidden_size": 450, "latent_size": 56, "depthT": 20, "depthG": 3}
        vocab = pd.read_csv(vocab_path,header=None).values.squeeze()
    vae = JTNNVAE(Vocab(vocab),
                  hyperparams["hidden_size"],
                  hyperparams["latent_size"],
                  hyperparams["depthT"],
                  hyperparams["depthG"],
                  device=device)

    # The error handling here is mostly for legacy models...
    try:
        vae.load_state_dict(vae_dict["model_state_dict"], strict=False)
    except:
        warnings.warn('Unable to load VAE state dict, this can occur in legacy models')

    if 'epoch' in vae_dict.keys():
        hyperparams['epoch'] = vae_dict['epoch']
    else:
        hyperparams['epoch'] = 'unknown'

    if 'exact_epoch' in vae_dict.keys():
        hyperparams['exact_epoch'] = vae_dict['exact_epoch']
    else:
        hyperparams['exact_epoch'] = 'unknown'
    return vae, hyperparams


#This should probably be unified with the tensorize code in preprocess and the device managment code in vae_train and set up as a data util
def tensorize(SMILES, workers=1, assm=True, device='cpu'):
    if isinstance(SMILES, list) | isinstance(SMILES, np.ndarray):
        if workers > 1:
            with mp.Pool(workers) as p:
                mol_tree = p.map(tensorize, SMILES)
        else:
            mol_tree = [tensorize(smi) for smi in SMILES]
    else:
        mol_tree = MolTree(SMILES)
        mol_tree.recover()
        if assm:
            mol_tree.assemble()
            for node in mol_tree.nodes:
                if node.label not in node.cands:
                    node.cands.append(node.label)

        del mol_tree.mol
        for node in mol_tree.nodes:
            del node.mol
        mol_tree = [mol_tree]

    return mol_tree


def baseline_recovery(smiles,
                      vae_path=os.path.dirname(
                          os.path.abspath(__file__)) + '/icml18_jtnn/fast_molvae/moses-h450z56/model.iter-400000',
                      vocab_path=os.path.dirname(os.path.abspath(__file__)) + '/icml18_jtnn/data/moses/vocab.txt',
                      latent_size=56,
                      processes=mp.cpu_count(), #TEMPORARY comment for development
                      #processes=1,
                      timeout=60,
                      verbose=True,
                      prob_decode=True):
    evaluator = DistributedEvaluator(device='cuda:0', vae_path=vae_path, vocab_path=vocab_path, verbose=verbose)
    evaluator_cpu = DistributedEvaluator(device='cpu', vae_path=vae_path, vocab_path=vocab_path, verbose=verbose)

    mols = tensorize(smiles, workers=processes)
    dataset = MolTreeDataset(mols, evaluator.vae.vocab)

    #init_set_len = len(dataset.data)
    dataset, invalid_compounds = dataset.remove_invalid_vocab(verbose=verbose, workers=processes)

    removed = invalid_compounds  # need to know this value later on, so storing it in the object
    if len(removed) > 0:
        print('Removed %s compounds from set incompatible with JT VAE vocabulary' % (len(removed)))

    # Need to fix collate function to support larger batches
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.workers, collate_fn=lambda x: x[0])
    all_smiles_list = []

    for data in tqdm(dataloader):
        jtenc_holder = data[1]
        mpn_holder = data[2]
        latent_mean, latent_var = evaluator.vae.encode_latent(jtenc_holder, mpn_holder)

        tree_mean = latent_mean[:, :int(latent_size / 2)]
        mol_mean = latent_mean[:, int(latent_size / 2):]

        tree_log_var = latent_var[:, :int(latent_size / 2)]
        mol_log_var = latent_var[:, int(latent_size / 2):]

        all_smiles_list.append(baseline_recovery_job(evaluator=evaluator,
                                                     evaluator_cpu=evaluator_cpu,
                                                     tree_mean=tree_mean,
                                                     mol_mean=mol_mean,
                                                     tree_log_var=tree_log_var,
                                                     mol_log_var=mol_log_var,
                                                     latent_size=latent_size,
                                                     processes=processes,
                                                     timeout=timeout,
                                                     prob_decode=prob_decode))  # NOTE: for now assuming that input is always 1 by N

    return {"smiles": dataset.get_smiles(), "eval": all_smiles_list, "removed": removed}


def baseline_recovery_job(evaluator, evaluator_cpu, tree_mean, mol_mean, tree_log_var, mol_log_var, latent_size,
                          processes, timeout, prob_decode):
    latent_vec_list = []  # keep this to store all vecs that are then passed to parallelized decode function

    for i in range(10):
        tree_epsilon = torch.randn(1, int(latent_size / 2)).cuda()

        tree_vec = tree_mean + torch.exp(tree_log_var / 2) * tree_epsilon

        mol_epsilon = torch.randn(1, int(latent_size / 2)).cuda()

        mol_vec = mol_mean + torch.exp(mol_log_var / 2) * mol_epsilon

        latent_vec = torch.cat([mol_vec, tree_vec], dim=1)
        latent_vec_list.extend([latent_vec] * 10)

    latent_vec_batch = torch.cat(latent_vec_list)
    latent_vec_batch = latent_vec_batch.cpu().data.numpy()

    output_smiles_list = []

    output_smiles_list.extend(
        evaluator_cpu.decode_smiles(evaluator_cpu.vae, latent_vec_batch, processes=processes, timeout=timeout,
                                    verbose=False, prob_decode=True))

    return output_smiles_list


class DistributedEvaluator(object):

    def __init__(self,
                 vae     = None,
                 vocab   = None, # Only need to specify this for legacy reasons
                 device  = 'cpu',
                 verbose = False,
                 workers = mp.cpu_count(), #TEMPORARY comment for development
                 #workers = 1,
                 timeout = 60):
        self.device  = device
        self.verbose = verbose
        self.workers = workers
        self.timeout = timeout

        if (not torch.cuda.is_available()) & (device is not 'cpu'): #TEMPORARY comment for development
             device = 'cpu'
             warnings.warn('Cuda unavalable, DistributedEvaluator defaulting to cpu mode')
        #device = 'cpu'

        if isinstance(vae,str):
            self.vae, self.hyperparams = load_vae(vae, vocab_path=vocab, device=self.device, verbose=self.verbose)

        else:
            self.vae = vae
            self.hyperparams = []

        self.vae = self.vae.eval()  # we're not training so put model into eval mode

    def encode_smiles(self, SMILES):
        
        if isinstance(SMILES,str):
            SMILES = [SMILES]
        
        #self.removed = None # TODO: DELETE after testing
        mols = tensorize(SMILES, workers=self.workers, device=self.device)
        dataset = MolTreeDataset(mols, self.vae.vocab)

        init_set_len = len(dataset.data)
        dataset, invalid_compounds = dataset.remove_invalid_vocab(verbose=self.verbose, workers=self.workers)
        
        keep_compounds = [smiles not in invalid_compounds for smiles in SMILES]


        #self.removed = invalid_compounds  # TODO: DELETE after testing
        if len(invalid_compounds) > 0:
            print('Removed %s compounds from set incompatible with JT VAE vocabulary' % (len(invalid_compounds)))
 
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.workers, collate_fn=lambda x:x[0])

        latent = []
        for data in tqdm(dataloader, desc='encoding smiles', disable=True):
            data = mol_tensor_to_device(data,self.device)
            jtenc_holder = data[1]
            mpn_holder   = data[2]
            latent_mean, latent_var = self.vae.encode_latent(jtenc_holder, mpn_holder)
            latent.append(latent_mean.cpu().data.numpy())

        latent = np.asarray(latent).squeeze()
        return latent, keep_compounds, dataset
    
    
    def _submit_decode_thread(self, latent_space, prob_decode = False):
        # Use threadpool executor to set a timeout on decode time, force single torch thread for multiprocess efficiency
        #torch.set_num_threads(1)
        os.environ["OMP_NUM_THREADS"] = "1"
        try:
            with ThreadPoolExecutor(max_workers=1) as tpe:
                res = tpe.submit(self._decode_helper, latent_space, prob_decode=prob_decode)
                out = res.result(timeout=self.timeout)  # Wait timeout seconds for func to complete.
        except TimeoutError:
            out = ([''], [self.timeout])  # return value is expected to (smiles, time)
        return out

    
    def _decode_helper(self,
                       latent_space,
                       return_timing=True,
                       prob_decode=False,
                       verbose=False):
        
        # Ensure correct typeof latent tensor
        if not isinstance(latent_space, torch.Tensor):
            latent_space = torch.Tensor(latent_space)
        if len(latent_space.size()) == 1:
            latent_space = latent_space.reshape((1, len(latent_space)))

        tree_size = int(latent_space.shape[1] / 2)
        mol_size  = int(latent_space.shape[1])

        smiles, times = [], []  # empty list for appending the smiles
        for vec in latent_space:
            
            begin = time.time()
            # adding in the required dimensionality for mol_trees and graph_trees and send to device
            tree_vec = vec[0:tree_size].unsqueeze(0).to(self.vae.device())
            mol_vec  = vec[tree_size:mol_size].unsqueeze(0).to(self.vae.device())
            try:
                smi = self.vae.decode(tree_vec, mol_vec, prob_decode)
            except Exception as e:
                print('Unable to decode latent vec :' + str(vec))
                sys.stdout.flush()
                smi = traceback.print_exc()

            smiles.append(smi)
            elapsed = time.time() - begin
            times.append(elapsed)

        if return_timing:
            return smiles, times
        else:
            return smiles

        
    def decode_smiles(self, latent_space, prob_decode=False):
        """Method for decoding a given latent_space using the vae model in the evaluator.

           Args:
               latent_space (NxN array of floats): The latent space representation to be decoded
               prob_decode (bool): Flag for probabilistic decoding (True to turn prob decode on)

           Returns:
               tuple:
                    smiles (np.array): An array list of the decoded smiles strings
                    times (np.array): An array of the decoding times.
        """
 
        # NOTE (derek): this is here to make sure that the rng is seeded according to the wall clock for both torch and 
        # numpy o.w. random sequences are identical
        np.random.seed()
        torch.manual_seed(np.random.get_state()[1][0])
        
        if np.ndim(latent_space) == 1:
            latent_space = latent_space.reshape((1, len(latent_space)))

        decode_func = self._decode_helper
        
        if (self.device == 'cpu') & (self.workers > 1):

            smiles = []
            times = []
            #chunksize = max(1, int(len(latent_space) / self.workers))
            chunksize = 1
            with ProcessPool(max_workers=self.workers) as p:
                torch.set_num_threads(1)
                os.environ["OMP_NUM_THREADS"] = "1"
                fut = p.map(decode_func, latent_space, chunksize=chunksize, timeout=chunksize*self.timeout)
                res_iter = fut.result()
                res_num = 0
                while True:
                    try:
                        res_smiles, res_time = next(res_iter)
                        #print(f"Decoding succeeded for latent vector {res_num}, result={res_smiles}, time={res_time}")
                        smiles.extend(res_smiles)
                        times.extend(res_time)
                    except StopIteration:
                        break
                    except TimeoutError:
                        #print(f"Decoding timed out for latent vector {res_num}")
                        smiles.extend(['<Decoding Timed Out>'] * chunksize)
                        times.extend([self.timeout] * chunksize)
                    res_num += 1
        else:
            out = [decode_func(l) for l in latent_space]

            # Unpack results
            if len(out) == 1:
                out = out[0]
            else:
                out = list(zip(*out))

            smiles = np.array(out[0]).squeeze()
            times  = np.array(out[1]).squeeze()
        
        #print("Done with decoding")
        sys.stdout.flush()

        return np.array(smiles), np.array(times)

    
    def to(self, device):
        self.vae          = self.vae.to(device)
        self.vae.jtnn     = self.vae.jtnn.to(device)
        self.vae.jtmpn    = self.vae.jtmpn.to(device)
        self.vae.decoder  = self.vae.decoder.to(device)
        self.vae.jtnn.GRU = self.vae.jtnn.GRU.to(device) 
        self.device       = device
        return self

        
    def recover(self,
                smiles):

        if isinstance(smiles, str):
            smiles = [smiles]

        start = time.time()
        latent, dataset = self.encode_smiles(smiles)
        if self.verbose:
            print('Encoding successful, took: ' + str(time.time() - start) + ' seconds')

        # Move vae to cpu for parallel decode
        device = self.vae.device()
        if self.vae.device is not 'cpu':
            self.vae.to('cpu')

        start = time.time()
        recovered, decode_time = self.decode_smiles(latent)
        if self.verbose:
            print('Decoding took: ' + str(time.time() - start) + ' seconds')

        self.to(device)  # Return evaluator to initial device

        output = pd.DataFrame({'Input': dataset.get_smiles(),
                               'Output': recovered})
        dist = []
        for row in output.iterrows():
            try:
                dist.append(
                    calc_dist_smiles('ecfp', 'tanimoto', [row[1]['Input']], [row[1]['Output']], calc_type='all'))
            except:
                dist.append([1])

        output['Tanimoto Distance'] = np.array(dist).squeeze()
        output['Recovered']         = output['Input'] == output['Output']
        output['Decode Time']       = decode_time
        if self.verbose:
            print("Recovery rate = {:3.1f} %".format(sum(output['Recovered'].values) / len(output) * 100))
            

        return output


if __name__ == "__main__":
    # TODO: can the main script be removed?
    """
    fname = sys.argv[1]
    with open(fname, "r") as file_handle:
        smiles = file_handle.readlines()
        smiles = [smi.rstrip('\n') for smi in smiles]
    # smiles=smiles[:10]

    # recover(smiles)
    result = baseline_recovery(smiles, prob_decode=True)

    with open("testing_baseline_recovery.pkl", 'wb') as handle:
        pickle.dump(result, handle)
    """
def test_encode():
    smiles = [
        "CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1",
        "C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1",
        "CC1(C)CC[C@H](CNC(=O)Cn2ncc3ccccc3c2=O)c2ccccc21"
    ]

    vae = DistributedEvaluator(
            #device=self.device,
            timeout=15, #TODO: is this a parameter likely to be tweaked?,
            vae='/mnt/projects/ATOM/blackst/GenGMD/source/generative_models/JTNN/icml18_jtnn/fast_molvae/moses-h450z56/model.iter-400000',
            vocab='/mnt/projects/ATOM/blackst/GenGMD/source/generative_models/JTNN/icml18_jtnn/data/moses/vocab.txt'
        )
    
    latent, dataset = vae.encode_smiles(smiles)
    print(type(latent), len(latent))
    print(type(dataset))

