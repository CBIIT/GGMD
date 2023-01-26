import numpy as np
import time, warnings, torch

from .vocab import Vocab
from .jtnn_vae import JTNNVAE
from rdkit.Chem import MolFromSmiles, MolToSmiles
from joblib import Parallel, delayed

class encode_smiles():

    def __init__(self, params):
        self.hidden_size = 450 #default
        self.latent_size = 128 #default
        self.depthT = 20 #default
        self.depthG = 3 #default
        self.n_cpus = 36 #default
        self.bat_size = 200 #default
        self.verbose = False #default
        self.vocab_path = params.vocab_path
        self.model_path = params.model_path
        #self.lat_out #Remove: we are returning rather than saving to file

    #JI - Wrapper for encode_latent_mean 
    def wrap_encode(self, smiles):
        warnings.filterwarnings("ignore")
        mol_v = self.model.encode_latent_mean(smiles)
        return mol_v

    def encode(self, smiles):
        start_time = time.time()

        #hidden_size = int(opts.hidden_size)
        #latent_size = int(opts.latent_size)
        #depthT = int(opts.depthT)
        #depthG = int(opts.depthG)
        #n_cpus = int(opts.ncpus)
        #bat_size = int(opts.bats)
        #verbose = bool(opts.verbose)
        #lat_out = opts.lat_out

        if self.verbose == True:
            n_verb = 100  
        else:
            n_verb = 0

        #with open(opts.data_path) as f:
        #    smiles_list = [line.strip("\r\n ").split()[0] for line in f]

        #JI - We generate smiles_rdkit here now rather than later
        #JI - Note that currently this Fast-JTVAE doesn't use stereochemistry
    
        smiles_rdkit = []
        for s in smiles:
            mol = MolFromSmiles(s)
            smi = MolToSmiles(mol,isomericSmiles=False)
            smiles_rdkit.append(smi)

        n_data = len(smiles_rdkit)
        vocab = [x.strip("\r\n ") for x in open(self.vocab_path)] 
        vocab = Vocab(vocab)

        #JI - Load Fast-JTVAE Model/Autoencoder

        self.model = JTNNVAE(vocab, self.hidden_size, self.latent_size, self.depthT, self.depthG)
        self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        self.model = self.model.cpu()

        curr_time = time.time()

        latent_points = []
        batches = [smiles_rdkit[i:i+self.bat_size] for i in range(0, n_data, self.bat_size)]

        #JI - Encode SMILES in parallel

        all_vec = Parallel(n_jobs=self.n_cpus,batch_size=1,max_nbytes=None,mmap_mode=None,verbose=n_verb)\
                (delayed(self.wrap_encode)(batch) for batch in batches)
        print ('Encoding computation time, Total time = %.0f, %0.f seconds' % \
            ((time.time() - curr_time), (time.time() - start_time)))
        #curr_time = time.time()

        for i in range(0, len(all_vec)):
            latent_points.append(all_vec[i].data.cpu().numpy())

        latent_points = np.vstack(latent_points)

        return latent_points
























