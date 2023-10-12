import numpy as np
import time, warnings, torch

from .nnutils import create_var
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
        self.vocab = Vocab([x.strip("\r\n ") for x in open(params.vocab_path)])
        self.model_path = params.model_path
        self.counter = 0

    #JI - Wrapper for encode_latent_mean 
    def wrap_encode(self, smiles):
        warnings.filterwarnings("ignore")
        mol_v = self.model.encode_latent_mean(smiles)
        return mol_v

    def encode(self, smiles):
        start_time = time.time()

        if self.verbose == True:
            n_verb = 100
        else:
            n_verb = 0

        #JI - We generate smiles_rdkit here now rather than later
        #JI - Note that currently this Fast-JTVAE doesn't use stereochemistry
    
        smiles_rdkit = []
        for s in smiles:
            mol = MolFromSmiles(s)
            smi = MolToSmiles(mol,isomericSmiles=False)
            smiles_rdkit.append(smi)

        n_data = len(smiles_rdkit)

        #JI - Load Fast-JTVAE Model/Autoencoder

        self.model = JTNNVAE(self.vocab, self.hidden_size, self.latent_size, self.depthT, self.depthG)
        self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        self.model = self.model.cpu()

        curr_time = time.time()

        latent_points = []
        batches = [smiles_rdkit[i:i+self.bat_size] for i in range(0, n_data, self.bat_size)]
        print(len(batches), " batches of smiles")

        #JI - Encode SMILES in parallel
        """
        #all_vec = Parallel(n_jobs=self.n_cpus,batch_size='auto',max_nbytes=None,mmap_mode=None,verbose=n_verb)\
        """
        all_vec = Parallel(n_jobs=8,batch_size='auto',max_nbytes=None,mmap_mode=None,verbose=n_verb)\
                (delayed(self.wrap_encode)(batch) for batch in batches)
        
        #all_vec = self.wrap_encode(smiles_rdkit) #Non-Parellel STB
        #print ('Encoding computation time, Total time = %.0f, %0.f seconds' % \
        #    ((time.time() - curr_time), (time.time() - start_time)))
        #curr_time = time.time()

        for i in range(0, len(all_vec)):
            latent_points.append(all_vec[i].data.cpu().numpy())

        latent_points = np.vstack(latent_points)

        return latent_points
    
def test_encoder():
    vocab = '/mnt/projects/ATOM/blackst/GMD/LOGP-JTVAE-PAPER/Vocabulary/all_vocab.txt'
    model = '/mnt/projects/ATOM/blackst/GMD/LOGP-JTVAE-PAPER/Train/MODEL-TRAIN/model.epoch-35'

    encoder = encode_smiles({"vocab_path": vocab,
                             "model_path": model})
    
    fname = '/mnt/projects/ATOM/blackst/GMD/LOGP-JTVAE-PAPER/Raw-Data/ZINC/all.txt'
    # Encoding time for  249456  molecules:  511.4566116333008  seconds
    # 
    #with open(args.smiles_input_file) as f:
    with open(fname) as f:
        smiles_list = [line.strip("\r\n ").split()[0] for line in f]

    smiles_list = smiles_list[:100]

    latent = encoder.encode(smiles_list)

    


class decoder():
    def __init__(self, args):
        self.model_path = args.model_path
        self.vocab = Vocab([x.strip("\r\n ") for x in open(args.vocab_path)])
        self.hidden_size = 450 #default
        self.latent_size = 128 #default
        self.depthT = 20 #default
        self.depthG = 3 #default
        self.n_cpus = 8 #default 36
        self.verbose = False #default

        if self.verbose == True:
            self.n_verb = 100
        else:
            self.n_verb = 0
    
    def wrap_decode_2(self, latent_vectors):
        
        smiles_recon = []
        for i in range(len(latent_vectors)):
            vect = latent_vectors
            #vect = torch.tensor(latent_vectors[i])
            print(type(vect))
            print(vect)
            #print(type(latent_vectors[i]))
            #print(latent_vectors[i])
            #tree_vec,mol_vec = torch.split(torch.reshape(latent_vectors[i], (1,128)), 64, dim=1)
            tree_vec,mol_vec = torch.split(torch.reshape(vect, (1,128)), 64, dim=1)
            smiles = self.model.decode_2(tree_vec, mol_vec, prob_decode=False)
            smiles_recon.append(smiles) 
        
        return smiles_recon

    def decode(self, latent):

        #vocab = [x.strip("\r\n ") for x in open(self.vocab)] #TODO: When combining these two classes, can combine vocab
        #vocab = Vocab(vocab)

        model = JTNNVAE(self.vocab, self.hidden_size, self.latent_size, self.depthT, self.depthG)
        model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        self.model = model.cpu()

        #smi_recon = Parallel(n_jobs=self.n_cpus,batch_size=1,max_nbytes=None,mmap_mode=None,verbose=self.n_verb)\
        #    (delayed(self.wrap_decode_2)(latent[i]) for i in range(len(latent)))
        smi_recon = Parallel(n_jobs=self.n_cpus,batch_size=1,max_nbytes=None,mmap_mode=None,verbose=self.n_verb)\
            (delayed(self.wrap_decode_2)(torch.tensor(latent[i])) for i in range(len(latent)))
        #smi_recon = self.wrap_decode_2(latent)

        smi_list = np.concatenate(smi_recon).flat

    def wrap_decode_simple(self, l):

        all_vec = l.reshape((1,-1))
        tree_vec,mol_vec = np.hsplit(all_vec, 2)
        tree_vec = create_var(torch.from_numpy(tree_vec).float())
        mol_vec = create_var(torch.from_numpy(mol_vec).float())
        s = self.model.decode_2(tree_vec, mol_vec, prob_decode=False)
        if s is not None: #TODO: Need to handle when a smiles is not valid? if s is None
            return s

    def decode_simple_2(self, latent_list):
        warnings.filterwarnings("ignore")
        #vocab = [x.strip("\r\n ") for x in open(self.vocab)] #TODO: When combining these two classes, can combine vocab
        #vocab = Vocab(vocab)

        model = JTNNVAE(self.vocab, self.hidden_size, self.latent_size, self.depthT, self.depthG)
        model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        self.model = model.cpu()

        smi_recon = Parallel(n_jobs=self.n_cpus,batch_size=5,max_nbytes=50000,mmap_mode=None,verbose=self.n_verb)\
            (delayed(self.wrap_decode_simple)(latent_list[i]) for i in range(len(latent_list)))
        
        return smi_recon

    def decode_simple(self, latent):

        #vocab = [x.strip("\r\n ") for x in open(self.vocab)] #TODO: When combining these two classes, can combine vocab
        #vocab = Vocab(vocab)

        model = JTNNVAE(self.vocab, self.hidden_size, self.latent_size, self.depthT, self.depthG)
        model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        model = model.cpu()

        i = 0

        valid_smiles = []
        new_features = []
        for l in latent:
            #print(i)
            all_vec = l.reshape((1,-1))
            tree_vec,mol_vec = np.hsplit(all_vec, 2)
            tree_vec = create_var(torch.from_numpy(tree_vec).float())
            mol_vec = create_var(torch.from_numpy(mol_vec).float())
            s = model.decode_2(tree_vec, mol_vec, prob_decode=False)
            if s is not None:
                
                valid_smiles.append(s)
                new_features.append(all_vec)
            i += 1
        
        return valid_smiles
        




def test_decoder(args):
    with open(args.smiles_input_file) as f:
        smiles = [line.strip("\r\n ").split()[0] for line in f]
    
    smiles = smiles[0]

    smiles_encoder = encode_smiles(args)
    latent_decoder = decoder(args)

    #Encode 
    latent = smiles_encoder.encode(smiles)
    print(latent)

if __name__ == "__main__":
    """
    import argparse
    import yaml
    from yaml import Loader

    #Test the decoder
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', help="Config file location *.yml", action='append', required=True)
    args = parser.parse_args()

    for conf_fname in args.config:
        with open(conf_fname, 'r') as f:
            parser.set_defaults(**yaml.load(f, Loader=Loader))

    args = parser.parse_args()

    test_decoder(args)

    

    test_encoder()
    """ 