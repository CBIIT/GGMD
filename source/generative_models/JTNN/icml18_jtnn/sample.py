import sys
import numpy as np
import torch
import torch.nn as nn
from atomsci.glo.generative_networks.VAEUtils import DistributedEvaluator
import math, random, sys
import argparse
from scipy import stats
from atomsci.glo.generative_networks.icml18_jtnn.fast_jtnn import *
import rdkit
import copy

import pandas as pd
import random
import numpy as np

# torch.manual_seed(0)
# np.random.seed(0)
# random.seed(0)


lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


def sample_from_latent_vector(
    model, z_tree, z_mol, k, return_value=None, prob_decode=True
):
    decodings = []
    for i in range(k):

        # NOTE: use keyword args to pass z_tree and z_graph to model
        decodings.append(
            model.decode(x_tree_vecs=z_tree, x_mol_vecs=z_mol, prob_decode=True)
        )

    if return_val == "mode":
        return stats.mode(decodings).mode

    elif return_val == "all":
        return decodings

    else:
        raise NotImplementedError


def sample_from_prior(model, k, prob_decode=True, std_out=False):
    # decodings = []

    # model.cpu()
    # for i in range(k):
    smiles = model.sample_prior(prob_decode=prob_decode)
    # decodings.append(smiles)
    # if std_out:
    # sys.stdout.flush()
    # sys.stdout.write(smiles)
    # sys.stdout.write("\n")

    # return decodings
    return smiles


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prob-decode", action="store_true", default=False)
    parser.add_argument("--hidden_size", type=int, default=450)
    parser.add_argument("--latent_size", type=int, default=56)
    parser.add_argument("--depthT", type=int, default=20)
    parser.add_argument("--depthG", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--k-decodings", type=int, required=True)
    args = parser.parse_args()

    TIMEOUT = args.timeout

    # evaluator_cpu = DistributedEvaluator(vae_path="/g/g13/jones289/ATOM/active_learning/icml18_jtnn/kevin_model/model.iter-11000_dict",
    #                                                vocab_path="/g/g13/jones289/ATOM/active_learning/icml18_jtnn/data/aurk/aurk_base_vocab.txt", device=False, workers=64)

    evaluator_cpu = DistributedEvaluator(
        vae_path=args.model, vocab_path=args.vocab, usecuda=False, workers=64
    )

    for k in range(args.k_decodings):
        print(sample_from_prior(evaluator_cpu.vae, k, prob_decode=True, std_out=False))

    """
    #vocab = [x.strip("\r\n") for x in open("/g/g13/jones289/ATOM/active_learning/icml18_jtnn/data/aurk/aurk_base_vocab.txt")] 
    #vocab = Vocab(vocab)

    #model = JTNNVAE(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG)
    #model.load_state_dict(torch.load(args.model)["model_state_dict"])
    #gpu_model.cuda()

    # load some latent vectors
    model_eval_df = pd.read_csv(args.eval_path)

    tree_cols = ['t%.02d' % i for i in range(int(56/2))]
    graph_cols = ['g%.02d' % i for i in range(int(56/2))]

    tree_vecs = torch.from_numpy(model_eval_df[tree_cols].values)
    mol_vecs = torch.from_numpy(model_eval_df[graph_cols].values)
    latent = torch.cat([tree_vecs, mol_vecs], dim=1).float()
 

    # testing decode with one smiles at a time
    print("DECODING SINGLE VECTOR AT A TIME")
    print("===========================  using prob_decode=True ===========================")
    for idx, z in enumerate(iter(latent)):
        #z_tree = z[:int(args.latent_size/2)].view(1, -1).cuda()
        #z_mol = z[int(args.latent_size/2):].view(1, -1).cuda()
        
        #for i in range(args.k_decodings):
           
        #gpu_smiles =  gpu_model.decode(z_tree, z_mol, prob_decode=True)
        z_ = z.view(1, -1).cpu()
        
        cpu_smiles = evaluator_cpu.decode_SMILES_cpu_python_parallel(z_, prob_decode=True, TIMEOUT=TIMEOUT, k=args.k_decodings)
        #print("latent idx:{} decoding i={} \t{}\t{}".format(idx, i, model.decode(z_tree, z_mol, prob_decode=True), evaluator_cpu.decode_SMILES_cpu_python_parallel(z.view(1, -1).cpu(), prob_decode=True, TIMEOUT=TIMEOUT)[0]))
        #print("latent idx:{} decoding i={}\t{}".format(idx, i, evaluator_cpu.decode_SMILES_cpu_python_parallel(z.view(1, -1).cpu(), prob_decode=True, TIMEOUT=TIMEOUT)[0]))
        #print("latent idx:{}\tdecoding i={}\tgpu_smiles={}\tcpu_smiles={}".format(idx, i, gpu_smiles, cpu_smiles))
        print("latent idx:{}\tcpu_smiles={}".format(idx, cpu_smiles))
        #print("latent idx:{}\tcpu_smiles={}".format(idx, evaluator_cpu.decode_SMILES_cpu_python_parallel(z.view(1, -1).cpu(), prob_decode=True, TIMEOUT=TIMEOUT,
        #                                                                            k=args.k_decodings)[0]))


    #print("=========================== using prob_decode=False ===========================")
    #for idx, z in enumerate(iter(latent)):
    #    z_tree = z[:int(args.latent_size/2)].view(1, -1)
    #    z_mol = z[int(args.latent_size/2):].view(1, -1)
       
    #    for i in range(args.k_decodings):
    #        print("latent idx:{} decoding i={} \t{}\t{}".format(idx, i, model.decode(z_tree, z_mol, prob_decode=False), evaluator_cpu.decode_SMILES_cpu_python_parallel(z.view(1, -1).cpu(), prob_decode=False, TIMEOUT=TIMEOUT)[0])) 

    # testing batch decode

    print("DECODING MULTIPLE VECTORS AT A TIME (BATCH MODE)")
    #print("===========================  using prob_decode=True ===========================")
    #for i in range(args.k_decodings): 
    #    print("decoding i={} \t{}".format(i, evaluator_cpu.decode_SMILES_cpu_python_parallel(latent.cpu(), prob_decode=True)))

    print("===========================  using prob_decode=False ===========================")
    for i in range(args.k_decodings):
        print("decoding i={} \t{}".format(i, evaluator_cpu.decode_SMILES_cpu_python_parallel(latent.cpu(), prob_decode=False)))


    """


if __name__ == "__main__":
    main()
