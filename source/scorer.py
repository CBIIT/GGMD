from rdkit.Chem import MolFromSmiles, MolToSmiles, Descriptors, rdmolops
import sascorer
import networkx as nx
import numpy as np


def create_scorer(args):
    if args.scorer_type == 'FNL':
        return FNL_Scorer(args)
    else:
        raise ValueError("Unknown scorer_type: %s" % args.scorer_type)

class Scorer():
    def __init__(self, args):
        self.params = args
    
    def score(self):
        """
        Evaluate compounds not implemented in super class
        """
        raise NotImplementedError
    
class FNL_Scorer(Scorer):
    def __init__(self, args):
        pass
    def score(self, population):

        smiles = population['smiles']
        smiles_rdkit = []
        for s in smiles:
            mol = MolFromSmiles(s)
            smi = MolToSmiles(mol,isomericSmiles=False)
            smiles_rdkit.append(smi)
        
        logP_values = []
        for i in range(len(smiles_rdkit)):
            logP_values.append(Descriptors.MolLogP(MolFromSmiles(smiles_rdkit[ i ])))
        
        SA_scores = []
        for i in range(len(smiles_rdkit)):
            SA_scores.append(-sascorer.calculateScore(MolFromSmiles(smiles_rdkit[ i ])))
        
        cycle_scores = []
        
        for i in range(len(smiles_rdkit)):
            cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(smiles_rdkit[ i ]))))
            
            if len(cycle_list) == 0:
                cycle_length = 0
            else:
                cycle_length = max([ len(j) for j in cycle_list ])
            if cycle_length <= 6:
                cycle_length = 0
            else:
                cycle_length = cycle_length - 6
            cycle_scores.append(-cycle_length)
        
        SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
        logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)
        cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)

        #STB: Added below three lines to handle nan's in cycle score. Any time you attempt np.nan + 5 (or 
        # any real number like that), the result will be nan, so summing SA score, logP values, and cycle scores 
        # was giving a fitness score of nan. Seems to only happen with certain molecules and not all?
        SA_scores_normalized[np.isnan(SA_scores_normalized)] = 0.0
        logP_values_normalized[np.isnan(logP_values_normalized)] = 0.0
        cycle_scores_normalized[np.isnan(cycle_scores_normalized)] = 0.0

        targets = SA_scores_normalized + logP_values_normalized + cycle_scores_normalized
        population['fitness'] = targets 
        
        return population
