from generative_models.FNL_JTNN.fast_jtnn.mol_tree import Vocab, MolTree
from generative_models.FNL_JTNN.fast_jtnn.chemutils import tree_decomp, get_clique_mol, get_mol

file = "/mnt/projects/ATOM/blackst/FNLGMD/examples/LogP_JTVAE/all.txt"

with open(file, 'r') as f:
    smiles = []
    for line in f:
        smiles.append(line.split()[0])
len(smiles)

for s in smiles:
    mol = get_mol(s)

    cliques, edges = tree_decomp(mol)
    for i,c in enumerate(cliques):
        cmol = get_clique_mol(mol, c)