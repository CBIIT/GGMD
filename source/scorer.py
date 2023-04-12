from rdkit.Chem import MolFromSmiles, MolToSmiles, Descriptors, rdmolops
import sascorer, os
import networkx as nx
import numpy as np
from costinfo import CostInfo


def create_scorer(args):
    if args.scorer_type == 'FNL':
        return FNL_Scorer(args)
    elif args.scorer_type == 'LC':
        return CostinfoScorer(args)
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



class CostinfoScorer(Scorer):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)

        # costinfo file which specifies cost terms for each model
        self.costinfo_path  = self.params.costinfo_path
        # instantiate CostInfo for accessing the model cost terms
        self.costinfo = CostInfo(self.costinfo_path)

        # scoring_criteria is a list of model names indicating which
        # models to use when scoring. This needs to be a subset of
        # the model names specified in the costinfo. The default is
        # to use the list of models costinfo file
        self.scoring_criteria = list(self.costinfo.model_names)

        # use scoring criteria from params if specified (override)
        if self.params.scoring_criteria:
            self.scoring_criteria = self.params.scoring_criteria

        # use scoring criteria from cost_scoring_config (override)
        cost_scoring_config = kwargs.get('cost_scoring_config')
        if cost_scoring_config is not None:
            self.scoring_criteria = cost_scoring_config.model_names

        # panic if there is no configured scoring critera
        assert self.scoring_criteria, "No cost scoring criteria specified"

        # validate scoring criteria
        valid, err_str = self.costinfo.validate(
            model_names=self.scoring_criteria
        )
        if not valid:
            #Log.error("Invalid cost scoring configuration: {err_str}") #STB
            raise Exception(err_str)
    
    def compute_cost(self, values, cost_terms):
            if cost_terms.cost_function == 'exp':
                cost_func = rectified_exp
            elif cost_terms.cost_function == 'linear':
                cost_func = rectified_linear
            elif cost_terms.cost_function == 'binary':
                cost_func = binary_cost
            else:
                raise Exception(
                    "Unrecognized Cost Function: {cost_terms.cost_function}"
                )

            cost = cost_func(values, **cost_terms.as_dict()) * cost_terms.weight

            return cost

    def score(self, cmpd_df, return_costs=True):
        """
        Compute an overall cost score for each compound listed in cmpd_df.
        The columns of cmpd_df are expected to include the input criteria in
        the costinfo table associated with this Scorer object; the column names
        should match the values in the costinfo Name column. Returns a copy
        of cmpd_df with an additional 'cost' column holding the cost score.
        If return_costs is True, it will also contain the individual terms in
        the cost function, with names of the form '<criterion>_cost'.
        """
        # one cost score for each compound in dataframe
        cost = np.zeros(len(cmpd_df))

        # the result dataframe
        result_df = cmpd_df.copy(deep=False)

        # compute cost score for each model in the scoring criteria
        for model_name in self.scoring_criteria:

            # get the costinfo for the given model to compute the score
            # for each compound in cmpd_df
            model_costinfo = self.costinfo.get_model_costinfo(model_name)

            # compute cost for each compound in cmpd_df
            # using the cost_terms item for the the model in the costinfo
            for cost_terms in model_costinfo.cost_terms:
                # reset cost_df for each cost_terms item
                cost_df = cmpd_df

                # get the model_result_id to retrieve the model results
                # from cmpd_df to use when computing the cost based on
                # the given cost_terms for the model
                model_result_id = cost_terms.model_result_id

                # ensure the model_result_id is in the cmpd_df.
                # if not, then log the error
                if model_result_id not in cmpd_df:
                    err_str = (
                        f"Missing model results required for cost scoring. "
                        f"model_result_id: {model_result_id} not present in "
                        f"cmpd_df dataframe."
                    )
                    #Log.error(err_str) #STB
                    err_file = os.path.join(os.getcwd(), "cmpd_df.csv")
                    #Log.error(f"Saving cmpd_df to file: {err_file}") #STB
                    try:
                        cmpd_df.to_csv(err_file)
                    except:
                        #Log.error(f"Failed to create {err_file}") #STB
                        pass
                    raise Exception(f"{err_str}")

                score_agg_func = cost_terms.score_agg_func
                # if a score_agg_func is specified in costinfo for a model,
                # aggregate multiple values returned from the model
                if score_agg_func is not None:
                    agg_series = cmpd_df[model_result_id].apply(
                        lambda x: score_agg_func(x)
                        if isinstance(x, list) else x
                    )
                    cost_df = pd.DataFrame({model_result_id: agg_series})

                # get the model result values of all compounds
                # for the given model_result_id
                values = cost_df[model_result_id].values

                # compute the cost of all compounds for the given cost_terms
                dcost = self.compute_cost(values, cost_terms)

                # store the non-accumulated cost for the given model cost_term
                if return_costs:
                    cost_col = f"{model_name}:{cost_terms.name}:cost"
                    result_df[cost_col] = dcost

                # accumulate the cost of this model's cost_terms
                cost += dcost

        # store the computed cost for the compounds in the result dataframe
        #result_df['cost'] = cost

        return result_df

#TODO: Need to move below functions into the LC CostInfoScorer
def binary_cost(values, target_min, **kwargs):
    """Use Target_min for desired classification value."""
    dcost = np.array(values != target_min, dtype=float)
    return dcost

def rectified_linear(values, target_min, target_max, scale=1.0, allow_neg=False,
        **kwargs):
    """
    Computes a rectified linear function of the differences between values and
    either targ_min or targ_max, whichever is not NaN. 'Rectified' means that
    the cost function isn't negative for values within the (targ_min, targ_max)
    range, unless 'allow_neg' is True. Either targ_min or targ_max can be NaN,
    in which case the other range constraint is enforced only.
    """
    if is_valid_value(target_min):
        if not is_valid_value(target_max):
            cost = (target_min - values)/scale
        else:
            cost = np.maximum(
                (target_min - values)/scale, (values - target_max)/scale
            )

    elif is_valid_value(target_max):
        cost = (values - target_max)/scale
    else:
        raise Exception(
            'target_min or target_max must be specified in costinfo table'
        )

    return cost if allow_neg else np.maximum(cost, 0.0)

def rectified_exp(values, target_min, target_max, scale=1.0, allow_neg=False,
        **kwargs):
    """
    Computes a rectified shifted exponential cost function of the differencs
    between values and targ_min and/or targ_max. 'Rectified' means that the
    cost function isn't negative for values within the (targ_min, targ_max)
    range, unless 'allow_neg' is True. Either targ_min or targ_max can be NaN,
    in which case the other range constraint is enforced only.
    """
    if is_valid_value(target_min):
        if not is_valid_value(target_max):
            cost = np.exp((target_min - values)/scale) - 1.0
        else:
            cost = np.exp(
                np.maximum(
                    (target_min - values)/scale, (values - target_max)/scale
                )
            ) - 1.0

    elif is_valid_value(target_max):
        cost = np.exp((values - target_max)/scale) - 1.
    else:
        raise Exception(
            'target_min or target_max must be specified in costinfo table'
        )

    return cost if allow_neg else np.maximum(cost, 0.0)

def is_valid_value(val):
    """Tests if a value from costinfo (target_min/max) is valid"""
    return val is not None and not np.isnan(val)
