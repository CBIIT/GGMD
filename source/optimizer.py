import os
import numpy as np
import pandas as pd
from copy import deepcopy

import logging
import pickle

Log = logging.getLogger(__name__)


def create_optimizer(params):
    """
    Factory function for creating optimizer objects of the correct subclass for params.optimizer_type.
    Args:
        params: parameters passed to the optimizers
    Returns:
        optimizer (object):  wrapper for optimizers
    Raises:
        ValueError: Only params.optimizer_type = "GeneticOptimizer"  is supported
    """
    if params.optimizer_type.lower() == "geneticoptimizer":
        return GeneticOptimizer(params)
    elif params.optimizer_type.lower() == "particleswarmoptimizer":
        return ParticleSwarmOptimizer(params)
    else:
        raise ValueError("Unknown optimizer_type %s" % params.optimizer_type)


class Optimizer(object):
    def __init__(self, params, **kwargs):
        self.params = params
        # to keep the cmpds that already have their costs.
        self.retained_population = pd.DataFrame()

    def optimize(self):
        """
        Optimize the molecule cost in the latent space and returns optimized latent variables
        Args:
            latent_cost_df (dataframe): molecules presented by latent variables and calculated cost of the molecules.
            latent_cost_df must have columns ['latent', 'cost']
        Raises:
            NotImplementedError: The method is implemented by subclasses
        """
        raise NotImplementedError


class GeneticOptimizer(Optimizer):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.optimizer_type = self.params.optimizer_type.lower()
        self.selection_type = self.params.selection_type.lower()
        #self.crossover_type = self.params.crossover_type.lower()
        #self.mutation_type = self.params.mutation_type.lower()
        self.input_file = self.params.smiles_input_file 
        self.set_max_population_size()
        self.tourn_size = self.params.tourn_size
        self.mate_prob = self.params.mate_prob
        self.ind_mutate_prob = self.params.ind_mutate_prob
        self.gene_mutate_prob = self.params.gene_mutate_prob
        self.max_clones = self.params.max_clones
        self.mutation_std = self.params.mutate_std
        self.memetic_frac = self.params.memetic_frac
        #self.memetic_delta = self.params.memetic_delta
        self.memetic_delta = 1 #Need to figure out what the file path is about
        self.model_type = self.params.model_type
        if self.model_type == 'jtnn':
            #self.tree_sd = self.params.tree_sd #TEMPORARY comment for development
            self.tree_sd = 4.86
            #self.molg_sd = self.params.molg_sd #TEMPORARY comment for development
            self.molg_sd = 4

        """
        if os.path.isfile(self.memetic_delta):
            self.memetic_delta = pd.read_csv(self.memetic_delta)['Std'].values
        else:
            Log.info('memetic delta file not specified or not found; setting delta equal to 1')
            self.memetic_delta = 1
        """
        self.memetic_delta = 1 #TODO: Debugging
        self.memetic_delta_scale = self.params.memetic_delta_scale
        self.memetic_size = self.params.memetic_size

        self.optimization = self.params.optimization
        self.verbose = self.params.verbose

    def set_max_population_size(self):
        """
        Set the target max population size based on the max_population, compound_count and
        input_file parameters.

        self.max_population defaults to self.compound_count, which in turn defaults to the number of
        SMILES strings in the input file. Both parameters are set by this function.
        """

        # Count lines in the input file, not counting the header if a CSV file
        smiles_count = 0
        with open(self.input_file, 'r') as fd:
            for line in fd:
                smiles_count += 1
        if self.input_file.endswith('.csv'):
            smiles_count -= 1

        if self.params.compound_count == 0:
            self.compound_count = smiles_count
        else:
            # compound_count can't be greater than number of SMILES strings
            self.compound_count = min(self.params.compound_count, smiles_count)

        if self.params.max_population == 0:
            self.max_population = self.compound_count
        else:
            self.max_population = self.params.max_population

    def set_population(self, latent_cost_df): #TODO: Ideally I would remove the self.population
        self.population = latent_cost_df
        #self.population = self.population.rename(columns={"latent": "chromosome"}) #TODO: chomosome
        self.population = self.population.append(
            self.retained_population, ignore_index=True
        ) #TODO: Need to align to prepare for future pandas versions:
        # FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
        self.population = self.population.reset_index(drop=True)
        self.population_size = len(self.population)
        print(f"latent_optimizer population_size: {self.population_size}, latent_cost_df size: {len(latent_cost_df)}, retained_df size: {len(self.retained_population)}, target population size: {self.max_population}")
        Log.info(f"latent_optimizer population_size: {self.population_size}, latent_cost_df size: {len(latent_cost_df)}, retained_df size: {len(self.retained_population)}, target population size: {self.max_population}")
        #self.chromosome_length = len(self.population["chromosome"].iloc[0]) #TODO: chomosome
        self.chromosome_length = len(self.population["latent"].iloc[0])
        if self.model_type == 'jtnn':
            self.tree_len = int(self.chromosome_length / 2)
            self.molg_len = self.chromosome_length - self.tree_len
        return self

    def select_individual_from_candidates(self, candidates):
        """
        Function that performs tournament selection from a data frame of candidates.
        Returns a data frame containing only the row with the best cost score.
        """
        if self.optimization.lower() == "minimize":
            try:
                selected_individual = candidates.loc[candidates.cost.idxmin()]
            except TypeError:
                Log.info(
                    "cost value missing in all of the selection candidates. Randomly sample one. Check input."
                )
                selected_individual = candidates.sample(n=1).squeeze()

        elif self.optimization.lower() == "maximize":
            try:
                selected_individual = candidates.loc[candidates.cost.idxmax()]
            except TypeError:
                Log.info(
                    "cost value missing in all of the selection candidates. Randomly sample one. Check input."
                )
                selected_individual = candidates.sample(n=1).squeeze()

        else:
            raise ValueError(
                'Unknown optimization type, please set to "minimize" or "maximize"'
            )
        return selected_individual


    def tournament_selection(self, tournament):
        """
        Alternative function for tournament selection that returns the index of the winner rather than a one row data frame
        """
        if self.optimization.lower() == "minimize":
            return tournament.cost.idxmin()
        elif self.optimization.lower() == "maximize":
            return tournament.cost.idxmax()
        else:
            raise ValueError(f"Unknown optimization type {self.optimization}, should be 'minimize' or 'maximize'")


    def select(self):
        """
        Select the molecules/latent vectors that will be carried forward to the next generation or used
        as parents for crossover or mutation. Several variants of tournament selection are supported, and
        are specified via the selection_type parameter.
        """
        if self.selection_type == 'tournament':
            n_tourn = int((1 - self.mate_prob - self.memetic_frac) * self.population_size)
            selection_pool = self.population.copy(deep=True)
            self.population = None  # clear the variable.
            # make sure the index is completely reset for proper selection tracking
            # TODO: check if the followings are redundant
            selection_pool.reset_index(drop=True, inplace=True)
            selection_pool.reset_index(inplace=True)
            selection_pool.set_index("index", inplace=True)

            selected_population = pd.DataFrame()
            # get gene0 from the pool's chromosomes, for quick screen
            #selection_pool["gene0"] = np.hstack(selection_pool["chromosome"])[ #TODO: chomosome
            selection_pool["gene0"] = np.hstack(selection_pool["latent"])[
                0 :: self.chromosome_length
            ]

            if self.tourn_size >= len(selection_pool):
                raise Exception(
                    "self.tourn_size {} has to be less than the pool size {}.".format(
                        self.tourn_size, len(selection_pool)
                    )
                )

            while len(selected_population) < n_tourn:
                candidates = selection_pool.sample(self.tourn_size, replace=False)
                selected_individual = self.select_individual_from_candidates(candidates)

                # to populate the selected_population df
                if len(selected_population) == 0:
                    selected_population = selected_population.append(
                        selected_individual
                    ) #TODO: Need to align to prepare for future pandas versions:
                    # FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.

                    # quick screen of gene0 occurrence in the selected population
                recurred_gene0_idx = np.where(
                    selected_population["gene0"].values == selected_individual["gene0"]
                )
                recurred_gene0_count = len(recurred_gene0_idx[0])
                # if gene0 reach max_clones occurrence, going to check the whole chromosomes that are flagged from the selected_pop
                if (self.max_clones > 0) & (recurred_gene0_count >= self.max_clones):
                    # get the chromosome from selected individual.
                    #selected_ind_chrom = selected_individual["chromosome"] #TODO: chomosome
                    selected_ind_chrom = selected_individual["latent"]

                    # get the flagged individuals and the chromosomes from the selected population
                    flagged_inds = selected_population.iloc[recurred_gene0_idx]
                    #flagged_chromosomes = np.hstack(flagged_inds["chromosome"]).reshape( #TODO: chomosome
                    flagged_chromosomes = np.hstack(flagged_inds["latent"]).reshape(
                        recurred_gene0_count, self.chromosome_length
                    )

                    # calculate repeated chrom counts of the selected_individual from the selected populations
                    recurred_chrom_count = np.sum(
                        np.all(
                            np.equal(selected_ind_chrom, flagged_chromosomes), axis=1
                        )
                    )

                    # if the selected_individual indeed reached max_clones in the selected_pop,
                    # going to remove that chromosome from the pool.
                    if (self.max_clones > 0) & (
                        recurred_chrom_count >= self.max_clones
                    ):
                        # remove the index from the selection_pool

                        mask = flagged_inds.index.values
                        # try to remove the flagged indice from the pool.
                        # ignoring error if already removed from the pool.
                        for m in mask:
                            try:
                                selection_pool = selection_pool.drop(m)
                            except (ValueError, KeyError):
                                pass
                        try:
                            selection_pool = selection_pool.drop(
                                selected_individual.name
                            )
                        except (ValueError, KeyError):
                            pass
                    else:
                        #  add the selected_individual to the selected population
                        selected_population = selected_population.append(
                            selected_individual
                        )

                else:
                    selected_population = selected_population.append(
                        selected_individual
                    ) #TODO: Need to align to prepare for future pandas versions:
                    # FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
            selected_population.pop("gene0")
            self.population = selected_population.reset_index(drop=True)
        #######################################################################################################################
        # Perform tournament selection with checking of SMILES strings so that no more than max_clones latent vectors
        # decoding to the same SMILES string are selected.
        elif self.selection_type == 'smiles_tournament':
            n_tourn = int( self.population_size * (1 - self.mate_prob) / (1 + self.memetic_frac))

            selection_pool = self.population.copy(deep=True)
            self.population = None  # clear the variable.

            # make sure the index is completely reset for proper selection tracking
            selection_pool.reset_index(drop=True, inplace=True)
            selection_pool.reset_index(inplace=True)
            selection_pool.set_index("index", inplace=True)

            selected_population = pd.DataFrame()

            # exclude the nan cost rows from the selection
            initial_pool = len(selection_pool)
            selection_pool = selection_pool.dropna(subset=['cost'])
            pool_size = len(selection_pool)
            excluded_rows = initial_pool - pool_size
            Log.info(f"smiles_tournament: {n_tourn} tournaments, pool = {pool_size} after excluding {excluded_rows} with no cost")

            # check len for quick screen
            selection_pool["smiles_len"] = [
                len(S) for S in selection_pool["smiles"].values
            ]

            if self.tourn_size >= pool_size:
                raise Exception(
                    "self.tourn_size {} has to be less than the pool size {}.".format(
                        self.tourn_size, pool_size
                    )
                )

            dropped_smiles_count = 0
            while len(selected_population) < n_tourn and len(selection_pool) >= self.tourn_size:
                # Note that there's nothing here to prevent the same latent vector from being sampled multiple times
                # in successive iterations.
                candidates = selection_pool.sample(self.tourn_size, replace=False)
                selected_individual = self.select_individual_from_candidates(candidates)
                # Initialize the selected_population df
                if len(selected_population) == 0:
                    selected_population = selected_population.append(selected_individual)
                    continue

                # quick screen of substring occurrence in the selected population
                recurred_len_idx = np.where(
                    selected_population["smiles_len"].values
                    == selected_individual["smiles_len"]
                )
                recurred_len_count = len(recurred_len_idx[0])
                # if smiles lengths reach max_clones occurrence,
                # going to check the whole smiles that are flagged from the selected_pop
                if (self.max_clones > 0) & (recurred_len_count >= self.max_clones):
                    # get the smiles from selected individual.
                    selected_ind_smiles = selected_individual["smiles"]

                    # get the flagged individuals and the chromosomes from the selected population
                    flagged_inds = selected_population.iloc[recurred_len_idx]
                    flagged_smiles = flagged_inds["smiles"].values

                    # calculate repeated chrom counts of the selected_individual from the selected populations
                    recurred_smiles_count = np.sum(
                        selected_ind_smiles == flagged_smiles
                    )

                    # if the selected_individual indeed reached max_clones in the selected_pop,
                    # going to remove that smiles from the pool.
                    if (self.max_clones > 0) & (
                        recurred_smiles_count >= self.max_clones
                    ):
                        dropped_smiles_count += 1
                        # remove the index from the selection_pool

                        mask = flagged_inds.index.values
                        # try to remove the flagged indice from the pool.
                        # ignoring error if already removed from the pool.
                        for m in mask:
                            try:
                                selection_pool = selection_pool.drop(m)
                            except (ValueError, KeyError):
                                pass
                        try:
                            selection_pool = selection_pool.drop(
                                selected_individual.name
                            )
                        except (ValueError, KeyError):
                            pass
                    else:
                        #  add the selected_individual to the selected population
                        selected_population = selected_population.append(selected_individual)

                else:
                    selected_population = selected_population.append(selected_individual)

            selected_population.pop("smiles_len")
            pool_size = len(selection_pool)
            Log.info(f"After tournament selection: {len(selected_population)} in population, {pool_size} in selection pool, {dropped_smiles_count} smiles dropped from pool")

            self.population = selected_population.reset_index(drop=True)

        #######################################################################################################################
        # Perform tournament selection without replacement, so that each latent vector appears only once in the mating pool.
        # As with smiles, limit the number of vectors decoding to the same smiles string.

        elif self.selection_type == 'tournament_wo_replacement':

            selection_pool = self.population.copy(deep=True)

            # make sure the index is completely reset for proper selection tracking
            selection_pool.reset_index(drop=True, inplace=True)

            # Sanity check: Input latents should all have compound_id, smiles and cost
            initial_len = len(selection_pool)
            selection_pool = selection_pool.dropna(subset=['cost', 'compound_id', 'smiles'])
            pool_size = len(selection_pool)
            Log.info(f"Tournament selection without replacement: input pop size {initial_len}, {len(set(selection_pool.smiles.values))} distinct smiles")
            excluded_rows = initial_len - pool_size
            if excluded_rows > 0:
                Log.warning(f"tournament_wo_replacement: {excluded_rows} rows excluded because cost, compound_id or smiles missing")

            # Sanity check: Compound IDs should be unique
            selection_pool = selection_pool.drop_duplicates(subset=['compound_id'])
            if len(selection_pool) < pool_size:
                Log.warning(f"Eliminated {pool_size - len(selection_pool)} duplicate compound IDs from input population")
                pool_size = len(selection_pool)

            # Check that we can hold at least one tournament
            if self.tourn_size >= pool_size:
                raise Exception(f"Selection pool size {pool_size} too small for tournament size {self.tourn_size}")

            # Set number of tournaments
            n_tourn = int( pool_size * (1 - self.mate_prob) / (1 + self.memetic_frac))
            Log.info(f"tournament_wo_replacement: {n_tourn} tournaments, pool = {pool_size}")

            dropped_smiles_count = 0
            smiles_count = {}
            selected_population = pd.DataFrame()

            # Repeat tournament selection until we have enough latent vectors selected. Remove selected
            # latents from the pool so they can't be selected again. Also make sure that no more than
            # max_clones latent vectors mapping to the same smiles string get selected.
             
            while len(selected_population) < n_tourn and len(selection_pool) >= self.tourn_size:
                tournament = selection_pool.sample(self.tourn_size, replace=False)
                #selected_individual = self.select_individual_from_candidates(tournament)
                winner_idx = self.tournament_selection(tournament)
                winner = selection_pool.loc[winner_idx].copy()
                best_smiles = winner.smiles
                selected_population = selected_population.append(winner)
                selection_pool = selection_pool.drop(winner_idx)
                smiles_count[best_smiles] = smiles_count.get(best_smiles, 0) + 1
                if self.max_clones > 0:
                    if smiles_count[best_smiles] >= self.max_clones:
                        dropped_smiles_count += 1
                        selection_pool = selection_pool[selection_pool.smiles != best_smiles]

            pool_size = len(selection_pool)
            Log.info(f"After tournament selection: {len(selected_population)} selected, {pool_size} remaining in pool, {dropped_smiles_count} smiles dropped from pool")

            self.population = selected_population.reset_index(drop=True)

        #######################################################################################################################
        else:
            raise ValueError("Non-implemented selection method: " + self.selection_type)

        #######################################################################################################################
        return self

    def crossover(self):
        # Uniform crossover
        #         n_offspring = self.population_size - len(self.population)
        n_offspring = int(self.mate_prob * self.max_population)
        parent_idx = np.random.randint(0, len(self.population), (n_offspring, 2))
        child_chrom = []
        cols = self.population.columns.values.tolist()
        Log.info(f"Optimizer.crossover(): population columns = {', '.join(cols)}")
        moms = []
        pops = []
        for i in range(0, n_offspring):
            #child_chrom.append(np.zeros(self.chromosome_length))
            parents = self.population.iloc[parent_idx[i]]

            # to prevent drawing the two clones when multiple clones exist (max_clones > 1)
            while np.array_equal(
                #parents.iloc[0]["chromosome"], parents.iloc[1]["chromosome"] #TODO: chomosome
                parents.iloc[0]["latent"], parents.iloc[1]["latent"]
            ):
                redraw_idx = np.random.randint(0, len(self.population), (1, 2))
                parents = self.population.iloc[redraw_idx[0]]

            #parent_chrom = np.vstack(parents["chromosome"].values) #TODO: chomosome
            parent_chrom = np.vstack(parents["latent"].values)
            moms.append(parents.compound_id.values[0])
            pops.append(parents.compound_id.values[1])

            selected_genes = np.random.randint(0, 2, self.chromosome_length)
            child_chromosome = np.where(selected_genes, parent_chrom[1], parent_chrom[0])
            child_chrom.append(child_chromosome)

        #children = pd.DataFrame( {"chromosome": child_chrom, "cost": np.full(n_offspring, np.nan)}) #TODO: chomosome
        children = pd.DataFrame( {"latent": child_chrom, "cost": np.full(n_offspring, np.nan)})
        children['parent1_id'] = moms
        children['parent2_id'] = pops
        self.population = self.population.append(children) #TODO: Need to align to prepare for future pandas versions:
        # FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
        self.population = self.population.reset_index(drop=True)
        Log.info(f"After crossover: {len(child_chrom)} offspring added, total population {len(self.population)}")
        return self

    def mutate(self):
        # save a copy of the top population for memetic
        memetic_pop_size = 0
        if self.memetic_frac > 0:
            memetic_candidates = self.population.dropna(subset=['cost'])
            memetic_pop_size = int(self.memetic_frac * len(memetic_candidates))
            memetic_candidates = memetic_candidates.sort_values(by='cost', ascending=True).iloc[:memetic_pop_size]
            #memetic_chromosomes = memetic_candidates['chromosome'].values #TODO: chomosome
            memetic_chromosomes = memetic_candidates['latent'].values
            memetic_parents = memetic_candidates['compound_id'].values
            # workaround of the issue with pandas' deepcopy (it is not that deep)
            copy_memetic_chromosomes = deepcopy(memetic_chromosomes)

            # set the upper bound of memetic_size if it is over the chromosome length
            if self.memetic_size > self.chromosome_length:
                self.memetic_size = self.chromosome_length

        # random mutation
        ind_idx = np.where(np.random.rand(len(self.population)) < self.ind_mutate_prob)[0]
        n_mutated = len(ind_idx)
        n_offspring_mutated = 0
        for idx in ind_idx:
            #chromosome = deepcopy(self.population["chromosome"].iloc[idx]) #TODO: chomosome
            chromosome = deepcopy(self.population["latent"].iloc[idx])
            if self.model_type == 'jtnn':
                # If we are using a JTNN autoencoder, split the latent vectors into tree and
                # molecular graph parts and adjust the scale parameter according to the tree_sd
                # and molg_sd parameters.
                tree_chr = chromosome[:self.tree_len]
                molg_chr = chromosome[self.tree_len:]
                tree_mut_pts = np.where(np.random.rand(self.tree_len) < self.gene_mutate_prob)[0]
                if len(tree_mut_pts) > 0:
                    tree_chr[tree_mut_pts] = np.random.normal(loc=tree_chr[tree_mut_pts], scale=self.tree_sd * self.mutation_std)
                molg_mut_pts = np.where(np.random.rand(self.molg_len) < self.gene_mutate_prob)[0]
                if len(molg_mut_pts) > 0:
                    molg_chr[molg_mut_pts] = np.random.normal(loc=molg_chr[molg_mut_pts], scale=self.molg_sd * self.mutation_std)
                chromosome = np.concatenate([tree_chr, molg_chr])
                n_mutation_pts = len(tree_mut_pts) + len(molg_mut_pts)
            else:
                mutation_pts = np.where(np.random.rand(self.chromosome_length) < self.gene_mutate_prob)[0]
                n_mutation_pts = len(mutation_pts)
                if n_mutation_pts > 0:
                    chromosome[mutation_pts] = np.random.normal(loc=chromosome[mutation_pts], scale=self.mutation_std)

            # Only update the identifiers if the latent vector was actually mutated
            if n_mutation_pts > 0:
                n_mutated += 1
                if np.isnan(self.population.at[idx, 'cost']):
                    # Cost is already nan, so this latent vector is the result of a crossover.
                    # In this case, the parent IDs have already been set.
                    n_offspring_mutated += 1
                else:
                    # Otherwise, the mutant has just one parent
                    self.population.at[idx, "parent1_id"] = self.population.at[idx, "compound_id"]
                    self.population.at[idx, "parent2_id"] = np.nan
                #self.population.at[idx, "chromosome"] = chromosome #TODO: chomosome
                self.population.at[idx, "latent"] = chromosome
                self.population.at[idx, "cost"] = np.nan
                self.population.at[idx, "smiles"] = np.nan
                self.population.at[idx, "compound_id"] = np.nan

        # single point fixed size mutation on memetic_candidates

        # This has the same issue as random mutations for JTNN, namely that the delta applied to the mutated element
        # needs to have a different scale depending on whether the element is in the tree or graph part of the latent vector.

        if (memetic_pop_size > 0) and (self.memetic_frac > 0):
            for idx, selected_memetic_chromosome in enumerate(copy_memetic_chromosomes):
                memetic_pts = np.random.choice(a=self.chromosome_length,
                                           size=self.memetic_size,
                                           replace=False)
                copy_selected_memetic_chromosome = deepcopy(selected_memetic_chromosome)
                # update each mutation point over the chromosome
                if not isinstance(self.memetic_delta, np.ndarray):
                    if self.memetic_delta == 1:
                        self.memetic_delta = np.array([self.memetic_delta] * self.chromosome_length)
                assert self.memetic_delta.size == self.chromosome_length, f'The length of self.memetic_delta {self.memetic_delta.size} not equal to self.chromosome_length {self.chromosome_length}'
                for memetic_pt in memetic_pts:
                    if self.model_type == 'jtnn':
                        if memetic_pt < self.tree_len:
                            delta_sd = self.tree_sd
                        else:
                            delta_sd = self.molg_sd
                    else:
                        delta_sd = 1.0

                    if np.random.randint(2) == 1:
                        copy_selected_memetic_chromosome[memetic_pt] += self.memetic_delta[memetic_pt] * self.memetic_delta_scale * delta_sd
                    else:
                        copy_selected_memetic_chromosome[memetic_pt] -= self.memetic_delta[memetic_pt] * self.memetic_delta_scale * delta_sd
                #memetic_dict = dict(chromosome=copy_selected_memetic_chromosome, cost=np.nan, parent1_id=memetic_parents[idx], #TODO: chomosome
                                        #parent2_id=np.nan) #TODO: chomosome
                memetic_dict = dict(latent=copy_selected_memetic_chromosome, cost=np.nan, parent1_id=memetic_parents[idx],
                                        parent2_id=np.nan)
                self.population = self.population.append(memetic_dict, ignore_index= True) #TODO: Need to align to prepare for future pandas versions:
                # FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.

        Log.info(f"After mutation: {n_mutated-n_offspring_mutated} scored latents mutated, {n_offspring_mutated} crossover offspring mutated, "
                 f"{memetic_pop_size} added by memetic, total population {len(self.population)}")

        return self

    def optimize(self, population):
        """
        Optimize the molecule cost in the latent space and returns optimized latent variables
        Args:
            latent_cost_df (dataframe): molecules presented by latent variables and calculated cost of the molecules.
            latent_cost_df must have columns ['latent', 'cost']
        return new population
        side effect set self.retained_population
        """
        self.set_population(population)
        self.select()
        self.crossover()
        self.mutate()
        self.retained_population = self.population.dropna(subset=['cost']).copy()
        #self.population = self.population.rename(columns={"chromosome": "latent"}) #TODO: chomosome
        Log.info(f"After optimization, population columns are {', '.join(self.population.columns.values)}")
        print(f"After optimization, population columns are {', '.join(self.population.columns.values)}")
        Log.info(f"After optimization: retained population {len(self.retained_population)}, total population {len(self.population)}")
        print(f"After optimization: retained population {len(self.retained_population)}, total population {len(self.population)}")
        return self.population

#######################################################################################################################
class ParticleSwarmOptimizer(Optimizer):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)

        # optmizer params
        self.optimizer_type = self.params.optimizer_type.lower()
        self.optimization = self.params.optimization.lower()

        # PSO params
        self.upper_bound = self.params.upper_bound
        self.lower_bound = self.params.lower_bound
        self.inertia = self.params.inertia
        self.init_velocity_scale = self.params.init_velocity_scale
        self.self_attraction = self.params.self_attraction
        self.swarm_attraction = self.params.swarm_attraction

        # misc params
        self.verbose = self.params.verbose

        self.swarm_best_position = None
        self.result_df =  None
        self.swarm = None
        self.swarm_trace = None
        self.swarm_size = None
        self.position_size = None
        self.positions = None
        self.pos_range = None
        self.velocities = None


    def set_swarm(self, latent_cost_df):
        if "swarm_particle_ID" not in latent_cost_df.columns:
            # initiate swarm_trace
            self.swarm_trace = pd.DataFrame(
                {
                    "swarm_particle_ID": np.arange(len(latent_cost_df)),
                    "particle_best_position": latent_cost_df["latent"],
                    "particle_best_cost": latent_cost_df["cost"],
                    "velocity": np.nan,
                }
            )

            self.swarm = pd.DataFrame(
                {
                    "swarm_particle_ID": np.arange(len(latent_cost_df)),
                    "position": latent_cost_df["latent"],
                    "cost": latent_cost_df["cost"],
                }
            )

        else:
            self.swarm = pd.DataFrame(
                {
                    "swarm_particle_ID": latent_cost_df["swarm_particle_ID"],
                    "position": latent_cost_df["latent"],
                    "cost": latent_cost_df["cost"],
                }
            )

        # update swarm with the trace.
        # left join to handle possible missing particle_ID.
        self.swarm = pd.merge(
            self.swarm, self.swarm_trace, on="swarm_particle_ID", how="left"
        )

        self.swarm_size = len(self.swarm)
        self.position_size = len(self.swarm["position"].iloc[0])

        # convert the poistions into np arrays
        self.positions = np.hstack(self.swarm["position"]).reshape(
            self.swarm_size, self.position_size
        )

        # set initial velocities if velocity is NaN.
        if self.swarm.velocity.isna().all():
            self.pos_range = np.max(self.positions, axis=0) - np.min(
                self.positions, axis=0
            )
            self.velocities = (
                np.random.randn(self.swarm_size, self.position_size)
                * self.pos_range
                * self.init_velocity_scale
            )
        else:
            self.velocities = np.hstack(self.swarm['velocity']).reshape(self.swarm_size, self.position_size)
        self._update_particle_best_position_cost()
        self._update_swarm_best_position_cost()
        self.swarm = self.swarm.reset_index(drop=True)
        return self

    def _update_particle_best_position_cost(self):
        # update self.swarm with the best_position and best_cost.
        if self.optimization == "minimize":
            # update particle best trace
            update_idx = self.swarm.index[
                self.swarm["cost"] < self.swarm["particle_best_cost"]
            ]
            self.swarm.loc[update_idx, "particle_best_cost"] = self.swarm.loc[
                update_idx, "cost"
            ]
            self.swarm.loc[update_idx, "particle_best_position"] = self.swarm.loc[
                update_idx, "position"
            ]

        elif self.optimization == "maximize":
            # update particle best trace
            update_idx = self.swarm.index[
                self.swarm["cost"] > self.swarm["particle_best_cost"]
            ]
            self.swarm.loc[update_idx, "particle_best_cost"] = self.swarm.loc[
                update_idx, "cost"
            ]
            self.swarm.loc[update_idx, "particle_best_position"] = self.swarm.loc[
                update_idx, "position"
            ]

        elif (self.optimization != "minimize") & (self.optimization != "maximize"):
            raise Exception(
                'Invalid optimization type "'
                + self.optimization
                + '" valid selections are minimize or maximize'
            )
        return self

    def _update_swarm_best_position_cost(self):
        # update self.swarm_best_position
        if self.optimization == "minimize":
            # update swarm best trace
            current_swarm_best_cost = self.swarm.iloc[
                self.swarm["particle_best_cost"].idxmin()
            ]["particle_best_cost"]
            current_swarm_best_position = np.hstack(
                self.swarm.iloc[self.swarm["particle_best_cost"].idxmin()][
                    "particle_best_position"
                ]
            )
            if (self.swarm_best_position is None) or (
                current_swarm_best_cost < self.swarm_best_cost
            ):
                self.swarm_best_position = current_swarm_best_position
                self.swarm_best_cost = current_swarm_best_cost

        elif self.optimization == "maximize":
            # update swarm best trace
            current_swarm_best_cost = self.swarm.iloc[
                self.swarm["particle_best_cost"].idxmax()
            ]["particle_best_cost"]
            current_swarm_best_position = np.hstack(
                self.swarm.iloc[self.swarm["particle_best_cost"].idxmax()][
                    "particle_best_position"
                ]
            )
            if (self.swarm_best_position is None) or (
                current_swarm_best_cost > self.swarm_best_cost
            ):
                self.swarm_best_position = current_swarm_best_position
                self.swarm_best_cost = current_swarm_best_cost

        elif (self.optimization != "minimize") & (self.optimization != "maximize"):
            raise Exception(
                'Invalid optimization type "'
                + self.optimization
                + '" valid selections are minimize or maximize'
            )

        return self

    def _update_velocities(self):
        r_swarm = self.swarm_attraction * np.random.rand(self.swarm_size,
                                                         self.position_size)  # Swarm best position velocity factor
        r_self = self.self_attraction * np.random.rand(self.swarm_size,
                                                       self.position_size)  # Self  best position velocity factor
        self.particle_best_positions = np.hstack(self.swarm['particle_best_position']).reshape(self.swarm_size,
                                                                                               self.position_size)
        new_velocities = self.inertia * self.velocities + \
                         r_swarm * (np.array([self.swarm_best_position, ] * self.swarm_size) - self.positions) + \
                         r_self * (self.particle_best_positions - self.positions)
        return new_velocities

    def update_positions(self):
        # Need to implement bounds
        if self.upper_bound is not None:
            raise Exception("Bounds not yet implemented")
        if self.lower_bound is not None:
            raise Exception("Bounds not yet implemented")

        new_velocities = self._update_velocities()

        self.positions = self.positions + new_velocities

        self.swarm["velocity"] = new_velocities.tolist()
        self.swarm["position"] = self.positions.tolist()

        # update the swarm trace.
        self.swarm_trace.set_index("swarm_particle_ID", inplace=True)
        self.swarm_trace.update(self.swarm.set_index("swarm_particle_ID"))
        self.swarm_trace.reset_index(inplace=True)

        self.swarm = self.swarm.reset_index(drop=True)

        return self

    def optimize(self, latent_cost_df):
        """
        Optimize the molecule cost in the latent space and returns optimized latent variables, together with position and velocities
        Args:
            latent_cost_df (dataframe): molecules presented by latent variables and calculated cost of the molecules.
            latent_cost_df must have columns ['latent', 'cost']
        Action: generate self.result_df

        """
        self.set_swarm(latent_cost_df)
        #         print (f'swarm best cost: {self.swarm_best_cost}')
        self.update_positions()
        self.result_df = self.swarm[['swarm_particle_ID', 'position']].copy(deep=False)
        self.result_df['cost'] = np.nan
        self.result_df.rename(columns={"position": "latent"}, inplace=True)
        return self.result_df

