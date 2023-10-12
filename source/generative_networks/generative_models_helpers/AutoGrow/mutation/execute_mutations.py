"""
This function should contain all the info for executing the mutation functions
"""

import __future__

import random
import copy
import pandas as pd


import generative_networks.generative_models_helpers.AutoGrow.mutation.smiles_click_chem.smiles_click_chem as SmileClickClass
#from generative_network import GenerativeModel

#STB Development notes TODO:
"""
- Need to figure out a way to sanitize the tracking of generated smiles here and 
    in the smiles_click_chem.py class
"""



#######################################
# Functions for creating molecular models
##########################################

class Mutator():
    def __init__(self, params):
        """
        :param int number_of_processors: number of processors as specified by the
            user
        """
        self.number_of_processors = params.number_of_processors
        self.rxn_library = params.rxn_library
        self.rxn_library_file = params.rxn_library_file
        self.function_group_library = params.function_group_library
        self.complementary_mol_dir = params.complementary_mol_dir

        # initialize the smileclickclass
        self.a_smiles_click_chem_object = SmileClickClass.SmilesClickChem(
            params)



    def make_mutants(self, generation_num, num_mutants_to_make, parent_population, new_generation_df):
        """
        Make mutant compounds in a list to be returned

        This runs SmileClick and returns a list of new molecules

        Inputs:
        :param dict vars: a dictionary of all user variables
        :param int generation_num: generation number
        :param int num_mutants_to_make: number of mutants to return
        :param list ligands_list: list of ligand/name pairs which are the order in
            which to be sampled 
        :param list new_mutation_smiles_list: is the list of
            mutants made for the current generation being populated but in a previous
            iteration of the loop in Operations
        :param list rxn_library_variables: a list of user variables which define
            the rxn_library, rxn_library_file, and function_group_library. ie.
            rxn_library_variables = [vars['rxn_library'], vars['rxn_library_file'],
            vars['function_group_library']]

        Returns:
        :returns: list new_ligands_list: ligand/name pairs OR returns None if
            sufficient number was not generated. None: bol if mutations failed
        """

        loop_counter = 0
        self.ligands_list = parent_population

        while loop_counter < 2000 and len(new_generation_df) < num_mutants_to_make:

            #react_list = copy.deepcopy(ligands_list)

            while len(new_generation_df) < num_mutants_to_make and len(self.ligands_list) > 0:

                self.a_smiles_click_chem_object.update_list_of_already_made_smiles(new_generation_df)
                num_to_make = num_mutants_to_make - len(new_generation_df)

                # to minimize a big loop of running a single mutation at a time we
                # will make 1 new lig/processor. This will help to prevent wasting
                # reasources and time.
                if num_to_make < self.number_of_processors:
                    num_to_make = self.number_of_processors

                #smile_pairs = [
                #    react_list.pop() for x in range(num_to_make) if len(react_list) > 0
                #]

                #smile_pairs = [
                #    react_list.pop() for x in range(num_to_make) if len(react_list) > 0
                #]
                
                #smile_inputs = [x[0] for x in smile_pairs]
                #smile_names = [x[1] for x in smile_pairs]

                #job_input = tuple(
                #    [tuple([smile, self.a_smiles_click_chem_object]) for smile in smile_inputs]
                #)

                # STB
                #results = vars["parallelizer"].run(
                #    job_input, run_smiles_click_for_multithread
                #)
                #results = self.run_smiles_click_for_multithread(job_input)

                results = self.run_smiles_click_linear(self.ligands_list)

                #TODO: STB choose new variable name to replace i. Don't like using i for complex variables
                for index, i in enumerate(results): 
                    if i is not None:

                        # Get the new molecule's (aka the Child lig) Smile string
                        new_smile = i[0]
                        reaction_id = i[1]
                        zinc_id = i[2]
                        parent1_id = i[3]


                        if new_smile not in new_generation_df:
                            new_generation_df.loc[len(new_generation_df.index)] = {
                                "smiles": new_smile, 
                                "parent1_id": int(parent1_id), 
                                "reaction_id": reaction_id, 
                                "zinc_id": zinc_id
                                }
                        
                        if len(new_generation_df) >= num_mutants_to_make:
                            break
                            
                return new_generation_df
                
            break #STB: TODO: breaks input because when running non parallel, we only need to loop once.
                    #During parallel, each loop only runs one smile per available node

            loop_counter = loop_counter + 1
        
        # once the number of mutants we need is generated return the list
        return new_generation_df



    def run_smiles_click_for_multithread(self, smile):
        """
        This function takes a single smilestring and performs SmileClick on it.

        This is necessary for Multithreading as it is unable to execute
        multithread on a class function, but can thread a class run within a
        function.

        Inputs:
        :param str smile: a SMILES string

        Returns:
        :returns: str result_of_run: either a smile string of a child mol or None
            if the reactions failed
        """

        result_of_run = self.a_smiles_click_chem_object.run_smiles_click(smile)

        return result_of_run



    def run_smiles_click_linear(self, population): #Dataframe version
        
        smiles = population['smiles'].tolist()
        ids = population['compound_id'].tolist()

        results = []
        for smile, id in zip(smiles, ids):
            result = self.a_smiles_click_chem_object.run_smiles_click(smile)
            
            if result is not None:
                results.append([result[0], result[1], result[2], id])

        return results