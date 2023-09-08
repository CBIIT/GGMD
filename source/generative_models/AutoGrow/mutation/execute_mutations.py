"""
This function should contain all the info for executing the mutation functions
"""

import __future__

import random
import copy


import generative_models.AutoGrow.mutation.smiles_click_chem.smiles_click_chem as SmileClickClass
#from generative_network import GenerativeModel

#STB Development notes TODO:
"""
- Need to figure out a way to sanitize the tracking of generated smiles here and 
    in the smiles_click_chem.py class
- Add in the tracking of unique name used here in the autogrow method. Would be a useful addition to the data to 
    track the synthesis path of each ligand
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

    #def make_mutants(self, vars, generation_num,
    #                num_mutants_to_make, ligands_list,
    #                new_mutation_smiles_list):
    def make_mutants(self, generation_num, num_mutants_to_make, ligands_list, new_mutation_smiles_list):
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

        if len(new_mutation_smiles_list) == 0:
            new_ligands_list = []
        else:
            new_ligands_list = new_mutation_smiles_list

        loop_counter = 0

        #number_of_processors = int(vars["parallelizer"].return_node()) #STB

        #STB
        # initialize the smileclickclass
        #a_smiles_click_chem_object = SmileClickClass.SmilesClickChem(
        #    rxn_library_variables, new_mutation_smiles_list, vars["filter_object_dict"]
        #)

        while loop_counter < 2000 and len(new_ligands_list) < num_mutants_to_make:

            react_list = copy.deepcopy(ligands_list)

            while len(new_ligands_list) < num_mutants_to_make and len(react_list) > 0:

                self.a_smiles_click_chem_object.update_list_of_already_made_smiles(new_ligands_list)
                num_to_make = num_mutants_to_make - len(new_ligands_list)

                # to minimize a big loop of running a single mutation at a time we
                # will make 1 new lig/processor. This will help to prevent wasting
                # reasources and time.
                if num_to_make < self.number_of_processors:
                    num_to_make = self.number_of_processors

                #smile_pairs = [
                #    react_list.pop() for x in range(num_to_make) if len(react_list) > 0
                #]

                smile_pairs = [
                    react_list.pop() for x in range(num_to_make) if len(react_list) > 0
                ]
                
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

                results = self.run_smiles_click_linear(smile_pairs)

                #TODO: STB choose new variable name to replace i. Don't like using i for complex variables
                for index, i in enumerate(results): 
                    if i is not None:

                        # Get the new molecule's (aka the Child lig) Smile string
                        child_lig_smile = i[0]

                        if child_lig_smile not in new_ligands_list:
                            new_ligands_list.append(child_lig_smile)

                        """ #STB
                        # get the reaction id number
                        #reaction_id_number = i[1] #STB

                        # get the ID for the parent of a child mol and the
                        # complementary parent mol. comp mol could be None or a
                        # zinc database ID
                        #parent_lig_id = smile_names[index] #STB
                        #zinc_id_comp_mol = i[2] #STB

                        # Make a list of all smiles and smile_id's of all
                        # previously made smiles in this generation
                        list_of_already_made_smiles = []
                        #list_of_already_made_id = [] #STB

                        # fill lists of all smiles and smile_id's of all
                        # previously made smiles in this generation
                        for x in new_ligands_list:
                            list_of_already_made_smiles.append(x[0])
                            #list_of_already_made_id.append(x[1])

                        if child_lig_smile not in list_of_already_made_smiles:
                            # if the smiles string is unique to the list of
                            # previous smile strings in this round of reactions
                            # then we append it to the list of newly created
                            # ligands we append it with a unique ID, which also
                            # tracks the progress of the reactant
                            is_name_unique = False
                            while is_name_unique is False:
                                # make unique ID with the 1st number being the
                                # parent_lig_id for the derived mol, Followed by
                                # Mutant, folowed by the generationnumber,
                                # followed by a unique.

                                # get the unique ID (last few diget ID of the
                                # parent mol
                                
                                parent_lig_id = parent_lig_id.split(")")[-1]

                                random_id_num = random.randint(100, 1000000)
                                if zinc_id_comp_mol is None:
                                    new_lig_id = "({})Gen_{}_Mutant_{}_{}".format(
                                        parent_lig_id,
                                        generation_num,
                                        reaction_id_number,
                                        random_id_num,
                                    )
                                else:
                                    new_lig_id = "({}+{})Gen_{}_Mutant_{}_{}".format(
                                        parent_lig_id,
                                        zinc_id_comp_mol,
                                        generation_num,
                                        reaction_id_number,
                                        random_id_num,
                                    )
                                
                                # check name is unique
                                #if new_lig_id not in list_of_already_made_id: #STB
                                #    is_name_unique = True

                            # make a temporary list containing the smiles string
                            # of the new product and the unique ID
                            ligand_info = [child_lig_smile, new_lig_id]

                            # append the new ligand smile and ID to the list of
                            # all newly made ligands
                            new_ligands_list.append(ligand_info)

                        """

            loop_counter = loop_counter + 1
        
        if len(new_ligands_list) < num_mutants_to_make:
            return None

        # once the number of mutants we need is generated return the list
        return new_ligands_list


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
    
    def run_smiles_click_linear(self, smiles_list):

        results = []
        for smile in smiles_list:
            result = self.a_smiles_click_chem_object.run_smiles_click(smile)
            if result is not None:
                results.append(result)
        return results
