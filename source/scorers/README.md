# Scoring Code

This directory contains all the code related to computing the fitness values. Currently there is only one scoring function implemented and that is for the logP Octanol water Coefficient. 

Any scoring function that is implemented should be a class that inherrits the type called "Scorer". Any implemented scorer
has only one requirement and that is to have a function called "score" that accepts and returns a pandas DataFrame. The implemented scorer
can have any number of helper functions, but must at a minimum have an init and score function. 

Example scoring functions can include synthetic accessability score, docking scores, binding free energy, and predictive machine learning model scores.