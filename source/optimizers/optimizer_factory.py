import argparse, yaml
import pandas as pd
from yaml import Loader

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
        from optimizers.genetic_optimizer import GeneticOptimizer
        return GeneticOptimizer(params)
    elif params.optimizer_type.lower() == "particleswarmoptimizer":
        from optimizers.particle_swarm import ParticleSwarmOptimizer
        return ParticleSwarmOptimizer(params)
    else:
        raise ValueError("Unknown optimizer_type %s" % params.optimizer_type)


def test_selection():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', help="Config file location *.yml", action='append', required=True)
    args = parser.parse_args()

    for conf_fname in args.config:
        with open(conf_fname, 'r') as f:
            parser.set_defaults(**yaml.load(f, Loader=Loader))

    params = parser.parse_args()

    population = pd.read_csv("/mnt/projects/ATOM/blackst/FNLGMD/workspace/development/debug_population.csv")
    
    optimizer = create_optimizer(params)
    population = optimizer.optimize(population)

    population.to_csv("/mnt/projects/ATOM/blackst/FNLGMD/workspace/development/optimized.csv", index=False)

if __name__ == "__main__":
    test_selection()
