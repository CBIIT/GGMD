# trainer.py
from collections import Counter
import os
import sys
import time
import datetime
import logging
from ray import tune
from ray.tune import track
from ray.tune.suggest.ax import AxSearch
from ray.tune import Experiment
import os
import pandas as pd
import pickle
import ray
import ray.tune as tune
import numpy as np
from ax.service.ax_client import AxClient
from ax.utils.measurement.synthetic_functions import hartmann6
from ray.tune.schedulers import AsyncHyperBandScheduler
from atomsci.glo.generative_networks.icml18_jtnn.jtnn_experiments import JTNNTrainable


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-samples", type=int, default=30, help="number of trials to run"
    )
    parser.add_argument("--redis-password", help="password for redis cluster")
    parser.add_argument(
        "--exp-dir",
        help="path to directory to store experiment output, logs, and other items",
    )
    parser.add_argument(
        "--resume-exp",
        action="store_true",
        help="boolean that indicates whether to resume an experiment. useful for cases in which experiments may run beyond the slurm time limit",
    )
    return parser.parse_args()


def main():

    args = get_args()
    redis_password = args.redis_password

    ray.init(address=os.environ["ip_head"], redis_password=redis_password)

    print("Nodes in the Ray cluster: {}".format(ray.nodes()))

    # define the config for the ray.Tune Trainable

    config = {
        "num_samples": args.num_samples,
        "config": {"iterations": 100,},  # TODO: where is this being used?
        "stop": {"total_step": 10},
    }

    # initialize Client

    ax = AxClient(enforce_sequential_optimization=False)
    ax.create_experiment(
        name="jtnn_hyperparam_optimization",
        parameters=[
            {
                "name": "train",
                "type": "fixed",
                "value": "/g/g13/jones289/ATOM/active_learning/icml18_jtnn/fast_molvae/aurk/aurk_base_train_processed",
                "parameter_type": "str",
            },
            {
                "name": "test",
                "type": "fixed",
                "value": "/g/g13/jones289/ATOM/active_learning/icml18_jtnn/fast_molvae/aurk/aurk_base_test_processed",  # THIS SHOULD BE FOR VAL but we all we have is a train/test split at the moment..
                "parameter_type": "str",
            },
            {
                "name": "vocab",
                "type": "fixed",
                "value": "/g/g13/jones289/ATOM/glo/atomsci/glo/generative_networks/icml18_jtnn/data/aurk/aurk_base_rdkit/aurk_base_vocab.txt",
                "parameter_type": "str",
            },
            {
                "name": "hidden_size",
                "type": "range",
                "bounds": [100, 650],
                "value_type": "int",  # Optional, defaults to inference from type of "bounds".
                "log_scale": False,  # Optional, defaults to False.
            },
            {
                "name": "batch_size",
                "type": "range",
                "value_type": "int",
                "bounds": [16, 128],
            },
            {
                "name": "latent_size",
                "type": "range",
                "value_type": "int",
                "bounds": [28, 128],
            },
            {
                "name": "depthT",
                "type": "range",
                "value_type": "int",
                "bounds": [1, 20],
            },
            {"name": "depthG", "type": "range", "value_type": "int", "bounds": [1, 3],},
            {
                "name": "lr",
                "type": "range",
                "value_type": "float",
                "bounds": [1e-5, 1e-2],
            },
            {
                "name": "clip_norm",
                "type": "range",
                "value_type": "float",
                "bounds": [20.0, 70.0],
            },
            {
                "name": "beta",
                "type": "range",
                "value_type": "float",
                "bounds": [0, 1e-3],
            },
            {
                "name": "step_beta",
                "type": "range",
                "value_type": "float",
                "bounds": [1e-5, 1e-2],
            },
            {
                "name": "max_beta",
                "type": "range",
                "value_type": "float",
                "bounds": [1e-3, 1.0],  # TODO: put some more thought into this
            },
            {
                "name": "warmup",
                "type": "range",
                "value_type": "int",
                "bounds": [10, 10000],  # TODO: this might be a bit much
            },
            {
                "name": "anneal_rate",
                "type": "range",
                "value_type": "float",
                "bounds": [9e-3, 9e-1],
            },
            {
                "name": "anneal_iter",
                "type": "range",
                "value_type": "int",
                "bounds": [10, 10000],
            },
            {
                "name": "kl_anneal_iter",
                "type": "range",
                "value_type": "int",
                "bounds": [10, 9000],
            },
        ],
        objective_name="loss",
        minimize=True,
    )

    tune.run(
        JTNNTrainable,
        search_alg=AxSearch(
            ax, max_concurrent=2 * len(ray.nodes())
        ),  # Note that the argument here is the `AxClient`. max concurrent is equal to number of possible simulataneous trials
        verbose=2,  # Set this level to 1 to see status updates and to 2 to also see trial results. # To use GPU, specify: resources_per_trial={"gpu": 1}.
        # scheduler=AsyncHyperBandScheduler(metric="loss", mode="min"),  # leaving this unspecified gives the default FIFOScheduler
        reuse_actors=True,  # this is supposed to reduce overhead costs
        local_dir=args.exp_dir,
        resume=args.resume_exp,
        queue_trials=True,  # Whether to queue trials when the cluster does not currently have enough resources to launch one.
        resources_per_trial={
            "cpu": 16,
            "gpu": 1,
        },  # allocate 16 cpus and 1 gpu to each trial..
        **config
    )

    now = datetime.now()
    outfile = "testing_ray_ax_hyperopt_aurk_{}.pkl".format(
        "{}_{}_{}_{}_{}_{}".format(
            now.month, now.day, now.year, now.hour, now.minute, now.second
        )
    )

    with open(outfile, "wb") as handle:
        pickle.dump(ax, handle)


if __name__ == "__main__":

    # TODO: test fault tolerance capability
    # TODO: impose timelimits on trials
    # TODO: make sure that best values are being kept/updated
    # TODO: add JTNN training loop
    main()
