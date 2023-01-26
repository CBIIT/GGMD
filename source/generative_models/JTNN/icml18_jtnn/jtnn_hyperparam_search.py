import os
import tempfile
import argparse
import pandas as pd
import numpy as np

np.random.seed(1)
from scipy.stats import randint, uniform

random_state = np.random.RandomState(
    1
)  # TODO: if these functions are to be imported elsewhere, may cause problems...


def optimize_params(experiment_config):
    # the following parameters are found via random search
    batch_size = randint(10, 65)  # integers between 10 and 64
    latent_size = randint(4, 256)  # integers between 4 and 255
    hidden_size = randint(4, 600)  # integers between 4 and 599
    warmup = randint(500, 6001)  # integers between 5000 and 6000
    step_beta = uniform(loc=1e-5, scale=1e-4)  # floats between 1e-5 and 1e-5 + 1e-4
    clip_norm = randint(10, 100)  # integers between 10 and 99
    anneal_rate = uniform(loc=0.1, scale=0.9)  # floats between 1e-1 and 1e-1 + 9e-1
    anneal_iter = randint(500, 4001)  # integers between 500 and 4000
    kl_anneal_iter = randint(10, 1001)  # integers between 10 and 1000

    # these are found via grid search instead in order to constrain the realm of possibilities
    lr = lambda x: np.random.choice([1e-2, 1e-3, 1e-4], x)
    beta = lambda x: np.random.choice(np.geomspace(1e-6, 5e-1, num=12), x)
    max_beta = lambda x: np.random.choice(np.geomspace(6e-1, 1, num=12), x)

    # seed the distributions
    batch_size.random_state = random_state
    latent_size.random_state = random_state
    hidden_size.random_sate = random_state
    step_beta.random_state = random_state
    warmup.random_state = random_state

    # Keep these parameters fixed to the defaults

    depthT = 20
    depthG = 3

    # define a dictionary where we will store all of the necessary information to run a trial, including param samples
    trial_config = dict()

    trial_config["experiment"] = experiment_config["experiment"]

    # fixed input parameters
    trial_config["train"] = experiment_config["train"]
    trial_config["vocab"] = experiment_config["vocab"]
    trial_config["epoch"] = experiment_config["epochs_per_trial"]
    trial_config["save_iter"] = experiment_config["save_iter"]
    trial_config["load_epoch"] = 0
    trial_config["print_iter"] = 50

    trial_list = []
    with tempfile.TemporaryDirectory() as tempdir:

        for trial in range(experiment_config["n_trials"]):

            trial_config["trial_num"] = int(trial)

            trial_config["batch_size"] = batch_size.rvs()
            trial_config["latent_size"] = latent_size.rvs()
            trial_config["hidden_size"] = hidden_size.rvs()
            trial_config["warmup"] = warmup.rvs()
            trial_config["step_beta"] = step_beta.rvs()
            trial_config["lr"] = lr(1)[0]
            trial_config["beta"] = beta(1)[0]
            trial_config["max_beta"] = max_beta(1)[0]
            trial_config["depthT"] = depthT
            trial_config["depthG"] = depthG
            trial_config["clip_norm"] = clip_norm.rvs()
            trial_config["anneal_rate"] = anneal_rate.rvs()
            trial_config["anneal_iter"] = anneal_iter.rvs()
            trial_config["kl_anneal_iter"] = kl_anneal_iter.rvs()

            # these parameters are sampled for each trial
            trial_config["save_dir"] = (
                experiment_config["output_dir"]
                + "/"
                + experiment_config["experiment"]
                + "/"
                + str(trial)
            )

            # print(trial_config["save_dir"])

            """
            if not os.path.exists(trial_config["save_dir"]):
                os.makedirs(trial_config["save_dir"])
            

            pd.DataFrame(trial_config, index=[trial]).to_csv(trial_config["save_dir"] + "/config.csv")
            """
            trial_script = script_trial(trial_config=trial_config)
            with open("{}/{}.sh".format(tempdir, trial), "w") as handle:
                # write the job string to a temporary file
                handle.write(trial_script)

        slurm_job = (
            "#!/usr/bin/sh\n"
            "#SBATCH --job-name=jtnn_hyperopt_{}".format(trial_config["experiment"])
            + "#SBATCH --time=1-00:00:00\n"
            "#SBATCH --partition=pbatch\n"
            + "cd /p/lscratchh/jones289/ATOM/active_learning/icml18_jtnn/fast_molvae; "
            "export PATH=/usr/workspace/wsa/jones289/miniconda3/bin:$PATH; source activate glo;"
            "export PYTHONPATH=/p/lscratchh/jones289/ATOM/active_learning/icml18_jtnn;"
            "#SBATCH --array=0-{}%{}\n".format(len(trial_list), args.max_jobs + 1)
            + "./{}/".format(tempdir)
            + "${SLURM_ARRAY_TASK_ID}.sh\n"
        )

        tmp_job_file = tempdir + "/hyperopt_job.sl"
        with open(tmp_job_file, "w") as handle:
            handle.write(slurm_job)

        print(os.listdir(tempdir))
        os.system("sbatch -p {} {}".format(args.partition, tmp_job_file))


def script_trial(trial_config):
    train = trial_config["train"]
    vocab = trial_config["vocab"]
    save_dir = trial_config["save_dir"]
    load_epoch = trial_config["load_epoch"]
    hidden_size = trial_config["hidden_size"]
    batch_size = trial_config["batch_size"]
    latent_size = trial_config["latent_size"]
    depthT = trial_config["depthT"]
    depthG = trial_config["depthG"]
    lr = trial_config["lr"]
    clip_norm = trial_config["clip_norm"]
    beta = trial_config["beta"]
    step_beta = trial_config["step_beta"]
    max_beta = trial_config["max_beta"]
    warmup = trial_config["warmup"]
    epoch = trial_config["epoch"]
    anneal_rate = trial_config["anneal_rate"]
    anneal_iter = trial_config["anneal_iter"]
    kl_anneal_iter = trial_config["kl_anneal_iter"]
    print_iter = trial_config["print_iter"]
    save_iter = trial_config["save_iter"]

    trial_job = (
        "#!/bin/bash\n"
        "python vae_train.py --train={} --vocab={} --save_dir={} --load_epoch={} --hidden_size={} "
        "--batch_size={} --latent_size={} --depthT={} --depthG={} --lr={} --clip_norm={} --beta={} "
        "--step_beta={} --max_beta={} --warmup={} --epoch={} --anneal_rate={} --anneal_iter={} "
        "--kl_anneal_iter={} --print_iter={} --save_iter={}".format(
            train,
            vocab,
            save_dir,
            load_epoch,
            hidden_size,
            batch_size,
            latent_size,
            depthT,
            depthG,
            lr,
            clip_norm,
            beta,
            step_beta,
            max_beta,
            warmup,
            epoch,
            anneal_rate,
            anneal_iter,
            kl_anneal_iter,
            print_iter,
            save_iter,
        )
    )

    # return trial_job

    return "echo hello world"


def search_param_space(args):
    # define a dictionary containing parameters that not specific to any experiment

    arg_config = vars(args)

    # TODO: make sure that all fields are being supplied...
    # WAIT, why is this a standalone function? hmm.....
    optimize_params(arg_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=str, default=None, help="path to training data")
    parser.add_argument("--vocab", type=str, default=None, help="path to vocabulary")
    parser.add_argument(
        "--experiment", required=True, help="title of experiment"
    )  # for ex. one of ["aurk_base", "aurk_base_train", "aurk_neutral", "aurk_neutral_train"]
    parser.add_argument(
        "--n-trials", type=int, default=10, help="number of trials to sample"
    )
    parser.add_argument(
        "--epochs-per-trial",
        type=int,
        default=10,
        help="maximum number of training epochs per trial",
    )
    parser.add_argument(
        "--output-dir", default=os.getcwd(), help="root path to the dir to save results"
    )
    parser.add_argument("--save-iter", type=int, default=1000)
    parser.add_argument(
        "--max-jobs",
        type=int,
        help="maximum number of concurrent slurm jobs",
        default=100,
    )
    parser.add_argument("--partition", help="slurm partition to submit search to")

    args = parser.parse_args()

    search_param_space(args)
