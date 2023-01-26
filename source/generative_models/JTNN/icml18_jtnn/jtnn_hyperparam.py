import numpy as np
import pandas as pd
import os, sys, shutil
import pickle
from argparse import ArgumentParser


def define_hyperparam_runs(hyperparams, n_models):

    runs = pd.DataFrame()

    for key in hyperparams.keys():
        sample = hyperparams[key]["sample"]
        if (sample == "uniform") | (sample == "uniformint"):
            minimum = hyperparams[key]["min"]
            maximum = hyperparams[key]["max"]

            if hyperparams[key]["dist"] == "log":
                minimum = np.log(minimum)
                maximum = np.log(maximum)

            vals = np.random.rand(n_models) * (maximum - minimum) + minimum
        else:
            print("Unrecognized sampling type " + sample)

        # Round to 2 sig figs and untranform
        if hyperparams[key]["dist"] == "log":
            vals = np.array(
                ["{:0.3g}".format(np.exp(val)) for val in vals], dtype=float
            )
        else:
            vals = np.array(["{:0.3g}".format(val) for val in vals], dtype=float)

        if hyperparams[key]["sample"] == "uniformint":
            vals = np.array(np.round(vals, 0), dtype=int)

        runs[key] = vals

    return runs


def submit_jobs(
    params,
    hyperparam_runs,
    save_dir,
    partition="pbatch",
    bank="baasic",
    time_limit=1440,
    directory=os.getcwd(),
):
    fname = "vae_hyperparam_run.sh"
    with open(fname, "w+") as f:
        f.write(
            "#!/bin/bash \n"
            + "#SBATCH -t "
            + str(time_limit)
            + " \n"
            + "#SBATCH -p "
            + partition
            + " \n"
            + "#SBATCH -A "
            + bank
            + " \n"
            + "#SBATCH -D "
            + directory
            + " \n"
            + "#SBATCH --array=0-"
            + str(len(hyperparam_runs) - 1)
            + "\n"
        )

        f.write("\n\n")
        # Write hyperparameter variables as bash arrays
        for col in hyperparam_runs.columns:
            vals = (
                str(list(hyperparam_runs[col].values))
                .replace("[", "")
                .replace("]", "")
                .replace(",", "")
            )
            f.write(col + "=(" + vals + ") \n")
        f.write("\n\n")

        # Main function call command
        f.write("python -m atomsci.glo.generative_networks.icml18_jtnn.vae_train \\\n")

        for p in params:
            f.write("--" + p + " " + str(params[p]) + " \\\n")

        for col in hyperparam_runs.columns:
            f.write("--" + col + " ${" + col + "[${SLURM_ARRAY_TASK_ID}]} \\\n")

        f.write("--save_dir model${SLURM_ARRAY_TASK_ID}")

    os.system("sbatch " + fname)

    hyperparam_runs.to_csv(save_dir + "/run_summary.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--n_models',      type=int,  default=10,
                        help='Number of models (parameter sets) to generate')
    parser.add_argument('--save_dir',      type=str,  default=os.getcwd() + '/hyperparam_run',
                        help='Parent directory of directories where model checkpoints will be saved')
    parser.add_argument('--partition',     type=str,  default='pbatch',
                        help='Batch queue for training jobs')
    parser.add_argument('--bank',     type=str,  default='baasic',
                        help='Bank for training jobs')
    parser.add_argument('--time_limit',    type=int,  default=1440,
                        help='Time limit in minutes for training batch jobs')
    parser.add_argument('--train_split',   type=str,  default='train_split',
                        help='Path to directory of tensorized training data')
    parser.add_argument('--vocab',         type=str,  default='vocab.txt',
                        help='Path to vocabulary file')
    parser.add_argument('--workers',       type=int,  default=36,
                        help='Number of workers for multiprocessing tasks')
    parser.add_argument('--epoch',         type=int,  default=200,
                        help='Maximum number of epochs to train for')
    parser.add_argument('--print_iter',    type=int,  default=250,
                        help='Number of iterations (minibatches) between log messages during training')
    parser.add_argument('--save_interval', type=int,  default=10,
                        help='Number of epochs between checkpoints')
    parser.add_argument('--batch_size',    type=int,  default=36,
                        help='Number of SMILES strings per iteration (minibatch)')
    parser.add_argument('--warmup',        type=int,  default=None,
                        help='Number of iterations to wait before starting to increase beta')
    parser.add_argument('--shuffle',       type=bool, default=True,
                        help='Shuffle the order of the training SMILES strings')

    args = parser.parse_args()

    # Figure out the training set size
    train_size = 0
    for fn in os.listdir(args.train_split):
        if fn.endswith('.pkl'):
            pkl_path = os.path.join(args.train_split, fn)
            with open(pkl_path, 'rb') as f:
                pkl_data = pickle.load(f)
                train_size += len(pkl_data)
    print("Training set size = %d" % train_size)


    params = {
        "train": args.train_split,
        "vocab": args.vocab,
        "save_by": "epoch",
        "print_iter": args.print_iter,
        "save_interval": args.save_interval,
        "epoch": args.epoch,
        "beta": 0,
        "max_beta": 1,
        "workers": args.workers,
        "batch_size": args.batch_size,
        "shuffle": args.shuffle,
    }
    if args.warmup is None:
        params['warmup'] = 4000 * train_size // 66385
    else:
        params['warmup'] = args.warmup

    try:
        shutil.copytree(args.train_split, args.save_dir + "/" + args.train_split)
    except:
        shutil.rmtree(args.save_dir + "/" + args.train_split)
        shutil.copytree(args.train_split, args.save_dir + "/" + args.train_split)

    try:
        shutil.copy(args.vocab, args.save_dir + "/" + args.vocab)
    except:
        shutil.rm(args.save_dir + '/' + args.vocab)
        shutil.copy(args.vocab,args.save_dir + '/' + args.vocab)

    # Set ranges for hyperparams. Scale iteration counts by the training set size.
    hyperparams = {
        'lr':             {
            'sample': 'uniform',
            'dist':   'log',
            'max':    1E-3,
            'min':    2E-4},
        'step_beta':      {
            'sample': 'uniform',
            'dist':   'log',
            'max':    4E-4,
            'min':    3E-4},
        'kl_anneal_iter': {
            'sample': 'uniformint',
            'dist':   'log',
            'max':    11000 * train_size // 66385,
            'min':     9000 * train_size // 66385},
        'anneal_iter':    {
            'sample': 'uniformint',
            'dist':   'log',
            'max':     4000 * train_size // 66385,
            'min':     1000 * train_size // 66385},
        'hidden_size':    {
            'sample': 'uniformint',
            'dist':   'lin',
            'max':     900,
            'min':     450},
        'latent_size':    {
            'sample': 'uniformint',
            'dist':   'lin',
            'max':      80,
            'min':       40},
        'depthT':         {
            'sample': 'uniformint',
            'dist':   'lin',
            'max':       40,
            'min':       10},
        'depthG':         {
            'sample': 'uniformint',
            'dist':   'lin',
            'max':     10,
            'min':     3},
    }
    pd.set_option("display.max_columns", None)
    hprms = define_hyperparam_runs(hyperparams, args.n_models)
    print(hprms.head(len(hprms)))
    sys.stdout.flush()
    submit_jobs(
        params,
        hprms,
        args.save_dir,
        partition=args.partition,
        bank=args.bank,
        time_limit=args.time_limit,
        directory=args.save_dir,
    )
