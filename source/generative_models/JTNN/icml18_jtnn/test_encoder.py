"""
Module for testing JT-VAE autoencoder performance
"""

import sys, os, argparse, random
import re
import pandas as pd
import numpy as np
from atomsci.glo.generative_networks import VAEUtils
from atomsci.ddm.pipeline.chem_diversity import calc_dist_smiles
from atomsci.ddm.utils import llnl_utils


#-----------------------------------------------------------------------------------------------------------------------------------
def test_encoder(vae, test_path, max_size=None, timeout=10):
    """
    Compute recovery rate and average Tanimoto distance for a single VAE model checkpoint on a single list of SMILES strings.

    Args:
        vae: (DistributedEvaluator, JTNNVAE or str): Object used to evaluate recovery rate

        test_path (str): Path to file of test SMILES strings

        max_size (int): Maximum number of SMILES strings to test.

        timeout (int): Timeout for decode operations, in seconds.

    Return:
        recovery (DataFrame): Table of recovery results with columns:
            Input: input SMILES strings
            Output: decoded SMILES strings
            Tanimoto Distance: distance between input and output
            Recovered: binary, 1 if input == output
            Decode Time: seconds to encode and decode input
    """
    smiles = pd.read_csv(test_path).values
    smiles = smiles.tolist()
    smiles = [smi[0] for smi in smiles]

    if (max_size is not None) & (max_size < len(smiles)):
        random.shuffle(smiles)
        smiles = smiles[0:max_size]

    if isinstance(vae, VAEUtils.DistributedEvaluator):
        evaluator = vae
    else:
        evaluator = VAEUtils.DistributedEvaluator(vae=vae, timeout=timeout, verbose=False)
    recovery = evaluator.recover(smiles)

    return recovery

#-----------------------------------------------------------------------------------------------------------------------------------
def test_checkpoint(vae_path, train_path, test_path, max_size=None, timeout=10, show_recovery=True, verbose=False):
    """
    Compute recovery rate and average Tanimoto distance for a single VAE model checkpoint for the SMILES strings in
    both train_path and test_path files.
    """
    evaluator = VAEUtils.DistributedEvaluator(vae=vae_path, verbose=verbose, timeout=timeout)
    train_recovery = test_encoder(vae=evaluator, test_path=train_path, max_size=max_size, timeout=timeout)
    evaluator = VAEUtils.DistributedEvaluator(vae=vae_path, verbose=verbose, timeout=timeout)
    test_recovery = test_encoder(vae=evaluator, test_path=test_path, max_size=max_size, timeout=timeout)
    
    if show_recovery:
        train_recovery['Set'] = 'train'
        test_recovery['Set']  = 'test'
        rec_df = pd.concat([train_recovery,test_recovery], ignore_index=True)
        rec_file = f"{vae_path}_recovery.csv"
        rec_df.to_csv(rec_file, index=False)

    train_recovery = train_recovery[train_recovery.Output != '<Decoding Timed Out>']
    test_recovery = test_recovery[test_recovery.Output != '<Decoding Timed Out>']

    chk_df = pd.DataFrame(
            {
                "Checkpoint": [vae_path],
                "Epoch": [evaluator.hyperparams["epoch"]],
                "Train Recovery": [
                    sum(train_recovery["Recovered"]) / len(train_recovery) * 100
                ],
                "Train Distance": [
                    np.average(train_recovery["Tanimoto Distance"].values)
                ],
                "# Train Eval": [len(train_recovery)],
                "Test Recovery": [
                    sum(test_recovery["Recovered"]) / len(test_recovery) * 100
                ],
                "Test Distance": [
                    np.average(test_recovery["Tanimoto Distance"].values)
                ],
                "# Test Eval": [len(test_recovery)],
            }
        )
    if verbose:
        print(chk_df)
    return chk_df


#-----------------------------------------------------------------------------------------------------------------------------------
def test_training_run(run_path, train_path, test_path, max_size=None, show_recovery=True, verbose=True, timeout=10):
    """
    Compute recovery rate and Tanimoto distances at each checkpoint from a single model training run, for
    test SMILES strings and a sample of training SMILES strings. Tests all checkpoints within the current process.

    Args:
        run_path (str): Directory containing model checkpoints

        train_path (str): Path to training dataset file

        test_path (str): Path to test dataset file

        max_size (int): Maximum number of SMILES strings in each set to encode and decode.

        timeout (int): Timeout for decode operations, in seconds.

        verbose (bool): Print verbose log messages.

    Return:
        df (DataFrame): Table of results for each checkpoint. For training and test set compounds, lists
        the percentage of SMILES strings perfectly recovered and the average Tanimoto distance between
        input and decoded SMILES strings.
    """
    all_files = os.listdir(run_path)
    checkpoints = [x for x in all_files if ".iter" in os.fsdecode(x)] + [
        x for x in all_files if ".epoch" in os.fsdecode(x)
    ]
    run_df = pd.DataFrame()
    pd.set_option("display.max_columns", None)

    for chk in checkpoints:
        vae_path = run_path + "/" + chk

        if verbose:
            print(f"Testing checkpoint {chk}")
        evaluator = VAEUtils.DistributedEvaluator(vae=vae_path, timeout=timeout, verbose=False)
        train_recovery = test_encoder(evaluator, train_path, max_size=max_size, timeout=timeout)
        test_recovery = test_encoder(evaluator, test_path, max_size=max_size, timeout=timeout)

        if show_recovery:
            train_recovery['Set'] = 'train'
            test_recovery['Set']  = 'test'
            rec_df = pd.concat([train_recovery,test_recovery], ignore_index=True)
            rec_file = f"{vae_path}_recovery.csv"
            rec_df.to_csv(rec_file, index=False)

        chk_df = pd.DataFrame(
                {
                    "Checkpoint": [chk],
                    "Epoch": [evaluator.hyperparams["epoch"]],
                    "Train Recovery": [
                        sum(train_recovery["Recovered"]) / len(train_recovery) * 100
                    ],
                    "Train Distance": [
                        np.average(train_recovery["Tanimoto Distance"].values)
                    ],
                    "# Train Eval": [len(train_recovery)],
                    "Test Recovery": [
                        sum(test_recovery["Recovered"]) / len(test_recovery) * 100
                    ],
                    "Test Distance": [
                        np.average(test_recovery["Tanimoto Distance"].values)
                    ],
                    "# Test Eval": [len(test_recovery)],
                }
            )
        # Write out intermediate results for this checkpoint, and also append to run results
        chk_file = f"{vae_path}_summary.csv"
        chk_df.to_csv(chk_file, index=False)
        run_df.append(chk_df)
        if verbose:
            print(chk_df)

    return run_df


#-----------------------------------------------------------------------------------------------------------------------------------
def launch_training_run_tests(run_path, train_path, test_path, max_size=None, max_jobs=16, verbose=True, timeout=10,
                            time_limit=720, partition='pvis', bank='ncov2019', show_recovery=True):
    """
    Launch a series of batch jobs, one per checkpoint, to compute recovery rates and Tanimoto distances for
    each model saved by a training run for a given hyperparameter set. The results will be combined later
    after the batch jobs have finished.

    Args:
        run_path (str): Directory containing checkpoint files for a training run

        train_path (str): Path to training dataset file

        test_path (str): Path to test dataset file

        max_size (int): Maximum number of SMILES strings in each dataset to encode and decode.

        max_jobs (int): Maximum number of batch jobs to be on the queue at one time.

        timeout (int): Timeout for decode operations, in seconds.

        verbose (bool): Print verbose log messages.

        time_limit (int): Batch job time limit in minutes

        partition (str): Batch queue to run jobs on

        bank (str): Bank to charge compute time to

        show_recovery (bool): If true, output a detailed table of encoded and decoded SMILES strings and Tanimoto distances

    """
    # Get subdirectories of run_path
    if verbose:
        print("Evaluating runs under %s" % run_path)
    all_files = os.listdir(run_path)
    vae_paths = []
    epochs = []
    epoch_pat = re.compile(r'model.epoch-(\d+)')
    for file in all_files:
        m = epoch_pat.match(file)
        if m is not None:
            vae_paths.append(f"{run_path}/{file}")
            epochs.append(int(m.group(1)))
    chkpt_df = pd.DataFrame(dict(vae_path=vae_paths, epoch=epochs)).sort_values(by='epoch', ascending=False)
    username = llnl_utils.get_my_username()
    for vae_path, epoch in zip(chkpt_df.vae_path.values, chkpt_df.epoch.values):
        script_file = f"{run_path}/test_recovery_epoch_{epoch}.sh"
        with open(script_file, 'w+') as fp:
            fp.write('#!/bin/bash \n' +
                    '#SBATCH -t ' + str(time_limit) + ' \n' +
                    '#SBATCH -A ' + bank + ' \n' +
                    '#SBATCH -p ' + partition + ' \n' +
                    '#SBATCH -D ' + run_path + ' \n')
            fp.write('python -m atomsci.glo.generative_networks.icml18_jtnn.test_encoder \\\n')
            fp.write('  --mode checkpoint \\\n')
            fp.write('  --vae_path %s \\\n' % vae_path)
            fp.write('  --test_path %s \\\n' % test_path)
            fp.write('  --train_path %s \\\n' % train_path)
            fp.write('  --max_size %d \\\n' % max_size)
            fp.write('  --timeout %d \\\n' % timeout)
            if show_recovery:
                fp.write('  --show_recovery \\\n')
            if verbose:
                fp.write('  --verbose \\\n')
            run_perf_file = f"{run_path}/recovery_perf_epoch_{epoch}.csv"
            fp.write('  --output_file %s\n' % run_perf_file)

        if verbose:
            print('Wrote batch script %s' % script_file)
        llnl_utils.throttle_jobs(max_jobs, my_username=username, retry_time=60, verbose=True)
        os.system('sbatch ' + script_file)


#-----------------------------------------------------------------------------------------------------------------------------------
def test_hyperparam_run(path, train_path, test_path, max_size=None, verbose=True, show_recovery=False, output_file=None,
                        timeout=10):
    """
    Compute recovery rates and Tanimoto distances for a series of model training runs generated by a hyperparameter search.
    """
    df = pd.DataFrame()

    runs = os.listdir(path)
    runs = np.array([path + "/" + r for r in runs])
    # Filter to only directories
    idx = [os.path.isdir(f) for f in runs]
    runs = runs[idx]
    # Filter to only directories with models (i.e. ones that contain files with .epoch-)
    idx = (
        np.array(
            [sum([".iter-" in os.fsdecode(fname) for fname in os.listdir(run)]) for run in runs]
        )
        > 0
    ) + (
        np.array(
            [sum([".epoch-" in os.fsdecode(fname) for fname in os.listdir(run)]) for run in runs]
        )
        > 0
    )

    runs = runs[idx]

    for idx, run in enumerate(runs):
        if verbose:
            print(
                "Evaluating run for "
                + run
                + "("
                + str(idx + 1)
                + " of "
                + str(len(runs))
                + ")"
            )
        run_df = test_training_run(
            run, train_path, test_path, max_size=max_size, max_jobs=1, show_recovery=show_recovery, verbose=verbose, timeout=timeout
        )
        run_df["model"] = [run] * len(run_df)
        df = df.append(run_df)
        if verbose:
            print('........done')
        if output_file is not None:
            if idx == 0:
                run_df.to_csv(output_file)
            else:
                run_df.to_csv(output_file, header=False, mode="a")
    return df

#-----------------------------------------------------------------------------------------------------------------------------------
def launch_hyperparam_tests(path, train_path, test_path, max_size=None, verbose=True, timeout=10,
                            time_limit=1440, partition='pbatch', bank='baasic', max_jobs=8):
    """
    Launch a series of batch jobs, one per training run, to compute recovery rates and Tanimoto distances for
    the runs generated by a hyperparameter search. The results will be combined later with
    combine_hyperparam_results() after the batch jobs have finished.

    Args:
        path (str): Directory containing training run subdirectories.

        train_path (str): Path to training dataset file

        test_path (str): Path to test dataset file

        max_size (int): Maximum number of SMILES strings in each set to encode and decode.

        timeout (int): Timeout for decode operations, in seconds.

        verbose (bool): Print verbose log messages.

    """
    # Get subdirectories of path
    print("Evaluating runs under %s" % path)
    dir_files = np.array( [os.path.join(path, d) for d in os.listdir(path)])
    subdirs = np.array( [d for d in dir_files if os.path.isdir(d)] )
    print("%d subdirs" % len(subdirs))
    # Filter to only directories with model checkpoints (i.e. ones that contain files with .iter- or .epoch-)
    idx  = ((np.array([sum(['.iter-'  in os.fsdecode(fname) for fname in os.listdir(dir)]) for dir in subdirs]) > 0 ) +
            (np.array([sum(['.epoch-' in os.fsdecode(fname) for fname in os.listdir(dir)]) for dir in subdirs]) > 0 ))
    run_dirs = subdirs[idx]
    print("%d run dirs" % len(run_dirs))
    username = llnl_utils.get_my_username()

    for idx, run_dir in enumerate(run_dirs):
        run_path = os.path.join(path, run_dir)
        script_file = os.path.join(run_path, 'test_recovery.sh')
        with open(script_file, 'w+') as fp:
            fp.write('#!/bin/bash \n' +
                    '#SBATCH -t ' + str(time_limit) + ' \n' +
                    '#SBATCH -A ' + bank + ' \n' +
                    '#SBATCH -p ' + partition + ' \n' +
                    '#SBATCH -D ' + run_dir + ' \n')
            run_perf_file = os.path.join(run_dir, 'recovery_perf.csv')
            fp.write('python -m atomsci.glo.generative_networks.icml18_jtnn.test_encoder \\\n')
            fp.write('  --mode run \\\n')
            fp.write('  --vae_path %s \\\n' % run_path)
            fp.write('  --test_path %s \\\n' % test_path)
            fp.write('  --train_path %s \\\n' % train_path)
            fp.write('  --max_size %d \\\n' % max_size)
            fp.write('  --timeout %d \\\n' % timeout)
            if verbose:
                fp.write('  --verbose \\\n')
            fp.write('  --output_file %s\n' % run_perf_file)

        print('Wrote batch script %s' % script_file)
        llnl_utils.throttle_jobs(max_jobs, my_username=username, retry_time=60, verbose=True)
        os.system('sbatch ' + script_file)

#-----------------------------------------------------------------------------------------------------------------------------------
def combine_hyperparam_results(vae_path, output_file):
    """
    Combine the results from testing recovery performance on the training runs within a hyperparameter search.
    The performance stats will be merged with the saved hyperparameters and written to a CSV file.
    """
    param_file = os.path.join(vae_path, 'run_summary.csv')
    param_df = pd.read_csv(param_file)
    indices = param_df.index.values
    run_dirs = [os.path.join(vae_path, 'model%d' % i) for i in indices]
    param_df['model'] = run_dirs
    param_df['Run'] = indices
    perf_data = []
    for i, run_dir in enumerate(run_dirs):
        run_perf_file = os.path.join(run_dir, 'recovery_perf.csv')
        if os.path.exists(run_perf_file):
            run_perf_df = pd.read_csv(run_perf_file)
            run_perf_df['Epoch'] = [int(s.replace('model.epoch-', '')) for s in run_perf_df.Checkpoint.values]
            run_perf_df['Run'] = i
            perf_data.append(run_perf_df)
    perf_df = pd.concat(perf_data, ignore_index=True)
    #print('perf_df columns: %s' % ', '.join(perf_df.columns.values))
    #print('param_df columns: %s' % ', '.join(param_df.columns.values))

    combined_df = perf_df.merge(param_df, how='left', on='Run')
    combined_df.to_csv(output_file)
    print('Wrote combined data to %s' % output_file)

#-----------------------------------------------------------------------------------------------------------------------------------
def combine_training_run_results(run_path, output_file):
    """
    Combine the results from testing recovery performance on the checkpoints within a training run directory.
    The performance stats will be merged with the saved hyperparameters and written to a CSV file.

    Args:
        run_path (str): Directory containing checkpoint files for a training run

        output_file (str): Path to file to receive output table.

    Returns:
        None
    """
    # Figure out our run number within the hyperparameter search set
    run_dir = os.path.basename(run_path)
    model_pat = re.compile(r'model(\d+)')
    run_num = int(model_pat.match(run_dir).group(1))

    # Extract the hyperparameters for this run from the parameter table
    parent_dir = os.path.dirname(run_path)
    param_file = os.path.join(parent_dir, 'run_summary.csv')
    param_df = pd.read_csv(param_file)
    my_param_df = param_df.iloc[run_num]

    # Read the recovery performance output for each checkpoint
    all_files = os.listdir(run_path)
    epoch_pat = re.compile(r'model.epoch-(\d+)')
    perf_data = []
    for file in all_files:
        m = epoch_pat.match(file)
        if m is not None:
            epoch = m.group(1)
            run_perf_file = f"{run_path}/recovery_perf_epoch_{epoch}.csv"
            if os.path.exists(run_perf_file):
                perf_df = pd.read_csv(run_perf_file)
                perf_data.append(perf_df)
    perf_df = pd.concat(perf_data, ignore_index=True)

    # Add the parameter columns; these will be the same for all checkpoints
    param_cols = ['lr', 'step_beta', 'kl_anneal_iter', 'anneal_iter', 'hidden_size', 'latent_size', 'depthT', 'depthG']
    for col in param_cols:
        perf_df[col] = my_param_df[col]

    perf_df.to_csv(output_file, index=False)
    print(f"Wrote performance results for run {run_num} to {output_file}")


#-----------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test performance of JT-VAE autoencoder models.')
    parser.add_argument('--mode',    type=str, dest='mode',
                        choices=['hyperparam', 'run', 'checkpoint', 'combine', 'combine_run'], default='checkpoint',
                        help='Scope of models to assess performance for: single checkpoint, all checkpoints in run, or all runs\n' +
                             'in hyperparam search. If mode is "combine", combine the separate performance tables from all runs\n' +
                             'in a hyperparameter search. If mode is "combine_run", combine the results from all checkpoints\n' +
                             'in a training run.')
    parser.add_argument('--vae_path',    type=str, dest='vae_path',   required=True,
                        help='Path to directory or checkpoint file over which to assess performance')
    parser.add_argument('--test_path',   type=str, dest='test_path',  
                        help='Path to file containing test set SMILES strings')
    parser.add_argument('--train_path',  type=str, dest='train_path', 
                        help='Path to file containing training set SMILES strings')
    parser.add_argument('--max_size',    type=int, dest='max_size',   default=None,
                        help='Maximum number of SMILES strings to encode and decode in each set')
    parser.add_argument('--timeout',    type=int, dest='timeout',   default=10,
                        help='Number of seconds to wait for each decode operation before timing out')
    parser.add_argument('--verbose',        action='store_true',      default=False,
                        help='Print verbose log messages')
    parser.add_argument('--show_recovery',        action='store_true',      default=False,
                        help='Output detailed table of input and decoded SMILES for each checkpoint')
    parser.add_argument('--time_limit',    type=int, dest='time_limit',   default=1440,
                        help='Time limit in minutes for batch jobs')
    parser.add_argument('--partition',  type=str, dest='partition', default='pbatch',
                        help='Batch queue for batch jobs')
    parser.add_argument('--bank',  type=str, dest='bank', default='baasic',
                        help='Bank for batch jobs')
    parser.add_argument('--max_jobs',  type=int, dest='max_jobs', default=6,
                        help='Max number of batch jobs to have in queue or running')
    parser.add_argument('--training_run',   action='store_true',      default=False,
                        help='Like mode="run", but does not launch a job for each checkpoint')
    parser.add_argument('--hyperparam_run', action='store_true',      default=False,
                        help='Like mode="hyperparam", but does not launch a job for each training run')
    parser.add_argument('--output_file',    default=None,
                        help='Path to CSV file to receive output')

    args = parser.parse_args()
    if args.output_file is not None:
        if not os.path.exists(os.path.dirname(args.output_file)):
            print(f"Directory {os.path.dirname(args.output_file)} does not exist")
            sys.exit(1)
        print('Output file: %s' % args.output_file)

    if args.mode == 'checkpoint':
        out_df = test_checkpoint(vae_path=args.vae_path, train_path=args.train_path, test_path=args.test_path,
                              max_size=args.max_size, timeout=args.timeout, show_recovery=args.show_recovery,
                              verbose=args.verbose)
        if args.output_file is not None:
            out_df.to_csv(args.output_file)

    elif args.mode == 'run':
        print('Launching test jobs for training run %s' % args.vae_path)
        launch_training_run_tests(args.vae_path,
                                args.train_path,
                                args.test_path,
                                max_size = args.max_size,
                                max_jobs = args.max_jobs,
                                timeout = args.timeout,
                                verbose  = args.verbose,
                                time_limit=args.time_limit, 
                                partition=args.partition,
                                bank=args.bank,
                                show_recovery = args.show_recovery)
    elif args.mode == 'hyperparam':
        # Launch a separate batch job for each run (parameter set)
        launch_hyperparam_tests(args.vae_path, args.train_path, args.test_path, max_size=args.max_size, timeout=args.timeout,
                                verbose=args.verbose, time_limit=args.time_limit, partition=args.partition,
                                bank=args.bank, max_jobs=args.max_jobs)
    elif args.training_run:
        print('Testing training run %s' % args.vae_path)
        out_df = test_training_run(args.vae_path,
                                args.train_path,
                                args.test_path,
                                max_size = args.max_size,
                                timeout = args.timeout,
                                verbose  = args.verbose,
                                show_recovery = args.show_recovery)
        if args.output_file is not None:
            out_df.to_csv(args.output_file)
    elif args.hyperparam_run:
        # Test all checkpoints for all runs (parameter sets) within the current batch job. This may not be feasible
        # to complete given LC job time limits.
        out_df = test_hyperparam_run(args.vae_path,
                                  args.train_path,
                                  args.test_path,
                                  max_size    = args.max_size,
                                  timeout = args.timeout,
                                  verbose     = args.verbose,
                                  show_recovery = args.show_recovery,
                                  output_file = args.output_file)
    elif args.mode == 'combine':
        combine_hyperparam_results(args.vae_path, args.output_file)
    elif args.mode == 'combine_run':
        combine_training_run_results(args.vae_path, args.output_file)
    else:
        raise ValueError(f"Unrecognized mode parameter {args.mode}")
