Here are the steps for training a JTNN-VAE on a chemical library, to be used in the GMD loop. This assumes that you have cloned the glo repo from LC Bitbucket. The scripts to be run below are in the directory glo/atomsci/glo/generative_networks/icml18_jtnn (relative to your git root directory). The scripts neurocrine_train_1.sh, neurocrine_test_1.sh and neurocrine_combine_1.sh are examples of the sequences used in the Neurocrine project; you'll want to make your own versions to deal with different datasets.

1. Make a text file of the SMILES strings you want to use for training and testing - one per line, with no header. Edit the script prep_dataset.py and add an entry to the dset_data dictionary for your new dataset, with the 'raw_file' element pointing to your file of SMILES strings. The dictionary will initially look like this:

dset_data = dict(
    enamine_kinase = dict(
        raw_file = '/usr/workspace/atom/enamine/Enamine_Kinase.txt'
    ),
    neurocrine = dict(
        raw_file = '/usr/workspace/atom/neurocrine/vae_training_data/neurocrine_vae_smiles.txt'
    ),
    test = dict(
        raw_file = '/usr/workspace/atom/enamine/test.txt'
    )
)
Then run the script you just edited:

./prep_dataset.py --dset_name <dataset_name> --test_size <test_set_size>

This script shuffles the list of SMILES strings, then splits it into training and test sets; the --test_size argument controls how many go into the test set. It generates the vocabulary file and converts the training data into "tensorized" one-hot form. All the generated files go into a new directory ...icml18_jtnn/data/<dataset_name>, where dataset_name is the key you assigned in the dset_data dictionary.



2. Run a training hyperparam search. You can copy neurocrine_train_1.sh and modify it to change the directory paths to your own directories; let's call your version "my_train.sh". This is an sbatch script that calls jtnn_hyperparam.py, which in turn launches a set of training batch jobs with randomly varying hyperparameters; the --n_models argument specifies the number of jobs, and therefore the number of models you'll generate. Change the settings in my_train.sh as follows:

DATA_DIR should point to the directory you created under icml18_jtnn/data when you ran the prep_dataset script.
--workers specifies the number of threads per job; set it to the number of cores per node on the machine you are running on.
--save_dir specifies the parent directory under which all the model files get saved.
--train_split points to the 'processed' subdirectory of your new dataset directory.
--vocab points to the vocabulary file.
Also, create the --save_dir directory, and also the directory where the slurm output gets written. Run your training script:

bash ./my_train.sh

If the script failed with a dynamic linked library error, such as 'ImportError: /lib64/libstdc++.so.6: version "CXXABI_1.3.8" not found', try to clean your `$HOME/.local` directory, re-installed all AMPL/GMD packages, and run it again. If that doesn't help, try to remove and re-install the python environment. 
After the jobs are all finished, you should get a bunch of directories under `--save_dir`, with names as model0, each one contains a few model epoch files.


3. Run a test script to evaluate the models you trained. You can copy neurocrine_test_1.sh and modify the directory paths and parameters as needed; let's call your version "my_test.sh". Note that, during training, a checkpoint was saved after each epoch. The module called by your test script, test_encoder.py, runs the models from each checkpoint, encoding and then decoding the SMILES strings in your training and test sets. For each checkpoint, the script tabulates the number of decoded SMILES strings that exactly match the input and the average Tanimoto distance between the input and output strings. The test code runs the recovery tests for each hyperparameter set in a separate batch job; to throttle the number of batch jobs running at any one time, set the --max_jobs parameter in the script. To get the detailed test results - i.e. the input and output SMILES for each checkpoint - make sure the --show_recovery argument is set. When you are done editing the script, run it:



bash ./my_test.sh



4. After all the testing jobs are finished, you end up with a separate table of results for each "model" (i.e., each hyperparam set). To combine all the outputs into one table, so you can quickly sort it and find the best model, copy the script neurocrine_combine_1.sh, edit it and run it; for example:

bash ./my_combine.sh

The following parameters will need to be edited in the script:

--vae_path should point to the directory with the trained VAE models.
--output_file will be the output CSV file containing the combined VAE model performance stats.
5. Examine the output CSV file and identify the best VAE model - i.e. the best hyperparameter set and epoch. Copy the model file to the subdirectory of your glo_spl examples directory where you'll be putting the config files for your GMD runs.

