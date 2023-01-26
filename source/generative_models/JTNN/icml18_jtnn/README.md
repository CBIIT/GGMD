# Junction Tree Variational Autoencoder

Given a dataset in /path/to/data/smiles.txt, follow these steps in order to prepare the data and train the model. NOTE: the smiles data does not necessarily need to be a .txt or .csv, can be any extension as long as
it simply contains smiles strings seperated by \n and has no index column or header.

* 1: Extract the vocabulary:

    `$> python mol_tree.py < /path/to/data/smiles.txt > /path/to/data/vocab.txt`

    **NOTE**: one does not need to name the output as vocab.txt, can be something more specific or etc;

* 2.1: Preprocess the mol_tree data:

  `$> python preprocess.py -t /path/to/data/smiles.txt -n ${num_data_chunks} -j ${num_jobs}`

    where num_data_chunks determines the number of files the data will be split across and num_jobs is the number of python processes to use for preprocessing

* 2.2: Once preprocessing is done, one may either keep it in the cwd or move it to /path/some/preprocessed/data/ in which the files will be of the form tensors-0.pkl, tensors-1.pkl, ..., tensors-n-1.pkl.

    `$> mv tensors-*.pkl /path/to/some/preprocessed/data/`

    **NOTE**: For the remaining steps will assume that you have moved it to the /path/some/preprocessed/data/ directory.


* 3: Train the model:

    `$> python vae_train.py --train /path/to/some/preprocessed/data/ --vocab /path/to/data/vocab.txt --save_dir /path/to/checkpoints --epoch=10 --beta=0.0 --step_beta=0.001 --warmup=1000 --anneal_iter=1000 --kl_anneal_iter=20 --workers=16 --save_iter=1000 ...`


   ~~~~
   usage: vae_train.py [-h] --train TRAIN --vocab VOCAB --save_dir SAVE_DIR
                    [--load_epoch LOAD_EPOCH] [--workers WORKERS]
                    [--hidden_size HIDDEN_SIZE] [--batch_size BATCH_SIZE]
                    [--latent_size LATENT_SIZE] [--depthT DEPTHT]
                    [--depthG DEPTHG] [--lr LR] [--clip_norm CLIP_NORM]
                    [--beta BETA] [--step_beta STEP_BETA]
                    [--max_beta MAX_BETA] [--warmup WARMUP] [--epoch EPOCH]
                    [--anneal_rate ANNEAL_RATE] [--anneal_iter ANNEAL_ITER]
                    [--kl_anneal_iter KL_ANNEAL_ITER]
                    [--print_iter PRINT_ITER] [--save_iter SAVE_ITER]

        optional arguments:
        -h, --help            show this help message and exit
        --train TRAIN
        --vocab VOCAB
        --save_dir SAVE_DIR
        --load_epoch LOAD_EPOCH
        --workers WORKERS
        --hidden_size HIDDEN_SIZE
        --batch_size BATCH_SIZE
        --latent_size LATENT_SIZE
        --depthT DEPTHT
        --depthG DEPTHG
        --lr LR
        --clip_norm CLIP_NORM
        --beta BETA
        --step_beta STEP_BETA
        --max_beta MAX_BETA
        --warmup WARMUP
        --epoch EPOCH
        --anneal_rate ANNEAL_RATE
        --anneal_iter ANNEAL_ITER
        --kl_anneal_iter KL_ANNEAL_ITER
        --print_iter PRINT_ITER
        --save_iter SAVE_ITER

