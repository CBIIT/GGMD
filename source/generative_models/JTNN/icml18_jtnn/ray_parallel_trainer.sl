#!/bin/bash

#SBATCH --job-name=jtvae_hyperopt
##SBATCH --cpus-per-task=5
##SBATCH --mem-per-cpu=1GB
#SBATCH --nodes=3
#SBATCH --tasks-per-node 1

LC_ALL=en_US
export LC_ALL


worker_num=$((${SLURM_JOB_NUM_NODES}-1)) # Must be one less that the total number of nodes
num_samples=$1

echo "using ${worker_num} workers for ${num_samples} trials..."

# module load Langs/Python/3.6.4 # This will vary depending on your environment
# source venv/bin/activate

source activate glo

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node1=${nodes_array[0]}

ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address) # Making address
suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)

export ip_head # Exporting for latter access by trainer.py

srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head --redis-port=6379 --redis-password=$redis_password & # Starting the head
sleep 5

for ((  i=1; i<=$worker_num; i++ ))
do
  node2=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address=$ip_head --redis-password=$redis_password & # Starting the workers
  sleep 5
done

# TODO: num_nodes should be world_size

python -u trainer.py --redis-password $redis_password --num-samples $num_samples # Pass the total number of allocated CPUs

