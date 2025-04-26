#!/bin/bash
# FILENAME:  myjobsubmissionfile

#SBATCH -A ai240462-gpu       # allocation name
#SBATCH --nodes=2             # Total # of nodes 
#SBATCH --ntasks-per-node=4   # Number of MPI ranks per node (one rank per GPU)
#SBATCH --gpus-per-node=2     # Number of GPUs per node
#SBATCH --time=6:00:00        # Total run time limit (hh:mm:ss)
#SBATCH -J rajas_train_unet          # Job name
#SBATCH -o myjob.o%j          # Name of stdout output file
#SBATCH -e myjob.e%j          # Name of stderr error file
#SBATCH -p gpu                # Queue (partition) name
#SBATCH --qos=gpu

#SBATCH --mail-user=rachital@usc.edu
#SBATCH --mail-type=all       # Send email to above address at begin and end of job

# Manage processing environment, load compilers, and applications.
module purge
module load modtree/gpu
module load conda/2025.02
module list

# Activate the conda environment
conda activate denoise_pc
# Launch GPU code
python dpc_train.py --batch_size 8 --epochs 50 --warmup_epochs 10 --lr 3e-4 --run_name run_with_warmup_50eps_anvil