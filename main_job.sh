#!/bin/bash 
#SBATCH -p general ## Partition
#SBATCH -q public  ## QOS
#SBATCH -c 4       ## Number of Cores
#SBATCH -t 0-08:00:00   # 08 hour time in d-hh:mm:ss
#SBATCH --job-name=Cpu4_8h_gpu100-python
#SBATCH --gpus=a100:1
#SBATCH --output=slurm.%j.out  ## job /dev/stdout record (%j expands -> jobid)
#SBATCH --error=slurm.%j.err   ## job /dev/stderr record 
#SBATCH --export=NONE          ## keep environment clean
#SBATCH --mail-type=ALL        ## notify <asurite>@asu.edu for any job state change

echo "WHERE I AM FROM: $SLURM_SUBMIT_DIR"
echo "WHERE AM I NOW: $(pwd)"

echo "Loading python 3 from anaconda module"
module load mamba/latest
echo "Loading scientific computing python environment, scicomp"
source activate python3_6
echo "Running example python script"
export PYTHONPATH="/home/pbist/AlgoCompBio/Ergo_29Nov:$PYTHONPATH"
echo "Executing program"
python /home/pbist/AlgoCompBio/Ergo_1Dec/proj_ergo.py train cuda --train_data_path '/home/pbist/AlgoCompBio/data/BAP/tcr_split/train.csv' --test_data_path '/home/pbist/AlgoCompBio/data/BAP/tcr_split/test.csv' --roc_file 'roc_file' > output.txt 2>&1
echo "Executed program"
