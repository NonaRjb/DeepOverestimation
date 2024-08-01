#!/usr/bin/env bash
#SBATCH --mem  5GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --constrain "eowyn|galadriel|arwen"
#SBATCH --mail-type FAIL
#SBATCH --mail-user nonar@kth.se
#SBATCH --output /Midgard/home/%u/Overfitting/logs/cluster_logs/%A_%a_slurm.out
#SBATCH --error  /Midgard/home/%u/Overfitting/logs/cluster_logs/%A_%a_slurm.err
#SBATCH --array=1-2%2

# Check job environment
echo "JOB:  ${SLURM_JOB_ID}"
echo "TASK: ${SLURM_ARRAY_TASK_ID}"
echo "HOST: $(hostname)"
echo ""
# Activate conda
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
if [ "${SLURMD_NODENAME}" == "galadriel" ]; then
  conda activate cuda_11
elif [ "${SLURMD_NODENAME}" == "eowyn" ]; then
  conda activate cuda_11
elif [ "${SLURMD_NODENAME}" == "arwen" ]; then
  conda activate cuda_11
else
  conda activate eegnet_pytorch
fi

# f_array=("2d_1.txt" "3d_1.txt" "3d_2.txt" "64d_4.txt" "64d_16.txt")
f_array=("256d_4.txt" "256d_16.txt")

fname=${f_array[`expr $((SLURM_ARRAY_TASK_ID-1)) % ${#f_array[@]}`]}

python train.py -b 8 --lr 0.0001 --epochs 100 -n 1000 -d 2 --cov_file "$fname" --seed 42