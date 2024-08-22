#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --constrain "eowyn|galadriel|arwen"
#SBATCH --mail-type FAIL
#SBATCH --mail-user nonar@kth.se
#SBATCH --output /Midgard/home/%u/Overfitting/logs/cluster_logs/%A_%a_slurm.out
#SBATCH --error  /Midgard/home/%u/Overfitting/logs/cluster_logs/%A_%a_slurm.err
#SBATCH --array=1-112%20

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

# d_array=(4 8 16 32 64 128 256 512)
h_array=(4 8 16 32 64 128 256)
l_array=(1 2 4 8 16 32 64 128)
op_array=("sgd" "adamw")

# num_d=${#d_array[@]}
num_h=${#h_array[@]}
num_l=${#l_array[@]}
num_o=${#op_array[@]}

idx=$SLURM_ARRAY_TASK_ID-1

# Determine the indices
# d_index=$((idx % num_d))
op_index=$((idx % num_o))
h_index=$(((idx / num_o) % num_h))
l_index=$(((idx / (num_o * num_h)) % num_l))

# Get the corresponding values
# d=${d_array[$d_index]}
op=${op_array[$op_index]}
h=${h_array[$h_index]}
l=${l_array[$l_index]}

# Print values (or use them in your script)
echo "Running job with op=$op, h=$h, l=$l"

python train_random.py -b 64 --lr 0.0001 --epochs 500 --hidden_size "$h" -l "$l" -n 1000 --n_test 5000  \
-d 64 -r 2 --optim "$op" --seed 42 --save_path "/Midgard/home/nonar/data/Overfitting/" --experiment N1000_ohl