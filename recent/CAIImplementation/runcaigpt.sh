#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=CaiTestProfile
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --nodelist=node[8-9]
module load conda 
module load cuda/11.8
module load gcc/9
module load mpi
cd /scratch/dheemanth/caigpt/

# source /home/dheemanth/spack/share/spack/setup-env.sh
# spack env activate .

source /scratch/dheemanth/CaiBase1/bin/activate

set -x
# distplan in ["CAI_ZeRO1", "CAI_ZeRO2", "CAI_Gemini", "Pytorch_DDP", "Pytorch_ZeRO"]
export DISTPLAN=${DISTPLAN:-"CAI_Gemini"}

# The following options only valid when DISTPLAN="colossalai"
export GPUNUM=${GPUNUM:-1}
echo $GPUNUM
export BATCH_SIZE=${BATCH_SIZE:-16}
export MODEL_TYPE=${MODEL_TYPE:-"gpt2_7b"}
export TRAIN_STEP=${TRAIN_STEP:-10}
# export PYTHONPATH=$PWD:$PYTHONPATH

mkdir -p gemini_logs

# nsys profile \
#    -o my_profile \
#    --trace cuda,nvtx,osrt,cublas,cusparse,cudnn,openmp \
# PYTHONPATH=$PYTHONPATH:/scratch/dheemanth/caigpt/.spack-env/view/x86_64/lib/shared-papi-pthread-python-cupti-pdt \
# LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/dheemanth/caigpt/.spack-env/view/x86_64/lib/shared-papi-pthread-python-cupti-pdt \
export TORCH_LOGS="+all"
export TORCHDYNAMO_VERBOSE=1
export TORCHDYNAMO_REPRO_AFTER="dynamo"
torchrun --standalone --nproc_per_node=4 train_gpt_cai.py \
--model_type=${MODEL_TYPE} \
--batch_size=${BATCH_SIZE} \
--distplan=${DISTPLAN} \
--train_step=${TRAIN_STEP} \
2>&1 | tee ./gemini_logs/${MODEL_TYPE}_${DISTPLAN}_gpu_${4}_bs_${BATCH_SIZE}.log