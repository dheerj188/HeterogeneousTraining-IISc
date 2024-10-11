#!/bin/bash
set -x
# distplan in ["CAI_ZeRO1", "CAI_ZeRO2", "CAI_Gemini", "Pytorch_DDP", "Pytorch_ZeRO"]
export DISTPLAN=${DISTPLAN:-"CAI_Gemini"}

# The following options only valid when DISTPLAN="colossalai"
export GPUNUM=${GPUNUM:-1}
echo $GPUNUM
export BATCH_SIZE=${BATCH_SIZE:-32}
export MODEL_TYPE=${MODEL_TYPE:-"gpt2_40b"}
export TRAIN_STEP=${TRAIN_STEP:-16}
# export PYTHONPATH=$PWD:$PYTHONPATH

mkdir -p gemini_logs

mkdir -p logchrome9b

# nsys profile \
#    -o my_profile \
#    --trace cuda,nvtx,osrt,cublas,cusparse,cudnn,openmp \
# PYTHONPATH=$PYTHONPATH:/scratch/dheemanth/caigpt/.spack-env/view/x86_64/lib/shared-papi-pthread-python-cupti-pdt \
# LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/dheemanth/caigpt/.spack-env/view/x86_64/lib/shared-papi-pthread-python-cupti-pdt \
torchrun --standalone --nproc_per_node=4 /workspace/home/proj/24/cdsdhe/train_gpt_cai.py \
--model_type=${MODEL_TYPE} \
--batch_size=${BATCH_SIZE} \
--distplan=${DISTPLAN} \
--train_step=${TRAIN_STEP} \
2>&1 | tee ./gemini_logs/${MODEL_TYPE}_${DISTPLAN}_gpu_${4}_bs_${BATCH_SIZE}.log
