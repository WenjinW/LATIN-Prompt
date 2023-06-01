#!/bin/bash
# set -x
gpu_id=${1:-0}
model_name=${2:-"claude"}
dataset_name=${3:-"docvqa"}
prompt=${4:-"task_instruction_space"}  # plain or task_instruction_space or task_instruction or space
comment=${5:-""}

run_name=${model_name}__Prompt_${prompt}
if [ -n "${comment}" ]; then
    run_name=${run_name}__${comment}
fi
run_name=${run_name}__${dataset_name}

export CUDA_VISIBLE_DEVICES=${gpu_id}

if [ "${dataset_name}" = "docvqa" ]; then
    python examples/claude_docvqa.py \
        --model_name_or_path ${model_name} \
        --dataset_name ${dataset_name} \
        --output_dir "outputs" \
        --results_dir "results" \
        --wandb_project "Layout" \
        --run_name ${run_name} \
        --prompt ${prompt} \
        --per_device_eval_batch_size 1
elif [ "${dataset_name}" = "infographicvqa" ]; then
    python examples/claude_infographicvqa.py \
        --model_name_or_path ${model_name} \
        --dataset_name ${dataset_name} \
        --output_dir "outputs" \
        --results_dir "results" \
        --datas_dir ${DATAS_DIR} \
        --wandb_project "Layout" \
        --run_name ${run_name} \
        --prompt ${prompt} \
        --two_stage ${two_stage} \
        --per_device_eval_batch_size 1
elif [ "${dataset_name}" = "mpdocvqa" ]; then
    python examples/claude_mpdocvqa.py \
        --model_name_or_path ${model_name} \
        --dataset_name ${dataset_name} \
        --output_dir "outputs" \
        --results_dir "results" \
        --datas_dir ${DATAS_DIR} \
        --wandb_project "Layout" \
        --run_name ${run_name} \
        --prompt ${prompt} \
        --two_stage ${two_stage} \
        --per_device_eval_batch_size 1
else
    echo "wrong dataset: "${dataset_name}
fi
