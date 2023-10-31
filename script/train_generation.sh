ulimit -c unlimited

export WANDB_WATCH=false
export PYTHONPATH=.

MODEL_NAME='facebook/opt-350m'
TASK='section'
CONTEXT='all'
NEIGHBOR_MODE='raw'
PEFT_TYPE='none'
DESCRIPTION=${MODEL_NAME}-${TASK}-${CONTEXT}

python language_modelling/run_generation.py \
    --dataset wikiweb2m \
    --model_name_or_path ${MODEL_NAME} \
    --task ${TASK} \
    --context ${CONTEXT} \
    --neighbor_mode ${NEIGHBOR_MODE} \
    --peft_type ${PEFT_TYPE} \
    --max_input_length 512 \
    --max_output_length 128 \
    --epochs 50 \
    --steps_per_epoch 10000 \
    --val_steps_per_epoch 400 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 2 \
    --per_device_val_batch_size 2 \
    --dataloader_num_workers 8 \
    --grad_accumulation_steps 16 \
    --fp16 \
    --wandb_project MMHG \
    --wandb_run ${DESCRIPTION}
