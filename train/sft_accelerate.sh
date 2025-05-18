export WANDB_PROJECT=sft_kk
export GPUS=$(($(nvidia-smi --list-gpus | wc -l)))
nohup accelerate launch --config_file zero3.yaml --num_processes=$GPUS sft_trainer.py --config sftconfig.yaml  > training_log.out 2>&1 &
