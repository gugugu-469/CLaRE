export WANDB_DISABLED=True
now_time=$(command date +%m-%d-%H-%M-%S)
echo "now time ${now_time}"
cd ../
deepspeed --include localhost:0 --master_port=23091 src/train.py --deepspeed ./CLARE_scirpts/ds_config.json \
    --stage sft \
    --model_name_or_path xxxx \
    --do_train \
    --do_eval \
    --lora_rank 32 \
    --lora_alpha 64 \
    --dataset xxxx \
    --template xxxx \
    --finetuning_type lora \
    --lora_target query_key_value \
    --output_dir xxxx \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-4 \
    --num_train_epochs 4.0 \
    --val_size 0.1 \
    --plot_loss \
    --preprocessing_num_workers  48 \
    --bf16 \
    --cutoff_len 8100 \
    --ddp_timeout 180000 \
    --save_total_limit 5

