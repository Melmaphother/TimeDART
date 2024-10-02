for pred_len in 96 192 336 720; do
    python -u run.py \
        --task_name finetune \
        --dataset ETTm1 \
        --pretrain_dataset ETTm1 \
        --train_batch_size 64 \
        --val_batch_size 64 \
        --test_batch_size 64 \
        --input_len 336 \
        --num_features 7 \
        --position_encoding absolute \
        --lr_adjust_method step \
        --embedding patch \
        --d_model 32 \
        --num_heads 8 \
        --feedforward_dim 64 \
        --dropout 0.2 \
        --num_layers_casual 2 \
        --patch_len 2 \
        --stride 2 \
        --time_steps 1000 \
        --scheduler cosine \
        --head_dropout 0.1 \
        --num_layers_denoising 1 \
        --num_epochs_pretrain 50 \
        --eval_per_epochs_pretrain 1 \
        --pretrain_lr 0.0001 \
        --pretrain_lr_decay 0.95 \
        --finetune_mode fine_all \
        --num_epochs_finetune 10 \
        --eval_per_epochs_finetune 1 \
        --finetune_lr 0.0001 \
        --finetune_lr_decay 0.8 \
        --finetune_weight_decay 0.0 \
        --finetune_pct_start 0.2 \
        --pred_len $pred_len \
        --finetune_head_dropout 0.0 \
        --patience 3 \
        --device cuda:0 \
        --use_tqdm
done