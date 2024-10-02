for pred_len in 96 192 336 720; do
    python -u run.py \
        --task_name finetune \
        --dataset ETTm2 \
        --pretrain_dataset ETTm2 \
        --train_batch_size 64 \
        --val_batch_size 64 \
        --test_batch_size 64 \
        --input_len 336 \
        --num_features 7 \
        --position_encoding absolute \
        --lr_adjust_method step \
        --d_model 8 \
        --num_heads 8 \
        --feedforward_dim 16 \
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
        --pretrain_lr 0.001 \
        --pretrain_lr_decay 0.8 \
        --finetune_mode fine_all \
        --num_epochs_finetune 10 \
        --eval_per_epochs_finetune 1 \
        --finetune_lr 0.0001 \
        --finetune_lr_decay 0.5 \
        --pred_len $pred_len \
        --finetune_head_dropout 0.1 \
        --patience 3 \
        --device cuda:7
done