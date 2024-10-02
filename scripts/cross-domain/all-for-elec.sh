for pred_len in 96 192 336 720; do
    python -u run.py \
        --task_name finetune \
        --dataset electricity\
        --pretrain_dataset "electricity-ETTh1-ETTh2-ETTm1-ETTm2" \
        --train_batch_size 16 \
        --val_batch_size 16 \
        --test_batch_size 16 \
        --input_len 336 \
        --position_encoding absolute \
        --lr_adjust_method step \
        --d_model 128 \
        --num_heads 16 \
        --feedforward_dim 256 \
        --dropout 0.2 \
        --num_layers_casual 2 \
        --patch_len 8 \
        --stride 8 \
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
        --finetune_lr 0.0005 \
        --finetune_lr_decay 0.5 \
        --finetune_pct_start 0.2 \
        --pred_len $pred_len \
        --finetune_head_dropout 0.1 \
        --patience 3 \
        --use_tqdm \
        --device cuda:0
done