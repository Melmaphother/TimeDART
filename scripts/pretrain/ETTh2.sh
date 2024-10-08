python -u run.py \
    --task_name pretrain \
    --dataset ETTh2 \
    --train_batch_size 16 \
    --val_batch_size 16 \
    --test_batch_size 16 \
    --input_len 336 \
    --num_features 7 \
    --position_encoding absolute \
    --embedding patch \
    --d_model 8 \
    --num_heads 8 \
    --feedforward_dim 32 \
    --dropout 0.2 \
    --num_layers_casual 2 \
    --patch_len 2 \
    --stride 2 \
    --time_steps 1000 \
    --scheduler cosine \
    --num_layers_denoising 1 \
    --head_dropout 0.1 \
    --num_epochs_pretrain 50 \
    --eval_per_epochs_pretrain 1 \
    --pretrain_lr 0.0005 \
    --pretrain_lr_decay 0.95 \
    --patience 10 \
    --device cuda:7