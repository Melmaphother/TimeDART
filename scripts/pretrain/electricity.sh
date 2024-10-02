python -u run.py \
    --task_name pretrain \
    --dataset electricity \
    --train_batch_size 16 \
    --val_batch_size 16 \
    --test_batch_size 16 \
    --input_len 336 \
    --num_features 321 \
    --position_encoding absolute \
    --embedding patch \
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
    --patience 10 \
    --device cuda:0 \
    --use_tqdm