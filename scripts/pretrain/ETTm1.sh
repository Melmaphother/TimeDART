python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id ETTm1 \
    --model TimeDART \
    --data ETTm1 \
    --features M \
    --input_len 336 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --n_heads 8 \
    --d_model 32 \
    --d_ff 64 \
    --patch_len 2 \
    --stride 2 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.95 \
    --learning_rate 0.0001 \
    --batch_size 64 \
    --train_epochs 50 \
    --gpu 0
