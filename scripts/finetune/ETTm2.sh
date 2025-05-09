for pred_len in 96 192 336 720; do
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/ETT-small/ \
        --data_path ETTm2.csv \
        --model_id ETTm2 \
        --model TimeDART \
        --data ETTm2 \
        --features M \
        --input_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads 8 \
        --d_model 8 \
        --d_ff 16 \
        --patch_len 2 \
        --stride 2 \
        --dropout 0.2 \
        --head_dropout 0.0 \
        --batch_size 64 \
        --gpu 0 \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0001 \
        --pct_start 0.2
done