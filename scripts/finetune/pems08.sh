for pred_len in 12 24; do
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/PEMS/ \
        --data_path PEMS08.npz \
        --model_id PEMS08 \
        --model TimeDART \
        --data PEMS \
        --features M \
        --input_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 170 \
        --dec_in 170 \
        --c_out 170 \
        --n_heads 8 \
        --d_model 128 \
        --d_ff 256 \
        --patch_len 1 \
        --stride 1 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size 32 \
        --gpu 0 \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0001 \
        --pct_start 0.3 \
        --use_norm 1
done


for pred_len in 36 48; do
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/PEMS/ \
        --data_path PEMS08.npz \
        --model_id PEMS08 \
        --model TimeDART \
        --data PEMS \
        --features M \
        --input_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 170 \
        --dec_in 170 \
        --c_out 170 \
        --n_heads 8 \
        --d_model 128 \
        --d_ff 256 \
        --patch_len 1 \
        --stride 1 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size 32 \
        --gpu 0 \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate 0.0001 \
        --pct_start 0.3 \
        --use_norm 0
done
