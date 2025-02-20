python -u run.py \
    --task_name pretrain \
    --root_path ./datasets/electricity/ \
    --data_path electricity.csv \
    --model_id Electricity \
    --model TimeDART \
    --data Electricity \
    --features M \
    --input_len 336 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --patch_len 8 \
    --stride 8 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.95 \
    --learning_rate 0.0001 \
    --batch_size 16 \
    --train_epochs 50 \
    --gpu 6
