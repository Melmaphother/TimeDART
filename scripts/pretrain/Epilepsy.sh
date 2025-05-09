python -u run.py \
    --task_name pretrain \
    --downstream_task classification \
    --root_path datasets/Epilepsy/ \
    --model_id Epilepsy \
    --model TimeDART \
    --data Epilepsy \
    --e_layers 2 \
    --d_layers 1 \
    --input_len 206 \
    --enc_in 3 \
    --dec_in 3 \
    --c_out 3 \
    --num_classes 4 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --patch_len 6 \
    --stride 6 \
    --head_dropout 0.1 \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --lr_decay 0.95 \
    --learning_rate 0.001 \
    --batch_size 16 \
    --train_epochs 50 \
    --gpu 0
