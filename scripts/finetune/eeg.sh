python -u run.py \
    --task_name finetune \
    --downstream_task classification \
    --root_path datasets/eeg_no_big/ \
    --model_id EEG \
    --model TimeDART \
    --data EEG \
    --e_layers 2 \
    --d_layers 1 \
    --input_len 3000 \
    --enc_in 2 \
    --dec_in 2 \
    --c_out 2 \
    --num_classes 8 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --patch_len 75 \
    --stride 75 \
    --head_dropout 0.2 \
    --dropout 0.2 \
    --time_steps 1000 \
    --scheduler cosine \
    --batch_size 128 \
    --gpu 0 \
    --lr_decay 0.99 \
    --lradj decay \
    --scheduler cosine \
    --patience 100 \
    --learning_rate 0.001 \
    --pct_start 0.3 \
    --train_epochs 100
