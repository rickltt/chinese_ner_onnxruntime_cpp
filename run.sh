CUDA_VISIBLE_DEVICE=0 python run.py 
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --evaluate_after_epoch \
    --per_gpu_train_batch_size 64 \
    --per_gpu_eval_batch_size 64 \
    --dropout_prob 0.1 \
    --max_seq_length 256 \
    --learning_rate 3e-5 \
    --weight_decay 5e-5 \
    --num_train_epochs 10 \
    --seed 42