export TRITON_CACHE_DIR=/workspace/.triton_cache


export TORCHINDUCTOR_DISABLE=1
export USER=dung
export LOGNAME=dung
export TORCHINDUCTOR_CACHE_DIR=/workspace/.torchinductor


export HF_HOME=/workspace/.hf
export HF_DATASETS_CACHE=/workspace/.hf/datasets
export TRANSFORMERS_CACHE=/workspace/.hf/transformers
export XDG_CACHE_HOME=/workspace/.cache


export NEMORUN_HOME=/workspace/nemo_restore/nemo_run
CUDA_VISIBLE_DEVICES=2,3 python pretrain.py \
    --model-id qwen3_4b \
    --dir-name pretraining_nemo2_info \
    --name-recipe qwen3_4b_123025-16h \
    --dataset-root datasets/pretrain_datasets/qwen3/finewiki datasets/pretrain_datasets/qwen3/news \
    --model-path models/qwen3/Qwen3-4B \
    --seq-length 8192 \
    --val-check-interval 10000 \
    --log-every-n-steps 5000 \
    --max-steps 350000 \
    --tokenizer-path /workspace/Qwen3-4B-Base
