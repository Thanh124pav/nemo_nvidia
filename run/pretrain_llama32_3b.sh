export TRITON_CACHE_DIR=/workspace/.triton_cache


export TORCHINDUCTOR_DISABLE=1
export USER=dung
export LOGNAME=dung
export TORCHINDUCTOR_CACHE_DIR=/workspace/.torchinductor


export HF_HOME=/workspace/.hf
export HF_DATASETS_CACHE=/workspace/.hf/datasets
export TRANSFORMERS_CACHE=/workspace/.hf/transformers
export XDG_CACHE_HOME=/workspace/.cache
export FLASHINFER_WORKSPACE_BASE=/workspace/flashinfer_workspace


export NEMORUN_HOME=/workspace/nemo_restore/nemo_run

CUDA_VISIBLE_DEVICES=0 python pretrain.py \
    --model-id llama32_3b \
    --dir-name pretraining_nemo2_info \
    --name-recipe llama32_3b_010626-10h \
    --dataset-root datasets/pretrain_datasets/llama3/finewiki datasets/pretrain_datasets/llama3/news \
    --model-path models/llama/Llama-3.2-3B \
    --seq-length 8192 \
    --val-check-interval 3000 \
    --log-every-n-steps 1000 \
    --max-steps 50000 \
    --epochs 2 \
    --gpus-per-node 1 \
    --tensor-parallel 1 \
    --tokenizer-path /workspace/Llama-3.2-3B
