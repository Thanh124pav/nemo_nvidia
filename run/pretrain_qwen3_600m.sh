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
    --model-id qwen3_0.6b \
    --dir-name pretraining_nemo2_info \
    --name-recipe qwen3_600m_010726-15h \
    --dataset-root datasets/pretrain_datasets/qwen3/finewiki datasets/pretrain_datasets/qwen3/news \
    --model-path models/qwen3/Qwen3-0.6B-Base \
    --seq-length 16384 \
    --val-check-interval 10000 \
    --log-every-n-steps 5000 \
    --max-steps 50000 \
    --epochs 2 \
    --gpus-per-node 1 \
    --tensor-parallel 1 \
    --tokenizer-path /workspace/Qwen3-0.6B-Base 
