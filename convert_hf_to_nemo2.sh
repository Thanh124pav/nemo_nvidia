export TRITON_CACHE_DIR=/workspace/.triton_cache


export TORCHINDUCTOR_DISABLE=1
export USER=dung
export LOGNAME=dung
export TORCHINDUCTOR_CACHE_DIR=/workspace/.torchinductor


export HF_HOME=/workspace/.hf
export HF_DATASETS_CACHE=/workspace/.hf/datasets
export TRANSFORMERS_CACHE=/workspace/.hf/transformers
export XDG_CACHE_HOME=/workspace/.cache

FLASHINFER_WORKSPACE_BASE=/workspace/flashinfer_workspace python convert_hf_to_nemo2.py \
    --model-id qwen3_4b \
    --src hf:///workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B-Base \
    --dst models/qwen3/Qwen3-4B-Base

#FLASHINFER_WORKSPACE_BASE=/workspace/flashinfer_workspace python convert_hf_to_nemo2.py \
#    --model-id qwen3_1.7b \
#    --src hf:///workspace/Qwen3-1.7B-Base \
#    --dst models/qwen3/Qwen3-1.7B
