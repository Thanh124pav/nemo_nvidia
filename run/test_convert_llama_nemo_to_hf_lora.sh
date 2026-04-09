export TRITON_CACHE_DIR=/workspace/.triton_cache


export TORCHINDUCTOR_DISABLE=1
export USER=dung
export LOGNAME=dung
export TORCHINDUCTOR_CACHE_DIR=/workspace/.torchinductor


export HF_HOME=/workspace/.hf
export HF_DATASETS_CACHE=/workspace/.hf/datasets
export TRANSFORMERS_CACHE=/workspace/.hf/transformers
export XDG_CACHE_HOME=/workspace/.cache

FLASHINFER_WORKSPACE_BASE=/workspace/flashinfer_workspace NEMORUN_HOME=/workspace/nemo_restore/nemo_run/ python convert_llama_nemo_to_hf_lora.py
