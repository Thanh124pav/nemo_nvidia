export TRITON_CACHE_DIR=/workspace/.triton_cache


export TORCHINDUCTOR_DISABLE=1
export USER=dung
export LOGNAME=dung
export TORCHINDUCTOR_CACHE_DIR=/workspace/.torchinductor


export HF_HOME=/workspace/.hf
export HF_DATASETS_CACHE=/workspace/.hf/datasets
export TRANSFORMERS_CACHE=/workspace/.hf/transformers
export XDG_CACHE_HOME=/workspace/.cache





FLASHINFER_WORKSPACE_BASE=/workspace/flashinfer_workspace NEMORUN_HOME=/workspace/nemo_restore/nemo_run CUDA_VISIBLE_DEVICES=0 python distill.py \
    --teacher-model-path finetuning_nemo2_info/qwen3_14b_lora_merge \
    --student-model-path models/qwen3/Qwen3-0.6B \
    --distillation-config-path models/conf/distillation_config.yaml \
    --dir-name distillation_info \
    --name-recipe Qwen3-0.6B_Qwen3-14B_maxL-4096 \
    --data-path datasets/raw_data/finetune/processed/all_dataset_merged \
    --dataset-root datasets/finetuned_datasets/all_merged_qwen3_maxL-4096 \
    --tokenizer-path /workspace/Qwen3-14B-Base \
    --seq-length 4096 \
    --gpus-per-node 1 \
    --tensor-parallel 1 


