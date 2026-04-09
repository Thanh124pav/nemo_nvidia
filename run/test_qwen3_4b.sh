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

# With 1 GPU 
NEMORUN_HOME=/workspace/nemo_restore/nemo_run CUDA_VISIBLE_DEVICES=0,1 python finetune.py \
    --model-id qwen3_4b \
    --dir-name finetuning_nemo2_info \
    --name-recipe qwen3_4b_lora_010526-11h_globalBatch-8_seqLength-16384 \
    --model-path pretraining_nemo2_info/qwen3_4b_123025-16h/checkpoints/model_name=0--val_loss=1.51-step=14985-consumed_samples=479552.0-last \
    --global-batch-size 8 \
    --seq-length 16384 \
    --max-steps 50000 \
    --data-path datasets/raw_data/finetune/processed/all_dataset_merged \
    --dataset-root datasets/finetuned_datasets/merged_datset_Qwen3-4B_seqLength-16384 \
    --tokenizer-path /workspace/Qwen3-4B-Base \
    --gpus-per-node 2 \
    --tensor-parallel 2 \

# With 2 GPU 
#NEMORUN_HOME=/workspace/nemo_restore/nemo_run CUDA_VISIBLE_DEVICES=0,1 python finetune.py \
#    --model-id qwen3_8b \
#    --dir-name finetuning_nemo2_info \
#    --name-recipe qwen3_8b_lora_121225-14h_globalBatch-8_seqLength-3072 \
#    --model-path models/qwen3/Qwen3-8B \
#    --global-batch-size 8 \
#    --seq-length 3072 \
#    --max-steps 50000 \
#    --data-path datasets/raw_data/finetune/processed/all_dataset_merged \
#    --dataset-root datasets/finetuned_datasets/merged_datset_Qwen3-8B_seqLength-3072 \
#    --tokenizer-path /workspace/Qwen3-8B-Base \

