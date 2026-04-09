export FLASHINFER_WORKSPACE_BASE=/workspace/data-shared/nlp/dungdx4/flashinfer_workspace 
export TRITON_CACHE_DIR=/workspace/.triton_cache


export TORCHINDUCTOR_DISABLE=1
export USER=dung
export LOGNAME=dung
export TORCHINDUCTOR_CACHE_DIR=/workspace/.torchinductor


export HF_HOME=/workspace/.hf
export HF_DATASETS_CACHE=/workspace/.hf/datasets
export TRANSFORMERS_CACHE=/workspace/.hf/transformers
export XDG_CACHE_HOME=/workspace/.cache

NEMORUN_HOME=/workspace/data-shared/nlp/dungdx4/nemo_restore/nemo_run
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node=4 finetune.py \
    --model-id qwen3_14b \
    --dir-name finetuning_nemo2_info \
    --name-recipe qwen3_14b_012926-17h_globalBatch-8_seqLength-16384 \
    --model-path models/qwen3/Qwen3-14B \
    --global-batch-size 8 \
    --seq-length 16384 \
    --max-steps 56000 \
    --data-path datasets/raw_data/finetune/all_datasets_stratified/16384_aggregated \
    --dataset-root datasets/finetuned_datasets/stratified_dataset_Qwen3-14B_seqLength-16384 \
    --tokenizer-path models/qwen3/Qwen3-14B/context/nemo_tokenizer \
    --tensor-parallel 4 \
    --gpus-per-node 4


# CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 finetune.py \
#     --model-id qwen3_14b \
#     --dir-name finetuning_nemo2_info \
#     --name-recipe qwen3_14b_lora_012626-14h_globalBatch-8_seqLength-8192 \
#     --model-path models/qwen3/Qwen3-14B \
#     --global-batch-size 8 \
#     --seq-length 8192 \
#     --max-steps 56000 \
#     --data-path datasets/raw_data/finetune/all_datasets_stratified/8192_aggregated \
#     --dataset-root datasets/finetuned_datasets/stratified_dataset_Qwen3-14B_seqLength-8192 \
#     --tokenizer-path /workspace/Qwen3-14B-Base \
#     --peft-scheme lora \
#     --rank 16 \
#     --alpha 16 \
#     --tensor-parallel 1

