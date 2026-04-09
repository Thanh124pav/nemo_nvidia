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
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc-per-node=4  --master_port=31000 finetune.py \
    --model-id qwen3_8b \
    --dir-name finetuning_nemo2_info \
    --name-recipe qwen3_8b_012926-19h_globalBatch-8_seqLength-16384 \
    --model-path models/qwen3/Qwen3-8B \
    --global-batch-size 8 \
    --seq-length 16384 \
    --max-steps 56000 \
    --data-path datasets/raw_data/finetune/all_datasets_stratified/16384_aggregated \
    --dataset-root datasets/finetuned_datasets/stratified_dataset_Qwen3-8B_seqLength-16384 \
    --tokenizer-path models/qwen3/Qwen3-8B/context/nemo_tokenizer \
    --tensor-parallel 4 \
    --gpus-per-node 4 


# NEMORUN_HOME=/workspace/nemo_restore/nemo_run
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 finetune.py \
#     --model-id qwen3_8b \
#     --dir-name finetuning_nemo2_info \
#     --name-recipe qwen3_8b_lora_012826-11h_globalBatch-8_seqLength-16384 \
#     --model-path models/qwen3/Qwen3-8B \
#     --global-batch-size 8 \
#     --seq-length 16384 \
#     --max-steps 56000 \
#     --data-path datasets/raw_data/finetune/all_datasets_stratified/16384_aggregated \
#     --dataset-root datasets/finetuned_datasets/stratified_dataset_Qwen3-8B_seqLength-16384 \
#     --tokenizer-path /workspace/Qwen3-8B \
#     --peft-scheme lora \
#     --rank 16 \
#     --alpha 16 \
#     --tensor-parallel 1

