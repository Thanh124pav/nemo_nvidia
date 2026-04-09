export FLASHINFER_WORKSPACE_BASE=/workspace/data-shared/nlp/dungdx4/flashinfer_workspace 
# export TRITON_CACHE_DIR=/workspace/.triton_cache


# export TORCHINDUCTOR_DISABLE=1
# export USER=dung
# export LOGNAME=dung
# export TORCHINDUCTOR_CACHE_DIR=/workspace/.torchinductor


# export HF_HOME=/workspace/.hf
# export HF_DATASETS_CACHE=/workspace/.hf/datasets
# export TRANSFORMERS_CACHE=/workspace/.hf/transformers
# export XDG_CACHE_HOME=/workspace/.cache

NEMORUN_HOME=/workspace/data-shared/nlp/dungdx4/nemo_restore/nemo_run
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=8 --master_port=31000 finetune.py \
    --model-id gptoss_20b \
    --dir-name finetuning_nemo2_info \
    --name-recipe gptoss_20b_012926-18h_globalBatch-4_seqLength-3072 \
    --model-path models/gptoss \
    --global-batch-size 8 \
    --seq-length 1024 \
    --max-steps 56000 \
    --data-path datasets/raw_data/finetune/all_datasets_stratified/3072_aggregated \
    --dataset-root datasets/finetuned_datasets/stratified_dataset_GPTOSS-20B_seqLength-1024 \
    --tokenizer-path models/gptoss/context/nemo_tokenizer \
    --tensor-parallel 8 \
    --pipeline-model-parallel 1 \
    --gpus-per-node 8
