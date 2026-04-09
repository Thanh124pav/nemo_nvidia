export TRITON_CACHE_DIR=/workspace/.triton_cache


export TORCHINDUCTOR_DISABLE=1
export USER=dung
export LOGNAME=dung
export TORCHINDUCTOR_CACHE_DIR=/workspace/.torchinductor


export HF_HOME=/workspace/.hf
export HF_DATASETS_CACHE=/workspace/.hf/datasets
export TRANSFORMERS_CACHE=/workspace/.hf/transformers
export XDG_CACHE_HOME=/workspace/.cache

# Comment if finish tokenization
#mv nemo nemo_25.11 
#mv nemo_25.07 nemo
#python tokenize_dataset_for_pretrain.py \
#    --data-path /workspace/s3-data-tla-data/nlp/dungdx4/dungdx4_B200/raw_data/pretraining_stratified \
#    --dataset-root datasets/pretrain_datasets/qwen3/news \
#    --tokenizer-path models/qwen3/Qwen3-8B/context/nemo_tokenizer
#mv nemo nemo_25.07
#mv nemo_25.11 nemo
# 

export FLASHINFER_WORKSPACE_BASE=/workspace/data-shared/nlp/dungdx4/flashinfer_workspace 
export NEMORUN_HOME=/workspace/data-shared/nlp/dungdx4/nemo_restore/nemo_run
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python pretrain.py \
    --model-id qwen3_8b \
    --dir-name pretraining_nemo2_info \
    --name-recipe qwen3_8b_040826-11h30 \
    --dataset-root datasets/pretrain_datasets/qwen3/news \
    --model-path models/qwen3/Qwen3-8B \
    --seq-length 32768 \
    --val-check-interval 5000 \
    --log-every-n-steps 5000 \
    --max-steps 350000 \
    --gpus-per-node 8 \
    --tensor-parallel 8 \
    --tokenizer-path models/qwen3/Qwen3-8B/context/nemo_tokenizer
