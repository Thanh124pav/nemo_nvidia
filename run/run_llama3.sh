#!/bin/bash
set +e  # cho phép tiếp tục chạy khi có lỗi

# Convert HF to Nemo
python scripts/checkpoint_converters/convert_llama_hf_to_nemo.py  --input_name_or_path /home/dungdx4/BERT/llama3_1b_instruct  --output_path models/llama/llama3_1b_instruct.nemo

# LoRA finetuning for summarization
CUDA_VISIBLE_DEVICES=1,6 torchrun --nproc_per_node=2 --master_port=29500 finetune_nemo1.py \
    --data-path-ft datasets/finetuned_datasets/all_clean \
    --model-path-ft models/llama/llama3_1b_instruct \
    --log-dir pretraining_info/llama3_1b_instruct_all \
    --output-path-ft models/finetuned_dense/llama3_1b_instruct_lora.nemo \
    --scheme lora \
    --seq-length 2048 \
    --target-modules attention mlp \
    --alpha-r-ratio 1.2 \
    --answer-only-loss True \
    --log-dir finetuning_info/llama3_1b_instruct_all \
    --max-steps 40000 \
    --max-epochs 1 \
    --micro-batch-size 1 \
    --global-batch-size 4 \
    --val-check-interval 5000 \
    --devices 2

# Upcycle dense to moe 
NUMEXPR_MAX_THREADS=64 python upcycle_dense_to_moe.py --model models/pretrained_dense/llama3_1b_3584.nemo --num-experts 32 --moe-ffn-dim 512 --output-path models/pretrained_moe/llama3_1b_pt_e32g16sh0.nemo
CUDA_VISIBLE_DEVICES=1,2 python clear_cache.py


# Continual learning after upcycling 
CUDA_VISIBLE_DEVICES=1,6 torchrun --nproc_per_node=2 --master_port=31000 pretrain_nemo1.py     \
 --config-file megatron_llama_pretraining_config.yaml     \
 --restore-from-path models/pretrained_moe/llama3_1b_pt_ft_e32g16sh0.nemo     \
 --data-prefix datasets/pretrain_datasets/dantri_content_document/dantri_content_document     \
 --seq-length 2048     \
 --max-steps 100000     \
 --max-epochs 1  \
 --moe   \
 --micro-batch-size 1     \
 --global-batch-size 2     \
 --val-check-interval 5000     \
 --limit-val-batches 500     \
 --output-model-path models/pretrained_moe/llama3_1b_pt_ft_e32g16sh0_pt.nemo \
 --devices 2
# or
CUDA_VISIBLE_DEVICES=1,6 torchrun --nproc_per_node=2 --master_port=31000 pretrain_nemo1.py     \
 --config-file megatron_llama_pretraining_config.yaml     \
 --restore-from-path models/llama_moe/llama3_1b_e32g16sh0.nemo     \
 --data-prefix datasets/pretrain_datasets/dantri_content_document/dantri_content_document     \
 --seq-length 2048     \
 --max-steps 100000     \
 --max-epochs 1  \
 --moe   \
 --micro-batch-size 1     \
 --global-batch-size 2     \
 --val-check-interval 5000     \
 --limit-val-batches 500     \
 --output-model-path models/pretrained_moe/llama3_1b_pt_ft_e32g16sh0_pt.nemo \
 --devices 2

# #Full finetuning for language modelling (if not use continual learning after upcycling)
# CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=31000 finetune_nemo1.py \
#     --data-path-ft datasets/finetuned_datasets/language_modelling/dantri \
#     --model-path-ft models/pretrained_moe/llama3_1b_pt_ft_e32g16sh0_pt.nemo \
#     --log-dir pretraining_info/llama3_all \
#     --output-path-ft models/finetuned_moe/llama3_1b_pt_ft_e32g16sh0_pt_fsft.nemo \
#     --answer-only-loss False \
#     --max-steps 40000 \
#     --max-epochs 1 \
#     --micro-batch-size 1 \
#     --global-batch-size 4 \
#     --val-check-interval 1000 \
#     --devices 1
    
#Full finetuning for summarization
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=31000 finetune_nemo1.py \
    --data-path-ft datasets/finetuned_datasets/all_clean \
    --model-path-ft models/pretrained_moe/llama3_1b_pt_ft_e32g16sh0_pt.nemo \
    --log-dir pretraining_info/llama3_all \
    --output-path-ft models/finetuned_moe/llama3_1b_pt_ft_e32g16sh0_pt_fsft.nemo \
    --answer-only-loss True \
    --max-steps 80000 \
    --max-epochs 1 \
    --micro-batch-size 1 \
    --global-batch-size 2 \
    --val-check-interval 1000 \
    --devices 1
    
#Generating 
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1  generate_nemo1.py \
  --test-data-folder-path datasets/finetuned_datasets/vietnews_clean  \
  --model-path models/finetuned_moe/llama3_1b_pt_ft_e32g16sh0_pt_fsft.nemo   \
  --dir-name generating_info/generate_llama3_moe_vietnews  \
  --name vietnews     \
  --test-data-name vietnews 