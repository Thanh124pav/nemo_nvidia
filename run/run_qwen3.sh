#!/bin/bash
set +e  # cho phép tiếp tục chạy khi có lỗi

# # Convert HF to nemo
# python scripts/checkpoint_converters/convert_qwen3_600m_hf_to_nemo.py \
#  --input_name_or_path /home/dungdx4/BERT/Qwen3_600m \
#  --output_path models/qwen3/qwen3_600m
 

# #Continual learning with dense model
# CUDA_VISIBLE_DEVICES=1,6 torchrun --nproc_per_node=2 --master_port=31000 pretrain_nemo1.py \
#  --config-file qwen3_600m_pretraining_config.yaml \
#  --restore-from-path models/qwen3/qwen3_600m.nemo \
#  --data-prefix datasets/pretrain_datasets/dantri_qwen3/dantri_qwen3_content_document \
#  --seq-length 2048 \
#  --max-steps 50000 \
#  --max-epochs 1 \
#  --micro-batch-size 1 \
#  --global-batch-size 4 \
#  --val-check-interval 5000 \
#  --limit-val-batches 500 \
#  --output-model-path models/pretrained_dense/qwen3_600m.nemo

# #Upcycling dense to moe model
# NUMEXPR_MAX_THREADS=64 python upcycle_dense_to_moe.py --model models/qwen3/qwen3_600m.nemo --num-experts 24 --moe-ffn-dim 512 --output-path models/pretrained_moe/qwen3_600m_e24g6sh0.nemo

# Continual learning after upcycling
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=31000 pretrain_nemo1.py     \
 --config-file qwen3_600m_pretraining_config.yaml     \
 --restore-from-path models/pretrained_moe/qwen3_600m_e24g6sh0.nemo     \
 --data-prefix datasets/pretrain_datasets/dantri_qwen3/dantri_qwen3_content_document datasets/pretrain_datasets/all_finetuning_qwen3/all_finetuning_qwen3_text_pretraining_document    \
 --seq-length 2048     \
 --max-steps 25000     \
 --max-epochs 1  \
 --moe   \
 --micro-batch-size 1     \
 --global-batch-size 8    \
 --val-check-interval 1000     \
 --limit-val-batches 500     \
 --output-model-path models/pretrained_moe/qwen3_600m_e24g6sh0_pt.nemo \
 --devices 1
# # Continual learning after upcycling
# CUDA_VISIBLE_DEVICES=1,6 torchrun --nproc_per_node=2 --master_port=31000 pretrain_nemo1.py     \
#  --config-file qwen3_600m_pretraining_config.yaml     \
#  --restore-from-path models/pretrained_moe/qwen3_600m_e24g6sh0_pt.nemo     \
#  --data-prefix datasets/pretrain_datasets/all_finetuning_qwen3/all_finetuning_qwen3_text_pretraining_document     \
#  --seq-length 2048     \
#  --max-steps 50000     \
#  --max-epochs 1  \
#  --moe   \
#  --micro-batch-size 1     \
#  --global-batch-size 8     \
#  --val-check-interval 5000     \
#  --limit-val-batches 500     \
#  --output-model-path models/pretrained_moe/qwen3_600m_e24g6sh0_pt_ptft.nemo \
#  --devices 2

# # # Continual learning after upcycling
# CUDA_VISIBLE_DEVICES=1,6 torchrun --nproc_per_node=2 --master_port=31000 pretrain_nemo1.py     \
#  --config-file qwen3_600m_pretraining_config.yaml     \
#  --restore-from-path models/pretrained_moe/qwen3_600m_e24g6sh0_ptft     \
#  --data-prefix datasets/pretrain_datasets/BKAINewsCorpus_qwen3/BKAINewsCorpus_qwen3_0to200k_text_document datasets/pretrain_datasets/BKAINewsCorpus_qwen3/BKAINewsCorpus_qwen3_200kto400k_text_document      \
#  --seq-length 2048     \
#  --max-steps 25000     \
#  --max-epochs 1  \
#  --moe   \
#  --micro-batch-size 1     \
#  --global-batch-size 4     \
#  --val-check-interval 5000     \
#  --limit-val-batches 500     \
#  --output-model-path models/pretrained_moe/qwen3_600m_e24g6sh0_ptftBK.nemo \
#  --devices 2

#Full finetuning 
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=31000 finetune_nemo1.py \
    --data-path-ft datasets/finetuned_datasets/language_modelling/BKAI \
    --seq-length 2048 \
    --model-path-ft models/pretrained_moe/qwen3_600m_e24g6sh0_ptft \
    --log-dir pretraining_info/qwen3_allBK \
    --output-path-ft models/finetuned_moe/qwen3_600m_e24g6sh0_ptftBK.nemo \
    --answer-only-loss False \
    --moe \
    --pretraining
    --moe-router-topk 12 \
    --router-initialized \
    --max-steps 25000 \
    --max-epochs 1 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --val-check-interval 5000 \
    --devices 2

#Full finetuning 
CUDA_VISIBLE_DEVICES=1,6 torchrun --nproc_per_node=2 --master_port=31000 finetune_nemo1.py \
    --data-path-ft datasets/finetuned_datasets/all_clean \
    --seq-length 2048 \
    --model-path-ft models/pretrained_moe/qwen3_600m_e24g6sh0_ptftBK.nemo \
    --log-dir finetuning_info/qwen3BK_all \
    --output-path-ft models/finetuned_moe/qwen3_600m_e24g6sh0_ptft_fsft.nemo \
    --answer-only-loss True \
    --moe \
    --moe-router-topk 6 \
    --router-initialized \
    --max-steps 15000 \
    --max-epochs 1 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --val-check-interval 5000 \
    --devices 2

# Generating answer 
CUDA_VISIBLE_DEVICES=1,6 torchrun --nproc_per_node=2  generate_nemo1.py  \
    --test-data-folder-path datasets/finetuned_datasets/vietnews_clean_demo  \
    --model-path models/finetuned_moe/qwen3_600m_e24g6sh0_ptft_fsft.nemo \
    --lora-model-path models/finetuned_moe/qwen3_600m_e24g6sh0_ptft_fsft.nemo \
    --seq-length 2048 \
    --dir-name generating_info/generate_qwen3_moe_vietnews\
    --name qwen3_moe_vietnews   \
    --test-data-name vietnews  \
    --global-batch-size 8 \
    --micro-batch-size 2 \
    --devices 2

CUDA_VISIBLE_DEVICES=1,6 torchrun --nproc_per_node=2  generate_nemo1.py  \
    --test-data-folder-path datasets/finetuned_datasets/wikilingual_clean_demo  \
    --model-path models/finetuned_moe/qwen3_600m_e24g6sh0_ptft_fsft.nemo \
    --lora-model-path models/finetuned_moe/qwen3_600m_e24g6sh0_ptft_fsft.nemo \
    --seq-length 2048 \
    --dir-name generating_info/generate_qwen3_moe_wikilingual\
    --name qwen3_moe_wikilingual   \
    --test-data-name wikilingual  \
    --global-batch-size 8 \
    --micro-batch-size 2 \
    --devices 2

CUDA_VISIBLE_DEVICES=1,6 torchrun --nproc_per_node=2  generate_nemo1.py  \
    --test-data-folder-path datasets/finetuned_datasets/xlsum_clean  \
    --model-path models/finetuned_moe/qwen3_600m_e24g6sh0_ptft_fsft.nemo \
    --lora-model-path models/finetuned_moe/qwen3_600m_e24g6sh0_ptft_fsft.nemo \
    --seq-length 2048 \
    --dir-name generating_info/generate_qwen3_moe_xlsum\
    --name qwen3_moe_xlsum   \
    --test-data-name xlsum  \
    --global-batch-size 8 \
    --micro-batch-size 2 \
    --devices 2

CUDA_VISIBLE_DEVICES=1,6 torchrun --nproc_per_node=2  generate_nemo1.py  \
    --test-data-folder-path datasets/finetuned_datasets/law_clean  \
    --model-path models/finetuned_moe/qwen3_600m_e24g6sh0_ptft_fsft.nemo \
    --lora-model-path models/finetuned_moe/qwen3_600m_e24g6sh0_ptft_fsft.nemo \
    --seq-length 2048 \
    --dir-name generating_info/generate_qwen3_moe_law\
    --name qwen3_moe_law   \
    --test-data-name law  \
    --global-batch-size 8 \
    --micro-batch-size 2 \
    --devices 2
