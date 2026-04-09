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
# CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --master_port=31000 pretrain_nemo1.py     \
#  --config-file qwen3_600m_pretraining_config.yaml     \
#  --restore-from-path models/qwen3/qwen3_600m_e24g6sh0svd.nemo     \
#  --data-prefix datasets/pretrain_datasets/dantri_qwen3/dantri_qwen3_content_document datasets/pretrain_datasets/all_finetuning_qwen3/all_finetuning_qwen3_text_pretraining_document \
#  --log-dir pretraining_info/qwen3_return_vai \
#  --seq-length 2048     \
#  --max-steps 25000 \
#  --max-epochs 1  \
#  --moe   \
#  --moe-router-topk 12 \
#  --micro-batch-size 1     \
#  --global-batch-size 8    \
#  --val-check-interval 10000     \
#  --limit-val-batches 500     \
#  --output-model-path models/pretrained_moe/qwen3_600m_e24g6svd_ptvai.nemo \
#  --devices 2

CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --master_port=31000 pretrain_nemo1.py     \
 --config-file qwen3_600m_pretraining_config.yaml     \
 --restore-from-path models/pretrained_moe/qwen3_600m_e24g6svd_ptvai.nemo     \
 --data-prefix datasets/pretrain_datasets/BKAINewsCorpus_qwen3/BKAINewsCorpus_qwen3_0to200k_text_document \
 --log-dir pretraining_info/qwen3_return_vai \
 --seq-length 2048     \
 --max-epochs 1  \
 --max-steps 35000 \
 --moe   \
 --router-initialized \
 --moe-router-topk 12 \
 --micro-batch-size 1     \
 --global-batch-size 8    \
 --val-check-interval 10000     \
 --limit-val-batches 500     \
 --output-model-path models/pretrained_moe/qwen3_600m_e24g6svd_ptvaibk.nemo \
 --devices 2
 
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --master_port=31000 pretrain_nemo1.py     \
 --config-file qwen3_600m_pretraining_config.yaml     \
 --restore-from-path models/pretrained_moe/qwen3_600m_e24g6svd_ptvaibk.nemo     \
 --data-prefix datasets/pretrain_datasets/BKAINewsCorpus_qwen3/BKAINewsCorpus_qwen3_200kto400k_text_document  \
 --log-dir pretraining_info/qwen3_return_vai \
 --seq-length 2048     \
 --max-epochs 1  \
 --max-steps 45000 \
 --moe   \
 --router-initialized \
 --moe-router-topk 12 \
 --micro-batch-size 1     \
 --global-batch-size 8    \
 --val-check-interval 10000     \
 --limit-val-batches 500     \
 --output-model-path models/pretrained_moe/qwen3_600m_e24g6svd_ptvaibk.nemo \
 --devices 2

#Full finetuning 
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --master_port=31000 finetune_nemo1.py \
    --data-path-ft datasets/finetuned_datasets/all_clean \
    --seq-length 2048 \
    --model-path-ft models/pretrained_moe/qwen3_600m_e24g6svd_ptvaibk.nemo \
    --log-dir finetuning_info/qwen3_return \
    --output-path-ft models/finetuned_moe/qwen3_600m_e24g6svd_pt_fsft.nemo \
    --answer-only-loss False \
    --moe \
    --router-initialized \
    --max-steps 50000 \
    --moe-router-topk 4 \
    --max-epochs 1 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --val-check-interval 5000 \
    --devices 2

# Generating answer 
# Vietnews
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2  generate_nemo1.py  \
    --test-data-folder-path datasets/finetuned_datasets/vietnews_clean_demo  \
    --model-path models/finetuned_moe/qwen3_600m_e24g6svd_pt_fsft.nemo \
    --lora-model-path models/finetuned_moe/qwen3_600m_e24g6svd_pt_fsft.nemo \
    --seq-length 2048 \
    --dir-name generating_info/generate_qwen3_moe_upgrade_vietnews\
    --name qwen3_moe_upgrade_vietnews   \
    --test-data-name vietnews  \
    --micro-batch-size 1 \
    --global-batch-size 16 \
    --devices 2
    
python eval/peft_eval.py --pred_file test_results/qwen3_moe_upgrade_vietnews.jsonl --eval_path test_results/qwen3_moe_upgrade_vietnews.txt
    
# Wikilingual
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2  generate_nemo1.py  \
    --test-data-folder-path datasets/finetuned_datasets/wikilingual_clean_demo  \
    --model-path models/finetuned_moe/qwen3_600m_e24g6svd_pt_fsft.nemo \
    --lora-model-path models/finetuned_moe/qwen3_600m_e24g6svd_pt_fsft.nemo \
    --seq-length 2048 \
    --dir-name generating_info/generate_qwen3_moe_upgrade_wikilingual\
    --name qwen3_moe_upgrade_wikilingual   \
    --test-data-name wikilingual  \
    --micro-batch-size 1 \
    --global-batch-size 16 \
    --devices 2
    
python eval/peft_eval.py --pred_file test_results/qwen3_moe_upgrade_wikilingual.jsonl --eval_path test_results/qwen3_moe_upgrade_wikilingual.txt
# Xlsum
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2  generate_nemo1.py  \
    --test-data-folder-path datasets/finetuned_datasets/xlsum_clean_demo  \
    --model-path models/finetuned_moe/qwen3_600m_e24g6svd_pt_fsft.nemo \
    --lora-model-path models/finetuned_moe/qwen3_600m_e24g6svd_pt_fsft.nemo \
    --seq-length 2048 \
    --dir-name generating_info/generate_qwen3_moe_upgrade_xlsum\
    --name qwen3_moe_upgrade_xlsum   \
    --test-data-name xlsum  \
    --micro-batch-size 1 \
    --global-batch-size 16 \
    --devices 2

python eval/peft_eval.py --pred_file test_results/qwen3_moe_upgrade_xlsum.jsonl --eval_path test_results/qwen3_moe_upgrade_xlsum.txt
#Law
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2  generate_nemo1.py  \
    --test-data-folder-path datasets/finetuned_datasets/law_clean_demo  \
    --model-path models/finetuned_moe/qwen3_600m_e24g6svd_pt_fsft.nemo \
    --lora-model-path models/finetuned_moe/qwen3_600m_e24g6svd_pt_fsft.nemo \
    --seq-length 2048 \
    --dir-name generating_info/generate_qwen3_moe_upgrade_law\
    --name qwen3_moe_upgrade_law   \
    --test-data-name law  \
    --micro-batch-size 1 \
    --global-batch-size 16 \
    --devices 2

python eval/peft_eval.py --pred_file test_results/qwen3_moe_upgrade_law.jsonl --eval_path test_results/qwen3_moe_upgrade_law.txt
