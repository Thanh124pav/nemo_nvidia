#!/bin/bash
set +e  # cho phép tiếp tục chạy khi có lỗi


# Lora finetuning 
MASTER_PORT=31000

CUDA_VISIBLE_DEVICES=1 NUMEXPR_MAX_THREADS=64 torchrun --nproc_per_node=1 --master_port=$MASTER_PORT finetune_nemo1.py --scheme lora  --target-modules mlp --adapter-initialized xavier --router-initialized True --data-path-ft finetuned_datasets/vietnews --model-path-ft models/llama/llama3_1b.nemo --log-dir finetuning_lora_dense_vietnews --output-path-ft models/finetuned_dense/llama3_1b_lora_mlp_vietnews.nemo
# CUDA_VISIBLE_DEVICES=1,2 python clear_cache.py


CUDA_VISIBLE_DEVICES=1 NUMEXPR_MAX_THREADS=64 torchrun --nproc_per_node=1 --master_port=$MASTER_PORT  finetune_nemo1.py --scheme lora --target-modules mlp  --adapter-initialized xavier --router-initialized True --data-path-ft finetuned_datasets/xlsum --model-path-ft models/llama/llama3_1b.nemo --log-dir finetuning_lora_dense_xlsum --output-path-ft models/finetuned_dense/llama3_1b_lora_mlp_xlsum.nemo
# CUDA_VISIBLE_DEVICES=1,2 python clear_cache.py 



CUDA_VISIBLE_DEVICES=1 NUMEXPR_MAX_THREADS=64 torchrun --nproc_per_node=1 --master_port=$MASTER_PORT  finetune_nemo1.py --scheme lora --target-modules mlp --adapter-initialized xavier --router-initialized True --data-path-ft finetuned_datasets/wikilingual --model-path-ft models/llama/llama3_1b.nemo --log-dir finetuning_lora_dense_wikilingual --output-path-ft models/finetuned_dense/llama3_1b_lora_mlp_wikilingual.nemo