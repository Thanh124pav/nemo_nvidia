export FLASHINFER_WORKSPACE_BASE=/workspace/flashinfer_workspace 

#Lora 
NEMORUN_HOME=/workspace/nemo_restore/nemo_run CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --model-id gptoss_20b \
    --dir-name finetuning_nemo2_info \
    --name-recipe gptoss_20b \
    --model-path models/gptoss \
    --data-path datasets/raw_data/finetune/processed/all_dataset_merged \
    --dataset-root datasets/finetuned_datasets/all_merged_gptoss_20b \
    --tokenizer-path /workspace/gpt-oss-20b \
    --tensor-parallel 1 \
    --gpus-per-node 1 \
    --peft-scheme lora \
    --rank 16 \
    --alpha 16 \
    --seq-length 8192
    
#Full finetuning
#CUDA_VISIBLE_DEVICES=0,1 python finetune.py \
#    --model-id gptoss_20b \
#    --dir-name finetuning_nemo2_info \
#    --name-recipe gptoss_20b \
#    --model-path models/gptoss/GPT-OSS-20B \
#    --data-path datasets/raw_data/finetune/processed/all_dataset_merged \
#    --dataset-root datasets/finetuned_datasets/all_merged_gptoss_120b \
#    --tokenizer-path /home/dungdx4/BERT/gpt-oss-safeguard-20b \
#    --seq-length 8192
