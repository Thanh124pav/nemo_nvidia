export TRITON_CACHE_DIR=/workspace/.triton_cache


export TORCHINDUCTOR_DISABLE=1
export USER=dung
export LOGNAME=dung
export TORCHINDUCTOR_CACHE_DIR=/workspace/.torchinductor


export HF_HOME=/workspace/.hf
export HF_DATASETS_CACHE=/workspace/.hf/datasets
export TRANSFORMERS_CACHE=/workspace/.hf/transformers
export XDG_CACHE_HOME=/workspace/.cache

# Convert HF to Nemo1 - DONE 
#python scripts/checkpoint_converters/convert_llama_hf_to_nemo.py  \
#    --input_name_or_path /workspace/Llama-3.2-1B  \
#    --output_path models/llama/llama3_1b_instruct.nemo

#python scripts/checkpoint_converters/convert_llama_hf_to_nemo.py \
# --input_name_or_path /workspace/Meta-Llama-3-8B \
# --output_path models/llama/llama3_8b_instruct.nemo \
# --precision bf16 \
# --llama31 False
 
#Conver Nemo1 to Nemo2 - DONE
#python scripts/checkpoint_converters/convert_nemo1_to_nemo2.py  \
#--input_path=models/llama/llama3_1b.nemo   \
#--output_path=models/llama/llama3_1b_nemo2  \
#--model_id=meta-llama/Meta-Llama-3.2-1B --tokenizer_path=/workspace/Llama-3.2-1B

#python scripts/checkpoint_converters/convert_nemo1_to_nemo2.py  \
#--input_path=models/llama/llama3_1b.nemo   \
#--output_path=models/llama/llama3_1b_nemo2  \
#--model_id=meta-llama/Meta-Llama-3.2-1B --tokenizer_path=/workspace/Llama-3.2-1B


#CUDA_VISIBLE_DEVICES=0,1 python scripts/checkpoint_converters/convert_nemo1_to_nemo2.py  \
#--input_path=models/llama/llama3_8b.nemo   \
#--output_path=models/llama/llama3_8b_nemo2  \
#--model_id=meta-llama/Meta-Llama-3-8B --tokenizer_path=/workspace/Meta-Llama-3-8B


#LoRA finetuning 
#specific layer - NOT DONE
#NEMORUN_HOME=/workspace/nemo_restore/nemo_run CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 finetune.py --model-id llama3_8b --dir-name finetuning_nemo2_info --name-recipe llama3_8b --model-path models/llama/llama3_8b_nemo2 --data-path datasets/raw_data/finetune/processed/all_data --dataset-root datasets/finetuned_datasets/all_clean_llama3_8b --tokenizer-path /workspace/Meta-Llama-3-8B --layers 31 --peft-scheme lora --rank 4



#all layers - DONE
#CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 finetune.py --model-id llama3_8b --dir-name finetuning_nemo2_info --name-recipe llama3_8b --model-path models/llama/llama3_8b_nemo2 --data-path datasets/raw_data/finetune/processed/all_data --dataset-root datasets/finetuned_datasets/all_clean_llama3_8b --tokenizer-path /workspace/Meta-Llama-3-8B --peft-scheme lora --rank 4

#Full finetuning - DONE
#NEMORUN_HOME=/workspace/nemo_restore/nemo_run CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 finetune.py \
#    --model-id llama3_8b \
#    --dir-name finetuning_nemo2_info \
#    --name-recipe llama3_8b_sft_17h20_globalBatch-8_seqLength-3072 \
#    --model-path models/llama/llama3_8b_nemo2 \
#    --global-batch-size 8 \
#    --seq-length 3072 \
#    --max-steps 100000 \
#    --data-path datasets/raw_data/finetune/processed/all_data \
#    --dataset-root datasets/finetuned_datasets/all_clean_llama3_8b \
#    --tokenizer-path /workspace/Meta-Llama-3-8B 
# Full finetuning Qwen3-14B
NEMORUN_HOME=/workspace/nemo_restore/nemo_run CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 finetune.py \
    --model-id qwen3_14b \
    --dir-name finetuning_nemo2_info \
    --name-recipe qwen3_14b_lora_120525-16h_globalBatch-8_seqLength-8192 \
    --model-path models/qwen3/Qwen3-14B \
    --global-batch-size 8 \
    --seq-length 8192 \
    --max-steps 50000 \
    --data-path datasets/raw_data/finetune/processed/all_dataset_merged \
    --dataset-root datasets/finetuned_datasets/merged_datset_Qwen3-14B_seqLength-8192 \
    --tokenizer-path /workspace/Qwen3-14B-Base \
    --peft-scheme lora \
    --rank 32 \
    --alpha 32

#NEMORUN_HOME=/workspace/nemo_restore/nemo_run CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 finetune.py \
#    --model-id qwen3_8b \
#    --dir-name finetuning_nemo2_info \
#    --name-recipe qwen3_8b_lora_120825-11h30_globalBatch-8_seqLength-16384 \
#    --model-path models/qwen3/Qwen3-8B \
#    --global-batch-size 8 \
#    --seq-length 16384 \
#    --max-steps 50000 \
#    --data-path datasets/raw_data/finetune/processed/all_dataset_merged \
#    --dataset-root datasets/finetuned_datasets/merged_datset_Qwen3-8B_seqLength-16384 \
#    --tokenizer-path /workspace/Qwen3-8B-Base \
#    --peft-scheme lora \
#    --rank 32 \
#    --alpha 32  

 
#Pretraining - DONE
#CUDA_VISIBLE_DEVICES=0,1 python pretrain.py \
#    --model-id llama32_1b \
#    --dir-name pretraining_nemo2_info \
#    --name-recipe llama32_1b \
#    --data-path datasets/pretrain_datasets/dantri_content_document/dantri_content_document datasets/pretrain_datasets/ocr_summary_document/ocr_summary_document \
#    --model-path models/llama/llama3_1b_nemo2 \
#    --tokenizer-path /workspace/Llama-3.2-1B-Instruct

