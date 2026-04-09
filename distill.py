import nemo_run as run
from nemo.collections import llm
from nemo import lightning as nl
import argparse
from dataModule import ThanhFinetuningDataModule
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from datetime import timedelta
from nemo.collections.llm.modelopt.recipes import distillation_recipe

'''Example
FLASHINFER_WORKSPACE_BASE=/workspace/flashinfer_workspace NEMORUN_HOME=/workspace/nemo_restore/nemo_run CUDA_VISIBLE_DEVICES=0,1 python distill.py \
    --teacher-model-path finetuning_nemo2_info/qwen3_14b_lora_merge \
    --student-model-path models/qwen3/Qwen3-4B \
    --dir-name distillation_info \
    --name-recipe Qwen3-4B_Qwen3-14B \
    --model-path models/qwen3/Qwen3-4B-distilled \
    --data-path datasets/raw_data/finetune/processed/all_dataset_merged \
    --dataset-root datasets/finetuned_datasets/all_merged_qwen3_4b_14b \
    --tokenizer-path /workspace/Qwen3-14B-Base \
    --seq-length 4096
'''


def get_args():
    parser = argparse.ArgumentParser(description="Code to distill models")
    

    parser.add_argument("--student-model-path", type=str, required=True,
                        help="student model")
    parser.add_argument("--teacher-model-path", type=str, required=True,
                        help="teacher model path")
    parser.add_argument("--distillation-config-path", type=str, required=True,
                        help="distillation config path") 
    
    parser.add_argument("--model-path", default="unknown", help="model path")
    parser.add_argument("--dir-name", type=str, required=True,
                        help="folder to save log and checkpoint from pretraining")
    parser.add_argument("--name-recipe", type=str, required=True,
                        help = "name of finetunign recipe")
    parser.add_argument("--val-check-interval", type=int, default=5000,
                        help="Validation interval")
    parser.add_argument("--log-every-n-steps", type=int, default=1000,
                         help="Logging interval")
    parser.add_argument("--limit-val-batches", type=int, default=500,
                        help="Limit validation batches")
    parser.add_argument("--max-steps", type=int, default=50000,
                        help="Max steps")
    parser.add_argument("--micro-batch-size", type=int, default=1,
                        help="Micro batch size")
    parser.add_argument("--global-batch-size", type=int, default=4,
                        help="Global batch size")
    
    parser.add_argument("--data-path", type=str, required=True,
                        help="data path for finetuning")
    parser.add_argument("--dataset-root", type=str, required=True,
                        help="dataset root for finetuning")
    parser.add_argument("--tokenizer-path", type=str, required=True, 
                        help="path to local hugginface model to get tokenizer")
    
    parser.add_argument("--gpus-per-node", type=int, default=2,
                        help="num nodes of GPU") 
    parser.add_argument("--seq-length", type=int, default=2048,
                        help="max seq length")
    parser.add_argument("--tensor-parallel", type=int, default=2,
                        help="tensor model parallel size")
    args = parser.parse_args()
    
    return args

def configure_distillation_recipe(args, nodes: int = 1, gpus_per_node: int = 2):
    dir_name = args.dir_name
    name_recipe = args.name_recipe
    model_path = args.model_path
    data_path = args.data_path
    tokenizer_path = args.tokenizer_path
    dataset_root = args.dataset_root
    
    recipe  = distillation_recipe(
        student_model_path = args.student_model_path,
        teacher_model_path = args.teacher_model_path, 
        distillation_config_path = args.distillation_config_path,
        dir=dir_name,
        name=name_recipe,
        num_nodes = 1,
        num_gpus_per_node = args.gpus_per_node,
    )
    #recipe.log.ckpt.train_time_interval=run.Config(timedelta,minutes = 13*60 + 30)

    recipe.trainer.max_steps = args.max_steps
    recipe.trainer.max_epochs = 1
    recipe.trainer.num_sanity_val_steps = 0
    recipe.trainer.devices = args.gpus_per_node

    # Async checkpointing doesn't work with PEFT
    recipe.trainer.strategy.ckpt_async_save = False
    # Need to set this to 1 since the default is 2
    recipe.trainer.strategy.context_parallel_size = 1
    recipe.trainer.val_check_interval = args.val_check_interval
    recipe.trainer.limit_val_batches = args.limit_val_batches
    recipe.trainer.limit_test_batches = args.limit_val_batches
    recipe.trainer.log_every_n_steps = args.log_every_n_steps
    # This is currently required for LoRA/PEFT
    recipe.trainer.strategy.ddp = "megatron"
    recipe.trainer.strategy.tensor_model_parallel_size = args.tensor_parallel
    recipe.trainer.strategy.sequence_parallel = False if args.tensor_parallel == 1 else True
#     recipe.trainer.strategy.ckpt_load_strictness = False
    print(f"Trainer config: {recipe.trainer}")
    with open("trainer_config.txt", "w") as f:
        f.write(str(recipe.trainer))
    #recipe.resume = run.Config(
    #    nl.AutoResume, 
    #    restore_config = run.Config(
    #        nl.RestoreConfig, 
    #        path = model_path
    #    ),
    #    resume_if_exists = True,
    #)
    
    
    
    
    modules = ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2']  # ['*.layers.0.*.linear_qkv', '*.layers.1.*.linear_qkv']
    target_modules = []

    recipe.data = run.Config(
        ThanhFinetuningDataModule, 
        local_path = data_path,
        dataset_root = dataset_root,
        tokenizer = run.Config(AutoTokenizer, tokenizer_path), 
        dataset_kwargs={
        "prompt_template": "Question: {input} Answer: {output}",  # default is "{input} {output}" (naive concatenation)
        "answer_only_loss": False,  # default is True (only calculate loss on answer/output)
        },
        seq_length = args.seq_length, 
        micro_batch_size = args.micro_batch_size, 
        global_batch_size = args.global_batch_size, 
        num_workers = 64,
    )
    
    return recipe


def local_executor_torchrun(nodes: int = 1, devices: int = 2) -> run.LocalExecutor:
    
    # Env vars for jobs are configured here
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
    }

    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

    return executor


def run_finetuning(args):
    finetune = configure_distillation_recipe(args,nodes=1, gpus_per_node=args.gpus_per_node)

    executor = local_executor_torchrun(nodes=finetune.trainer.num_nodes, devices=finetune.trainer.devices)

    with run.Experiment(f"{args.name_recipe}", base_dir="experiments") as exp:
        exp.add(finetune, executor=executor, name=f"{args.name_recipe}_nemo2")
        exp.run(sequential=True, tail_logs=True)  # This will run the tasks 

if __name__ == "__main__":
    args = get_args()
    run_finetuning(args)
    
