import nemo_run as run
from nemo.collections import llm
from nemo import lightning as nl
import argparse
from dataModule import ThanhFinetuningDataModule
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from datetime import timedelta

'''Example
CUDA_VISIBLE_DEVICES=1,2 python finetune.py --model-id llama3_8b --dir-name finetuning_nemo2_info --name-recipe llama3_8b_instruct --model-path models/llama/llama3_8b_instruct --data-path datasets/raw_data/finetune/processed/all_data --dataset-root datasets/finetuned_datasets/all_clean_llama3_8b --tokenizer-path /home/dungdx4/BERT/Meta-Llama-3-8B-Instruct
'''

def get_args():
    # convert data
    parser = argparse.ArgumentParser(description="Code to finetune models")
    

    parser.add_argument("--model-id", type=str, required=True, choices = ["llama32_1b", "llama32_3b", "qwen3_4b", "llama3_8b","qwen3_8b", "qwen3_14b", "gptoss_20b"],
                        help="model ID")
    
    parser.add_argument("--dir-name", type=str, required=True,
                        help="folder to save log and checkpoint from pretraining")
    parser.add_argument("--name-recipe", type=str, required=True,
                        help = "name of finetunign recipe")
    parser.add_argument("--micro-batch-size", type=int, default=1,
                        help = "Micro batch size")
    parser.add_argument("--global-batch-size", type=int, default=8,
                        help = "Global batch size")
    parser.add_argument("--max-steps", type=int, default = 25000,
                        help = "Max steps")
    parser.add_argument("--max-epochs", type=int, default = 1,
                        help = "Max epochs")
    parser.add_argument("--val-check-interval", type=int, default=1000,
                        help="Validation interval")
    parser.add_argument("--log-every-n-steps", type=int, default=1000,
                         help="Logging interval")
    parser.add_argument("--limit-val-batches", type=int, default=500,
                        help="Limit validation batches")
    
    parser.add_argument("--model-path", type=str, required=True,
                        help="path to folder  of the finetuned model")
    parser.add_argument("--data-path", type=str, required=True,
                        help="data path for finetuning")
    parser.add_argument("--dataset-root", type=str, required=True,
                        help="dataset root for finetuning")
    parser.add_argument("--tokenizer-path", type=str, required=True, 
                        help="path to local hugginface model to get tokenizer")
    
    parser.add_argument("--peft-scheme", default=None, choices = [None, 'lora'],
                        help="Type of finetuning, None means full finetuning")
    parser.add_argument("--layers", type=int, nargs='+', default=None,
                        help="Idx layers to finetune")
    parser.add_argument("--rank", type=int, default=8, 
                        help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=8,
                        help="alpha coef in LoRA")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--warmup-steps", type=float, default=1000,
                        help="Warm up ratio")

    parser.add_argument("--gpus-per-node", type=int, default=1,
                        help="num nodes of GPU") 
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="tensor model parallel size")
    parser.add_argument("--pipeline-model-parallel", type=int, default=1,
                        help="pipeline model parallel size")
    parser.add_argument("--seq-length", type=int, default=2048,
                        help="max seq length")
    parser.add_argument("--default-root-dir", default = "/workspace/nemo_run",
                        help="directory to control experiments")
    args = parser.parse_args()
    
    return args

def configure_finetuning_recipe(args, nodes: int = 1, gpus_per_node: int = 2):
    dir_name = args.dir_name
    name_recipe = args.name_recipe
    model_path = args.model_path
    data_path = args.data_path
    tokenizer_path = args.tokenizer_path
    model_id = args.model_id
    dataset_root = args.dataset_root
    ## CHANGES WITH EACH MODEL
    if(model_id == "llama32_1b"  or model_id == "llama32_1b"):
        recipe = llm.llama32_1b.finetune_recipe(
            dir=dir_name,  # Path to store checkpoints
            name=name_recipe,
            num_nodes=nodes,
            num_gpus_per_node=gpus_per_node,
            peft_scheme = args.peft_scheme
        )
    elif model_id == "llama32_3b" or model_id == "llama32_3b_instruct":
        recipe = llm.llama32_3b.finetune_recipe(
            dir=dir_name,  # Path to store checkpoints
            name=name_recipe,
            num_nodes=nodes,
            num_gpus_per_node=gpus_per_node,
            peft_scheme = args.peft_scheme
        )
    elif model_id == "qwen3_4b":
        recipe = llm.qwen3_4b.finetune_recipe(
            dir = dir_name,
            name = name_recipe,
            num_nodes=nodes,
            num_gpus_per_node=gpus_per_node,
            peft_scheme=args.peft_scheme
        )
    elif model_id == "llama3_8b" or model_id == "llama3_8b_instruct":
        recipe = llm.llama3_8b.finetune_recipe(
            dir=dir_name,  # Path to store checkpoints
            name=name_recipe,
            num_nodes=nodes,
            num_gpus_per_node=gpus_per_node,
            peft_scheme = args.peft_scheme
        )
    elif model_id == "qwen3_8b":
        recipe = llm.qwen3_8b.finetune_recipe(
            dir=dir_name,
            name=name_recipe,
            num_nodes=nodes,
            num_gpus_per_node=gpus_per_node,
            peft_scheme=args.peft_scheme
        )
    elif model_id == "qwen3_14b":
        recipe = llm.qwen3_14b.finetune_recipe(
           dir=dir_name,
           name=name_recipe,
           num_nodes=nodes,
           num_gpus_per_node=gpus_per_node,
           peft_scheme = args.peft_scheme
        )
    elif model_id == "gptoss_20b":
        recipe = llm.gpt_oss_20b.finetune_recipe(
           dir=dir_name,
           name=name_recipe,
           num_nodes=nodes,
           num_gpus_per_node=gpus_per_node,
           peft_scheme=args.peft_scheme
        )
    else:
        raise ValueError(f"Model {model_id} is not supportted")
    
    #recipe.log.ckpt.train_time_interval=run.Config(timedelta,minutes = 13*60 + 30)
    recipe.optim.config.lr = args.lr                    # peak LR
    recipe.optim.lr_scheduler.warmup_steps = args.warmup_steps     # warmup steps
    recipe.optim.lr_scheduler.min_lr = 1e-6     
    recipe.trainer.max_steps = args.max_steps
    recipe.trainer.max_epochs = args.max_epochs
    recipe.trainer.num_sanity_val_steps = 0

    # Async checkpointing doesn't work with PEFT
    recipe.trainer.strategy.ckpt_async_save = False
    # Need to set this to 1 since the default is 2
    recipe.trainer.strategy.context_parallel_size = 1
    recipe.trainer.strategy.pipeline_model_parallel_size = args.pipeline_model_parallel
    recipe.trainer.strategy.sequence_parallel = True if 'gptoss' in  args.model_id else False
    recipe.trainer.val_check_interval = args.val_check_interval
    recipe.trainer.limit_val_batches = args.limit_val_batches
    recipe.trainer.limit_test_batches = args.limit_val_batches
    recipe.trainer.log_every_n_steps = args.log_every_n_steps
    # recipe.trainer.default_root_dir = args.default_root_dir
    # This is currently required for LoRA/PEFT
    recipe.trainer.strategy.ddp = "megatron"
    recipe.trainer.strategy.tensor_model_parallel_size = args.tensor_parallel
#     recipe.trainer.strategy.ckpt_load_strictness = False
    print(f"Trainer config: {recipe.trainer}")
    with open("trainer_config.txt", "w") as f:
        f.write(str(recipe.trainer))
    recipe.resume = run.Config(
        nl.AutoResume, 
        restore_config = run.Config(
            nl.RestoreConfig, 
            path = model_path
        ),
        resume_if_exists = True,
    )
    
    modules = ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2']  # ['*.layers.0.*.linear_qkv', '*.layers.1.*.linear_qkv']
    target_modules = []
    if args.layers is not None:
        print(f"fintuning {args.layers}-th layers")
        for layer in args.layers:
            fixed_modules = [f"*.layers.{layer}.*.{module}" for module in modules]
            target_modules.extend(fixed_modules)    
    else:
        target_modules = modules
        
    if args.peft_scheme is not None:
        recipe.peft.target_modules = target_modules
        print(f"Target modules for LoRA: {target_modules}")
        recipe.peft.dim = args.rank
        recipe.peft.alpha = args.alpha

    recipe.data = run.Config(
        ThanhFinetuningDataModule, 
        local_path = data_path,
        dataset_root = dataset_root,
        tokenizer = run.Config(AutoTokenizer, tokenizer_path), # "/home/dungdx4/BERT/Llama-3.2-1B/snapshots/model",
        dataset_kwargs={
        "prompt_template": "Question: {input} Answer: {output}",  # default is "{input} {output}" (naive concatenation)
        "answer_only_loss": True,  # default is True (only calculate loss on answer/output)
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
    finetune = configure_finetuning_recipe(args,nodes=1, gpus_per_node=args.gpus_per_node)
    # finetune.resume.restore_config.path = "/path/to/pretrained/NeMo-2/checkpoint"

    executor = local_executor_torchrun(nodes=finetune.trainer.num_nodes, devices=finetune.trainer.devices)
    ## CHANGES WITH EACH MODEL
    with run.Experiment(f"{args.name_recipe}", base_dir = "/workspace/nemo_run") as exp:
        exp.add(finetune, executor=executor, name=f"{args.name_recipe}_nemo2")
        exp.run(sequential=True, tail_logs=True)  # This will run the tasks sequentially and stream the logs


# Wrap the call in an if __name__ == "__main__": block to work with Python's multiprocessing module.
if __name__ == "__main__":
    args = get_args()
    run_finetuning(args)
    
    
'''Appendix
 target_modules (list[str], optional): A list of module names to apply LoRA to.
            Defaults to all linear layers ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2'].
                - 'linear_qkv': Apply LoRA to the fused linear layer used for query, key, and value projections
                                in self-attention.
                - 'linear_proj': Apply LoRA to the linear layer used for projecting the output of self-attention.
                - 'linear_fc1': Apply LoRA to the first fully-connected layer in MLP.
                - 'linear_fc2': Apply LoRA to the second fully-connected layer in MLP.
            Target modules can also contain wildcards. For example, you can specify
                target_modules=['*.layers.0.*.linear_qkv', '*.layers.1.*.linear_qkv'] to add LoRA to only linear_qkv
                on the first two layers.
'''
