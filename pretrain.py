import nemo_run as run
from nemo.collections import llm
from nemo import lightning as nl
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
import argparse
from datetime import timedelta
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
import subprocess
'''Example
CUDA_VISIBLE_DEVICES=1,2 python pretrain.py \
    --model-id llama32_1b \
    --dir-name pretraining_nemo2_info \
    --name-recipe llama32_1b_122825 \
    --dataset-root datasets/pretrain_datasets/llama3/finewiki datasets/pretrain_datasets/llama3/news datasets/pretrain_datasets/llama3/BKAINews \
    --dataset-weights 0.05 0.08 0.87 \
    --model-path models/pretrained_dense/llama3_1b_3584_v1 \
    --tokenizer-path /home/dungdx4/BERT/Llama-3.2-1B
'''
def get_args():
    parser = argparse.ArgumentParser(description="Code to pretrain models")
    
    
    parser.add_argument("--model-id", type=str, required=True, choices = ["llama32_1b", "llama32_3b", "llama3_8b", "qwen3_0.6b", "qwen3_1.7b", "qwen3_4b", "qwen3_14b", "qwen3_8b", "gptoss_20b"],
                        help="model ID")
    parser.add_argument("--dir-name", type=str, required=True,
                        help="folder to save log and checkpoint from pretraining")
    parser.add_argument("--name-recipe", type=str, required=True,
                        help = "name of pretraining recipe")
    
    parser.add_argument("--max-steps", type=int, default=None,
                        help="max steps")
    parser.add_argument("--epochs", type=int, default=2,
                        help="number of epochs")
    parser.add_argument("--val-check-interval", type=int, default=5000,
                        help="Validation interval")
    parser.add_argument("--log-every-n-steps", type=int, default=1000,
                         help="Logging interval")
    parser.add_argument("--limit-val-batches", type=int, default=500,
                        help="Limit validation batches")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--warmup-steps", type=float, default=1000,
                        help="Warm up ratio")
    
    parser.add_argument("--model-path", type=str, required=True,
                        help="path to .nemo file of the pretrained model")
    parser.add_argument("--resume-from-path", type=str, default=None,
                        help="path to a specific checkpoint to resume from (weights only, no optimizer)")
    parser.add_argument("--no-load-optim", action="store_true", default=False,
                        help="skip loading optimizer state from checkpoint (use when optimizer format is incompatible)")
    parser.add_argument("--ckpt-legacy-format", action="store_true", default=False,
                        help="load checkpoint saved with old dp_zero_gather_scatter format (PyTorch < 2.6 / pre-mcore-0.14)")
    parser.add_argument("--data-path", type=str, nargs = "+",
                        help="path to raw data for preprocessing")
    parser.add_argument("--dataset-root", type=str, required=True, nargs = '+',
                        help="path to data for pretraining (preprocessed data, with .bin and .idx files)")
    parser.add_argument("--dataset-weights", type=float, default=None, nargs = '+',
                        help="weights of data proportion in pretraining")
    parser.add_argument("--tokenizer-path", type=str, required=True, 
                        help="path to local hugginface model to get tokenizer")
    parser.add_argument("--suffix", type=str, default='_text_document',
                        help="suffix of the pretrained dataset")
  
    parser.add_argument("--gpus-per-node", type=int, default=2,
                        help="num nodes of GPU") 
    parser.add_argument("--seq-length", type=int, default=4096,
                        help="max seq length")
    parser.add_argument("--accumulate-grad-batches", type=int, default=4,
                        help="accumulate grad batches")
    parser.add_argument("--tensor-parallel", type=int, default=2,
                        help="tensor model parallel size")


    args = parser.parse_args()
    
    return args
    
def configure_recipe(args, nodes: int = 1):
    dir_name = args.dir_name
    name_recipe = args.name_recipe
    model_path = args.model_path
    dataset_root = args.dataset_root
    tokenizer_path = args.tokenizer_path
    model_id = args.model_id
    gpus_per_node = args.gpus_per_node
    ## CHANGES WITH EACH MODEL
    if(model_id == "llama32_1b"  or model_id == "llama32_1b"):
        recipe = llm.llama32_1b.pretrain_recipe(
            dir= dir_name, # "/checkpoints/llama3" Path to store checkpoints
            name=name_recipe, # "llama3_pretraining",
            num_nodes=nodes,
            num_gpus_per_node=gpus_per_node,

        )
    elif model_id == "llama32_3b" or model_id == "llama32_3b_instruct":
        recipe = llm.llama32_3b.pretrain_recipe(
            dir= dir_name, # "/checkpoints/llama3" Path to store checkpoints
            name=name_recipe, # "llama3_pretraining",
            num_nodes=nodes,
            num_gpus_per_node=gpus_per_node,

        )
    elif model_id == "llama3_8b" or model_id == "llama3_8b_instruct":
        recipe = llm.llama3_8b.pretrain_recipe(
            dir= dir_name, # "/checkpoints/llama3" Path to store checkpoints
            name=name_recipe, # "llama3_pretraining",
            num_nodes=nodes,
            num_gpus_per_node=gpus_per_node,

        )
    elif model_id == "qwen3_0.6b":
        recipe = llm.qwen3_600m.pretrain_recipe(
            dir=dir_name,
            name=name_recipe,
            num_nodes=nodes,
            num_gpus_per_node=gpus_per_node
        )
    elif model_id == "qwen3_1.7b":
        recipe = llm.qwen3_1p7b.pretrain_recipe(
            dir=dir_name,
            name=name_recipe,
            num_nodes=nodes,
            num_gpus_per_node=gpus_per_node
        )
    elif model_id == "qwen3_4b":
        recipe = llm.qwen3_4b.pretrain_recipe(
            dir=dir_name,
            name=name_recipe,
            num_nodes=nodes,
            num_gpus_per_node=gpus_per_node
        )
    elif model_id == "qwen3_8b":
        recipe = llm.qwen3_8b.pretrain_recipe(
            dir=dir_name,  # Path to store checkpoints
            name=name_recipe,
            num_nodes=nodes,
            num_gpus_per_node=gpus_per_node,
        )
    elif model_id == "qwen3_14b":
        recipe = llm.qwen3_14b.pretrain_recipe(
            dir=dir_name,  # Path to store checkpoints
            name=name_recipe,
            num_nodes=nodes,
            num_gpus_per_node=gpus_per_node,
        )
    elif model_id == "gptoss_20b":
        recipe = llm.gpt_oss_20b.pretrain_recipe(
            dir=dir_name,  # Path to store checkpoints
            name=name_recipe,
            num_nodes=nodes,
            num_gpus_per_node=gpus_per_node,
        )
    recipe.optim.config.lr = args.lr                    # peak LR
    recipe.optim.lr_scheduler.warmup_steps = args.warmup_steps     # warmup steps
    recipe.optim.lr_scheduler.min_lr = 1e-6     
    recipe.trainer.strategy.ckpt_load_strictness = False
    recipe.trainer.strategy.ckpt_load_optimizer = not args.no_load_optim
    if args.ckpt_legacy_format:
        recipe.trainer.strategy.ckpt_save_pre_mcore_014 = True
    recipe.trainer.devices = gpus_per_node
    recipe.model.config.seq_length = args.seq_length
    recipe.trainer.log_every_n_steps = args.log_every_n_steps
    #recipe.trainer.accumulate_grad_batches = args.accumulate_grad_batches
    recipe.trainer.val_check_interval = args.val_check_interval
    recipe.trainer.limit_val_batches = args.limit_val_batches
    if args.max_steps is not None:
        recipe.trainer.max_steps =  args.max_steps
    else:
        recipe.trainer.max_steps = -1
    recipe.trainer.max_epochs = args.epochs
    recipe.trainer.strategy.context_parallel_size = 1
    recipe.trainer.strategy.tensor_model_parallel_size = args.tensor_parallel
    # if model_id == "llama32_1b":
    #     recipe.trainer.strategy.ckpt_load_optimizer = False 
    recipe.log.ckpt.train_time_interval = run.Config(timedelta, minutes=7*24*60)
    recipe.log.ckpt.every_n_epochs = None
    recipe.log.ckpt.save_top_k = 3
    recipe.log.ckpt.every_n_train_steps = args.val_check_interval
    
    
    if args.resume_from_path:
        recipe.resume = run.Config(
            nl.AutoResume,
            restore_config=run.Config(
                nl.RestoreConfig,
                path=args.resume_from_path,
                load_optim_state=False,
            ),
            resume_if_exists=False,
        )
    else:
        recipe.resume = run.Config(
            nl.AutoResume,
            restore_config=run.Config(
                nl.RestoreConfig,
                path=model_path
            ),
            resume_if_exists=True,
        )

    suffix = args.suffix
#     new_paths = [data_path + suffix] if type(data_path) == str else data_path
    dataset_root = [data + suffix for data in dataset_root]
    if args.dataset_weights is not None:
        assert len(args.dataset_weights) == len(dataset_root), "weights must be set for all dataset"
        new_paths = []
        for w, dataset in zip(args.dataset_weights, dataset_root):
            new_paths.extend([w, dataset])
    else:
        new_paths = dataset_root
    print(f"Data: {new_paths}")
    recipe.data = run.Config(
        llm.PreTrainingDataModule, 
        tokenizer = run.Config(AutoTokenizer, tokenizer_path),
        paths = new_paths, 
        global_batch_size = 32,
        micro_batch_size = 2,
        seq_length = args.seq_length,
        split = "90,5,5",
        num_workers = 64,
        index_mapping_dir = 'datasets/cache',
    )
    print(recipe.trainer)

    return recipe

def local_executor_torchrun(nodes: int = 1, devices: int = 2) -> run.LocalExecutor:
    # Env vars for jobs are configured here
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256",
        "CUDA_LAUNCH_BLOCKING": "0", 
        "TORCH_CUDA_EMPTY_CACHE": "1",
    }

    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

    return executor

def run_finetuning(args):
    finetune = configure_finetuning_recipe(args,nodes=1, gpus_per_node=args.gpus_per_node)
    # finetune.resume.restore_config.path = "/path/to/pretrained/NeMo-2/checkpoint"

    executor = local_executor_torchrun(nodes=finetune.trainer.num_nodes, devices=finetune.trainer.devices)
    ## CHANGES WITH EACH MODEL
    with run.Experiment(f"{args.name_recipe}", base_dir="experiments") as exp:
        exp.add(finetune, executor=executor, name=f"{args.name_recipe}_nemo2")
        exp.run(sequential=True, tail_logs=True)  # This will run the tasks sequentially and stream the logs
def run_pretraining(args):
    recipe = configure_recipe(args)
    executor = local_executor_torchrun(nodes=recipe.trainer.num_nodes, devices=recipe.trainer.devices)
    with run.Experiment(f"{args.name_recipe}", base_dir="experiments") as exp:
        exp.add(recipe, executor=executor, name=f"{args.name_recipe}")
        exp.run(sequential=True, tail_logs=True)  # This will run the tasks sequentially and stream the logs

    

# This condition is necessary for the script to be compatible with Python's multiprocessing module.
if __name__ == "__main__":
    args = get_args()
    #preprocess_datasets(args)
    run_pretraining(args) 
