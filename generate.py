from megatron.core.inference.common_inference_params import CommonInferenceParams
from pathlib import Path 
import argparse 
from nemo import lightning as nl
import nemo_run as run
from nemo.collections import llm
from megatron.core.optimizer import OptimizerConfig
import torch
import lightning.pytorch as pl
from pathlib import Path
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed
from dataModule import ThanhFinetuningDataModule
from eval.peft_eval import evalSum
import os

"""
CUDA_VISIBLE_DEVICES=1,2 python generate.py --dataset_root finetuned_datasets/all --output_path results/ft_llama3_8b_instruct_generate.json --tokenizer_path llama3_8b --eval_path ft_llama3_8b_instruct.json
"""

def get_args():
    parser = argparse.ArgumentParser(description="Code to generate predictions from finetuned models")
    

    parser.add_argument("--dataset_root", type=str, required=True,
                        help="data path in json format")
    parser.add_argument("--peft_ckpt_path", type=str, required=False,
                        default = "finetuning_info/llama3_8b_instruct/checkpoints/model_name=0--val_loss=1.05-step=19999-consumed_samples=160000.0-last",
                        help = "path to checkpoint path")
    parser.add_argument("--output_path", type=str,required = True,
                        help = "path .json to save output")
    parser.add_argument("--tokenizer_path", type=str, required = True,
                        help = "tokenizer path")
    parser.add_argument("--eval_path", type=str,default = None,
                        help = "path to save evaluation")
    
    parser.add_argument(
        '--pred_field',
        type=str,
        help="The field in the json file that contains the prediction tokens",
        default="prediction",
    )
    parser.add_argument(
        '--label_field',
        type=str,
        help="The field in the json file that contains the ground truth tokens",
        default="label",
    )

    args = parser.parse_args()
    
    return args

def setup_tokenizer(name_tokenizer):
    dataset_dir = "/home/dungdx4/BERT/"
    if name_tokenizer == "llama3_1b":
        return dataset_dir + "Llama-3.2-1B/snapshots/model"
    elif name_tokenizer == "llama3_3b":
        return dataset_dir + "Llama-3.2-3B-Instruct"
    elif name_tokenizer == "llama3_8b":
        return dataset_dir + "Meta-Llama-3-8B-Instruct"
    else:
        return name_tokenizer
    

def thanh_data(args) -> run.Config[pl.LightningDataModule]:
    return  run.Config(
        ThanhFinetuningDataModule, 
        dataset_root = args.dataset_root,
        tokenizer = setup_tokenizer(args.tokenizer_path),
        seq_length = 3072, 
        micro_batch_size = 1, 
        global_batch_size = 8, 
        num_workers = 48,
    )

def trainer() -> run.Config[nl.Trainer]:
    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=2,
    )
    trainer = run.Config(
        nl.Trainer,
        accelerator="gpu",
        devices=2,
        num_nodes=1,
        strategy=strategy,
        plugins=bf16_mixed(),
    )
    return trainer

def configure_inference(args):
    return run.Partial(
        llm.generate,
        path=str(args.peft_ckpt_path),
        trainer=trainer(),
        input_dataset=thanh_data(args),
        inference_params=CommonInferenceParams(num_tokens_to_generate=100, top_k=5, top_p=0.9),
        output_path=args.output_path,
    )


def local_executor_torchrun(nodes: int = 1, devices: int = 2) -> run.LocalExecutor:
    # Env vars for jobs are configured here
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
    }

    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

    return executor

if __name__ == '__main__':
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("GPU count visible:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} name: {torch.cuda.get_device_name(i)}")
    args = get_args()
    print("We will load PEFT checkpoint from:", args.peft_ckpt_path)
    run.run(configure_inference(args), executor=local_executor_torchrun())
    if args.eval_path is not None:
        if not hasattr(args, "pred_file"):
            args.pred_file = args.output_path
        evalSum(args)
        