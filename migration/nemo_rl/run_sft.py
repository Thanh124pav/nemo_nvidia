"""
run_sft.py — NeMo RL drop-in for finetune.py (SFT mode)

Replaces NeMo 2 finetune.py when:
  • You need multi-node FSDP scaling (> 4 nodes)
  • You need DTensor-based sequence/context parallelism in finetuning
  • You want the same Ray-based worker isolation as GRPO/DPO pipelines

For single-node LoRA finetuning, finetune.py is still simpler and recommended.

Source repo: nemo-rl/ (at nemo_nvidia/nemo-rl/)
Requires: pip install -e nemo-rl

Usage:
    python migration/nemo_rl/run_sft.py \\
        --config migration/nemo_rl/configs/sft_llama3_8b.yaml \\
        [Hydra overrides, e.g. policy.model_name=/path/to/local/model]

CLI override examples:
    policy.model_name=models/llama/llama3_8b_instruct
    policy.dtensor_cfg.tensor_parallel_size=2
    policy.dtensor_cfg.lora_cfg.enabled=true policy.dtensor_cfg.lora_cfg.dim=16
    data.train.data_path=/path/to/train.jsonl
    sft.max_num_steps=5000
    cluster.gpus_per_node=8 cluster.num_nodes=2

Compared to finetune.py:
─────────────────────────────────────────────────────────────────────────────
  finetune.py                        run_sft.py (NeMo RL)
  ─────────────────────────────────  ─────────────────────────────────────
  --model-path (NeMo ckpt)           policy.model_name (HF name or path)
  --peft-scheme lora                 policy.dtensor_cfg.lora_cfg.enabled=true
  --rank 8 --alpha 8                 policy.dtensor_cfg.lora_cfg.dim=8 .alpha=32
  --layers 0 1 2                     policy.dtensor_cfg.lora_cfg.target_modules=[...]
  --tensor-parallel N                policy.dtensor_cfg.tensor_parallel_size=N
  --seq-length 2048                  policy.max_total_sequence_length=2048
  --global-batch-size 8              policy.train_global_batch_size=8
  --micro-batch-size 1               policy.train_micro_batch_size=1
  --lr 5e-6                          policy.optimizer.kwargs.lr=5e-6
  --warmup-steps 1000                (set in scheduler section of YAML)
  --max-steps 25000                  sft.max_num_steps=25000
  --dataset-root path/               data.train.data_path=path/train.jsonl
  ThanhFinetuningDataModule          data.train.processor=sft_processor
    prompt_template: "Q: {i} A: {o}" data.default.prompt_file=path/to/template.txt
    answer_only_loss: True           (handled by sft_processor by default)
─────────────────────────────────────────────────────────────────────────────

Data format for NeMo RL SFT (JSONL, one example per line):
    {"task_name": "finetune", "messages": [
        {"role": "user", "content": "Question: ..."},
        {"role": "assistant", "content": "Answer: ..."}
    ]}

Or using ResponseDataset format (input/output keys):
    {"input": "Question: ...", "output": "Answer: ..."}
"""

import argparse
import os
import sys

# Add nemo-rl to path if installed as source (adjust path as needed)
NEMO_RL_DIR = os.path.join(os.path.dirname(__file__), "../../nemo-rl")
if os.path.exists(NEMO_RL_DIR):
    sys.path.insert(0, NEMO_RL_DIR)

# ── NeMo RL imports ──────────────────────────────────────────────────────────
from functools import partial

from datasets import concatenate_datasets
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from nemo_rl.algorithms.sft import MasterConfig, setup, sft_train
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import (
    AllTaskProcessedDataset,
    load_response_dataset,
    update_single_dataset_config,
)
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "configs", "sft_llama3_8b.yaml")


def parse_args():
    parser = argparse.ArgumentParser(description="NeMo RL SFT — replaces finetune.py")
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to YAML config (default: configs/sft_llama3_8b.yaml)"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def setup_data(tokenizer, data_config: DataConfig):
    """Build train/val datasets from config. Same logic as nemo-rl/examples/run_sft.py."""
    assert "train" in data_config, (
        "data.train must be set. See migration/nemo_rl/configs/sft_llama3_8b.yaml for examples."
    )

    task_data_processors = {}
    task_data_preprocessors = {}
    data_list = []

    if isinstance(data_config["train"], dict):
        data_config["train"] = [data_config["train"]]

    for cfg in data_config["train"]:
        if "default" in data_config and data_config["default"] is not None:
            update_single_dataset_config(cfg, data_config["default"])
        data = load_response_dataset(cfg)
        data_list.append(data)
        processor = partial(
            data.processor,
            add_bos=data_config["add_bos"],
            add_eos=data_config["add_eos"],
            add_generation_prompt=data_config["add_generation_prompt"],
        )
        task_data_processors[data.task_name] = (data.task_spec, processor)
        if hasattr(data, "preprocessor") and data.preprocessor is not None:
            task_data_preprocessors[data.task_name] = data.preprocessor

    merged = concatenate_datasets([d.dataset for d in data_list])
    dataset = AllTaskProcessedDataset(
        merged, tokenizer, None, task_data_processors,
        task_data_preprocessors=task_data_preprocessors,
        max_seq_length=data_config["max_input_seq_length"],
    )
    print(f"  Train dataset: {len(dataset)} samples")

    # Validation
    val_data_list = []
    val_processors = {}
    val_preprocessors = {}

    for data in data_list:
        if hasattr(data, "val_dataset") and data.val_dataset is not None:
            val_data_list.append(data.val_dataset)
            val_processors[data.task_name] = task_data_processors[data.task_name]

    if "validation" in data_config and data_config["validation"] is not None:
        if isinstance(data_config["validation"], dict):
            data_config["validation"] = [data_config["validation"]]
        for cfg in data_config["validation"]:
            if "default" in data_config and data_config["default"] is not None:
                update_single_dataset_config(cfg, data_config["default"])
            vdata = load_response_dataset(cfg)
            val_data_list.append(vdata.dataset)
            vprocessor = partial(
                vdata.processor,
                add_bos=data_config["add_bos"],
                add_eos=data_config["add_eos"],
                add_generation_prompt=data_config["add_generation_prompt"],
            )
            val_processors[vdata.task_name] = (vdata.task_spec, vprocessor)

    val_dataset = None
    if val_data_list:
        merged_val = concatenate_datasets(val_data_list)
        val_dataset = AllTaskProcessedDataset(
            merged_val, tokenizer, None, val_processors,
            task_data_preprocessors=val_preprocessors,
            max_seq_length=data_config["max_input_seq_length"],
        )
        print(f"  Val dataset:   {len(val_dataset)} samples")

    return dataset, val_dataset


def main():
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"Log dir: {config['logger']['log_dir']}")

    init_ray()

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    dataset, val_dataset = setup_data(tokenizer, config["data"])

    (
        policy, cluster, train_dl, val_dl,
        loss_fn, logger, checkpointer, sft_save_state, master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    sft_train(
        policy, train_dl, val_dl, tokenizer, loss_fn,
        master_config, logger, checkpointer, sft_save_state,
    )


if __name__ == "__main__":
    main()
