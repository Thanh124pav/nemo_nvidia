"""
run_dpo.py — DPO preference learning (new capability, không có trong NeMo 2)

Direct Preference Optimization cho alignment từ preference data.
Không cần reward model hay RL training loop.

Source repo: nemo-rl/ (at nemo_nvidia/nemo-rl/)
Requires: pip install -e nemo-rl

Usage:
    python migration/nemo_rl/run_dpo.py \\
        --config migration/nemo_rl/configs/dpo_llama3_8b.yaml \\
        [Hydra overrides]

Override examples:
    policy.model_name=models/llama/llama3_8b_instruct
    dpo.reference_policy_kl_penalty=0.1
    dpo.preference_loss_weight=1.0
    data.train.dataset_name=HelpSteer3
    cluster.gpus_per_node=8

DPO Data Format (JSONL, PreferenceDataset):
    {"messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "...", "rank": 0}  <- chosen (rank=0)
    ]}
    {"messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "...", "rank": 1}  <- rejected (rank=1)
    ]}

Or BinaryPreferenceDataset (pairwise):
    {"prompt": "...", "chosen": "...", "rejected": "..."}

Supported public datasets (set data.train.dataset_name):
    • "HelpSteer3"
    • "Tulu3Preference"
    • Custom: data.train.data_path=/path/to/local.jsonl

DPO vs NeMo 2 distill.py:
─────────────────────────────────────────────────────────────────────────────
  distill.py (offline KD)                  run_dpo.py (preference learning)
  ─────────────────────────────────────    ──────────────────────────────────
  Student/teacher logit matching           Chosen vs rejected response pairs
  Requires teacher model running           Only needs preference labels
  Good for: capability distillation        Good for: alignment, safety, style
  Static dataset                           Can use human or synthetic labels
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import sys

NEMO_RL_DIR = os.path.join(os.path.dirname(__file__), "../../nemo-rl")
if os.path.exists(NEMO_RL_DIR):
    sys.path.insert(0, NEMO_RL_DIR)

from omegaconf import OmegaConf

from nemo_rl.algorithms.dpo import MasterConfig, dpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "configs", "dpo_llama3_8b.yaml")


def parse_args():
    parser = argparse.ArgumentParser(description="NeMo RL DPO — preference alignment")
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to YAML config (default: configs/dpo_llama3_8b.yaml)"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


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

    (
        policy, ref_policy, cluster,
        train_dataloader, val_dataloader,
        loss_fn, logger, checkpointer, dpo_save_state, master_config,
    ) = setup(config, tokenizer)

    dpo_train(
        policy, ref_policy,
        train_dataloader, val_dataloader,
        tokenizer, loss_fn,
        master_config, logger, checkpointer, dpo_save_state,
    )


if __name__ == "__main__":
    main()
