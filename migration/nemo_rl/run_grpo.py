"""
run_grpo.py — GRPO post-training (new capability, không có trong NeMo 2)

Group Relative Policy Optimization cho math reasoning, instruction following,
và các task cần RL reward signal.

Source repo: nemo-rl/ (at nemo_nvidia/nemo-rl/)
Requires: pip install -e nemo-rl[vllm]

Usage:
    python migration/nemo_rl/run_grpo.py \\
        --config migration/nemo_rl/configs/grpo_llama3_8b.yaml \\
        [Hydra overrides]

Override examples:
    policy.model_name=models/llama/llama3_8b_instruct
    grpo.num_prompts_per_step=32
    grpo.num_generations_per_prompt=16
    policy.generation.vllm_cfg.tensor_parallel_size=2
    data.train.dataset_name=OpenMathInstruct-2
    cluster.gpus_per_node=8

GRPO Pipeline Overview:
─────────────────────────────────────────────────────────────────────────────
  1. Sample prompts from dataset (num_prompts_per_step)
  2. Generate num_generations_per_prompt responses per prompt via vLLM/Megatron
  3. Compute rewards from environment (math verifier, LM judge, etc.)
  4. Compute advantages = (reward - mean_reward) / std_reward  [GRPO baseline]
  5. Update policy with PPO-clip loss + KL penalty
  6. Reload weights into vLLM for next generation step
─────────────────────────────────────────────────────────────────────────────

Reward Environments (set in data.train.env_name):
  • "math"          — exact match math verifier (for OpenMathInstruct-2, DeepScaler)
  • "lm_judge"      — LM-as-judge for open-ended responses
  • "custom"        — implement nemo_rl.envs.base.Environment
  • See nemo-rl/nemo_rl/envs/ for all available environments

DAPO variant (better for long reasoning chains):
  Set grpo.use_dynamic_sampling=true, grpo.clip_higher=true in config or overrides.
  Full DAPO config: nemo-rl/examples/configs/recipes/llm/dapo-qwen2.5-7b.yaml
"""

import argparse
import os
import sys

NEMO_RL_DIR = os.path.join(os.path.dirname(__file__), "../../nemo-rl")
if os.path.exists(NEMO_RL_DIR):
    sys.path.insert(0, NEMO_RL_DIR)

from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "configs", "grpo_llama3_8b.yaml")


def parse_args():
    parser = argparse.ArgumentParser(description="NeMo RL GRPO — post-training RL")
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to YAML config (default: configs/grpo_llama3_8b.yaml)"
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
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"],
        tokenizer,
        has_refit_draft_weights=bool(config["policy"]["draft"]["enabled"]),
    )

    dataset, val_dataset, task_to_env, val_task_to_env = setup_response_data(
        tokenizer, config["data"], config["env"]
    )

    (
        policy, policy_generation, cluster,
        dataloader, val_dataloader, loss_fn,
        logger, checkpointer, grpo_state, master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    # Check async mode
    async_cfg = config["grpo"].get("async_grpo", {})
    if async_cfg.get("enabled", False):
        from nemo_rl.algorithms.grpo import async_grpo_train
        print("Running async GRPO")
        async_grpo_train(
            policy=policy,
            policy_generation=policy_generation,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            tokenizer=tokenizer,
            loss_fn=loss_fn,
            task_to_env=task_to_env,
            val_task_to_env=val_task_to_env,
            logger=logger,
            checkpointer=checkpointer,
            grpo_save_state=grpo_state,
            master_config=master_config,
            max_trajectory_age_steps=async_cfg["max_trajectory_age_steps"],
        )
    else:
        print("Running synchronous GRPO")
        grpo_train(
            policy, policy_generation, dataloader, val_dataloader,
            tokenizer, loss_fn, task_to_env, val_task_to_env,
            logger, checkpointer, grpo_state, master_config,
        )


if __name__ == "__main__":
    main()
