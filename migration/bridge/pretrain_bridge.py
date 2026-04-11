"""
pretrain_bridge.py — Megatron-Bridge equivalent of pretrain.py

Replaces NeMo 2 pattern:
    MegatronStrategy + nemo_run.Experiment + AutoResume
with:
    ConfigContainer + megatron.bridge.training.pretrain.pretrain()

Source repo: nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
Requires: pip install -e nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
Launch:
    torchrun --nproc-per-node=<GPUS> migration/bridge/pretrain_bridge.py \\
        --model-id llama3_8b \\
        --checkpoint-load-dir models/llama/llama3_8b \\
        --checkpoint-save-dir experiments/pretrain_bridge/llama3_8b \\
        --dataset-blend path/to/data1 0.6 path/to/data2 0.4 \\
        --tokenizer-path /path/to/tokenizer \\
        [--tensor-parallel 2] [--pipeline-parallel 1] [--max-steps 10000]

Hydra-style overrides are also supported after the script args, e.g.:
    torchrun ... pretrain_bridge.py ... model.tensor_model_parallel_size=4
"""

import argparse
import sys
from typing import Iterator

# ── Megatron-Bridge imports ──────────────────────────────────────────────────
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.config import ConfigContainer

# Model-specific recipe functions (mirrors pretrain.py model choices)
# Each returns a ConfigContainer with sane defaults for that model.
from megatron.bridge.recipes.llama import (
    llama32_1b_pretrain_config,
    llama32_3b_pretrain_config,
    llama3_8b_pretrain_config,
)
from megatron.bridge.recipes.qwen import (
    qwen3_0_6b_pretrain_config,
    qwen3_1_7b_pretrain_config,
    qwen3_4b_pretrain_config,
    qwen3_8b_pretrain_config,
    qwen3_14b_pretrain_config,
)

# ── Argument parser (mirrors pretrain.py CLI) ────────────────────────────────

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretraining with Megatron-Bridge (drop-in replacement for pretrain.py)"
    )
    parser.add_argument(
        "--model-id", type=str, required=True,
        choices=["llama32_1b", "llama32_3b", "llama3_8b",
                 "qwen3_0.6b", "qwen3_1.7b", "qwen3_4b", "qwen3_8b", "qwen3_14b"],
        help="Model ID — same choices as pretrain.py"
    )
    parser.add_argument(
        "--checkpoint-load-dir", type=str, required=True,
        help="Path to initial checkpoint (NeMo or HF format). "
             "Megatron-Bridge: cfg.checkpoint.load_dir  ← was pretrain.py --model-path"
    )
    parser.add_argument(
        "--checkpoint-save-dir", type=str, required=True,
        help="Directory to save checkpoints. "
             "Megatron-Bridge: cfg.checkpoint.save_dir  ← was pretrain.py --dir-name/--name-recipe"
    )
    # Dataset: list of (path, weight) pairs, e.g. path1 0.6 path2 0.4
    parser.add_argument(
        "--dataset-blend", nargs="+", required=True,
        help="Alternating path/weight pairs for the dataset blend. "
             "Megatron-Bridge: cfg.dataset.blend"
    )
    parser.add_argument(
        "--tokenizer-path", type=str, required=True,
        help="HuggingFace tokenizer path. Megatron-Bridge: cfg.tokenizer.tokenizer_model"
    )
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Max training iterations. cfg.train.train_iters")
    parser.add_argument("--global-batch-size", type=int, default=None,
                        help="Global batch size. cfg.train.global_batch_size")
    parser.add_argument("--micro-batch-size", type=int, default=1,
                        help="Micro batch size per GPU. cfg.train.micro_batch_size")
    parser.add_argument("--seq-length", type=int, default=None,
                        help="Sequence length. cfg.model.seq_length + cfg.dataset.seq_length")
    parser.add_argument("--lr", type=float, default=None,
                        help="Peak learning rate. cfg.optimizer.lr")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="LR warmup steps. cfg.scheduler.lr_warmup_iters")
    parser.add_argument("--tensor-parallel", type=int, default=None,
                        help="TP size. cfg.model.tensor_model_parallel_size  "
                             "← was recipe.trainer.strategy.tensor_model_parallel_size")
    parser.add_argument("--pipeline-parallel", type=int, default=None,
                        help="PP size. cfg.model.pipeline_model_parallel_size  "
                             "← was recipe.trainer.strategy.pipeline_model_parallel_size")
    parser.add_argument("--context-parallel", type=int, default=1,
                        help="CP size. cfg.model.context_parallel_size")
    parser.add_argument("--eval-interval", type=int, default=2000,
                        help="Eval interval. cfg.validation.eval_interval")
    # Pass remaining args as Hydra-style overrides (e.g. model.seq_length=4096)
    args, remaining = parser.parse_known_args()
    args.overrides = remaining
    return args


# ── Recipe builder ────────────────────────────────────────────────────────────

def build_config(args: argparse.Namespace) -> ConfigContainer:
    """
    Build a Megatron-Bridge ConfigContainer from CLI args.

    NeMo 2 → Megatron-Bridge config mapping
    ─────────────────────────────────────────────────────────────────
    recipe.trainer.strategy.tensor_model_parallel_size  → cfg.model.tensor_model_parallel_size
    recipe.trainer.strategy.pipeline_model_parallel_size → cfg.model.pipeline_model_parallel_size
    recipe.trainer.strategy.context_parallel_size        → cfg.model.context_parallel_size
    recipe.trainer.strategy.sequence_parallel            → cfg.model.sequence_parallel
    recipe.trainer.max_steps                             → cfg.train.train_iters
    recipe.trainer.val_check_interval                    → cfg.validation.eval_interval
    recipe.optim.config.lr                               → cfg.optimizer.lr
    recipe.optim.lr_scheduler.warmup_steps               → cfg.scheduler.lr_warmup_iters
    nl.AutoResume(restore_config=nl.RestoreConfig(path)) → cfg.checkpoint.load_dir
    recipe.log (NeMoLogger dir)                          → cfg.checkpoint.save_dir
    recipe.data (PreTrainingDataModule paths+weights)    → cfg.dataset.blend
    AutoTokenizer(tokenizer_path)                        → cfg.tokenizer.*
    """
    model_id = args.model_id

    # Select recipe function matching model choice
    recipe_map = {
        "llama32_1b":  llama32_1b_pretrain_config,
        "llama32_3b":  llama32_3b_pretrain_config,
        "llama3_8b":   llama3_8b_pretrain_config,
        "qwen3_0.6b":  qwen3_0_6b_pretrain_config,
        "qwen3_1.7b":  qwen3_1_7b_pretrain_config,
        "qwen3_4b":    qwen3_4b_pretrain_config,
        "qwen3_8b":    qwen3_8b_pretrain_config,
        "qwen3_14b":   qwen3_14b_pretrain_config,
    }
    cfg: ConfigContainer = recipe_map[model_id]()

    # ── Checkpoint paths ──────────────────────────────────────────────────────
    # NeMo 2: nl.AutoResume(restore_config=nl.RestoreConfig(path=model_path))
    cfg.checkpoint.load_dir = args.checkpoint_load_dir
    cfg.checkpoint.save_dir = args.checkpoint_save_dir

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    # NeMo 2: AutoTokenizer(tokenizer_path) inside PreTrainingDataModule
    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    cfg.tokenizer.tokenizer_model = args.tokenizer_path

    # ── Dataset blend ─────────────────────────────────────────────────────────
    # NeMo 2: paths = [w1, path1, w2, path2, ...]  (interleaved weight/path)
    # Megatron-Bridge: blend = [(paths_list, weight), ...]
    blend_args = args.dataset_blend
    if len(blend_args) % 2 != 0:
        raise ValueError(
            "--dataset-blend must be alternating path/weight pairs: "
            "path1 0.6 path2 0.4 ..."
        )
    blend = []
    for i in range(0, len(blend_args), 2):
        path = blend_args[i]
        weight = float(blend_args[i + 1])
        blend.append(([path], weight))
    cfg.dataset.blend = blend

    # ── Training iterations ───────────────────────────────────────────────────
    # NeMo 2: recipe.trainer.max_steps
    if args.max_steps is not None:
        cfg.train.train_iters = args.max_steps

    if args.global_batch_size is not None:
        cfg.train.global_batch_size = args.global_batch_size
    cfg.train.micro_batch_size = args.micro_batch_size

    # ── Sequence length ───────────────────────────────────────────────────────
    if args.seq_length is not None:
        cfg.model.seq_length = args.seq_length
        cfg.dataset.seq_length = args.seq_length

    # ── Learning rate ─────────────────────────────────────────────────────────
    # NeMo 2: recipe.optim.config.lr / recipe.optim.lr_scheduler.warmup_steps
    if args.lr is not None:
        cfg.optimizer.lr = args.lr
    if args.warmup_steps is not None:
        cfg.scheduler.lr_warmup_iters = args.warmup_steps

    # ── Parallelism ───────────────────────────────────────────────────────────
    # NeMo 2: recipe.trainer.strategy.tensor_model_parallel_size
    # KEY CHANGE: parallelism now lives on cfg.model, NOT on strategy
    if args.tensor_parallel is not None:
        cfg.model.tensor_model_parallel_size = args.tensor_parallel
    if args.pipeline_parallel is not None:
        cfg.model.pipeline_model_parallel_size = args.pipeline_parallel
    cfg.model.context_parallel_size = args.context_parallel

    # Enable sequence_parallel automatically when TP > 1
    if cfg.model.tensor_model_parallel_size > 1:
        cfg.model.sequence_parallel = True

    # ── Validation ────────────────────────────────────────────────────────────
    # NeMo 2: recipe.trainer.val_check_interval
    cfg.validation.eval_interval = args.eval_interval

    # ── Hydra-style overrides (e.g. "model.seq_length=8192") ─────────────────
    if args.overrides:
        _apply_overrides(cfg, args.overrides)

    return cfg


def _apply_overrides(cfg: ConfigContainer, overrides: list[str]) -> None:
    """Apply dot-notation overrides, e.g. 'model.tensor_model_parallel_size=4'."""
    for override in overrides:
        if "=" not in override:
            print(f"[WARNING] Skipping override with no '=': {override}")
            continue
        key, value = override.split("=", 1)
        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        # Auto-cast value
        field_name = parts[-1]
        current = getattr(obj, field_name, None)
        if isinstance(current, int):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        elif isinstance(current, bool):
            value = value.lower() in ("true", "1", "yes")
        setattr(obj, field_name, value)
        print(f"[override] {key} = {value}")


# ── Forward step function ────────────────────────────────────────────────────

def gpt_forward_step(data_iterator: Iterator, model) -> tuple:
    """
    Minimal GPT forward step for causal LM pretraining.

    Megatron-Bridge's pretrain() expects:
        forward_step(data_iterator, model) -> loss (tensor)

    For a full production forward step (with loss masking, vocab parallelism, etc.),
    see Megatron-LM's examples/gpt3/train_gpt3_175b_distributed.py
    """
    batch = next(data_iterator)

    tokens     = batch["tokens"].cuda()
    labels     = batch["labels"].cuda()
    loss_mask  = batch["loss_mask"].cuda()
    attention_mask = batch.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.cuda()

    output = model(
        input_ids=tokens,
        position_ids=None,
        attention_mask=attention_mask,
        labels=labels,
    )
    # output is a dict or tensor depending on model; extract loss
    if isinstance(output, dict):
        loss = output["loss"]
    else:
        loss = output

    # Apply loss mask (padding tokens should not contribute)
    loss = (loss * loss_mask).sum() / loss_mask.sum()
    return loss, {"lm_loss": loss.detach()}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = get_args()
    cfg = build_config(args)

    print("=" * 70)
    print("Megatron-Bridge Pretrain Config")
    print(f"  model:              {args.model_id}")
    print(f"  checkpoint.load_dir: {cfg.checkpoint.load_dir}")
    print(f"  checkpoint.save_dir: {cfg.checkpoint.save_dir}")
    print(f"  train.train_iters:  {cfg.train.train_iters}")
    print(f"  TP={cfg.model.tensor_model_parallel_size}  "
          f"PP={cfg.model.pipeline_model_parallel_size}  "
          f"CP={cfg.model.context_parallel_size}")
    print("=" * 70)

    pretrain(cfg, gpt_forward_step)
