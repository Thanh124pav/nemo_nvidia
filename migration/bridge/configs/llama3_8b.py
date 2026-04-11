"""
Megatron-Bridge ConfigContainer for Llama-3-8B.

Maps each argument from pretrain.py to the corresponding ConfigContainer field.
Import and call build_llama3_8b_config() from pretrain_bridge.py or standalone.

NeMo 2 recipe equivalent: nemo/collections/llm/recipes/llama3_8b.py
"""

from megatron.bridge.recipes.llama import llama3_8b_pretrain_config
from megatron.bridge.training.config import ConfigContainer


def build_llama3_8b_config(
    *,
    checkpoint_load_dir: str,
    checkpoint_save_dir: str,
    dataset_blend: list,           # [(["path/to/data"], weight), ...]
    tokenizer_path: str,
    # ── Training ──────────────────────────────────────────────────────────────
    max_steps: int = 500_000,
    global_batch_size: int = 1024,
    micro_batch_size: int = 1,
    seq_length: int = 4096,
    lr: float = 3e-4,
    warmup_steps: int = 2000,
    min_lr: float = 3e-5,
    # ── Parallelism ───────────────────────────────────────────────────────────
    # NeMo 2 equivalent: recipe.trainer.strategy.*
    tensor_parallel: int = 2,
    pipeline_parallel: int = 1,
    context_parallel: int = 1,
    # ── Validation ───────────────────────────────────────────────────────────
    eval_interval: int = 2000,
) -> ConfigContainer:
    """
    Build a ConfigContainer for Llama-3-8B pretraining.

    Parallelism mapping (NeMo 2 → Megatron-Bridge):
      recipe.trainer.strategy.tensor_model_parallel_size   → cfg.model.tensor_model_parallel_size
      recipe.trainer.strategy.pipeline_model_parallel_size → cfg.model.pipeline_model_parallel_size
      recipe.trainer.strategy.context_parallel_size        → cfg.model.context_parallel_size
      recipe.trainer.strategy.sequence_parallel            → cfg.model.sequence_parallel (auto)

    Optimizer mapping:
      recipe.optim.config.lr                        → cfg.optimizer.lr
      recipe.optim.lr_scheduler.warmup_steps        → cfg.scheduler.lr_warmup_iters
      recipe.optim.lr_scheduler.min_lr              → cfg.scheduler.min_lr

    Checkpoint mapping:
      nl.AutoResume(restore_config=nl.RestoreConfig(path=X)) → cfg.checkpoint.load_dir = X
      recipe.log dir                                          → cfg.checkpoint.save_dir
    """
    cfg: ConfigContainer = llama3_8b_pretrain_config()

    # Checkpoint
    cfg.checkpoint.load_dir = checkpoint_load_dir
    cfg.checkpoint.save_dir = checkpoint_save_dir

    # Tokenizer (was AutoTokenizer in NeMo 2 inside DataModule)
    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    cfg.tokenizer.tokenizer_model = tokenizer_path

    # Dataset blend (was PreTrainingDataModule paths/weights)
    cfg.dataset.blend = dataset_blend
    cfg.dataset.seq_length = seq_length

    # Training
    cfg.train.train_iters = max_steps
    cfg.train.global_batch_size = global_batch_size
    cfg.train.micro_batch_size = micro_batch_size
    cfg.model.seq_length = seq_length

    # Optimizer (was OptimizerModule with cosine annealing)
    cfg.optimizer.lr = lr
    cfg.scheduler.lr_warmup_iters = warmup_steps
    cfg.scheduler.min_lr = min_lr

    # Parallelism (KEY CHANGE: no longer on strategy, now on model)
    cfg.model.tensor_model_parallel_size = tensor_parallel
    cfg.model.pipeline_model_parallel_size = pipeline_parallel
    cfg.model.context_parallel_size = context_parallel
    cfg.model.sequence_parallel = tensor_parallel > 1

    # Validation
    cfg.validation.eval_interval = eval_interval

    return cfg
