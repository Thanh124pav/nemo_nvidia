"""
Megatron-Bridge ConfigContainer for Llama-3.2-1B.

NeMo 2 recipe equivalent: nemo/collections/llm/recipes/llama32_1b.py
"""

from megatron.bridge.recipes.llama import llama32_1b_pretrain_config
from megatron.bridge.training.config import ConfigContainer


def build_llama32_1b_config(
    *,
    checkpoint_load_dir: str,
    checkpoint_save_dir: str,
    dataset_blend: list,
    tokenizer_path: str,
    max_steps: int = 1_000_000,
    global_batch_size: int = 512,
    micro_batch_size: int = 1,
    seq_length: int = 8192,
    lr: float = 3e-4,
    warmup_steps: int = 2000,
    min_lr: float = 3e-5,
    tensor_parallel: int = 1,
    pipeline_parallel: int = 1,
    context_parallel: int = 1,
    eval_interval: int = 2000,
) -> ConfigContainer:
    """
    Build ConfigContainer for Llama-3.2-1B pretraining.

    Note: Llama-3.2-1B uses TP=1 by default (model is small enough).
    Recommended parallelism per Megatron-Bridge docs: TP=1, PP=1, CP=1.
    """
    cfg: ConfigContainer = llama32_1b_pretrain_config()

    cfg.checkpoint.load_dir = checkpoint_load_dir
    cfg.checkpoint.save_dir = checkpoint_save_dir

    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    cfg.tokenizer.tokenizer_model = tokenizer_path

    cfg.dataset.blend = dataset_blend
    cfg.dataset.seq_length = seq_length

    cfg.train.train_iters = max_steps
    cfg.train.global_batch_size = global_batch_size
    cfg.train.micro_batch_size = micro_batch_size
    cfg.model.seq_length = seq_length

    cfg.optimizer.lr = lr
    cfg.scheduler.lr_warmup_iters = warmup_steps
    cfg.scheduler.min_lr = min_lr

    cfg.model.tensor_model_parallel_size = tensor_parallel
    cfg.model.pipeline_model_parallel_size = pipeline_parallel
    cfg.model.context_parallel_size = context_parallel
    cfg.model.sequence_parallel = tensor_parallel > 1

    cfg.validation.eval_interval = eval_interval

    return cfg
