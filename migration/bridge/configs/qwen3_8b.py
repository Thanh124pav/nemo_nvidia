"""
Megatron-Bridge ConfigContainer for Qwen3-8B.

NeMo 2 recipe equivalent: nemo/collections/llm/recipes/qwen3_8b.py
"""

from megatron.bridge.recipes.qwen import qwen3_8b_pretrain_config
from megatron.bridge.training.config import ConfigContainer


def build_qwen3_8b_config(
    *,
    checkpoint_load_dir: str,
    checkpoint_save_dir: str,
    dataset_blend: list,
    tokenizer_path: str,
    max_steps: int = 500_000,
    global_batch_size: int = 1024,
    micro_batch_size: int = 1,
    seq_length: int = 4096,
    lr: float = 3e-4,
    warmup_steps: int = 2000,
    min_lr: float = 3e-5,
    tensor_parallel: int = 2,
    pipeline_parallel: int = 1,
    context_parallel: int = 1,
    eval_interval: int = 2000,
) -> ConfigContainer:
    """
    Build ConfigContainer for Qwen3-8B pretraining.

    Qwen3 uses grouped-query attention; Megatron-Bridge handles QKV fusion
    automatically via QKVMapping in the QwenBridge.
    """
    cfg: ConfigContainer = qwen3_8b_pretrain_config()

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
