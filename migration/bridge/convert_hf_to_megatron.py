"""
convert_hf_to_megatron.py — Megatron-Bridge drop-in for convert_hf_to_nemo.py

Converts a HuggingFace checkpoint to Megatron Core distributed format.

AutoBridge replaces the old nemo.collections.nlp conversion scripts by streaming
parameters directly without loading the full model into memory.

Usage:
    python migration/bridge/convert_hf_to_megatron.py \\
        --hf-model-path  meta-llama/Llama-3-8B          \\   # HF hub OR local dir
        --save-dir       experiments/converted/llama3_8b  \\
        [--tensor-parallel 2] [--pipeline-parallel 1]

Notes:
  • Supports all models with a registered bridge:
    Llama 2/3/3.1/3.2, Qwen 2/2.5/3, DeepSeek v2/v3, Gemma, Mistral, Mixtral, etc.
  • The saved directory can be loaded via cfg.checkpoint.load_dir in pretrain_bridge.py.
  • For multi-GPU conversion, launch with torchrun (TP > 1 requires distributed init).
"""

import argparse

from megatron.bridge import AutoBridge


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace → Megatron Core checkpoint via AutoBridge"
    )
    parser.add_argument(
        "--hf-model-path", type=str, required=True,
        help="HuggingFace model name or local directory path"
    )
    parser.add_argument(
        "--save-dir", type=str, required=True,
        help="Output directory for the Megatron Core checkpoint"
    )
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="Tensor model parallel size for the converted checkpoint")
    parser.add_argument("--pipeline-parallel", type=int, default=1,
                        help="Pipeline model parallel size for the converted checkpoint")
    parser.add_argument("--context-parallel", type=int, default=1,
                        help="Context parallel size")
    parser.add_argument("--no-weights", action="store_true",
                        help="Convert architecture only (no weight loading), for testing")
    return parser.parse_args()


def main() -> None:
    args = get_args()

    print(f"Loading HuggingFace model from: {args.hf_model_path}")
    bridge = AutoBridge.from_hf_pretrained(
        args.hf_model_path,
        # load_weights=False means architecture-only (faster, for testing)
        load_weights=not args.no_weights,
    )

    print("Building Megatron provider ...")
    provider = bridge.to_megatron_provider()

    # Configure parallelism BEFORE finalizing
    # NeMo 2 equivalent: recipe.trainer.strategy.tensor_model_parallel_size = N
    provider.tensor_model_parallel_size = args.tensor_parallel
    provider.pipeline_model_parallel_size = args.pipeline_parallel
    provider.context_parallel_size = args.context_parallel

    # finalize() computes derived fields (like sequence_parallel)
    provider.finalize()

    print(f"Building distributed model (TP={args.tensor_parallel}, "
          f"PP={args.pipeline_parallel}) ...")
    model = provider.provide_distributed_model(wrap_with_ddp=False)

    print(f"Saving Megatron Core checkpoint to: {args.save_dir}")
    # The bridge saves in Megatron Core distributed format (sharded per TP/PP rank)
    bridge.save_megatron_checkpoint(model, args.save_dir)
    print("Done.")


if __name__ == "__main__":
    main()
