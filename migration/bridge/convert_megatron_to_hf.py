"""
convert_megatron_to_hf.py — Megatron-Bridge drop-in for convert_nemo_to_hf.py

Converts a Megatron Core distributed checkpoint back to HuggingFace format.

Useful after pretraining/finetuning with Megatron-Bridge to export the model
for inference with vLLM, TGI, or HuggingFace pipelines.

Usage:
    python migration/bridge/convert_megatron_to_hf.py \\
        --hf-reference-model  meta-llama/Llama-3-8B              \\  # for architecture
        --megatron-ckpt-dir   experiments/pretrain_bridge/llama3_8b/iter_0100000 \\
        --save-dir            exports/llama3_8b_hf               \\
        [--tensor-parallel 2]  # must match TP used during training

Notes:
  • --hf-reference-model provides the architecture config; weights come from
    --megatron-ckpt-dir.
  • The output is a standard HF safetensors directory usable with
    AutoModelForCausalLM.from_pretrained().
"""

import argparse

from megatron.bridge import AutoBridge


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Megatron Core checkpoint → HuggingFace via AutoBridge"
    )
    parser.add_argument(
        "--hf-reference-model", type=str, required=True,
        help="HF model name/path used only for architecture config (not weights)"
    )
    parser.add_argument(
        "--megatron-ckpt-dir", type=str, required=True,
        help="Path to the Megatron Core sharded checkpoint directory"
    )
    parser.add_argument(
        "--save-dir", type=str, required=True,
        help="Output directory for the HuggingFace checkpoint"
    )
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="TP size that was used during Megatron training")
    parser.add_argument("--pipeline-parallel", type=int, default=1,
                        help="PP size that was used during Megatron training")
    return parser.parse_args()


def main() -> None:
    args = get_args()

    print(f"Loading bridge from HF reference: {args.hf_reference_model}")
    bridge = AutoBridge.from_hf_pretrained(
        args.hf_reference_model,
        load_weights=False,   # We'll load weights from the Megatron ckpt
    )

    print("Building Megatron provider ...")
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = args.tensor_parallel
    provider.pipeline_model_parallel_size = args.pipeline_parallel
    provider.finalize()

    print(f"Loading Megatron Core checkpoint from: {args.megatron_ckpt_dir}")
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    bridge.load_megatron_checkpoint(model, args.megatron_ckpt_dir)

    print(f"Saving HuggingFace checkpoint to: {args.save_dir}")
    bridge.save_hf_pretrained(model, args.save_dir)
    print("Done. Load with: AutoModelForCausalLM.from_pretrained('{}')".format(
        args.save_dir
    ))


if __name__ == "__main__":
    main()
