"""
Generate predictions from finetuned models using vLLM.
Drop-in replacement for NeMo in-framework generate.py.

Supports:
  - HuggingFace checkpoints directly
  - NeMo 2.0 checkpoints (auto-export to HF)
  - NeMo PEFT/LoRA checkpoints (auto-convert to HF LoRA)
  - Base model + separate LoRA adapter

Usage (same args as original generate.py where possible):

    # HF model directly
    CUDA_VISIBLE_DEVICES=1,2 python generate_vllm.py \
        --dataset_root finetuned_datasets/all \
        --model_path /home/dungdx4/BERT/Meta-Llama-3-8B-Instruct \
        --output_path results/ft_llama3_8b_instruct_generate.json \
        --tokenizer_path llama3_8b \
        --eval_path ft_llama3_8b_instruct.json

    # NeMo PEFT checkpoint (like original script)
    CUDA_VISIBLE_DEVICES=1,2 python generate_vllm.py \
        --dataset_root finetuned_datasets/all \
        --model_path /home/dungdx4/BERT/Meta-Llama-3-8B-Instruct \
        --peft_ckpt_path finetuning_info/llama3_8b_instruct/checkpoints/model_name=0--val_loss=1.05-step=19999-consumed_samples=160000.0-last \
        --output_path results/ft_llama3_8b_instruct_generate.json \
        --tokenizer_path llama3_8b \
        --eval_path ft_llama3_8b_instruct.json

    # NeMo 2.0 full checkpoint (auto-export)
    CUDA_VISIBLE_DEVICES=1,2 python generate_vllm.py \
        --dataset_root finetuned_datasets/all \
        --from_nemo \
        --model_path /path/to/nemo2_checkpoint.nemo \
        --output_path results/generate.json \
        --tensor_parallel_size 2
"""

import os
import json
import argparse
import glob
from pathlib import Path
from typing import List, Dict, Optional

from vllm import LLM, SamplingParams
from tqdm import tqdm


# =============================================================================
# 1. ARGUMENT PARSING (matching original generate.py args)
# =============================================================================

def get_args():
    parser = argparse.ArgumentParser(description="Generate predictions using vLLM")

    # --- Same as original generate.py ---
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Data path (directory or single json/jsonl file)")
    parser.add_argument("--peft_ckpt_path", type=str, default=None,
                        help="Path to PEFT/LoRA checkpoint (NeMo or HF format)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path .json to save output")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Tokenizer name/path (supports shortcuts like 'llama3_8b')")
    parser.add_argument("--eval_path", type=str, default=None,
                        help="Path to save evaluation results")
    parser.add_argument("--pred_field", type=str, default="prediction",
                        help="Field name for predictions in output json")
    parser.add_argument("--label_field", type=str, default="label",
                        help="Field name for ground truth in output json")

    # --- New: model path (replaces implicit base model from NeMo config) ---
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to base HF model or NeMo checkpoint")
    parser.add_argument("--from_nemo", action="store_true",
                        help="Export NeMo checkpoint to HF before serving")
    parser.add_argument("--nemo_export_path", type=str, default=None,
                        help="Where to save exported HF model")

    # --- Generation params (matching original CommonInferenceParams) ---
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="num_tokens_to_generate equivalent")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Prompts per batch for progress tracking")

    # --- vLLM engine config ---
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "auto"])
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--tensor_parallel_size", type=int, default=2,
                        help="Number of GPUs (default=2, matching original TP=2)")
    parser.add_argument("--max_model_len", type=int, default=None,
                        help="Max sequence length (default: from model config)")

    args = parser.parse_args()
    return args


# =============================================================================
# 2. TOKENIZER SETUP (matching original setup_tokenizer)
# =============================================================================

# Shortcut mapping — same as your original code
TOKENIZER_SHORTCUTS = {
    "llama3_1b": "/home/dungdx4/BERT/Llama-3.2-1B/snapshots/model",
    "llama3_3b": "/home/dungdx4/BERT/Llama-3.2-3B-Instruct",
    "llama3_8b": "/home/dungdx4/BERT/Meta-Llama-3-8B-Instruct",
}


def setup_tokenizer(name_tokenizer: Optional[str]) -> Optional[str]:
    """Resolve tokenizer shortcut to full path."""
    if name_tokenizer is None:
        return None
    return TOKENIZER_SHORTCUTS.get(name_tokenizer, name_tokenizer)


# =============================================================================
# 3. NEMO EXPORT
# =============================================================================

def export_nemo_to_hf(nemo_path: str, output_path: str) -> str:
    """Export NeMo 2.0 checkpoint to HuggingFace format."""
    print(f"[Export] NeMo → HF")
    print(f"  Source: {nemo_path}")
    print(f"  Target: {output_path}")

    from nemo.collections.llm import export_ckpt
    export_ckpt(
        path=Path(nemo_path),
        target="hf",
        output_path=Path(output_path),
    )
    print(f"[Export] Done: {output_path}")
    return output_path


def convert_nemo_peft_to_hf(peft_path: str, base_model_path: str,
                             output_path: str) -> str:
    """
    Convert NeMo PEFT checkpoint to HF LoRA format.
    Returns path to converted HF LoRA adapter.
    """
    hf_lora_path = output_path or (peft_path.rstrip("/") + "_hf_lora")

    if Path(hf_lora_path).exists():
        print(f"[PEFT] HF LoRA already exists: {hf_lora_path}")
        return hf_lora_path

    print(f"[PEFT] Converting NeMo PEFT → HF LoRA")
    print(f"  PEFT source: {peft_path}")
    print(f"  Base model:  {base_model_path}")
    print(f"  Output:      {hf_lora_path}")

    try:
        from nemo.collections.llm import export_ckpt
        export_ckpt(
            path=Path(peft_path),
            target="hf",
            output_path=Path(hf_lora_path),
        )
        print(f"[PEFT] Conversion done: {hf_lora_path}")
        return hf_lora_path
    except Exception as e:
        print(f"[PEFT] Auto-conversion failed: {e}")
        print(f"[PEFT] Trying to merge PEFT into base model instead...")
        return merge_peft_into_base(peft_path, base_model_path, hf_lora_path)


def merge_peft_into_base(peft_path: str, base_model_path: str,
                          output_path: str) -> str:
    """
    Fallback: merge NeMo PEFT weights into base model and save as full HF model.
    This avoids the need for LoRA at inference time.
    """
    from nemo.collections.llm import export_ckpt
    from pathlib import Path

    merged_path = output_path + "_merged"
    print(f"[PEFT] Merging into: {merged_path}")

    export_ckpt(
        path=Path(peft_path),
        target="hf",
        output_path=Path(merged_path),
    )
    return merged_path


# =============================================================================
# 4. DATA LOADING
# =============================================================================

def load_dataset(dataset_root: str) -> List[Dict]:
    """
    Load dataset from json/jsonl files.
    Compatible with ThanhFinetuningDataModule's data format.
    """
    dataset_path = Path(dataset_root)
    samples = []

    if dataset_path.is_file():
        files = [dataset_path]
    elif dataset_path.is_dir():
        files = sorted(dataset_path.glob("*.json")) + sorted(dataset_path.glob("*.jsonl"))
    else:
        raise FileNotFoundError(f"Dataset not found: {dataset_root}")

    for fpath in files:
        print(f"  Loading: {fpath.name}")
        with open(fpath, "r", encoding="utf-8") as f:
            if fpath.suffix == ".jsonl":
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
            else:
                data = json.load(f)
                if isinstance(data, list):
                    samples.extend(data)
                else:
                    samples.append(data)

    print(f"  Total: {len(samples)} samples from {len(files)} file(s)")
    return samples


def extract_prompts(samples: List[Dict]) -> List[str]:
    """
    Extract prompts from dataset.
    Tries common field names used in Vietnamese summarization datasets.
    """
    prompts = []
    # Try field names in order of priority
    prompt_fields = ["input", "prompt", "text", "content", "source"]

    for s in samples:
        prompt = None
        for field in prompt_fields:
            if field in s and isinstance(s[field], str):
                prompt = s[field]
                break
        if prompt is None:
            # Fallback: first long string field
            for v in s.values():
                if isinstance(v, str) and len(v) > 20:
                    prompt = v
                    break
        if prompt:
            prompts.append(prompt)

    return prompts


# =============================================================================
# 5. MAIN GENERATION
# =============================================================================

def run_generation(args):
    # --- Resolve model path ---
    model_path = args.model_path

    # Export NeMo full checkpoint if needed
    if args.from_nemo and model_path:
        export_path = args.nemo_export_path or (model_path.rstrip("/") + "_hf")
        if not Path(export_path).exists():
            model_path = export_nemo_to_hf(model_path, export_path)
        else:
            print(f"[Model] HF export exists: {export_path}")
            model_path = export_path

    # --- Resolve tokenizer ---
    tokenizer_path = setup_tokenizer(args.tokenizer_path)

    # If no model_path but have tokenizer, use tokenizer as model
    # (for backward compat with original script where base model was implicit)
    if model_path is None and tokenizer_path:
        model_path = tokenizer_path
        print(f"[Model] Using tokenizer path as model: {model_path}")

    if model_path is None:
        raise ValueError("Either --model_path or --tokenizer_path must be provided")

    # --- Handle PEFT checkpoint ---
    use_lora = False
    lora_path = None

    if args.peft_ckpt_path:
        peft_path = Path(args.peft_ckpt_path)

        # Check if it's already HF LoRA format
        if (peft_path / "adapter_config.json").exists():
            print(f"[PEFT] HF LoRA format detected")
            use_lora = True
            lora_path = str(peft_path)
        else:
            # NeMo PEFT — try to convert or merge
            print(f"[PEFT] NeMo format detected, converting...")
            converted = convert_nemo_peft_to_hf(
                str(peft_path), model_path,
                str(peft_path) + "_hf_lora"
            )
            # Check if conversion produced LoRA adapter or merged model
            if Path(converted, "adapter_config.json").exists():
                use_lora = True
                lora_path = converted
            else:
                # Merged model — use as model_path directly
                model_path = converted
                print(f"[PEFT] Using merged model: {model_path}")

    # --- Load dataset ---
    print(f"\n[Data] Loading from: {args.dataset_root}")
    samples = load_dataset(args.dataset_root)
    prompts = extract_prompts(samples)
    print(f"[Data] {len(prompts)} prompts extracted")

    if len(prompts) == 0:
        raise ValueError("No prompts found in dataset. Check field names.")

    # --- Initialize vLLM ---
    print(f"\n[vLLM] Initializing engine...")
    print(f"  Model:     {model_path}")
    print(f"  TP size:   {args.tensor_parallel_size}")
    print(f"  dtype:     {args.dtype}")
    print(f"  LoRA:      {lora_path or 'None'}")

    engine_kwargs = dict(
        model=model_path,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )

    if tokenizer_path and tokenizer_path != model_path:
        engine_kwargs["tokenizer"] = tokenizer_path

    if args.max_model_len:
        engine_kwargs["max_model_len"] = args.max_model_len

    if use_lora:
        engine_kwargs["enable_lora"] = True

    llm = LLM(**engine_kwargs)

    # --- Sampling params (matching original CommonInferenceParams) ---
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    # --- LoRA request ---
    lora_request = None
    if use_lora and lora_path:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("peft_adapter", 1, lora_path)

    # --- Generate ---
    print(f"\n[Generate] {len(prompts)} samples, max_tokens={args.max_tokens}")

    all_predictions = []
    total_batches = (len(prompts) + args.batch_size - 1) // args.batch_size

    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i:i + args.batch_size]
        batch_num = i // args.batch_size + 1
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} prompts)")

        if lora_request:
            outputs = llm.generate(batch, sampling_params, lora_request=lora_request)
        else:
            outputs = llm.generate(batch, sampling_params)

        for output in outputs:
            all_predictions.append(output.outputs[0].text)

    # --- Build results (same format as original output) ---
    results = []
    for sample, prediction in zip(samples, all_predictions):
        result = {args.pred_field: prediction}

        if args.label_field in sample:
            result[args.label_field] = sample[args.label_field]

        # Preserve other useful fields
        for key in ["input", "prompt", "text", "content", "source", "id"]:
            if key in sample and key not in result:
                result[key] = sample[key]

        results.append(result)

    # --- Save ---
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[Output] Saved {len(results)} predictions → {args.output_path}")

    return results


# =============================================================================
# 6. MAIN
# =============================================================================

if __name__ == "__main__":
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    import torch
    print("GPU count visible:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} name: {torch.cuda.get_device_name(i)}")

    args = get_args()

    if args.peft_ckpt_path:
        print("PEFT checkpoint:", args.peft_ckpt_path)
    print("Model path:", args.model_path)

    # --- Generate ---
    run_generation(args)

    # --- Eval (same as original) ---
    if args.eval_path is not None:
        try:
            from eval.peft_eval import evalSum
            if not hasattr(args, "pred_file"):
                args.pred_file = args.output_path
            evalSum(args)
        except ImportError:
            print("[Warning] eval.peft_eval not found, skipping evaluation")
        except Exception as e:
            print(f"[Warning] Evaluation failed: {e}")