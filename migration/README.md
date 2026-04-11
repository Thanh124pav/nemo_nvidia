# Migration Guide: NeMo 2 → Megatron-Bridge + NeMo RL

## Cấu trúc thư mục

```
nemo_nvidia/
├── pretrain.py          # GIỮ NGUYÊN — NeMo 2 pretraining
├── finetune.py          # GIỮ NGUYÊN — NeMo 2 finetuning
├── distill.py           # GIỮ NGUYÊN — NeMo 2 offline KD
├── dataModule.py        # GIỮ NGUYÊN
│
├── nemo-rl/             # CLONED — NeMo RL repo (bao gồm Megatron-Bridge submodule)
│   └── 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/
│
└── migration/           # MỚI — migration files (không sửa code cũ)
    ├── README.md        # File này
    ├── bridge/
    │   ├── pretrain_bridge.py        # Thay pretrain.py — dùng Megatron-Bridge
    │   ├── convert_hf_to_megatron.py # Thay convert_hf_to_nemo.py
    │   ├── convert_megatron_to_hf.py # Thay convert_nemo_to_hf.py
    │   └── configs/
    │       ├── llama3_8b.py          # ConfigContainer cho Llama-3-8B
    │       ├── llama32_1b.py         # ConfigContainer cho Llama-3.2-1B
    │       └── qwen3_8b.py           # ConfigContainer cho Qwen3-8B
    └── nemo_rl/
        ├── run_sft.py                # Thay finetune.py (khi cần multi-node/FSDP)
        ├── run_grpo.py               # MỚI — GRPO post-training RL
        ├── run_dpo.py                # MỚI — DPO preference alignment
        └── configs/
            ├── sft_llama3_8b.yaml
            ├── grpo_llama3_8b.yaml
            └── dpo_llama3_8b.yaml
```

---

## Cài đặt (trên server, không phải máy local)

### Megatron-Bridge

```bash
# Megatron-Bridge đã có sẵn như submodule của NeMo RL
cd nemo_nvidia/nemo-rl
pip install uv
uv venv .venv && source .venv/bin/activate
uv sync

# Hoặc cài riêng Megatron-Bridge
pip install -e nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
```

### NeMo RL

```bash
cd nemo_nvidia/nemo-rl
uv venv .venv && source .venv/bin/activate
uv sync                  # base
uv sync --extra vllm     # + vLLM cho GRPO generation
uv sync --extra mcore    # + Megatron Core backend

export HF_HOME=/path/to/hf_cache
export WANDB_API_KEY=your_key     # nếu dùng W&B
```

---

## Part 1: Megatron-Bridge

### Khi nào dùng Megatron-Bridge thay NeMo 2?

| Tình huống | Dùng |
|-----------|------|
| Pretraining thông thường | NeMo 2 (`pretrain.py`) — đơn giản hơn |
| Cần HF ↔ Megatron conversion nhanh, ít RAM | **Megatron-Bridge** (`AutoBridge`) |
| Cần FP8 training + comm overlap mới nhất | **Megatron-Bridge** |
| Cần recipe built-in cho Llama/Qwen/DeepSeek | **Megatron-Bridge** |

### API thay đổi chính

```
NeMo 2                                    Megatron-Bridge
──────────────────────────────────────    ──────────────────────────────────────
recipe.trainer.strategy                →  cfg.model  (KEY CHANGE)
  .tensor_model_parallel_size             .tensor_model_parallel_size
  .pipeline_model_parallel_size           .pipeline_model_parallel_size
  .context_parallel_size                  .context_parallel_size
  .sequence_parallel                      .sequence_parallel (auto khi TP>1)

nl.AutoResume(restore_config=          →  cfg.checkpoint.load_dir = path
  nl.RestoreConfig(path=X))

recipe.log (NeMoLogger dir+name)       →  cfg.checkpoint.save_dir

recipe.optim.config.lr                 →  cfg.optimizer.lr
recipe.optim.lr_scheduler.warmup_steps →  cfg.scheduler.lr_warmup_iters
recipe.optim.lr_scheduler.min_lr       →  cfg.scheduler.min_lr

recipe.data (PreTrainingDataModule)    →  cfg.dataset.blend (list of (paths, weight))
AutoTokenizer(tokenizer_path)          →  cfg.tokenizer.tokenizer_model

run.Experiment(...).run()              →  pretrain(cfg, forward_step_func)

convert_hf_to_nemo.py                  →  AutoBridge.from_hf_pretrained()
convert_nemo_to_hf.py                  →  bridge.save_hf_pretrained()
```

### Ví dụ: pretrain với Megatron-Bridge

```bash
# Thay vì:
CUDA_VISIBLE_DEVICES=0,1 python pretrain.py \
    --model-id llama3_8b \
    --model-path models/llama/llama3_8b \
    --dir-name pretraining_info --name-recipe run1 \
    --dataset-root data/train --dataset-weights 1.0 \
    --tokenizer-path /path/to/tokenizer \
    --tensor-parallel 2

# Dùng:
torchrun --nproc-per-node=2 migration/bridge/pretrain_bridge.py \
    --model-id llama3_8b \
    --checkpoint-load-dir models/llama/llama3_8b \
    --checkpoint-save-dir pretraining_info/run1 \
    --dataset-blend data/train_text_document 1.0 \
    --tokenizer-path /path/to/tokenizer \
    --tensor-parallel 2
```

### Ví dụ: checkpoint conversion

```bash
# HF → Megatron (thay convert_hf_to_nemo.py):
python migration/bridge/convert_hf_to_megatron.py \
    --hf-model-path meta-llama/Llama-3-8B \
    --save-dir models/llama3_8b_megatron \
    --tensor-parallel 2

# Megatron → HF (thay convert_nemo_to_hf.py):
python migration/bridge/convert_megatron_to_hf.py \
    --hf-reference-model meta-llama/Llama-3-8B \
    --megatron-ckpt-dir experiments/pretrain/iter_0100000 \
    --save-dir exports/llama3_8b_hf \
    --tensor-parallel 2
```

---

## Part 2: NeMo RL

### Khi nào dùng NeMo RL?

| Task | Tool |
|------|------|
| SFT đơn giản, single-node | `finetune.py` (NeMo 2) — giữ nguyên |
| SFT multi-node (> 4 nodes) hoặc cần FSDP | `run_sft.py` |
| GRPO / DAPO post-training RL | `run_grpo.py` ← MỚI |
| DPO preference alignment | `run_dpo.py` ← MỚI |
| Offline KD (teacher/student logits) | `distill.py` (NeMo 2) — giữ nguyên |

### Pipeline đề xuất

```
HF Model (Llama/Qwen)
       │
       ▼
  finetune.py (SFT, domain adaptation)    [NeMo 2, giữ nguyên]
       │
       ▼
  run_grpo.py (RL alignment, math/coding) [NeMo RL, MỚI]
       │  hoặc
  run_dpo.py  (preference alignment)      [NeMo RL, MỚI]
       │
       ▼
  convert_megatron_to_hf.py (export)     [Megatron-Bridge, khi cần]
```

### SFT — finetune.py → run_sft.py

```bash
# Thay vì:
CUDA_VISIBLE_DEVICES=0,1 python finetune.py \
    --model-id llama3_8b \
    --model-path models/llama/llama3_8b_instruct \
    --peft-scheme lora --rank 16 --alpha 16 \
    --dataset-root datasets/train \
    --tokenizer-path /path/to/tokenizer \
    --seq-length 2048 --global-batch-size 8

# Dùng (sau khi convert checkpoint sang HF format):
python migration/nemo_rl/run_sft.py \
    --config migration/nemo_rl/configs/sft_llama3_8b.yaml \
    policy.model_name=models/llama/llama3_8b_instruct \
    policy.dtensor_cfg.lora_cfg.enabled=true \
    policy.dtensor_cfg.lora_cfg.dim=16 \
    data.train.data_path=datasets/train.jsonl \
    cluster.gpus_per_node=2
```

### GRPO — Post-training RL

```bash
python migration/nemo_rl/run_grpo.py \
    --config migration/nemo_rl/configs/grpo_llama3_8b.yaml \
    policy.model_name=models/llama/llama3_8b_sft \
    grpo.num_prompts_per_step=32 \
    cluster.gpus_per_node=8
```

### DPO — Preference Alignment

```bash
python migration/nemo_rl/run_dpo.py \
    --config migration/nemo_rl/configs/dpo_llama3_8b.yaml \
    policy.model_name=models/llama/llama3_8b_sft \
    data.train.data_path=datasets/preference_data/train.jsonl \
    cluster.gpus_per_node=2
```

---

## Data Format: finetune.py → run_sft.py

### Trước (ThanhFinetuningDataModule):
```python
# dataModule.py format (preprocessed binary + prompt template in Python)
dataset_kwargs = {
    "prompt_template": "Question: {input} Answer: {output}",
    "answer_only_loss": True,
}
```

### Sau (NeMo RL JSONL):
```jsonl
{"task_name": "finetune", "messages": [{"role": "user", "content": "Question: ..."}, {"role": "assistant", "content": "Answer: ..."}]}
{"task_name": "finetune", "messages": [{"role": "user", "content": "Question: ..."}, {"role": "assistant", "content": "Answer: ..."}]}
```

Hoặc format đơn giản hơn (input/output keys):
```jsonl
{"input": "Question: ...", "output": "Answer: ..."}
{"input": "Question: ...", "output": "Answer: ..."}
```

Script convert từ binary dataset sang JSONL (nếu cần):
```python
# Sử dụng dataModule.ThanhFinetuningDataModule để load rồi export
from dataModule import ThanhFinetuningDataModule
import json

dm = ThanhFinetuningDataModule(
    local_path="datasets/raw_data/...",
    dataset_root="datasets/finetuned_datasets/...",
    tokenizer=...,
    dataset_kwargs={"prompt_template": "Question: {input} Answer: {output}"},
)
dm.setup()
with open("datasets/train.jsonl", "w") as f:
    for item in dm.train_dataset:
        f.write(json.dumps({"input": item["input"], "output": item["output"]}) + "\n")
```

---

## Lưu ý quan trọng

1. **NeMo 2 vs Megatron-Bridge: môi trường khác nhau** — không share cùng virtualenv.
2. **NeMo RL checkpoint** lưu theo định dạng HF — không phải NeMo `.nemo` format.
3. **NeMo RL cần Ray** — overhead cho single-node; dùng `finetune.py` cho ≤ 2 nodes.
4. **GRPO cần model đã SFT** — raw pretrained model thường không ổn định với GRPO.
5. **distill.py** (offline KD) vẫn tốt hơn on-policy distillation khi dataset đã có sẵn.
6. **Megatron-Bridge TP/PP config** nằm trên `cfg.model`, không phải `strategy` như NeMo 2.
