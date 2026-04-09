from pathlib import Path

from nemo.collections.llm import export_ckpt

from nemo.collections.llm import peft

if __name__ == "__main__":
    peft.merge_lora(
        base_model_restore_path=Path("/workspace/nemo_restore/models/qwen3/Qwen3-14B"),
        lora_checkpoint_path="finetuning_nemo2_info/qwen3_14b_lora_120525-16h_globalBatch-8_seqLength-8192/checkpoints/model_name=0--val_loss=0.90-step=49999-consumed_samples=400000.0-last",
        output_path="models/qwen3/Qwen3-14B-LoRA-merged",
    #)
    #export_ckpt(
    #    path=Path("/raid/models/models_rsync/nemo_saved/qwen3_14b_lora_merge"),
    #    target="hf",
    #    output_path=Path("/raid/models/models_rsync/nemo_saved/qwen3_14b_lora_hf"),
    #)
    
    ################################
    #peft.merge_lora(
    #    lora_checkpoint_path=Path("/raid/models/models_rsync/nemo_saved/qwen3_8b_lora/checkpoints/model_name=0--val_loss=0.95-step=25599-consumed_samples=204800.0-last"),
    #    output_path=Path("/raid/models/models_rsync/nemo_saved/qwen3_8b_lora_merge"),
    #)
    #export_ckpt(
    #    path=Path("/raid/models/models_rsync/nemo_saved/qwen3_8b_lora_merge"),
    #    target="hf",
    #    output_path=Path("/raid/models/models_rsync/nemo_saved/qwen3_8b_lora_hf"),
    #)
    
    ################################
    #peft.merge_lora(
    #    lora_checkpoint_path=Path("/raid/models/models_rsync/nemo_saved/qwen3_4b/checkpoints/model_name=0--val_loss=1.00-step=49999-consumed_samples=400000.0-last"),
    #    output_path=Path("/raid/models/models_rsync/nemo_saved/qwen3_4b_lora_merge"),
    #)
    #export_ckpt(
    #    path=Path("/raid/models/models_rsync/nemo_saved/qwen3_4b_lora_merge"),
    #    target="hf",
    #    output_path=Path("/raid/models/models_rsync/nemo_saved/qwen3_4b_lora_hf"),
    #)

