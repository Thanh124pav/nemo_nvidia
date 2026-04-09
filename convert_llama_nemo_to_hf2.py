from pathlib import Path

from nemo.collections.llm import export_ckpt

if __name__ == "__main__":
    #export_ckpt(
    #    path=Path("/"),
    #    target="hf",
    #    output_path=Path("/"),
    #)

    export_ckpt(
        #path="/workspace/nemo_restore/finetuning_nemo2_info/qwen3_4b_lora_010526-11h_globalBatch-8_seqLength-16384/checkpoints/model_name=0--val_loss=0.92-step=49999-consumed_samples=400000.0-last",
        path="/workspace/data-shared/nlp/dungdx4/nemo_restore/finetuning_nemo2_info/qwen3_14b_012926-17h_globalBatch-8_seqLength-16384/checkpoints/model_name=0--val_loss=0.53-step=55999-consumed_samples=448000.0-last",
        target="hf",
        #output_path=Path("/workspace/nemo_restore/finetuning_nemo2_info/qwen3_4b_continual_hf"),
        output_path=Path("/workspace/data-shared/nlp/dungdx4/nemo_restore/models_covert/Qwen3-14B-hf"),
    )
