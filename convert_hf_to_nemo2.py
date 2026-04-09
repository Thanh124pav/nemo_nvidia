import os
os.environ['FLASHINFER_WORKSPACE_BASE'] = '/workspace/flashinfer_workspace' 
import pathlib
import argparse
import nemo
print(nemo.__file__)
from nemo.collections import llm

FLASHINFER_BASE_DIR: pathlib.Path = pathlib.Path(
    os.getenv("FLASHINFER_WORKSPACE_BASE", pathlib.Path.home().as_posix())
)

FLASHINFER_CACHE_DIR: pathlib.Path = FLASHINFER_BASE_DIR / ".cache" / "flashinfer"
_package_root: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]
print(FLASHINFER_CACHE_DIR)

def get_args():
    parser = argparse.ArgumentParser(description="Convert models from huggingface to Nemo 2")
    parser.add_argument("--model-id", required=True, choices = ["llama3.2_1b", "llama3.2_3b", "qwen3_0.6b", "qwen3_1.7b", "qwen3_4b", "qwen3_8b", "qwen3_14b", "gpt_oss_20b"], help = "Model ID")
    parser.add_argument("--src", required=True, help="Path to huggingface model")
    parser.add_argument("--dst", required=True, help="Path to nemo 2 model")

    args = parser.parse_args()
    return args 

if __name__ == "__main__":
#    src_qwen3_14b ="hf:///workspace/Qwen3-14B-Base"
#    src_qwen3_8b ="hf:///workspace/Qwen3-8B-Base"
#    dst_qwen3_14b = "/workspace/nemo_restore/models/qwen3/Qwen3-14B"
#    dst_qwen3_8b = "/workspace/nemo_restore/models/qwen3/Qwen3-8B"
#    llm.import_ckpt(model=llm.Qwen3Model(llm.Qwen3Config14B()), source = src_qwen3_14b, output_path = dst_qwen3_14b )
#    llm.import_ckpt(model=llm.Qwen3Model(llm.Qwen3Config8B()), source = src_qwen3_8b, output_path = dst_qwen3_8b )
#    src_gptoss_20b = "hf:///workspace/gpt-oss-20b"
#    llm.import_ckpt(model=llm.GPTOSSModel(llm.GPTOSSConfig20B()), source = src_gptoss_20b, output_path = "/workspace/nemo_restore/models/gptoss")
    args = get_args()
    model_id = args.model_id
    if model_id == "llama3.2_1b":
        llm.import_ckpt(model=llm.LlamaModel(llm.Llama32Config1B()), source = args.src, output_path = args.dst) 
    elif model_id == "llama3.2_3b":
        llm.import_ckpt(model=llm.LlamaModel(llm.Llama32Config3B()), source = args.src, output_path = args.dst)
    elif model_id == "qwen3_0.6b":
        llm.import_ckpt(model=llm.Qwen3Model(llm.Qwen3Config600M()), source = args.src, output_path = args.dst)
    elif model_id == "qwen3_1.7b":
        llm.import_ckpt(model=llm.Qwen3Model(llm.Qwen3Config1P7B()), source = args.src, output_path = args.dst)
    elif model_id == "qwen3_4b":
        llm.import_ckpt(model=llm.Qwen3Model(llm.Qwen3Config4B()), source = args.src, output_path = args.dst)
    elif model_id == "qwen3_8b":
        llm.import_ckpt(model=llm.Qwen3Model(llm.Qwen3Config8B()), source = args.src, output_path = args.dst)
    elif model_id == "qwen3_14b":
        llm.import_ckpt(model=llm.Qwen3Model(llm.Qwen3Config14B()), source = args.src, output_path = args.dst)
    elif model_id == "gpt_oss_20b":
        llm.import_ckpt(model=llm.GPTOSSModel(llm.GPTOSSConfig20B()), source = args.src, output_path = args.dst)
    else:
        print(f"{model_id} is unsupported") 
