import nemo_run as run
from nemo.collections import llm
from nemo import lightning as nl
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
import argparse
from datetime import timedelta
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
import subprocess
import os
'''
mv nemo nemo_25.11 
mv nemo_25.07 nemo
python tokenize_dataset_for_pretrain.py \
    --data-path datasets/raw_data/pretrain/finewiki datasets/raw_data/pretrain/sub_news_pretraining \
    --dataset-root datasets/pretrain_datasets/llama3/finewiki datasets/pretrain_datasets/llama3/news \
    --tokenizer-path /workspace/Llama-3.2-3B-Instruct
mv nemo nemo_25.07
mv nemo_25.11 nemo
'''
os.makedirs('datasets/pretrain_datasets/llama3', exist_ok=True)
def get_args():
    parser = argparse.ArgumentParser(description="Code to preprocess dataset")

    parser.add_argument("--data-path", type=str, nargs = "+",
                        help="path to raw data for preprocessing")
    parser.add_argument("--dataset-root", type=str, required=True, nargs = '+',
                        help="path to data for pretraining (preprocessed data, with .bin and .idx files)")
    parser.add_argument("--tokenizer-path", type=str, required=True, 
                        help="path to local hugginface model to get tokenizer")

    args = parser.parse_args()
    
    return args
def preprocess_single_dataset(data_path, dataset_root, tokenizer_path):
    print("start tokenizing datasets!\n")
    subprocess.run([
        "python", "scripts/nlp_language_modeling/preprocess_data_for_megatron.py",
        f"--input={data_path}",
        "--json-keys=text",
        "--tokenizer-library=huggingface",
        f"--tokenizer-type={tokenizer_path}",
        "--dataset-impl=mmap",
        f"--output-prefix={dataset_root}",
        "--preproc-folder",
        "--append-eod",
        "--workers=128"
    ])
    print(f"Finish preprocessing {data_path} dataset, saved at {dataset_root}")

def preprocess_datasets(args):
    assert len(args.data_path) == len(args.dataset_root), "Each dataset must have its own dataset root to save data"
    tokenizer_path = args.tokenizer_path
    for i in range(len(args.data_path)):
        data_path = args.data_path[i]
        dataset_root = args.dataset_root[i]
        preprocess_single_dataset(data_path, dataset_root, tokenizer_path)
    print(f"Finish preprocessing all datasets!")

if __name__ == "__main__":
    args = get_args()
    preprocess_datasets(args)
