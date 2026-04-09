# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import shutil
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import os
import numpy as np
from datasets import load_dataset

from nemo.collections.llm.gpt.data.core import get_dataset_root
from nemo.collections.llm.gpt.data.fine_tuning import FineTuningDataModule
from nemo.lightning.io.mixin import IOMixin
from nemo.utils import logging
from gen_context import gen_context

from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.llm.gpt.data.packed_sequence import PackedSequenceSpecs

from pathlib import Path

import random

def get_contexts(contexts_file_txt):
    with open(contexts_file_txt, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
    contexts = []
    for line in lines:
        if line.strip():
            contexts.append(line)
    print(f"Get {len(contexts)} contexts")
    return contexts

class ThanhFinetuningDataModule(FineTuningDataModule, IOMixin):
    """A data module for fine-tuning on the abstractive summarization dataset.

    This class inherits from the `FineTuningDataModule` class and is specifically designed for fine-tuning models on the
    "databricks/databricks-dolly-15k" dataset. It handles data download, preprocessing, splitting, and preparing the data
    in a format suitable for training, validation, and testing.

    Args:
        force_redownload (bool, optional): Whether to force re-download the dataset even if it exists locally. Defaults to False.
        delete_raw (bool, optional): Whether to delete the raw downloaded dataset after preprocessing. Defaults to True.
        See FineTuningDataModule for the other args
    """
    def __init__(
        self,
        local_path = "datasets/finetune/processed/law",
        dataset_root="finetuned_datasets/law",
        seq_length: int = 4096,
        tokenizer: Optional["TokenizerSpec"] = None,
        micro_batch_size: int = 1,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        force_redownload: bool = False,
        delete_raw: bool = True,
        seed: int = 1234,
        memmap_workers: int = 1,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        packed_sequence_specs: Optional["PackedSequenceSpecs"] = None,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
        contexts_file = 'datasets/prompts.txt',
    ):
        self.force_redownload = force_redownload
        self.delete_raw = delete_raw

        super().__init__(
            dataset_root=dataset_root,
            seq_length=seq_length,
            tokenizer=tokenizer,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
            seed=seed,
            memmap_workers=memmap_workers,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            packed_sequence_specs=packed_sequence_specs,
            dataset_kwargs=dataset_kwargs,
        )
        self.local_path = local_path
        print("Start generating contexts")
        if contexts_file is not None:
            self.contexts = get_contexts(contexts_file)
        else:
            self.contexts = gen_context()
        print("Finish generating contexts")

    def prepare_data(self) -> None:
        if self.train_path.exists():
            print("data exists!")
            return
        if Path(self.local_path).is_file():
            dset = load_dataset("json", data_files={"train": self.local_path})
            self._preprocess_and_split_data(dset)
        else:
            train_path = self.local_path + '/train.jsonl'
            validation_path = self.local_path + '/val.jsonl'
            test_path = self.local_path + '/test.jsonl'
            dset = load_dataset("json", data_files={"train": train_path, "validation": validation_path, "test": test_path})
            self._preprocess_data(dset)
        super().prepare_data()


    def _preprocess_and_split_data(self, dset, train_ratio: float = 0.80, val_ratio: float = 0.15):
        logging.info(f"Preprocessing {self.__class__.__name__} to jsonl format and splitting...")

        test_ratio = 1 - train_ratio - val_ratio
        save_splits = {}
        dataset = dset.get('train')
        split_dataset = dataset.train_test_split(test_size=val_ratio + test_ratio, seed=self.seed)
        split_dataset2 = split_dataset['test'].train_test_split(
            test_size=test_ratio / (val_ratio + test_ratio), seed=self.seed
        )
        save_splits['training'] = split_dataset['train']
        save_splits['validation'] = split_dataset2['train']
        save_splits['test'] = split_dataset2['test']
        print("sjdbfdsbfd\n")
        for split_name, dataset in save_splits.items():
            os.makedirs(self.dataset_root, exist_ok = True)
            output_file = self.dataset_root / f"{split_name}.jsonl"
            with output_file.open("w", encoding="utf-8") as f:
                for example in dataset:
                    if(example['content']==""):
                        continue
                    # context = example["context"].strip()
                    context = random.choice(self.contexts)
                    if context != "":
                        instruction = example["content"].strip()
                        assert instruction != ""
                        _output = example["summary"]
                        if '[Văn bản gốc]' in context:
                            _input = context.replace('[Văn bản gốc]', instruction)
                        else: 
                            # Randomize context and instruction order.
                            context_first = np.random.randint(0, 10)
                            if context_first <= 7:
                                _input = f"{context}\n\n{instruction}"
                            else:
                                _input = f"{instruction}\n\n{context}"
                    else:
                        _input = example["content"]
                        _output = example["summary"]

                    f.write(json.dumps({"input": _input, "output": _output, 
                                        "category": example.get("category","all domain")}, ensure_ascii = False) + "\n")

            logging.info(f"{split_name} split saved to {output_file}")

        if self.delete_raw:
            for p in self.dataset_root.iterdir():
                if p.is_dir():
                    shutil.rmtree(p)
                elif '.jsonl' not in str(p.name):
                    p.unlink()

    def _preprocess_data(self, dset):
        logging.info(f"Preprocessing {self.__class__.__name__}")


        save_splits = {}
        train_dataset = dset.get('train')
        valid_dataset = dset.get('validation')
        test_dataset = dset.get('test')
        save_splits['training'] = train_dataset
        save_splits['validation'] = valid_dataset
        save_splits['test'] = test_dataset
        for split_name, dataset in save_splits.items():
            os.makedirs(self.dataset_root, exist_ok = True)
            output_file = self.dataset_root / f"{split_name}.jsonl"
            with output_file.open("w", encoding="utf-8") as f:
                for example in dataset:
                    if(example['input'].strip()==""):
                        continue
                    # context = example["context"].strip()
                    context = random.choice(self.contexts)
                    if context != "":
                        # Randomize context and instruction order.
                        context_first = np.random.randint(0, 2) == 0
                        if context_first:
                            instruction = example["input"].strip()
                            assert instruction != ""
                            _input = f"{context}\n\n{instruction}"
                            _output = example["output"]
                        else:
                            instruction = example["input"].strip()
                            assert instruction != ""
                            _input = f"{instruction}\n\n{context}"
                            _output = example["output"]
                    else:
                        _input = example["input"]
                        _output = example["output"]

                    f.write(json.dumps({"input": _input, "output": _output, 
                                        "category": example.get("category", "all domain")}, ensure_ascii = False) + "\n")

            logging.info(f"{split_name} split saved to {output_file}")

        if self.delete_raw:
            for p in self.dataset_root.iterdir():
                if p.is_dir():
                    shutil.rmtree(p)
                elif '.jsonl' not in str(p.name):
                    p.unlink()

if __name__ == "__main__": 
    dataModule = ThanhFinetuningDataModule(
        local_path="datasets/finetune/processed/law",
        dataset_root="finetuned_datasets/law",
        dataset_kwargs={
        "prompt_template": "Question: {input} Answer: {output}",  # default is "{input} {output}" (naive concatenation)
        "answer_only_loss": True,  # default is True (only calculate loss on answer/output)
        },
        # packed_sequence_specs=PackedSequenceSpecs(
        #     packed_sequence_size=4096,
        #     tokenizer_model_name="/home/dungdx4/BERT/Llama-3-8B-Instruct",
        # ),
    )
    dataModule.prepare_data()
    print("finish preprocessing")