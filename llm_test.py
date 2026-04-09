import json
import os
import aiohttp
import asyncio
import traceback
from transformers import AutoTokenizer

from tqdm.asyncio import tqdm

from abc import ABC, abstractmethod

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

class VllmCrawler(ABC):

    def __init__(
            self,
            endpoint_ip: str = "",
            model_name: str = "",
            eos_token: str = None,
            system_prompt: str = "",
            user_prompt_template: str = "",
            assistant_prompt_prefix: str = "",
    ):
        self.endpoint_ip = endpoint_ip
        self.model_name = model_name

        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.assistant_prompt_prefix = assistant_prompt_prefix

    @abstractmethod
    def extract_sample(sample) -> dict:
        pass

    async def get_answer(
            self,
            session,
            sample,
            max_new_tokens=768):

        user_prompt = self.user_prompt_template.format(
            **self.extract_sample(sample)
        )

        prompt = f"<|im_start|>system\n<|im_end|>\n<|im_start|>{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

        headers = {
            "Content-Type": "application/json",
            # "Host": "minimax-m2-fp8.llm-h200.viettelai.vn:30080"
            # "Host": "llama70b-fp8.llm-h200.viettelai.vn:30080"
            # "Host": "gptoss120b-fp8.llm-h200.viettelai.vn:30080"
            # "Host": "deepseek32-fp8.llm-h200.viettelai.vn:30080"
            "Host": "qwen235b-fp8.llm-h200.viettelai.vn:30080"
        }

        data = {
            "model": self.model_name,
            "prompt": prompt,
            "n": 1,
            "best_of": 1,
            "use_beam_search": False,
            "max_tokens": 8192,
            "repetition_penalty": 1.0,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 20,
            # "temperature": 0.6,
            # "top_p": 0.95,
            # "top_k": 20,
        }
        try:
            async with session.post(self.endpoint_ip.strip("/") + '/v1/completions', headers=headers, json=data,
                                    timeout=600000) as resp:
                resp = await resp.json()
                return prompt, resp["choices"][0]["text"]
        except:
            return prompt, "Failed: " + str(traceback.format_exc())

    async def generate_async(
            self,
            dataset: list,
            output_file: str = "tmp.jsonl",
            output_field: str = "output",
    ):
        semaphore = asyncio.BoundedSemaphore(16)
        active_count = 0  # ← thêm ở đây
        
        async def bounded_get_answer(session, sample):
            nonlocal active_count  # ← để access biến bên ngoài
            async with semaphore:
                active_count += 1
                print(f"Active requests: {active_count}")
                try:
                    return await self.get_answer(session, sample)
                finally:
                    active_count -= 1
        
        session_timeout = aiohttp.ClientTimeout(total=None)
        async with aiohttp.ClientSession(timeout=session_timeout) as session:
            tasks = []
            print(f"Len dataset: {len(dataset)}")
            for sample in dataset:
                tasks.append(asyncio.ensure_future(
                    bounded_get_answer(session, sample)
                ))
            results = await tqdm.gather(*tasks)
            print(f"Number of results: {len(results)}")
        
        with open(output_file, "w") as f:
            for sample, (input_prompt, output) in zip(dataset, results):
                record = {
                    **sample,
                    output_field: output,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return

import json
import asyncio
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="datasets/tiny",
        help="The absolute path to directory of .JSONL files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/tiny_gts",
        help="The absolute path to output file"
    )
    parser.add_argument(
        "--input_field",
        type=str,
        default="input",
        help="field to extract text"
    )
    args = parser.parse_args()
    
    # Sanity checks
    if args.input_dir is None or args.output_dir is None:
        raise ValueError("Need both a input file and a output file")

    return args


HEADER = r"""
"""

SYS_PROMPT = HEADER + r"""
"""

USER_PROMPT_TEMPLATE = """
Question: {question}
""".strip()


class VimathqaVllmCrawler(VllmCrawler):
    def extract_sample(self, sample) -> dict:
        question = sample
        return {
            "question": question,
        }


if __name__ == "__main__":

    args = parse_args()

    crawler = VimathqaVllmCrawler(
        endpoint_ip="http://10.254.138.192:9003",
        # model_name="MiniMax-M2",
        # model_name="Meta-Llama-3-70B-Instruct",
        # model_name="gpt-oss-120b",
        # model_name="DeepSeek-V3.2",
        model_name="Qwen3-235B-A22B-Instruct-2507",
        eos_token="<|eot_id|>",
        system_prompt=SYS_PROMPT,
        user_prompt_template=USER_PROMPT_TEMPLATE,
        assistant_prompt_prefix=""
    )

    folder = args.input_dir
    folder_out = args.output_dir
    os.makedirs(folder_out, exist_ok=True)
    
    for file in os.listdir(folder):
        print(file)
        file_path = os.path.join(folder, file)
        file_path_out = os.path.join(folder_out, file)
        files = os.listdir(folder_out)
        if file in files:
            continue

        data = open(file_path, "r", encoding="utf-8").readlines()
        input_dset = []
        for item in data:
            item = json.loads(item.strip())
            text = item[args.input_field]
            text = text.strip()
            words = text.split()
            new_text = " ".join(words)
            new_text = "Tóm tắt văn bản sau bằng tiếng Việt: " + new_text
            input_dset.append(new_text)

        asyncio.run(crawler.generate_async(
            dataset=input_dset,
            output_field="instruct_code",
            output_file=file_path_out
        ))

        # exit()






