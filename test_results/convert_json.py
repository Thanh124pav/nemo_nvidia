import os 
import json 
import argparse 

def get_args():
    parser = argparse.ArgumentParser(description = "Code to convert initial response")
    parser.add_argument("--src-file", type=str, required = True, help = "src file")
    parser.add_argument("--dst-file", type=str, required = True, help = "dst file")
    parser.add_argument("--output-file", type=str, required = True, help = "output after modification")
    args = parser.parse_args()
    return args

def get_samples(file_path):
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines: 
        sample = json.loads(line.strip())
        samples.append(sample)
    return samples

def modify_samples(src_file, dst_file):
    src_samples = get_samples(src_file)
    dst_samples = get_samples(dst_file)
    res = []
    assert len(src_samples) == len(dst_samples), "Invalid src and dist files"
    for i in range(len(src_samples)):
        src_sample = src_samples[i]
        dst_sample = dst_samples[i] 
        question = src_sample['input']
        resp = dst_sample['sentence']
        pos = resp.find(question)
        if pos == -1:
            print("fjdsfsd")
            continue
        else:
            resp = resp[pos + len(question): ]
            data = {"inputs": question, "preds": resp, "label": src_sample['output'], "category": src_sample["category"]}
            res.append(data)
    return res

def save_data(dataset, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"save successfully, number of samples: {len(dataset)}")
    
if __name__ == "__main__":
    args = get_args()
    data = modify_samples(args.src_file, args.dst_file)
    save_data(data, args.output_file)