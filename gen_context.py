import random

base_contexts = [
    "Tóm tắt nội dung",
    "Viết bản tóm tắt cho đoạn văn bản này",
    "Tóm lược thông tin",
    "Viết phần tóm tắt cho văn bản",
    "Tóm tắt ý chính của đoạn văn bản",
    "Rút gọn nội dung văn bản",
    "Trình bày ngắn gọn nội dung",
    "Tổng hợp ý chính của văn bản",
    "Tóm tắt thông tin chính của đoạn văn",
    "Viết tóm tắt cho nội dung",
    "Tóm tắt đoạn nội dung",
    "Tóm lược đoạn văn bản này",
    "Tóm tắt các ý quan trọng trong văn bản",
    "Viết phần tổng kết cho đoạn văn",
    "Trình bày nội dung chính của đoạn văn",
    "Tóm tắt văn bản"
]

prefixes = [
    "Xin hãy", "Bạn có thể", "Vui lòng", "Đề nghị", "Yêu cầu", "Giúp tôi", "Làm ơn", "Mời bạn"
]
suffixes = [
    "một cách ngắn gọn.", "đầy đủ ý chính.", "cho người đọc dễ hiểu.", "bằng tiếng Việt.", "theo cách súc tích nhất.", "với độ dài tối thiểu.", "bằng một đoạn văn.", "một cách cô đọng"
]

def gen_context():
    contexts = set(base_contexts)
    while len(contexts) < 1000:
        prefix = random.choice(prefixes)
        base = random.choice(base_contexts)
        suffix = random.choice(suffixes)
        new_context = f"{prefix} {base.lower()} {suffix}"
        contexts.add(new_context)
    contexts_list = list(contexts)[:1000]
    return contexts_list
if __name__ == "__main__":
    contexts_list = gen_context()
    # Lưu ra file nếu cần
    with open("context_prompts.txt", "w", encoding="utf-8") as f:
        for ctx in contexts_list:
            f.write(ctx + "\n")