import subprocess
import sys
from rouge_score import rouge_scorer
subprocess.check_call([sys.executable, "-m", "pip", "install", "bert-score" ,"summac", "openai"])
import bert_score
from summac.model_summac import SummaCZS
import openai

# ===== HÀM ĐÁNH GIÁ =====
def evaluate_summaries(data, references=None, use_g_eval=False, openai_api_key=None):
    """
    data: list các dict {"input": source_text, "output": model_summary}
    references: list tóm tắt chuẩn (human reference), hoặc None nếu không có
    use_g_eval: bool, nếu True sẽ gọi OpenAI GPT-4o để chấm điểm G-Eval
    openai_api_key: API key OpenAI (bắt buộc nếu use_g_eval=True)
    """
    results = []

    # Init scorers
    rouge = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    summaC = SummaCZS(granularity="paragraph", model_name="vitc")

    if use_g_eval and openai_api_key:
        openai.api_key = openai_api_key

    # Lặp qua từng sample
    for idx, sample in enumerate(data):
        source = sample["input"]
        candidate = sample["output"]
        reference = references[idx] if references else None

        metrics = {}

        # ---- ROUGE ----
        if reference:
            r_scores = rouge.score(reference, candidate)
            metrics["ROUGE-1_F"] = r_scores["rouge1"].fmeasure
            metrics["ROUGE-2_F"] = r_scores["rouge2"].fmeasure
            metrics["ROUGE-L_F"] = r_scores["rougeL"].fmeasure

        # ---- BERTScore ----
        # if reference:
        #     _, _, f1 = bert_score.score([candidate],[reference], lang="en")
        #     metrics["BERTScore_F1"] = f1.mean().item()

        # ---- SummaC factuality ----
        # s_score = summaC.score([source],[candidate])["scores"][0]
        # metrics["SummaC_Fact"] = s_score

        # ---- G-Eval ----
        if use_g_eval and openai_api_key:
            prompt = f"""
            You are an expert summarization evaluator.
            Source: {source}
            Summary: {candidate}

            Rate the summary from 1 to 5 on:
            1. Relevance
            2. Coherence
            3. Factual consistency
            4. Fluency

            Answer strictly in JSON with keys: relevance, coherence, factuality, fluency
            """
            resp = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role":"user","content":prompt}],
                temperature=0
            )
            try:
                import json
                metrics["G-Eval"] = json.loads(resp.choices[0].message["content"])
            except:
                metrics["G-Eval"] = resp.choices[0].message["content"]

        results.append(metrics)

    return results
if __name__ == "__main__":
    data = [
        {
            "input": "Hai người thân của em cũng đang bị ốm và hiện còn đang đợi kết quả xét nghiệm cho biết liệu họ có bị cúm hay không. Bệnh cúm gà đã làm ít nhất 76 người tại châu Á tử vong kể từ khi bùng phát hồi tháng 12 năm 2003. Các nước khác như Thổ Nhĩ Kỳ cũng bị ảnh hưởng. Hàng triệu con gà vịt đã chết bệnh hoặc bị tiêu hủy để phòng bệnh dịch lây lan. Các chuyên gia lo ngại rằng virus cúm gà có thể biến chuyển thành dạng có thể truyền từ người sang người và gây ra một đại dịch trên diện rộng. Sắp Năm mới Trường hợp tử vong mới nhất vào hôm thứ Bảy là một em gái 13 tuổi tại Indraymayu trên đảo Java. Quan chức Bộ y tế Indonesia Hariadi Wibisono nói các xét nghiệm cho thấy em nhiễm virus H5N1, tuy nhiên còn phải chờ xét nghiệm của WHO tại Hong Kong. Ở chính Hong Kong các biện pháp cũng đang được tăng cường để phòng nguy cơ xảy ra dịch cúm. Nhà chức trách đã hạn chế số lượng gà sống nhập khẩu từ lục địa xuống còn 30.000 và tất cả số này đều phải được tiêm phòng. Tuy nhiên trong dịp Tết âm lịch sắp tới số gà nhập khẩu sẽ phải tăng lên vì nhu cầu tiêu thụ, có thể tới 50 hoặc 70 ngàn. Các khoa học gia cảnh báo càng nhiều gà vịt được chuyên chở bằng lồng tại các nước trong châu Á thì nguy cơ nạn dịch cúm gà Họ yêu cầu người dân cẩn trọng trong việc tiêu thụ hoặc tiếp xúc với các loại gà vịt ốm bệnh thế nhưng nói ăn thịt gà đã nấu chín thì không có vấn đề gì.",
            "output": "Indonesia cho biết một em gái vừa tử vong vì cúm gia cầm, đây là người thứ 13 thiệt mạng do căn bệnh này nếu như được xác nhận bởi Tổ chức Y tế Thế giới."
        }
    ]
    references = ["Các quan chức Indonesia cho biết một em gái vừa tử vong vì cúm gia cầm và là người thứ 13 thiệt mạng vì căn bệnh này nếu như kết quả xét nghiệm được Tổ chức Y tế Thế giới xác nhận"]
    scores = evaluate_summaries(
        data, 
        references=references,
        use_g_eval=False  # đổi True nếu muốn dùng GPT-4o
    )

    print(scores)
