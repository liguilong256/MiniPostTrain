import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_NUM_SAMPLES = 100
DEFAULT_MAX_NEW_TOKENS = 256
RESULTS_PATH = Path("results/gsm8k_results.jsonl")


def parse_args():
    """读取命令行参数。"""
    parser = argparse.ArgumentParser(description="在 GSM8K test 数据集上评估语言模型")
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Hugging Face 模型名称（默认：{DEFAULT_MODEL_NAME}）",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"评估前多少道题（默认：{DEFAULT_NUM_SAMPLES}）",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"每道题最多生成多少个新 token（默认：{DEFAULT_MAX_NEW_TOKENS}）",
    )
    args = parser.parse_args()

    if args.num_samples <= 0:
        parser.error("--num_samples 必须大于 0")
    if args.max_new_tokens <= 0:
        parser.error("--max_new_tokens 必须大于 0")

    return args


def build_prompt(question):
    """为一道数学题构造简单提示词。"""
    return (
        "Solve the following math problem step by step. "
        "End your response with the final answer in the format: #### number\n\n"
        f"Question: {question}"
    )


def format_for_model(tokenizer, prompt):
    """如果模型提供聊天模板，就按 instruct 模型需要的格式包装提示词。"""
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def normalize_number(number_text):
    """去掉千位逗号，并将 42.0 之类的写法统一为 42。"""
    cleaned = number_text.replace(",", "").strip()
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return None

    # format(..., "f") 可以避免 Decimal 输出科学计数法。
    normalized = format(number, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized == "-0":
        normalized = "0"
    return normalized


def extract_answer(text):
    """从文本中提取最终数字答案；找不到时返回 None。"""
    number_pattern = r"-?\d[\d,]*(?:\.\d+)?"

    # GSM8K 标准答案和我们的 prompt 都使用 #### 标记最终答案。
    marked_answers = re.findall(rf"####\s*\$?\s*({number_pattern})", text)
    if marked_answers:
        return normalize_number(marked_answers[-1])

    # 模型偶尔不遵守格式，此时使用它输出的最后一个数字作为答案。
    all_numbers = re.findall(number_pattern, text)
    if all_numbers:
        return normalize_number(all_numbers[-1])
    return None


def choose_device():
    """优先使用 CUDA；没有 GPU 时使用 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def generate_answer(model, tokenizer, prompt, device, max_new_tokens):
    """让模型为一道题生成回答，并且只返回新生成的部分。"""
    model_input = format_for_model(tokenizer, prompt)
    inputs = tokenizer(model_input, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]

    # evaluation 不需要计算梯度，可以节省显存和内存。
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = output_ids[0, prompt_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main():
    args = parse_args()

    # 从 Hugging Face 下载并加载 GSM8K 的 test split。
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    sample_count = min(args.num_samples, len(dataset))
    dataset = dataset.select(range(sample_count))

    device = choose_device()
    print(f"Loading model: {args.model_name}")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    # 如果 results 目录还不存在，就先创建它。
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    correct_count = 0
    with RESULTS_PATH.open("w", encoding="utf-8") as result_file:
        for example in tqdm(dataset, desc="Evaluating"):
            question = example["question"]
            prompt = build_prompt(question)
            model_output = generate_answer(
                model,
                tokenizer,
                prompt,
                device,
                args.max_new_tokens,
            )

            predicted_answer = extract_answer(model_output)
            ground_truth = extract_answer(example["answer"])
            # 没提取到模型答案时必须判错，不能让 None == None 被算作正确。
            is_correct = (
                predicted_answer is not None and predicted_answer == ground_truth
            )
            if is_correct:
                correct_count += 1

            # 每完成一道题就写入一行 JSON，方便中途查看结果。
            result = {
                "question": question,
                "model_output": model_output,
                "predicted_answer": predicted_answer,
                "ground_truth": ground_truth,
                "correct": is_correct,
            }
            result_file.write(json.dumps(result, ensure_ascii=False) + "\n")

    accuracy = correct_count / sample_count * 100
    print(f"Correct: {correct_count}")
    print(f"Total: {sample_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
