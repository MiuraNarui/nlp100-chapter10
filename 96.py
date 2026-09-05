import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# モデルを読み込む
model_name = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

# SST-2をダウンロードして展開する
data_dir = Path("data")
dev_path = data_dir / "SST-2" / "dev.tsv"
data_dir.mkdir(exist_ok=True)

if not dev_path.exists():
    zip_path = data_dir / "SST-2.zip"

    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip",
        zip_path,
    )

    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(data_dir)

# 開発データを読み込む
dev = pd.read_csv(dev_path, sep="\t")

# 感情分析用のプロンプトを作る
def make_prompt(sentence):
    messages = [{
        "role": "user",
        "content": (
            "Classify the sentiment of the following sentence as positive or negative. "
            "Answer with only one word: positive or negative.\n\n"
            f"Sentence: {sentence}"
        ),
    }]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

label_text = {0: "negative", 1: "positive"}
predictions = []
batch_size = 16

# 16文ずつまとめて推論する
for start in range(0, len(dev), batch_size):
    sentences = dev["sentence"].iloc[start:start + batch_size]
    prompts = [make_prompt(sentence) for sentence in sentences]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(model.device)

    input_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )

    for ids in outputs[:, input_length:]:
        response = tokenizer.decode(
            ids,
            skip_special_tokens=True,
        ).strip().lower()

        if "positive" in response:
            predictions.append("positive")
        elif "negative" in response:
            predictions.append("negative")
        else:
            predictions.append("unknown")

# 正解率を計算する
gold = [label_text[label] for label in dev["label"]]
correct = sum(p == g for p, g in zip(predictions, gold))
accuracy = correct / len(gold)

print(f"正解数: {correct}/{len(gold)}")
print(f"正解率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"判定不能件数: {predictions.count('unknown')}")