import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


# GPT型の事前学習済み言語モデルを読み込む
model_name = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
)

# SST-2をダウンロードして展開する
data_dir = Path("data")
train_path = data_dir / "SST-2" / "train.tsv"
data_dir.mkdir(exist_ok=True)

if not train_path.exists():
    zip_path = data_dir / "SST-2.zip"

    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip",
        zip_path,
    )

    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(data_dir)

train = pd.read_csv(train_path, sep="\t")


# 問題96と同じ感情分析用のプロンプト
def make_user_message(sentence):
    return (
        "Classify the sentiment of the following sentence as positive or negative. "
        "Answer with only one word: positive or negative.\n\n"
        f"Sentence: {sentence}"
    )


# DPO用の選好データを作る
# chosen   : 正しい感情ラベル
# rejected : 間違った感情ラベル
preference_data = []

for _, row in train.iterrows():
    correct = "positive" if row["label"] == 1 else "negative"
    incorrect = "negative" if row["label"] == 1 else "positive"

    preference_data.append(
        {
            "prompt": [
                {
                    "role": "user",
                    "content": make_user_message(row["sentence"]),
                }
            ],
            "chosen": [
                {
                    "role": "assistant",
                    "content": correct,
                }
            ],
            "rejected": [
                {
                    "role": "assistant",
                    "content": incorrect,
                }
            ],
        }
    )

train_dataset = Dataset.from_list(preference_data)

print(f"選好データ数: {len(train_dataset)}")
print("\n選好データの例:")
print("prompt  :", train_dataset[0]["prompt"])
print("chosen  :", train_dataset[0]["chosen"])
print("rejected:", train_dataset[0]["rejected"])


# ランク16のLoRAを用いる
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)


# DPOの学習条件
training_args = DPOConfig(
    output_dir="out_99",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    beta=0.1,
    bf16=True,
    max_length=256,
    logging_steps=50,
    save_strategy="no",
    report_to="none",
)


# DPOTrainerを作成する
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

# 選好チューニングを実行する
trainer.train()

# 学習したLoRAアダプタを保存する
save_dir = "preference_tuned_model_99"
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"\n選好チューニング済みモデルを {save_dir} に保存しました。")