import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


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

# SST-2のラベル
label_text = {
    0: "negative",
    1: "positive",
}


# 問題96と同じ感情分析用のプロンプトを作る
def make_user_message(sentence):
    return (
        "Classify the sentiment of the following sentence as positive or negative. "
        "Answer with only one word: positive or negative.\n\n"
        f"Sentence: {sentence}"
    )


# プロンプトに対して正解ラベルを応答する学習データを作る
class SentimentDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        user_message = {
            "role": "user",
            "content": make_user_message(row["sentence"]),
        }
        assistant_message = {
            "role": "assistant",
            "content": label_text[row["label"]],
        }

        # モデルへの入力となるプロンプト
        prompt = tokenizer.apply_chat_template(
            [user_message],
            tokenize=False,
            add_generation_prompt=True,
        )

        # 正解ラベルまで含めた学習用テキスト
        full_text = tokenizer.apply_chat_template(
            [user_message, assistant_message],
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=256,
        )["input_ids"]

        input_ids = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=256,
        )["input_ids"]

        # 正解ラベル部分だけでLossを計算する
        labels = input_ids.copy()
        prompt_length = min(len(prompt_ids), len(labels))
        labels[:prompt_length] = [-100] * prompt_length

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }


# バッチ内の文章を同じ長さにそろえる
def collate_fn(batch):
    max_length = max(len(item["input_ids"]) for item in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        pad_length = max_length - len(item["input_ids"])

        input_ids.append(
            item["input_ids"]
            + [tokenizer.pad_token_id] * pad_length
        )
        attention_mask.append(
            item["attention_mask"]
            + [0] * pad_length
        )
        labels.append(
            item["labels"]
            + [-100] * pad_length
        )

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


train_dataset = SentimentDataset(train)

# ランク16のLoRAを用いて、少数の追加パラメータだけを学習する
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 学習条件
training_args = TrainingArguments(
    output_dir="out_98",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=50,
    save_strategy="no",
    report_to="none",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
)

# ファインチューニングを実行する
trainer.train()

# 学習したLoRAアダプタを保存する
save_dir = "fine_tuned_model_98"
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"\n学習済みモデルを {save_dir} に保存しました。")


# 学習後のモデルで簡単に動作確認する
model.eval()

test_messages = [
    {
        "role": "user",
        "content": make_user_message("This movie was absolutely wonderful."),
    }
]

test_prompt = tokenizer.apply_chat_template(
    test_messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(
    test_prompt,
    return_tensors="pt",
    add_special_tokens=False,
).to(model.device)

input_length = inputs["input_ids"].shape[1]

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=4,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(
    output_ids[0, input_length:],
    skip_special_tokens=True,
).strip()

print("\n動作確認:")
print("Sentence: This movie was absolutely wonderful.")
print(f"Prediction: {response}")