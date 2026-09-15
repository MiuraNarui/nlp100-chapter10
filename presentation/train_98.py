"""
問98 発表用：SFT + LoRA 学習
--------------------------------
SST-2 train を用いて Llama 3.2 1B Instruct を LoRA で学習する。

保存するもの
1. 学習済みLoRA        : fine_tuned_model_98/
2. 学習条件・結果      : presentation/results/training_98.json
3. 学習ログ            : presentation/results/training_log_98.csv

実行:
    uv run python presentation/train_98.py

既存モデルを上書きして再学習する場合:
    uv run python presentation/train_98.py --overwrite
"""

import argparse
import json
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ============================================================
# 1. 学習条件
# ============================================================

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

SEED = 42
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4
TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
MAX_LENGTH = 256

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

LABEL_TEXT = {
    0: "negative",
    1: "positive",
}


# ============================================================
# 2. パス
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
SST2_DIR = DATA_DIR / "SST-2"
SST2_ZIP = DATA_DIR / "SST-2.zip"
SST2_URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"

MODEL_DIR = PROJECT_ROOT / "fine_tuned_model_98_presentation"
OUTPUT_DIR = PROJECT_ROOT / "out_98_presentation"

RESULTS_DIR = PROJECT_ROOT / "presentation" / "results"
RESULT_JSON = RESULTS_DIR / "training_98.json"
LOG_CSV = RESULTS_DIR / "training_log_98.csv"


# ============================================================
# 3. 再現性
# ============================================================

def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


# ============================================================
# 4. SST-2 train の読み込み
# ============================================================

def load_train_data():
    train_path = SST2_DIR / "train.tsv"

    # データがなければ公式ZIPから取得
    if not train_path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        if not SST2_ZIP.exists():
            print("SST-2をダウンロードします...")
            urllib.request.urlretrieve(SST2_URL, SST2_ZIP)

        print("SST-2を展開します...")
        with zipfile.ZipFile(SST2_ZIP, "r") as zf:
            zf.extractall(DATA_DIR)

    return pd.read_csv(train_path, sep="\t")


# ============================================================
# 5. 問96と同じプロンプト
# ============================================================

def make_user_message(sentence: str):
    return (
        "Classify the sentiment of the following sentence as positive or negative. "
        "Answer with only one word: positive or negative.\n\n"
        f"Sentence: {sentence}"
    )


# ============================================================
# 6. SFT用データセット
# ============================================================

class SentimentDataset(Dataset):
    """
    user     : 感情分類の指示文
    assistant: 正解ラベル（positive / negative）

    user側はLoss計算から除外し、
    assistantの正解応答部分だけを学習する。
    """

    def __init__(self, dataframe, tokenizer):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        user_message = {
            "role": "user",
            "content": make_user_message(row["sentence"]),
        }

        assistant_message = {
            "role": "assistant",
            "content": LABEL_TEXT[int(row["label"])],
        }

        # assistantが答え始める直前まで
        prompt_text = self.tokenizer.apply_chat_template(
            [user_message],
            tokenize=False,
            add_generation_prompt=True,
        )

        # 正解応答まで含めた全文
        full_text = self.tokenizer.apply_chat_template(
            [user_message, assistant_message],
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LENGTH,
        )["input_ids"]

        full_ids = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LENGTH,
        )["input_ids"]

        labels = full_ids.copy()

        # prompt部分はLoss計算から除外
        prompt_length = min(len(prompt_ids), len(labels))
        labels[:prompt_length] = [-100] * prompt_length

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }


# ============================================================
# 7. バッチ内の長さをそろえる
# ============================================================

def make_collate_fn(tokenizer):

    def collate_fn(batch):
        max_len = max(len(x["input_ids"]) for x in batch)

        input_ids = []
        attention_mask = []
        labels = []

        for item in batch:
            pad_len = max_len - len(item["input_ids"])

            input_ids.append(
                item["input_ids"]
                + [tokenizer.pad_token_id] * pad_len
            )

            attention_mask.append(
                item["attention_mask"]
                + [0] * pad_len
            )

            labels.append(
                item["labels"]
                + [-100] * pad_len
            )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate_fn


# ============================================================
# 8. 学習
# ============================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存のfine_tuned_model_98を削除して再学習する",
    )
    args = parser.parse_args()

    # --------------------------------------------------------
    # 既存の学習済みモデルを保護
    # --------------------------------------------------------

    if MODEL_DIR.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"\n{MODEL_DIR} がすでに存在します。\n"
                "既存の学習結果を保護するため停止しました。\n"
                "再学習する場合のみ --overwrite を付けてください。"
            )

        shutil.rmtree(MODEL_DIR)

    fix_seed(SEED)

    # --------------------------------------------------------
    # データ
    # --------------------------------------------------------

    train_df = load_train_data()
    print(f"学習データ数: {len(train_df)}")

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # --------------------------------------------------------
    # Llama本体
    # --------------------------------------------------------

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    model.config.use_cache = False

    # --------------------------------------------------------
    # q_proj / v_proj にLoRAを追加
    # --------------------------------------------------------

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    print("\n学習対象パラメータ:")
    model.print_trainable_parameters()

    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    # --------------------------------------------------------
    # Dataset / Trainer
    # --------------------------------------------------------

    train_dataset = SentimentDataset(
        train_df,
        tokenizer,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        bf16=True,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        seed=SEED,
        data_seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=make_collate_fn(tokenizer),
    )

    # --------------------------------------------------------
    # SFT
    # --------------------------------------------------------

    print("\n===== SFT開始 =====")
    train_output = trainer.train()

    # --------------------------------------------------------
    # 学習済みLoRAを保存
    # --------------------------------------------------------

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    # --------------------------------------------------------
    # 学習ログをCSV保存
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        trainer.state.log_history
    ).to_csv(
        LOG_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # 学習条件と結果をJSON保存
    # --------------------------------------------------------

    result = {
        "problem": 98,
        "method": "SFT + LoRA",
        "base_model": MODEL_NAME,
        "dataset": "SST-2",
        "seed": SEED,

        "training_conditions": {
            "train_samples": len(train_df),
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "per_device_train_batch_size": TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size":
                TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
            "max_length": MAX_LENGTH,
            "precision": "bfloat16",
        },

        "lora": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "target_modules": LORA_TARGET_MODULES,
            "trainable_parameters": trainable_params,
            "total_parameters": total_params,
        },

        "training_results": {
            "train_loss":
                train_output.metrics.get("train_loss"),
            "train_runtime_sec":
                train_output.metrics.get("train_runtime"),
            "train_samples_per_second":
                train_output.metrics.get("train_samples_per_second"),
            "train_steps_per_second":
                train_output.metrics.get("train_steps_per_second"),
        },

        "saved_model": "fine_tuned_model_98",
    }

    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # 結果表示
    # --------------------------------------------------------

    print("\n===== 学習完了 =====")
    print(
        f"Train Loss: "
        f"{train_output.metrics.get('train_loss')}"
    )
    print(
        f"Training Time: "
        f"{train_output.metrics.get('train_runtime')} sec"
    )

    print("\n保存先")
    print(f"LoRAモデル : {MODEL_DIR}")
    print(f"学習結果    : {RESULT_JSON}")
    print(f"学習ログ    : {LOG_CSV}")


if __name__ == "__main__":
    main()