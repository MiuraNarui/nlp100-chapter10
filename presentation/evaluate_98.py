"""
問98 発表用：学習済みモデルの評価

presentation/train_98.py で作成した学習済みLoRAを使い、
SST-2 dev 872件の正解率を求める。

保存:
    presentation/results/predictions_98.csv
    presentation/results/result_98.json

実行:
    uv run python presentation/evaluate_98.py
"""

import json
import re
from pathlib import Path

import pandas as pd
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# 1. 評価条件・パス
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEV_PATH = PROJECT_ROOT / "data" / "SST-2" / "dev.tsv"

# train_98.pyで作成した学習済みLoRA
MODEL_DIR = PROJECT_ROOT / "fine_tuned_model_98_presentation"

RESULTS_DIR = PROJECT_ROOT / "presentation" / "results"
TRAINING_JSON = RESULTS_DIR / "training_98.json"
PREDICTIONS_CSV = RESULTS_DIR / "predictions_98.csv"
RESULT_JSON = RESULTS_DIR / "result_98.json"

BATCH_SIZE = 16
MAX_LENGTH = 256
MAX_NEW_TOKENS = 4

LABEL_TEXT = {
    0: "negative",
    1: "positive",
}


# ============================================================
# 2. 問96・train_98.pyと同じプロンプト
# ============================================================

def make_user_message(sentence: str):
    return (
        "Classify the sentiment of the following sentence as positive or negative. "
        "Answer with only one word: positive or negative.\n\n"
        f"Sentence: {sentence}"
    )


# ============================================================
# 3. 学習済みモデルを読み込む
# ============================================================

def load_model():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"{MODEL_DIR} が見つかりません。\n"
            "先に presentation/train_98.py を実行してください。"
        )

    # LoRA設定から、元のLlamaモデル名を取得
    peft_config = PeftConfig.from_pretrained(str(MODEL_DIR))
    base_model_name = peft_config.base_model_name_or_path

    # train_98.pyで保存したTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 生成時のバッチ処理では左padding
    tokenizer.padding_side = "left"

    # 元のLlamaを読み込む
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    # 学習済みLoRAを読み込む
    model = PeftModel.from_pretrained(
        base_model,
        str(MODEL_DIR),
    )

    model.eval()

    return model, tokenizer, base_model_name


# ============================================================
# 4. 生成結果から positive / negative を取り出す
# ============================================================

def parse_prediction(response: str):
    match = re.search(
        r"\b(positive|negative)\b",
        response.lower(),
    )

    if match is None:
        return "unknown"

    return match.group(1)


# ============================================================
# 5. SST-2 dev を評価する
# ============================================================

@torch.inference_mode()
def evaluate(model, tokenizer, dev_df):

    results = []

    for start in range(0, len(dev_df), BATCH_SIZE):

        batch_df = dev_df.iloc[
            start:start + BATCH_SIZE
        ]

        prompts = []

        # 各文章をLlamaのチャット形式にする
        for sentence in batch_df["sentence"]:

            messages = [
                {
                    "role": "user",
                    "content": make_user_message(sentence),
                }
            ]

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            prompts.append(prompt)

        # 文章をtoken IDへ変換
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            add_special_tokens=False,
        )

        device = next(model.parameters()).device

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # 学習はせず、positive / negative を生成
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # 入力promptを除き、生成部分だけ取り出す
        input_length = inputs["input_ids"].shape[1]

        responses = tokenizer.batch_decode(
            generated[:, input_length:],
            skip_special_tokens=True,
        )

        # 正解ラベルと予測ラベルを比較
        for (_, row), response in zip(
            batch_df.iterrows(),
            responses,
        ):

            true_label = LABEL_TEXT[
                int(row["label"])
            ]

            prediction = parse_prediction(response)

            results.append(
                {
                    "sentence": row["sentence"],
                    "true_label": true_label,
                    "prediction": prediction,
                    "correct": prediction == true_label,
                    "response": response.strip(),
                }
            )

        done = min(
            start + BATCH_SIZE,
            len(dev_df),
        )

        print(
            f"\r評価中: {done}/{len(dev_df)}",
            end="",
            flush=True,
        )

    print()

    return pd.DataFrame(results)


# ============================================================
# 6. 正解率を計算して保存する
# ============================================================

def save_results(
    predictions_df,
    base_model_name,
):

    total = len(predictions_df)

    correct = int(
        predictions_df["correct"].sum()
    )

    accuracy = correct / total

    unknown = int(
        (
            predictions_df["prediction"]
            == "unknown"
        ).sum()
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 872件それぞれの予測結果
    predictions_df.to_csv(
        PREDICTIONS_CSV,
        index=False,
    )

    # train_98.pyで保存した学習結果も読み込む
    training_result = None

    if TRAINING_JSON.exists():
        with open(
            TRAINING_JSON,
            "r",
            encoding="utf-8",
        ) as f:
            training_result = json.load(f)

    # 学習結果と評価結果を1つのJSONにまとめる
    result = {
        "problem": 98,
        "method": "SFT + LoRA",
        "base_model": base_model_name,

        "training": training_result,

        "evaluation": {
            "dataset": "SST-2",
            "split": "dev",
            "samples": total,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "max_new_tokens": MAX_NEW_TOKENS,
            "decoding": "greedy",
            "correct": correct,
            "accuracy": accuracy,
            "unknown": unknown,
        },
    }

    with open(
        RESULT_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return correct, total, accuracy, unknown


# ============================================================
# 7. main
# ============================================================

def main():

    print("===== 問98 発表用評価 =====")

    if not DEV_PATH.exists():
        raise FileNotFoundError(
            f"{DEV_PATH} が見つかりません。"
        )

    # SST-2 dev 872件
    dev_df = pd.read_csv(
        DEV_PATH,
        sep="\t",
    )

    print(
        f"評価データ数: {len(dev_df)}"
    )

    # 学習済みLoRAを読み込む
    model, tokenizer, base_model_name = (
        load_model()
    )

    # 推論
    predictions_df = evaluate(
        model,
        tokenizer,
        dev_df,
    )

    # 正解率計算・保存
    correct, total, accuracy, unknown = (
        save_results(
            predictions_df,
            base_model_name,
        )
    )

    print("\n===== 評価結果 =====")
    print(
        f"正解数      : {correct}/{total}"
    )
    print(
        f"Accuracy    : {accuracy * 100:.2f}%"
    )
    print(
        f"判定不能件数: {unknown}"
    )

    print("\n===== 保存先 =====")
    print(
        f"予測結果: {PREDICTIONS_CSV}"
    )
    print(
        f"結果要約: {RESULT_JSON}"
    )


if __name__ == "__main__":
    main()
