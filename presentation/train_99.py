"""
問99 発表用：DPO + LoRA 学習

SST-2 train を使って Llama 3.2 1B Instruct を DPO で学習する。

各文章について
    chosen   = 正しい感情ラベル
    rejected = 間違った感情ラベル
を作り、「正しい回答をより好む」ように学習する。

保存:
    preference_tuned_model_99_presentation/
    presentation/results/training_99.json
    presentation/results/training_log_99.csv

実行:
    uv run python presentation/train_99.py
"""

import json
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer


# ============================================================
# 1. 学習条件
# ============================================================

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

SEED = 42
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4

TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
MAX_LENGTH = 256

# DPOでchosenとrejectedの差をどの程度強く学習するか
BETA = 0.1

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

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "SST-2"
    / "train.tsv"
)

# 元の99.pyとは別名で保存
MODEL_DIR = (
    PROJECT_ROOT
    / "preference_tuned_model_99_presentation"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "out_99_presentation"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "presentation"
    / "results"
)

RESULT_JSON = (
    RESULTS_DIR
    / "training_99.json"
)

LOG_CSV = (
    RESULTS_DIR
    / "training_log_99.csv"
)


# ============================================================
# 3. 問96と同じプロンプト
# ============================================================

def make_user_message(sentence: str):
    return (
        "Classify the sentiment of the following sentence as positive or negative. "
        "Answer with only one word: positive or negative.\n\n"
        f"Sentence: {sentence}"
    )


# ============================================================
# 4. SST-2をDPO用データへ変換
# ============================================================

def make_dpo_dataset(train_df):
    """
    DPOでは各promptに対して、
    chosen（好ましい回答）と rejected（好ましくない回答）を与える。

    SST-2では、
    正解ラベル   -> chosen
    反対ラベル   -> rejected
    とする。
    """

    examples = []

    for _, row in train_df.iterrows():

        correct = LABEL_TEXT[
            int(row["label"])
        ]

        incorrect = (
            "negative"
            if correct == "positive"
            else "positive"
        )

        examples.append(
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": make_user_message(
                            row["sentence"]
                        ),
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

    return Dataset.from_list(examples)


# ============================================================
# 5. 学習
# ============================================================

def main():

    # --------------------------------------------------------
    # SST-2 train
    # --------------------------------------------------------

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"{TRAIN_PATH} が見つかりません。"
        )

    # 既存の発表用モデルがある場合は上書きしない
    if MODEL_DIR.exists():
        raise FileExistsError(
            f"{MODEL_DIR} がすでに存在します。\n"
            "既存結果を保護するため停止しました。"
        )

    set_seed(SEED)

    train_df = pd.read_csv(
        TRAIN_PATH,
        sep="\t",
    )

    print(
        f"学習データ数: {len(train_df)}"
    )

    # SST-2を chosen / rejected 形式へ変換
    train_dataset = make_dpo_dataset(
        train_df
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

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
    # LoRA設定
    # --------------------------------------------------------

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --------------------------------------------------------
    # DPO設定
    # --------------------------------------------------------

    training_args = DPOConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=(
            TRAIN_BATCH_SIZE
        ),
        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),
        learning_rate=LEARNING_RATE,
        beta=BETA,
        bf16=True,
        max_length=MAX_LENGTH,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        seed=SEED,
        data_seed=SEED,
    )

    # DPOTrainerがLoRAをモデルへ追加する
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print(
        "\n学習対象パラメータ:"
    )

    trainer.model.print_trainable_parameters()

    # --------------------------------------------------------
    # DPO学習
    # --------------------------------------------------------

    print("\n===== DPO開始 =====")

    train_output = trainer.train()

    # --------------------------------------------------------
    # 学習済みLoRAを保存
    # --------------------------------------------------------

    trainer.save_model(
        str(MODEL_DIR)
    )

    tokenizer.save_pretrained(
        str(MODEL_DIR)
    )

    # --------------------------------------------------------
    # 学習ログを保存
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        trainer.state.log_history
    ).to_csv(
        LOG_CSV,
        index=False,
    )

    # DPOの最後のreward指標を取得
    reward_accuracy = None
    reward_margin = None

    for log in reversed(
        trainer.state.log_history
    ):
        if "rewards/accuracies" in log:
            reward_accuracy = log.get(
                "rewards/accuracies"
            )
            reward_margin = log.get(
                "rewards/margins"
            )
            break

    # --------------------------------------------------------
    # 学習条件・結果をJSON保存
    # --------------------------------------------------------

    result = {
        "problem": 99,
        "method": "DPO + LoRA",
        "base_model": MODEL_NAME,
        "dataset": "SST-2",
        "seed": SEED,

        "training_conditions": {
            "train_samples": len(train_df),
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "per_device_train_batch_size":
                TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps":
                GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size":
                TRAIN_BATCH_SIZE
                * GRADIENT_ACCUMULATION_STEPS,
            "max_length": MAX_LENGTH,
            "beta": BETA,
            "precision": "bfloat16",
        },

        "lora": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "target_modules":
                LORA_TARGET_MODULES,
        },

        "training_results": {
            "train_loss":
                train_output.metrics.get(
                    "train_loss"
                ),
            "train_runtime_sec":
                train_output.metrics.get(
                    "train_runtime"
                ),
            "reward_accuracy":
                reward_accuracy,
            "reward_margin":
                reward_margin,
        },

        "saved_model":
            "preference_tuned_model_99_presentation",
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

    print(
        f"Reward Accuracy: "
        f"{reward_accuracy}"
    )

    print(
        f"Reward Margin: "
        f"{reward_margin}"
    )

    print("\n保存先")
    print(
        f"LoRAモデル: {MODEL_DIR}"
    )
    print(
        f"学習結果  : {RESULT_JSON}"
    )
    print(
        f"学習ログ  : {LOG_CSV}"
    )


if __name__ == "__main__":
    main()