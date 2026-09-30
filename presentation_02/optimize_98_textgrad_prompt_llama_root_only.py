"""
問98 発表用：Llama 3.2 1B Instruct + TextGrad Prompt Search

Llama 3.2 1B Instruct の Prompt-only 性能を、パラメータ更新なしで改善する。
Unknown（invalid output）と binary wrong、Parent→Child の fixed / regressed を
分離して分析し、Prompt の内容と構造を自動探索する。

Prompt形式は自由。箇条書き・文章・System only・System+User等を人手で指定しない。
few-shot、学習文の埋め込みは禁止。SST original の train root 文だけを使い、
部分木 phrase と neutral（0.4 < score <= 0.6）は除外する。

Default:
- Optimization / Validation / Prompt-check = 各2000（各ラベル1000/1000）
- Multi-start 10, Branch 4, Beam 4, Depth 3
- Smoke 128
- Accuracy early stop なし

実行:
uv run python presentation_02/optimize_98_textgrad_prompt_llama_root_only.py --max-api-cost-usd 15
"""

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import textgrad as tg
from textgrad.engine import EngineLM


# ============================================================
# 1. 共通設定
# ============================================================

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
SEED = 42

# ③ System + User と同じ Root。
# User instruction は空なので、User message は "Sentence: ..." のみ。
INITIAL_SYSTEM_PROMPT = (
    "You are a sentiment classification model for movie reviews. "
    "Classify each review as positive or negative. "
    "Return only one word: positive or negative."
)
INITIAL_USER_INSTRUCTION = ""

LABEL_TEXT = {
    0: "negative",
    1: "positive",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SST_ORIGINAL_DIR = PROJECT_ROOT / "data" / "SST-2" / "original"
DATASET_SENTENCES_PATH = SST_ORIGINAL_DIR / "datasetSentences.txt"
DATASET_SPLIT_PATH = SST_ORIGINAL_DIR / "datasetSplit.txt"
DICTIONARY_PATH = SST_ORIGINAL_DIR / "dictionary.txt"
SENTIMENT_LABELS_PATH = SST_ORIGINAL_DIR / "sentiment_labels.txt"

RESULT_DIR = (
    PROJECT_ROOT
    / "presentation_02"
    / "result"
    / "textgrad_prompt_llama32_1b_root_only"
)

BEST_PROMPT_PATH = RESULT_DIR / "best_prompt.json"
BEST_PROMPT_TXT_PATH = RESULT_DIR / "best_prompt.txt"
PROMPT_TREE_PATH = RESULT_DIR / "prompt_tree.csv"
BATCH_FEEDBACK_PATH = RESULT_DIR / "batch_feedback.csv"
GLOBAL_SYNTHESIS_PATH = RESULT_DIR / "global_synthesis.csv"
DELTA_FEEDBACK_PATH = RESULT_DIR / "delta_feedback.csv"
GRADIENTS_PATH = RESULT_DIR / "textual_gradients.csv"
SPLIT_MANIFEST_PATH = RESULT_DIR / "textgrad_split.csv"
ROOT_DATA_PATH = RESULT_DIR / "root_only_train.csv"
DATA_SUMMARY_PATH = RESULT_DIR / "data_summary.json"
API_USAGE_PATH = RESULT_DIR / "api_usage.csv"
RESULT_PATH = RESULT_DIR / "optimization_result.json"
PROMPT_CHECK_PATH = RESULT_DIR / "prompt_check_results.csv"


# ============================================================
# 2. .env
# ============================================================

def load_simple_env(env_path: Path) -> None:
    """単純な KEY=VALUE 形式の .env を読む。"""

    if not env_path.exists():
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if (
                len(value) >= 2
                and value[0] == value[-1]
                and value[0] in {"'", '"'}
            ):
                value = value[1:-1]

            os.environ.setdefault(key, value)


# ============================================================
# 3. 再現性
# ============================================================

def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 4. Prompt Design の表現
# ============================================================

def make_design_text(
    system_prompt: str,
    user_instruction: str,
) -> str:
    """
    TextGrad が1つの Variable として扱えるよう、
    System / User instruction を構造化した1文字列へする。
    """

    return (
        "<SYSTEM_PROMPT>\n"
        f"{system_prompt.strip()}\n"
        "</SYSTEM_PROMPT>\n"
        "<USER_INSTRUCTION>\n"
        f"{user_instruction.strip()}\n"
        "</USER_INSTRUCTION>"
    )


def parse_design_text(text: str) -> tuple[str, str]:
    """構造化 Prompt Design から System / User を取り出す。"""

    system_match = re.search(
        r"<SYSTEM_PROMPT>\s*(.*?)\s*</SYSTEM_PROMPT>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    user_match = re.search(
        r"<USER_INSTRUCTION>\s*(.*?)\s*</USER_INSTRUCTION>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if system_match is None or user_match is None:
        raise ValueError(
            "SYSTEM_PROMPT / USER_INSTRUCTION タグを解析できません。"
        )

    return (
        system_match.group(1).strip(),
        user_match.group(1).strip(),
    )


def make_user_prompt(
    sentence: str,
    user_instruction: str,
) -> str:
    """
    {sentence} 自体は最適化対象にしない。
    User instruction が空なら、第2版と完全に同じ "Sentence: ..." になる。
    """

    if user_instruction.strip():
        return (
            f"{user_instruction.strip()}\n\n"
            f"Sentence: {sentence}"
        )

    return f"Sentence: {sentence}"


def normalize_design(
    system_prompt: str,
    user_instruction: str,
) -> str:
    text = make_design_text(system_prompt, user_instruction)
    return re.sub(r"\s+", " ", text.strip()).lower()


def output_contract_is_present(
    system_prompt: str,
    user_instruction: str,
) -> bool:
    """
    positive / negative の2値かつ単一ラベル出力という条件が
    Prompt Design のどこかに残っているかを簡易確認する。
    """

    text = f"{system_prompt}\n{user_instruction}".lower()

    if "positive" not in text or "negative" not in text:
        return False

    output_markers = [
        "one word",
        "one label",
        "exactly one",
        "return only",
        "respond only",
        "output only",
        "only one",
    ]

    return any(marker in text for marker in output_markers)


def validate_design(
    system_prompt: str,
    user_instruction: str,
) -> tuple[bool, str]:
    """
    タスク自体を変える Candidate を早期に除外する。
    """

    if not system_prompt.strip() and not user_instruction.strip():
        return False, "empty_prompt_design"

    if "{sentence}" in system_prompt.lower():
        return False, "sentence_placeholder_in_system"

    if "{sentence}" in user_instruction.lower():
        return False, "sentence_placeholder_in_user_instruction"

    if not output_contract_is_present(
        system_prompt,
        user_instruction,
    ):
        return False, "output_contract_missing"

    return True, ""


# ============================================================
# 5. ローカル Llama Engine
# ============================================================

class HuggingFaceLlamaEngine(EngineLM):
    """
    Prompt-only 評価用のローカル Llama 3.2 1B Instruct。

    モデル重みは更新せず、System / User Promptだけを探索する。
    SST-2 sentence はコード側で User message に挿入する。
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 8,
        max_length: int = 256,
        max_new_tokens: int = 8,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU が見つかりません。")

        self.model_string = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

        print(f"分類モデルを読み込みます: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        # Llama Instructには標準でpad tokenがない場合があるため、
        # generation用途ではEOSをpadとして用いる。
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(
            "Tokenizer: "
            f"padding_side={self.tokenizer.padding_side}, "
            f"truncation_side={self.tokenizer.truncation_side}, "
            f"pad_token={self.tokenizer.pad_token!r}, "
            f"pad_token_id={self.tokenizer.pad_token_id}"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

        print(
            "Llama loaded: "
            f"device={next(self.model.parameters()).device}, "
            f"dtype={next(self.model.parameters()).dtype}"
        )

    def _make_chat_text(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: list[str],
        system_prompt: Optional[str] = None,
    ) -> list[str]:
        responses = []

        for start in range(
            0,
            len(prompts),
            self.batch_size,
        ):
            batch_prompts = prompts[
                start:start + self.batch_size
            ]

            chat_texts = [
                self._make_chat_text(
                    user_prompt=prompt,
                    system_prompt=system_prompt,
                )
                for prompt in batch_prompts
            ]

            inputs = self.tokenizer(
                chat_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )

            device = next(
                self.model.parameters()
            ).device

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
                if torch.is_tensor(value)
            }

            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            input_length = inputs[
                "input_ids"
            ].shape[1]

            decoded = self.tokenizer.batch_decode(
                generated[:, input_length:],
                skip_special_tokens=True,
            )

            responses.extend(
                response.strip()
                for response in decoded
            )

        return responses

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        return self.generate_batch(
            [prompt],
            system_prompt=system_prompt,
        )[0]

    def __call__(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        return self.generate(
            prompt,
            system_prompt=system_prompt,
            **kwargs,
        )


# ============================================================
# 6. API 使用量ロガー
# ============================================================

class APICostLimitExceeded(RuntimeError):
    pass


class LoggedEngine(EngineLM):
    """
    TextGrad / Candidate生成 / backward で使う API Engine を包み、
    各callの概算token数と概算料金を記録する。

    TextGrad内部から呼ばれる backward engine もこの wrapper を使うため、
    同一wrapper経由のcallは記録される。
    """

    def __init__(
        self,
        base_engine,
        csv_path: Path,
        input_price_per_million: float,
        output_price_per_million: float,
        max_cost_usd: float = 0.0,
    ):
        self.base_engine = base_engine
        self.model_string = getattr(
            base_engine,
            "model_string",
            "logged-api-engine",
        )

        self.csv_path = csv_path
        self.input_price_per_million = input_price_per_million
        self.output_price_per_million = output_price_per_million
        self.max_cost_usd = max_cost_usd

        self.context = "unspecified"
        self.records = []
        self.cumulative_cost_usd = 0.0
        self.call_index = 0

        self._encoding = None

        try:
            import tiktoken

            try:
                self._encoding = tiktoken.encoding_for_model(
                    "gpt-4o"
                )
            except Exception:
                self._encoding = tiktoken.get_encoding(
                    "o200k_base"
                )
        except Exception:
            self._encoding = None

    def set_context(self, context: str) -> None:
        self.context = context

    def _count_tokens(self, text: str) -> int:
        if self._encoding is not None:
            return len(self._encoding.encode(text))

        # tiktoken がない場合の安全な概算
        return max(1, math.ceil(len(text) / 4))

    def save(self) -> None:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.records).to_csv(
            self.csv_path,
            index=False,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        if (
            self.max_cost_usd > 0
            and self.cumulative_cost_usd >= self.max_cost_usd
        ):
            raise APICostLimitExceeded(
                f"API推定累積料金が上限 "
                f"${self.max_cost_usd:.2f} に到達しました。"
            )

        full_input = (
            f"{system_prompt or ''}\n\n{prompt}"
        )

        input_tokens = self._count_tokens(full_input)

        response = self.base_engine(
            prompt,
            system_prompt=system_prompt,
            **kwargs,
        )

        output_tokens = self._count_tokens(str(response))

        input_cost = (
            input_tokens
            * self.input_price_per_million
            / 1_000_000
        )
        output_cost = (
            output_tokens
            * self.output_price_per_million
            / 1_000_000
        )
        call_cost = input_cost + output_cost

        self.cumulative_cost_usd += call_cost
        self.call_index += 1

        self.records.append(
            {
                "call_index": self.call_index,
                "context": self.context,
                "input_tokens_est": input_tokens,
                "output_tokens_est": output_tokens,
                "input_cost_usd_est": input_cost,
                "output_cost_usd_est": output_cost,
                "call_cost_usd_est": call_cost,
                "cumulative_cost_usd_est":
                    self.cumulative_cost_usd,
                "token_counter": (
                    "tiktoken"
                    if self._encoding is not None
                    else "chars_div_4_fallback"
                ),
            }
        )

        self.save()

        if (
            self.max_cost_usd > 0
            and self.cumulative_cost_usd > self.max_cost_usd
        ):
            raise APICostLimitExceeded(
                f"API推定累積料金が上限 "
                f"${self.max_cost_usd:.2f} を超えました。"
            )

        return response

    def __call__(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs,
        )


# ============================================================
# 7. 出力ラベル
# ============================================================

def parse_prediction(response: str) -> str:
    match = re.search(
        r"\b(positive|negative)\b",
        response.lower(),
    )

    if match is None:
        return "unknown"

    return match.group(1)


# ============================================================
# 8. Optimization / Validation 固定分割
# ============================================================

def normalize_sst_phrase(text: str) -> str:
    """datasetSentences と dictionary の表記差だけを吸収する。"""
    normalized = str(text)
    # datasetSentences.txt の一部には UTF-8 を Latin-1 として読んだ
    # mojibake が残るため、変換可能な場合だけ元へ戻す。
    try:
        normalized = normalized.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    normalized = normalized.replace("\\/", "/")
    replacements = {
        "-LRB-": "(",
        "-RRB-": ")",
        "-LSB-": "[",
        "-RSB-": "]",
        "-LCB-": "{",
        "-RCB-": "}",
    }
    for before, after in replacements.items():
        normalized = normalized.replace(before, after)
    return " ".join(normalized.split())


def load_root_only_binary_train() -> pd.DataFrame:
    """SST original の train root 文を読み、binary label を付与する。"""
    required_paths = [
        DATASET_SENTENCES_PATH,
        DATASET_SPLIT_PATH,
        DICTIONARY_PATH,
        SENTIMENT_LABELS_PATH,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "SST original の必要ファイルが見つかりません:\n- "
            + "\n- ".join(missing)
        )

    sentences = pd.read_csv(
        DATASET_SENTENCES_PATH,
        sep="\t",
        keep_default_na=False,
    )
    sentences.columns = ["sentence_index", "sentence"]

    split_info = pd.read_csv(DATASET_SPLIT_PATH)
    split_info.columns = ["sentence_index", "splitset_label"]
    train_roots = sentences.merge(
        split_info,
        on="sentence_index",
        how="inner",
        validate="one_to_one",
    )
    train_roots = train_roots[train_roots["splitset_label"] == 1].copy()

    exact_phrase_to_id: dict[str, int] = {}
    normalized_phrase_to_id: dict[str, int] = {}
    ambiguous_normalized: set[str] = set()
    with open(DICTIONARY_PATH, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            phrase, phrase_id_text = line.rsplit("|", 1)
            phrase_id = int(phrase_id_text)
            exact_phrase_to_id[phrase] = phrase_id
            key = normalize_sst_phrase(phrase)
            previous = normalized_phrase_to_id.get(key)
            if previous is not None and previous != phrase_id:
                ambiguous_normalized.add(key)
            else:
                normalized_phrase_to_id[key] = phrase_id

    def find_phrase_id(sentence: str) -> Optional[int]:
        exact = exact_phrase_to_id.get(sentence)
        if exact is not None:
            return exact
        key = normalize_sst_phrase(sentence)
        if key in ambiguous_normalized:
            return None
        return normalized_phrase_to_id.get(key)

    train_roots["phrase_id"] = train_roots["sentence"].map(find_phrase_id)
    unmatched = train_roots[train_roots["phrase_id"].isna()]
    if len(unmatched):
        examples = unmatched["sentence"].head(5).tolist()
        raise ValueError(
            f"root 文 {len(unmatched)}件を dictionary.txt に照合できませんでした。"
            f" 例: {examples}"
        )
    train_roots["phrase_id"] = train_roots["phrase_id"].astype(int)
    train_roots["sentence_raw"] = train_roots["sentence"]
    train_roots["sentence"] = train_roots["sentence"].map(normalize_sst_phrase)

    labels = pd.read_csv(SENTIMENT_LABELS_PATH, sep="|")
    labels.columns = ["phrase_id", "sentiment_value"]
    train_roots = train_roots.merge(
        labels,
        on="phrase_id",
        how="left",
        validate="many_to_one",
    )
    if train_roots["sentiment_value"].isna().any():
        raise ValueError("root 文の sentiment value に欠損があります。")

    # SST-2 の定義: score <= 0.4 は negative、score > 0.6 は positive。
    binary = train_roots[
        (train_roots["sentiment_value"] <= 0.4)
        | (train_roots["sentiment_value"] > 0.6)
    ].copy()
    binary["label"] = (binary["sentiment_value"] > 0.6).astype(int)
    binary["source_index"] = binary["sentence_index"].astype(int)
    binary["data_scope"] = "root_only"
    return binary[
        [
            "source_index",
            "sentence_index",
            "phrase_id",
            "sentence",
            "sentence_raw",
            "sentiment_value",
            "label",
            "splitset_label",
            "data_scope",
        ]
    ].reset_index(drop=True)

def make_fixed_split(
    train_df: pd.DataFrame,
    optimization_size: int,
    validation_size: int,
    prompt_check_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Optimization / Validation / Prompt-check を層化抽出し重複させない。"""
    for name, size in [
        ("optimization_size", optimization_size),
        ("validation_size", validation_size),
        ("prompt_check_size", prompt_check_size),
    ]:
        if size % 2 != 0:
            raise ValueError(f"{name} は偶数にしてください。")

    opt_n = optimization_size // 2
    val_n = validation_size // 2
    check_n = prompt_check_size // 2
    opt_parts, val_parts, check_parts = [], [], []

    for label in [0, 1]:
        label_df = (
            train_df[train_df["label"] == label]
            .sample(frac=1, random_state=seed + label)
            .reset_index(drop=True)
        )
        required = opt_n + val_n + check_n
        if len(label_df) < required:
            raise ValueError(f"label={label} のデータ数が不足しています。")
        opt_parts.append(label_df.iloc[:opt_n])
        val_parts.append(label_df.iloc[opt_n:opt_n + val_n])
        check_parts.append(label_df.iloc[opt_n + val_n:required])

    optimization_df = pd.concat(opt_parts, ignore_index=True).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)
    validation_df = pd.concat(val_parts, ignore_index=True).sample(
        frac=1, random_state=seed + 100
    ).reset_index(drop=True)
    prompt_check_df = pd.concat(check_parts, ignore_index=True).sample(
        frac=1, random_state=seed + 200
    ).reset_index(drop=True)
    return optimization_df, validation_df, prompt_check_df


def save_split_manifest(
    optimization_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    prompt_check_df: pd.DataFrame,
) -> None:
    manifest_columns = [
        "source_index",
        "sentence_index",
        "phrase_id",
        "sentence",
        "sentence_raw",
        "sentiment_value",
        "label",
        "data_scope",
    ]
    opt = optimization_df[manifest_columns].copy()
    opt["split"] = "optimization"
    val = validation_df[manifest_columns].copy()
    val["split"] = "validation"
    check = prompt_check_df[manifest_columns].copy()
    check["split"] = "prompt_check"
    pd.concat([opt, val, check], ignore_index=True).to_csv(
        SPLIT_MANIFEST_PATH,
        index=False,
    )


# ============================================================
# 9. Prompt評価
# ============================================================

def evaluate_prompt(
    dataframe: pd.DataFrame,
    engine: HuggingFaceLlamaEngine,
    system_prompt: str,
    user_instruction: str,
) -> tuple[float, pd.DataFrame]:
    user_prompts = [
        make_user_prompt(
            sentence=sentence,
            user_instruction=user_instruction,
        )
        for sentence in dataframe["sentence"]
    ]

    raw_responses = engine.generate_batch(
        user_prompts,
        system_prompt=system_prompt,
    )

    records = []

    for (_, row), raw_response in zip(
        dataframe.iterrows(),
        raw_responses,
    ):
        true_label = LABEL_TEXT[int(row["label"])]
        prediction = parse_prediction(raw_response)

        if prediction == "unknown":
            error_type = "unknown"
        elif prediction != true_label:
            error_type = (
                f"{true_label}_to_{prediction}"
            )
        else:
            error_type = "correct"

        records.append(
            {
                "source_index":
                    int(row["source_index"]),
                "sentence":
                    row["sentence"],
                "true_label":
                    true_label,
                "prediction":
                    prediction,
                "error_type":
                    error_type,
                "correct":
                    prediction == true_label,
                "response":
                    raw_response,
            }
        )

    result_df = pd.DataFrame(records)
    accuracy = float(
        result_df["correct"].mean()
    )

    return accuracy, result_df


def make_balanced_smoke_df(
    validation_df: pd.DataFrame,
    smoke_size: int,
    seed: int,
) -> pd.DataFrame:
    smoke_size = min(
        smoke_size,
        len(validation_df),
    )

    half = smoke_size // 2

    negative = (
        validation_df[
            validation_df["label"] == 0
        ]
        .sample(
            n=min(
                half,
                int(
                    (
                        validation_df["label"] == 0
                    ).sum()
                ),
            ),
            random_state=seed,
        )
    )

    positive_target = (
        smoke_size - len(negative)
    )

    positive = (
        validation_df[
            validation_df["label"] == 1
        ]
        .sample(
            n=min(
                positive_target,
                int(
                    (
                        validation_df["label"] == 1
                    ).sum()
                ),
            ),
            random_state=seed + 1,
        )
    )

    smoke = pd.concat(
        [negative, positive],
        ignore_index=True,
    )

    if len(smoke) < smoke_size:
        used = set(
            smoke["source_index"].tolist()
        )

        remainder = (
            validation_df[
                ~validation_df[
                    "source_index"
                ].isin(used)
            ]
            .sample(
                frac=1,
                random_state=seed + 2,
            )
        )

        smoke = pd.concat(
            [
                smoke,
                remainder.head(
                    smoke_size - len(smoke)
                ),
            ],
            ignore_index=True,
        )

    return (
        smoke
        .sample(
            frac=1,
            random_state=seed + 3,
        )
        .reset_index(drop=True)
    )


# ============================================================
# 10. Token長
# ============================================================

def get_prompt_length_metrics(
    system_prompt: str,
    user_instruction: str,
    length_check_df: pd.DataFrame,
    engine: HuggingFaceLlamaEngine,
    percentile: float,
) -> dict:
    tokenizer = engine.tokenizer

    design_text = make_design_text(
        system_prompt,
        user_instruction,
    )

    design_token_count = len(
        tokenizer.encode(
            design_text,
            add_special_tokens=False,
        )
    )

    combined_lengths = []

    for sentence in length_check_df["sentence"]:
        user_prompt = make_user_prompt(
            sentence=sentence,
            user_instruction=user_instruction,
        )

        chat_text = engine._make_chat_text(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )

        token_ids = tokenizer.encode(
            chat_text,
            add_special_tokens=False,
        )

        combined_lengths.append(
            len(token_ids)
        )

    lengths = np.asarray(
        combined_lengths,
        dtype=np.int32,
    )

    return {
        "system_word_count":
            len(system_prompt.split()),
        "user_instruction_word_count":
            len(user_instruction.split()),
        "design_token_count":
            int(design_token_count),
        "combined_token_p95":
            float(
                np.percentile(
                    lengths,
                    percentile,
                )
            ),
        "combined_token_max":
            int(lengths.max()),
        "over_limit_count":
            int(
                (
                    lengths
                    > engine.max_length
                ).sum()
            ),
        "over_limit_rate":
            float(
                (
                    lengths
                    > engine.max_length
                ).mean()
            ),
    }


def extract_improved_variable(text: str) -> str:
    match = re.search(
        r"<IMPROVED_VARIABLE>(.*?)</IMPROVED_VARIABLE>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if match is None:
        raise ValueError(
            "<IMPROVED_VARIABLE> タグが見つかりません。"
        )

    return match.group(1).strip()


def repair_design_length(
    system_prompt: str,
    user_instruction: str,
    api_engine: LoggedEngine,
    length_check_df: pd.DataFrame,
    local_engine: HuggingFaceLlamaEngine,
    percentile: float,
    max_length: int,
    max_retries: int,
    context_prefix: str,
) -> tuple[
    str,
    str,
    dict,
    int,
    list[str],
]:
    current_system = system_prompt
    current_user = user_instruction
    repair_logs = []

    for attempt in range(
        max_retries + 1
    ):
        metrics = get_prompt_length_metrics(
            system_prompt=current_system,
            user_instruction=current_user,
            length_check_df=length_check_df,
            engine=local_engine,
            percentile=percentile,
        )

        if (
            metrics["combined_token_p95"]
            <= max_length
        ):
            return (
                current_system,
                current_user,
                metrics,
                attempt,
                repair_logs,
            )

        if attempt >= max_retries:
            return (
                current_system,
                current_user,
                metrics,
                attempt,
                repair_logs,
            )

        current_design = make_design_text(
            current_system,
            current_user,
        )

        api_engine.set_context(
            f"{context_prefix}:length_repair:{attempt + 1}"
        )

        request = (
            "Shorten the prompt design below while preserving its "
            "generalizable decision principles and its binary sentiment "
            "classification behavior. Remove redundancy rather than adding "
            "new rules. Do not add examples. The task must still return only "
            "one label, positive or negative. Keep the two-part structure. "
            "Return the complete revised variable inside "
            "<IMPROVED_VARIABLE>...</IMPROVED_VARIABLE>, and inside it use "
            "exactly <SYSTEM_PROMPT>...</SYSTEM_PROMPT> and "
            "<USER_INSTRUCTION>...</USER_INSTRUCTION>.\n\n"
            f"<CURRENT_VARIABLE>\n{current_design}\n"
            "</CURRENT_VARIABLE>"
        )

        raw = api_engine(request)
        repair_logs.append(raw)

        try:
            improved = extract_improved_variable(
                raw
            )
            repaired_system, repaired_user = (
                parse_design_text(improved)
            )
        except ValueError:
            continue

        valid, _ = validate_design(
            repaired_system,
            repaired_user,
        )

        if valid:
            current_system = repaired_system
            current_user = repaired_user

    metrics = get_prompt_length_metrics(
        system_prompt=current_system,
        user_instruction=current_user,
        length_check_df=length_check_df,
        engine=local_engine,
        percentile=percentile,
    )

    return (
        current_system,
        current_user,
        metrics,
        max_retries,
        repair_logs,
    )


# ============================================================
# 11. Batch-level TextGrad evaluator
# ============================================================

def make_textgrad_evaluator(
    evaluation_engine,
):
    """
    約50件の誤分類をまとめて分析する TextGrad evaluator。

    今回は次を同時に扱う。
    - binary-wrong と unknown/invalid-output を分離
    - すべての誤りを Prompt のせいにしない
    - 現在うまくいっている挙動をできるだけ保持
    - System/User/箇条書き/文章など Prompt 構造は事前に固定しない
    - ルール追加だけでなく、簡略化・再構成も改善候補に含める
    """

    evaluation_instruction = tg.Variable(
        (
            "Analyze the classification results jointly rather than treating "
            "each example as an independent prompt-editing problem. "

            "The task is zero-shot binary sentiment classification of movie reviews. "
            "The input movie-review sentence, the two allowed labels 'positive' and "
            "'negative', and the requirement to return exactly one valid label are fixed "
            "and must not be changed. "

            "The optimizable prompt may use System and/or User instructions in any structure. "
            "Do not assume in advance that prose, bullet points, System-only, User-heavy, "
            "System/User separation, or a longer prompt is better. Let the observed evidence "
            "determine the most useful prompt structure. "

            "Compare the model predictions with the ground-truth labels and identify "
            "prompt-level weaknesses that recur across multiple examples. "
            "If the error evidence shows a strong directional class bias, treat that "
            "as a potentially important prompt-level weakness, but do not assume or "
            "invent a bias that is not supported by the batch. "

            "Explicitly distinguish two failure modes: "
            "(A) binary-wrong: the model returns a valid label, but that label is incorrect; "
            "(B) unknown/invalid-output: the model does not return either valid label. "
            "Do not treat these two failure modes as the same problem. "

            "Do not assume that every classification error is caused by the prompt. "
            "Distinguish errors that are plausibly addressable through prompt design from "
            "errors that may primarily reflect model limitations, genuine ambiguity, "
            "annotation noise, or difficult examples. Do not create prompt rules merely "
            "to fit isolated or questionable examples. "

            "Preserve behavior that already works. A proposed improvement should aim for "
            "net accuracy gain: fix recurring, prompt-addressable errors while minimizing "
            "the risk of turning currently correct predictions into binary-wrong or "
            "unknown outputs. "

            "Focus on generalizable weaknesses in how the task is communicated, how the "
            "binary decision is framed, how ambiguous or mixed sentiment is handled, how "
            "output constraints are communicated, and whether the current prompt is "
            "unnecessarily complex or underspecified. "

            "Do not automatically solve errors by adding more rules or more detail. "
            "Consider simplification, restructuring, removing unnecessary instructions, "
            "or changing the allocation of information between System and User messages "
            "when the evidence supports it. "

            "Prioritize patterns supported by multiple examples over isolated cases. "
            "Do not copy or quote review sentences into the recommended prompt, do not "
            "create sentence-specific rules, and do not propose few-shot examples. "

            "Return concise, specific, and actionable feedback about the prompt design. "
            "When possible, prioritize the changes most likely to increase overall "
            "validation accuracy while keeping invalid outputs and regressions low."
        ),
        requires_grad=False,
        role_description=(
            "batch-level regression-aware and error-type-aware "
            "prompt-design evaluation instruction"
        ),
    )

    return tg.loss.MultiFieldEvaluation(
        evaluation_instruction=evaluation_instruction,
        role_descriptions=[
            (
                "current optimizable prompt design; its structure is not fixed "
                "in advance"
            ),
            "batch of input movie-review sentences",
            (
                "corresponding model predictions annotated with error types "
                "(binary-wrong or unknown/invalid-output)"
            ),
            "corresponding ground-truth binary sentiment labels",
        ],
        engine=evaluation_engine,
    )


def build_batch_fields(
    batch_df: pd.DataFrame,
) -> tuple[str, str, str]:
    sentence_lines = []
    prediction_lines = []
    gold_lines = []

    for i, (_, row) in enumerate(
        batch_df.iterrows(),
        start=1,
    ):
        item_id = (
            f"E{i:02d}"
            f"_src{int(row['source_index'])}"
        )

        sentence_lines.append(
            f"[{item_id}] {row['sentence']}"
        )
        prediction_lines.append(
            f"[{item_id}] prediction={row['prediction']}; "
            f"error_type={row.get('error_type', '')}"
        )
        gold_lines.append(
            f"[{item_id}] {row['true_label']}"
        )

    return (
        "\n".join(sentence_lines),
        "\n".join(prediction_lines),
        "\n".join(gold_lines),
    )


def sample_representative_errors(
    errors_df: pd.DataFrame,
    n: int,
    seed: int,
) -> pd.DataFrame:
    if len(errors_df) <= n:
        return errors_df.copy()

    # gold label をできるだけ両方残しつつ、
    # 特定の誤りタイプを人手で優先しない。
    half = n // 2

    pos = errors_df[
        errors_df["true_label"] == "positive"
    ]
    neg = errors_df[
        errors_df["true_label"] == "negative"
    ]

    pos_take = min(
        half,
        len(pos),
    )
    neg_take = min(
        n - pos_take,
        len(neg),
    )

    selected = []

    if pos_take > 0:
        selected.append(
            pos.sample(
                n=pos_take,
                random_state=seed,
            )
        )

    if neg_take > 0:
        selected.append(
            neg.sample(
                n=neg_take,
                random_state=seed + 1,
            )
        )

    out = pd.concat(
        selected,
        ignore_index=True,
    )

    if len(out) < n:
        used = set(
            out["source_index"].tolist()
        )

        remaining = errors_df[
            ~errors_df[
                "source_index"
            ].isin(used)
        ]

        out = pd.concat(
            [
                out,
                remaining.sample(
                    n=min(
                        n - len(out),
                        len(remaining),
                    ),
                    random_state=seed + 2,
                ),
            ],
            ignore_index=True,
        )

    return (
        out
        .sample(
            frac=1,
            random_state=seed + 3,
        )
        .reset_index(drop=True)
    )


def compact_text(text: str, max_words: int) -> str:
    """APIへ再投入する分析文をword数で安全に圧縮する。"""
    words = str(text).split()
    if len(words) <= max_words:
        return str(text).strip()
    return " ".join(words[:max_words]).strip() + " ... [truncated]"


def collect_batch_feedback_from_all_errors(
    depth: int,
    parent_id: str,
    system_prompt: str,
    user_instruction: str,
    prediction_df: pd.DataFrame,
    evaluator,
    api_engine: LoggedEngine,
    error_batch_size: int,
) -> tuple[
    tg.Variable,
    list[dict],
]:
    """
    誤分類を原則すべて使い、error_batch_size件ずつ evaluator に渡す。

    重要:
    各Batchのlossを tg.sum(...).backward() しない。
    それをすると全Batchの長文feedbackが1回のbackward requestへ集約され、
    GPT-4oのTPM上限を超えることがある。

    ここでは全Batchを分析してログへ保存するだけにし、
    後段でGlobal Synthesisへ圧縮した後、1回だけcompact backwardを行う。
    """

    errors = (
        prediction_df[
            ~prediction_df["correct"]
        ]
        .copy()
        .reset_index(drop=True)
    )

    prompt_design = tg.Variable(
        make_design_text(
            system_prompt,
            user_instruction,
        ),
        requires_grad=True,
        role_description=(
            "optimizable prompt design for binary movie-review sentiment "
            "classification, containing both system prompt and user task instruction"
        ),
    )

    if len(errors) == 0:
        return prompt_design, []

    logs = []

    num_batches = math.ceil(
        len(errors) / error_batch_size
    )

    for batch_index, start in enumerate(
        range(
            0,
            len(errors),
            error_batch_size,
        ),
        start=1,
    ):
        batch_df = errors.iloc[
            start:start + error_batch_size
        ].copy()

        (
            sentence_text,
            prediction_text,
            gold_text,
        ) = build_batch_fields(
            batch_df
        )

        sentences_var = tg.Variable(
            sentence_text,
            requires_grad=False,
            role_description=(
                "batch of movie-review sentences"
            ),
        )

        predictions_var = tg.Variable(
            prediction_text,
            requires_grad=False,
            role_description=(
                "model predictions for the batch"
            ),
        )

        gold_var = tg.Variable(
            gold_text,
            requires_grad=False,
            role_description=(
                "ground-truth labels for the batch"
            ),
        )

        api_engine.set_context(
            f"depth{depth}:{parent_id}:"
            f"batch_feedback:{batch_index}/{num_batches}"
        )

        evaluation = evaluator(
            [
                prompt_design,
                sentences_var,
                predictions_var,
                gold_var,
            ]
        )

        logs.append(
            {
                "depth": depth,
                "parent_id": parent_id,
                "batch_index": batch_index,
                "num_batches": num_batches,
                "batch_size": len(batch_df),
                "source_indices": ",".join(
                    str(int(v))
                    for v in batch_df[
                        "source_index"
                    ].tolist()
                ),
                "evaluation_feedback":
                    evaluation.value,
                "system_prompt":
                    system_prompt,
                "user_instruction":
                    user_instruction,
            }
        )

    return prompt_design, logs


def apply_compact_textgrad_backward(
    depth: int,
    parent_id: str,
    prompt_design: tg.Variable,
    global_synthesis: str,
    delta_feedback: str,
    api_engine: LoggedEngine,
) -> tuple[str, str]:
    """
    全Batchの分析結果をGlobal Synthesisへ圧縮した後に、
    その短い情報だけでTextGrad backwardを1回行う。

    これにより「全誤分類を見る」条件は維持したまま、
    1回のAPI requestが巨大化することを防ぐ。
    """

    compact_global = compact_text(
        global_synthesis,
        max_words=350,
    )
    compact_delta = compact_text(
        delta_feedback or (
            "No parent-child delta analysis is available for this prompt."
        ),
        max_words=220,
    )

    compact_instruction = tg.Variable(
        (
            "Use the global diagnosis and parent-child delta evidence to produce a concise "
            "textual gradient. Optimize NET accuracy gain: fix recurring prompt-addressable "
            "errors while minimizing regressions. Explicitly distinguish binary-wrong errors "
            "from unknown/invalid-output errors. Do not assume every remaining error is "
            "prompt-fixable; account for possible model limitations, genuine ambiguity, and "
            "annotation noise. Preserve behavior that already works. Prompt structure is free; "
            "do not assume prose, bullets, System/User separation, or extra detail is inherently "
            "better. Consider simplification or removal as well as additions. Do not add few-shot "
            "examples or sentence-specific rules. The positive/negative exactly-one-valid-label "
            "contract must remain fixed. Keep the feedback concise and actionable."
        ),
        requires_grad=False,
        role_description=(
            "instruction for compact TextGrad backward after global error synthesis"
        ),
    )

    compact_evaluator = tg.loss.MultiFieldEvaluation(
        evaluation_instruction=compact_instruction,
        role_descriptions=[
            "current optimizable prompt design",
            "global synthesis of all batch-level error analyses",
            "parent-child improvement/regression analysis",
        ],
        engine=api_engine,
    )

    global_var = tg.Variable(
        compact_global,
        requires_grad=False,
        role_description=(
            "compact global synthesis of all current misclassification analyses"
        ),
    )

    delta_var = tg.Variable(
        compact_delta,
        requires_grad=False,
        role_description=(
            "compact parent-child prompt-change analysis"
        ),
    )

    api_engine.set_context(
        f"depth{depth}:{parent_id}:compact_textgrad_evaluation"
    )

    compact_loss = compact_evaluator(
        [
            prompt_design,
            global_var,
            delta_var,
        ]
    )

    api_engine.set_context(
        f"depth{depth}:{parent_id}:compact_textgrad_backward"
    )
    compact_loss.backward()

    gradient = compact_text(
        prompt_design.get_gradient_text(),
        max_words=300,
    )

    return gradient, compact_loss.value


# ============================================================
# 12. Batch分析の全体統合
# ============================================================

def synthesize_batch_feedback(
    depth: int,
    parent_id: str,
    system_prompt: str,
    user_instruction: str,
    errors_df: pd.DataFrame,
    batch_feedback_logs: list[dict],
    api_engine: LoggedEngine,
    representative_count: int,
    seed: int,
) -> str:
    if len(errors_df) == 0:
        return (
            "No misclassified examples were found."
        )

    feedback_text = "\n\n".join(
        (
            f"[Batch {item['batch_index']}]\n"
            f"{item['evaluation_feedback']}"
        )
        for item in batch_feedback_logs
    )

    representatives = (
        sample_representative_errors(
            errors_df=errors_df,
            n=min(
                representative_count,
                len(errors_df),
            ),
            seed=seed,
        )
    )

    representative_text = "\n".join(
        (
            f"- source_index={int(row['source_index'])}; "
            f"gold={row['true_label']}; "
            f"prediction={row['prediction']}; "
            f"sentence={row['sentence']}"
        )
        for _, row in representatives.iterrows()
    )

    current_design = make_design_text(
        system_prompt,
        user_instruction,
    )

    unknown_count = int(
        (errors_df["prediction"] == "unknown").sum()
    )
    binary_wrong_count = int(
        len(errors_df) - unknown_count
    )

    gold_distribution = (
        errors_df["true_label"]
        .value_counts()
        .to_dict()
    )
    prediction_distribution = (
        errors_df["prediction"]
        .value_counts()
        .to_dict()
    )

    error_stats = (
        f"Total current errors: {len(errors_df)}\n"
        f"Unknown/invalid-output errors: {unknown_count}\n"
        f"Binary-wrong errors: {binary_wrong_count}\n"
        f"Gold-label distribution among errors: {gold_distribution}\n"
        f"Prediction distribution among errors: {prediction_distribution}"
    )

    request = (
        "Synthesize the batch-level analyses below into one global diagnosis "
        "of the current prompt design. The batch analyses collectively cover "
        "all current misclassifications in the optimization pool. "
        "Identify recurring, generalizable prompt-design weaknesses. Explicitly separate "
        "binary-wrong errors from unknown/invalid-output errors. Inspect the supplied "
        "gold-label and prediction distributions for systematic class-direction bias, "
        "but do not assume such a bias exists unless the evidence supports it. "
        "Do not invent categories that are not supported by the analyses. "
        "Do not assume every model error is fixable by prompting. "
        "Do not respond by merely adding many special-case rules. "
        "Consider whether simplification, restructuring, or changing how "
        "instructions are divided between System and User would help. "
        "The movie-review sentence, the labels positive/negative, and the "
        "one-label output requirement are fixed. Few-shot examples are forbidden. "
        "Prioritize NET accuracy and consider what currently-correct behavior could regress. "
        "Give actionable guidance for the next prompt revision.\n\n"
        "<ERROR_TYPE_SUMMARY>\n"
        f"{error_stats}\n"
        "</ERROR_TYPE_SUMMARY>\n\n"
        "<CURRENT_PROMPT_DESIGN>\n"
        f"{current_design}\n"
        "</CURRENT_PROMPT_DESIGN>\n\n"
        "<BATCH_ANALYSES>\n"
        f"{feedback_text}\n"
        "</BATCH_ANALYSES>\n\n"
        "<REPRESENTATIVE_RAW_ERRORS>\n"
        f"{representative_text}\n"
        "</REPRESENTATIVE_RAW_ERRORS>"
    )

    api_engine.set_context(
        f"depth{depth}:{parent_id}:"
        "global_synthesis"
    )

    response = api_engine(
        request + (
            "\n\nReturn no more than about 350 words. "
            "Do not reproduce the batch analyses verbatim."
        ),
        system_prompt=(
            "You are a rigorous prompt-optimization analyst. "
            "Generalize from evidence and avoid example-specific rules."
        ),
    )

    return compact_text(
        response,
        max_words=350,
    )


# ============================================================
# 13. Parent -> Child 差分分析
# ============================================================

def make_delta_feedback(
    depth: int,
    parent_node_id: str,
    parent_system: str,
    parent_user: str,
    child_node_id: str,
    child_system: str,
    child_user: str,
    parent_result_df: pd.DataFrame,
    child_result_df: pd.DataFrame,
    api_engine: LoggedEngine,
    max_examples: int,
    seed: int,
) -> tuple[str, dict]:
    """Parent→Childを fixed/regressed × unknown/binary-wrong に分解する。"""
    parent_cols = parent_result_df[
        ["source_index", "sentence", "true_label", "prediction", "correct"]
    ].rename(columns={"prediction": "parent_prediction", "correct": "parent_correct"})
    child_cols = child_result_df[
        ["source_index", "prediction", "correct"]
    ].rename(columns={"prediction": "child_prediction", "correct": "child_correct"})
    merged = parent_cols.merge(child_cols, on="source_index", how="inner")

    improved = merged[(~merged["parent_correct"]) & merged["child_correct"]].copy()
    regressed = merged[merged["parent_correct"] & (~merged["child_correct"])].copy()
    fixed_from_unknown = improved[improved["parent_prediction"] == "unknown"]
    fixed_from_binary_wrong = improved[improved["parent_prediction"] != "unknown"]
    regressed_to_unknown = regressed[regressed["child_prediction"] == "unknown"]
    regressed_to_binary_wrong = regressed[regressed["child_prediction"] != "unknown"]
    wrong_to_unknown = merged[
        (~merged["parent_correct"]) & (~merged["child_correct"])
        & (merged["parent_prediction"] != "unknown")
        & (merged["child_prediction"] == "unknown")
    ]
    unknown_to_binary_wrong = merged[
        (~merged["parent_correct"]) & (~merged["child_correct"])
        & (merged["parent_prediction"] == "unknown")
        & (merged["child_prediction"] != "unknown")
    ]

    def change_type(row):
        if (not row["parent_correct"]) and row["child_correct"]:
            return (
                "fixed_from_unknown"
                if row["parent_prediction"] == "unknown"
                else "fixed_from_binary_wrong"
            )
        if row["parent_correct"] and (not row["child_correct"]):
            return (
                "regressed_to_unknown"
                if row["child_prediction"] == "unknown"
                else "regressed_to_binary_wrong"
            )
        if (
            (not row["parent_correct"]) and (not row["child_correct"])
            and row["parent_prediction"] != "unknown"
            and row["child_prediction"] == "unknown"
        ):
            return "binary_wrong_to_unknown"
        if (
            (not row["parent_correct"]) and (not row["child_correct"])
            and row["parent_prediction"] == "unknown"
            and row["child_prediction"] != "unknown"
        ):
            return "unknown_to_binary_wrong"
        return "other"

    changed = merged[
        merged["parent_prediction"] != merged["child_prediction"]
    ].copy()
    changed["change_type"] = changed.apply(change_type, axis=1)

    priority_types = [
        "fixed_from_unknown",
        "fixed_from_binary_wrong",
        "regressed_to_unknown",
        "regressed_to_binary_wrong",
        "binary_wrong_to_unknown",
        "unknown_to_binary_wrong",
    ]
    chunks = []
    per_type = max(1, max_examples // len(priority_types))
    for i, t in enumerate(priority_types):
        subset = changed[changed["change_type"] == t]
        if len(subset):
            chunks.append(
                subset.sample(
                    n=min(per_type, len(subset)),
                    random_state=seed + i,
                )
            )

    sampled = pd.concat(chunks, ignore_index=True) if chunks else changed.head(0).copy()
    if len(sampled) < max_examples:
        used = set(sampled["source_index"].tolist())
        rem = changed[~changed["source_index"].isin(used)]
        if len(rem):
            sampled = pd.concat(
                [
                    sampled,
                    rem.sample(
                        n=min(max_examples - len(sampled), len(rem)),
                        random_state=seed + 99,
                    ),
                ],
                ignore_index=True,
            )

    examples_text = "\n".join(
        (
            f"- change={r['change_type']}; gold={r['true_label']}; "
            f"parent={r['parent_prediction']}; child={r['child_prediction']}; "
            f"sentence={r['sentence']}"
        )
        for _, r in sampled.iterrows()
    )

    counts_text = (
        f"fixed_total={len(improved)}\n"
        f"fixed_from_unknown={len(fixed_from_unknown)}\n"
        f"fixed_from_binary_wrong={len(fixed_from_binary_wrong)}\n"
        f"regressed_total={len(regressed)}\n"
        f"regressed_to_unknown={len(regressed_to_unknown)}\n"
        f"regressed_to_binary_wrong={len(regressed_to_binary_wrong)}\n"
        f"binary_wrong_to_unknown_without_fix={len(wrong_to_unknown)}\n"
        f"unknown_to_binary_wrong_without_fix={len(unknown_to_binary_wrong)}"
    )

    if len(changed) == 0:
        feedback = "No prediction changes between parent and child."
    else:
        request = (
            "Compare the parent and child using the transition counts and examples. "
            "Optimize NET accuracy gain, not merely fixed errors. Separate semantic "
            "binary errors from invalid unknown outputs. Identify what helped, what "
            "caused regressions, and especially what caused correct predictions to "
            "become unknown. Preserve useful behavior while reducing regressions. "
            "Do not assume more detail is better. Prompt structure is unrestricted; "
            "few-shot and sentence-specific rules are forbidden.\n\n"
            "<PARENT_DESIGN>\n"
            f"{make_design_text(parent_system, parent_user)}\n"
            "</PARENT_DESIGN>\n\n"
            "<CHILD_DESIGN>\n"
            f"{make_design_text(child_system, child_user)}\n"
            "</CHILD_DESIGN>\n\n"
            "<TRANSITION_COUNTS>\n"
            f"{counts_text}\n"
            "</TRANSITION_COUNTS>\n\n"
            "<SAMPLED_TRANSITIONS>\n"
            f"{examples_text}\n"
            "</SAMPLED_TRANSITIONS>"
        )
        api_engine.set_context(
            f"depth{depth}:{child_node_id}:parent_child_delta"
        )
        feedback = api_engine(
            request,
            system_prompt=(
                "You analyze prompt revisions by separating fixes, regressions, "
                "binary errors, and invalid outputs."
            ),
        )

    summary = {
        "depth": depth,
        "parent_id": parent_node_id,
        "child_id": child_node_id,
        "improved_count": len(improved),
        "fixed_from_unknown": len(fixed_from_unknown),
        "fixed_from_binary_wrong": len(fixed_from_binary_wrong),
        "regressed_count": len(regressed),
        "regressed_to_unknown": len(regressed_to_unknown),
        "regressed_to_binary_wrong": len(regressed_to_binary_wrong),
        "binary_wrong_to_unknown_without_fix": len(wrong_to_unknown),
        "unknown_to_binary_wrong_without_fix": len(unknown_to_binary_wrong),
        "changed_prediction_count": len(changed),
        "sampled_count": len(sampled),
        "feedback": feedback,
    }
    return feedback, summary


# ============================================================
# 14. TextGrad gradient から複数Candidate生成
# ============================================================

def generate_branch_candidates(
    prompt_design: tg.Variable,
    textual_gradient: str,
    api_engine: LoggedEngine,
    global_synthesis: str,
    delta_feedback: str,
    branch_factor: int,
    existing_designs: set[str],
    design_history: list[str],
    max_retries: int,
    depth: int,
    parent_id: str,
) -> list[dict]:
    """
    Compact TextGrad backwardで得た textual gradient を使い、
    System/UserのPrompt Design候補を複数生成する。

    TextualGradientDescent._update_prompt() は、Variableの計算グラフ全体を
    取り込みrequestが巨大化する可能性があるため使わない。
    代わりに、TextGradで得たgradientを明示的に短くして候補生成へ利用する。
    """

    current_design = str(prompt_design.value)
    compact_gradient = compact_text(
        textual_gradient,
        max_words=300,
    )
    compact_global = compact_text(
        global_synthesis,
        max_words=350,
    )
    compact_delta = compact_text(
        delta_feedback or (
            "No delta analysis is available for the root prompt."
        ),
        max_words=220,
    )

    constraints_text = (
        "The returned variable must use the two wrapper sections "
        "<SYSTEM_PROMPT>...</SYSTEM_PROMPT> and <USER_INSTRUCTION>...</USER_INSTRUCTION> "
        "so code can parse it, but either section may be empty. The wrapper does not prescribe "
        "prompt style. Structure is free: prose, bullets, System-only, User-heavy, or another "
        "design are allowed; do not prefer a format a priori. The task remains zero-shot binary "
        "movie-review sentiment classification. The only valid labels are positive and negative "
        "and exactly one must be returned. Do not include or paraphrase training sentences or "
        "correct answers. No few-shot examples. The review sentence is inserted by code, so do not "
        "include a {sentence} placeholder. Do not introduce a fixed default label unless empirical "
        "evidence supports it. Do not assume a longer prompt is better."
    )

    history_items = design_history[-12:]
    history_text = "\n\n".join(
        f"Previously tried design {i + 1}:\n{compact_text(item, 80)}"
        for i, item in enumerate(history_items)
    )

    generated = []
    generated_here = set()

    for branch_index in range(
        1,
        branch_factor + 1,
    ):
        candidate_system = None
        candidate_user = None
        candidate_design = None
        raw_response = None
        status = "generation_failed"
        rejection_reason = ""

        for retry in range(
            max_retries + 1
        ):
            previous_text = ""

            valid_previous = [
                item
                for item in generated
                if item.get(
                    "structured_design"
                )
            ]

            if valid_previous:
                previous_text = (
                    "\n\nPreviously generated candidates from this same parent "
                    "are listed below. Produce a meaningfully different design, not "
                    "a cosmetic paraphrase:\n"
                    + "\n\n".join(
                        (
                            f"Candidate {i + 1}:\n"
                            f"{item['structured_design']}"
                        )
                        for i, item in enumerate(
                            valid_previous
                        )
                    )
                )

            request = (
                "Revise the current Prompt Design using the TextGrad gradient and evidence. "
                "Generate one strong zero-shot candidate. The current prompt may already be strong: "
                "preserve useful behavior while targeting net accuracy gain. Each candidate must test a "
                "substantively different prompt-design hypothesis, not a cosmetic paraphrase or a minor "
                "rewriting of an earlier candidate. Prompt structure is free and should "
                "be chosen from evidence, not convention. Do not overfit individual reviews.\n\n"
                "<CURRENT_PROMPT_DESIGN>\n"
                f"{current_design}\n"
                "</CURRENT_PROMPT_DESIGN>\n\n"
                "<TEXTGRAD_GRADIENT>\n"
                f"{compact_gradient}\n"
                "</TEXTGRAD_GRADIENT>\n\n"
                "<GLOBAL_ERROR_SYNTHESIS>\n"
                f"{compact_global}\n"
                "</GLOBAL_ERROR_SYNTHESIS>\n\n"
                "<PARENT_CHILD_DELTA_ANALYSIS>\n"
                f"{compact_delta}\n"
                "</PARENT_CHILD_DELTA_ANALYSIS>\n\n"
                "<PREVIOUSLY_TRIED_DESIGNS>\n"
                f"{history_text}\n"
                "</PREVIOUSLY_TRIED_DESIGNS>\n\n"
                "<CONSTRAINTS>\n"
                f"{constraints_text}\n"
                "</CONSTRAINTS>\n\n"
                f"This is candidate {branch_index} of {branch_factor} from parent "
                f"{parent_id} at depth {depth}. Retry number: {retry}."
                f"{previous_text}\n\n"
                "Return only the complete revised variable inside "
                "<IMPROVED_VARIABLE>...</IMPROVED_VARIABLE>. Inside it use exactly:\n"
                "<SYSTEM_PROMPT>...</SYSTEM_PROMPT>\n"
                "<USER_INSTRUCTION>...</USER_INSTRUCTION>"
            )

            api_engine.set_context(
                f"depth{depth}:{parent_id}:"
                f"candidate_generation:{branch_index}:retry{retry}"
            )

            raw_response = api_engine(
                request,
                system_prompt=(
                    "You are a careful prompt optimizer. Search prompt structure as well as wording. "
                    "Do not default to the same design pattern across candidates. Follow constraints "
                    "exactly and return only the requested structured variable."
                ),
            )

            try:
                improved_variable = (
                    extract_improved_variable(
                        raw_response
                    )
                )
                (
                    candidate_system,
                    candidate_user,
                ) = parse_design_text(
                    improved_variable
                )
            except ValueError as exc:
                rejection_reason = (
                    f"parse_error:{exc}"
                )
                continue

            valid, reason = validate_design(
                candidate_system,
                candidate_user,
            )

            if not valid:
                rejection_reason = reason
                continue

            candidate_design = make_design_text(
                candidate_system,
                candidate_user,
            )

            normalized = normalize_design(
                candidate_system,
                candidate_user,
            )

            if (
                normalized in generated_here
                or normalized in existing_designs
            ):
                candidate_system = None
                candidate_user = None
                candidate_design = None
                status = "duplicate"
                rejection_reason = "duplicate"
                continue

            generated_here.add(normalized)
            existing_designs.add(normalized)
            design_history.append(candidate_design)
            status = "generated"
            rejection_reason = ""
            break

        generated.append(
            {
                "branch_index": branch_index,
                "system_prompt": candidate_system,
                "user_instruction": candidate_user,
                "structured_design": candidate_design,
                "raw_optimizer_response": raw_response,
                "generation_status": status,
                "rejection_reason": rejection_reason,
            }
        )

    return generated


# ============================================================
# 15. Beam node
# ============================================================

@dataclass
class PromptNode:
    prompt_id: str
    parent_id: Optional[str]
    depth: int
    branch_index: int

    system_prompt: str
    user_instruction: str

    validation_accuracy: float
    validation_correct_count: int

    system_word_count: int
    user_instruction_word_count: int
    design_token_count: int
    combined_token_p95: float
    combined_token_max: int
    over_limit_count: int
    over_limit_rate: float

    length_valid: bool
    search_valid: bool

    unknown_count: int
    unknown_rate: float
    rejection_reason: str
    length_repair_count: int


# ============================================================
# 16. Beam選択
# ============================================================

def choose_beam(
    nodes: list[PromptNode],
    beam_width: int,
    near_tie_examples: int,
) -> list[PromptNode]:
    """Accuracy first; near-tie は Unknown、次に短さで選ぶ。"""
    remaining = [node for node in nodes if node.search_valid]
    remaining = list({node.prompt_id: node for node in remaining}.values())
    selected = []
    while remaining and len(selected) < beam_width:
        best_correct = max(node.validation_correct_count for node in remaining)
        near_ties = [
            node for node in remaining
            if best_correct - node.validation_correct_count <= near_tie_examples
        ]
        chosen = min(
            near_ties,
            key=lambda node: (
                node.unknown_count,
                node.design_token_count,
                -node.validation_correct_count,
                node.prompt_id,
            ),
        )
        selected.append(chosen)
        remaining = [node for node in remaining if node.prompt_id != chosen.prompt_id]
    return selected


# ============================================================
# 17. 保存
# ============================================================

def node_to_record(
    node: PromptNode,
    generation_status: str,
    selected_for_next_depth: bool = False,
    parent_optimization_accuracy: Optional[float] = None,
    parent_error_count: Optional[int] = None,
) -> dict:
    return {
        "prompt_id": node.prompt_id,
        "parent_id": node.parent_id,
        "depth": node.depth,
        "branch_index": node.branch_index,
        "validation_accuracy":
            node.validation_accuracy,
        "validation_correct_count":
            node.validation_correct_count,
        "system_word_count":
            node.system_word_count,
        "user_instruction_word_count":
            node.user_instruction_word_count,
        "design_token_count":
            node.design_token_count,
        "combined_token_p95":
            node.combined_token_p95,
        "combined_token_max":
            node.combined_token_max,
        "over_limit_count":
            node.over_limit_count,
        "over_limit_rate":
            node.over_limit_rate,
        "length_valid":
            node.length_valid,
        "search_valid":
            node.search_valid,
        "unknown_count":
            node.unknown_count,
        "unknown_rate":
            node.unknown_rate,
        "rejection_reason":
            node.rejection_reason,
        "length_repair_count":
            node.length_repair_count,
        "generation_status":
            generation_status,
        "selected_for_next_depth":
            selected_for_next_depth,
        "parent_optimization_accuracy":
            parent_optimization_accuracy,
        "parent_error_count":
            parent_error_count,
        "system_prompt":
            node.system_prompt,
        "user_instruction":
            node.user_instruction,
        "structured_design":
            make_design_text(
                node.system_prompt,
                node.user_instruction,
            ),
    }


def save_records(
    prompt_tree_records: list[dict],
    batch_feedback_logs: list[dict],
    synthesis_logs: list[dict],
    delta_logs: list[dict],
    gradient_logs: list[dict],
) -> None:
    pd.DataFrame(
        prompt_tree_records
    ).to_csv(
        PROMPT_TREE_PATH,
        index=False,
    )

    pd.DataFrame(
        batch_feedback_logs
    ).to_csv(
        BATCH_FEEDBACK_PATH,
        index=False,
    )

    pd.DataFrame(
        synthesis_logs
    ).to_csv(
        GLOBAL_SYNTHESIS_PATH,
        index=False,
    )

    pd.DataFrame(
        delta_logs
    ).to_csv(
        DELTA_FEEDBACK_PATH,
        index=False,
    )

    pd.DataFrame(
        gradient_logs
    ).to_csv(
        GRADIENTS_PATH,
        index=False,
    )


# ============================================================
# 18. main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Q98: Llama 3.2 1B Instruct + TextGrad Prompt Design Search "
            "(short search: System + User instruction)"
        )
    )

    parser.add_argument(
        "--evaluation-engine",
        type=str,
        default="gpt-4o",
    )

    parser.add_argument(
        "--optimization-size",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--validation-size",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--prompt-check-size",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--prompt-check-top-k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--error-batch-size",
        type=int,
        default=50,
        help=(
            "誤分類を何件ずつまとめてTextGrad分析するか"
        ),
    )

    parser.add_argument(
        "--representative-errors",
        type=int,
        default=24,
        help=(
            "全Batch統合時に生の誤分類例を何件添えるか"
        ),
    )

    parser.add_argument(
        "--multistart-candidates",
        type=int,
        default=10,
        help=(
            "Depth 1でRootから生成する候補数"
        ),
    )

    parser.add_argument(
        "--branch-factor",
        type=int,
        default=4,
        help=(
            "Depth 2以降で1 Parentから生成する候補数"
        ),
    )

    parser.add_argument(
        "--beam-width",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--near-tie-examples",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--delta-max-examples",
        type=int,
        default=100,
        help=(
            "Parent→Child差分分析でGPTに見せる変更例の最大数"
        ),
    )

    parser.add_argument(
        "--candidate-retries",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--length-percentile",
        type=float,
        default=95.0,
    )

    parser.add_argument(
        "--length-repair-retries",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--smoke-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--smoke-max-unknown-rate",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--smoke-min-accuracy",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=None,
        help=(
            "互換性のため残している引数。この版では値を指定しても"
            "Accuracyによる早期終了は行わない。"
        ),
    )

    parser.add_argument(
        "--local-batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--api-input-price-per-million",
        type=float,
        default=2.5,
        help=(
            "API入力100万tokenあたりのUSD単価。"
            "実行時点の最新料金に必要なら変更する。"
        ),
    )

    parser.add_argument(
        "--api-output-price-per-million",
        type=float,
        default=10.0,
        help=(
            "API出力100万tokenあたりのUSD単価。"
            "実行時点の最新料金に必要なら変更する。"
        ),
    )

    parser.add_argument(
        "--max-api-cost-usd",
        type=float,
        default=15.0,
        help=(
            "推定累積API料金の停止上限。"
            "0なら上限なし。例: 15"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # 条件確認
    # --------------------------------------------------------
    if args.error_batch_size <= 0:
        raise ValueError(
            "error_batch_size は1以上にしてください。"
        )

    if args.multistart_candidates <= 0:
        raise ValueError(
            "multistart_candidates は1以上にしてください。"
        )

    if (
        args.branch_factor <= 0
        or args.beam_width <= 0
        or args.depth <= 0
    ):
        raise ValueError(
            "branch_factor / beam_width / depth は1以上にしてください。"
        )

    if not (
        0
        < args.length_percentile
        <= 100
    ):
        raise ValueError(
            "length_percentile は0〜100にしてください。"
        )

    if not (0 <= args.smoke_max_unknown_rate <= 1):
        raise ValueError("smoke_max_unknown_rate は0〜1にしてください。")
    if not (0 <= args.smoke_min_accuracy <= 1):
        raise ValueError("smoke_min_accuracy は0〜1にしてください。")
    if args.prompt_check_size <= 0 or args.prompt_check_size % 2 != 0:
        raise ValueError("prompt_check_size は正の偶数にしてください。")
    if args.prompt_check_top_k <= 0:
        raise ValueError("prompt_check_top_k は1以上にしてください。")

    # --------------------------------------------------------
    # API key / file
    # --------------------------------------------------------
    env_path = PROJECT_ROOT / ".env"
    load_simple_env(env_path)

    if not os.getenv(
        "OPENAI_API_KEY"
    ):
        raise RuntimeError(
            "OPENAI_API_KEY が見つかりません。\n"
            f"{env_path} に OPENAI_API_KEY=... を設定してください。"
        )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fix_seed(args.seed)

    # --------------------------------------------------------
    # データ
    # --------------------------------------------------------
    train_df = load_root_only_binary_train()
    train_df.to_csv(ROOT_DATA_PATH, index=False)

    label_counts = train_df["label"].value_counts().sort_index()
    data_summary = {
        "source": "SST original train root sentences only",
        "neutral_rule": "exclude 0.4 < sentiment_value <= 0.6",
        "negative_rule": "sentiment_value <= 0.4",
        "positive_rule": "sentiment_value > 0.6",
        "root_binary_count": int(len(train_df)),
        "negative_count": int(label_counts.get(0, 0)),
        "positive_count": int(label_counts.get(1, 0)),
        "unique_sentence_index_count": int(train_df["sentence_index"].nunique()),
        "all_rows_root_only": bool((train_df["data_scope"] == "root_only").all()),
    }
    with open(DATA_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(data_summary, f, ensure_ascii=False, indent=2)

    (
        optimization_df,
        validation_df,
        prompt_check_df,
    ) = make_fixed_split(
        train_df=train_df,
        optimization_size=
            args.optimization_size,
        validation_size=
            args.validation_size,
        prompt_check_size=
            args.prompt_check_size,
        seed=args.seed,
    )

    save_split_manifest(
        optimization_df,
        validation_df,
        prompt_check_df,
    )

    smoke_df = make_balanced_smoke_df(
        validation_df=validation_df,
        smoke_size=args.smoke_size,
        seed=args.seed + 5000,
    )

    print(
        "\n===== TextGrad Prompt Design Search ====="
    )
    print(
        f"Optimization pool : {len(optimization_df)}"
    )
    print(
        "  positive / negative: "
        f"{int((optimization_df['label'] == 1).sum())} / "
        f"{int((optimization_df['label'] == 0).sum())}"
    )
    print(
        f"Validation pool   : {len(validation_df)}"
    )
    print(
        "  positive / negative: "
        f"{int((validation_df['label'] == 1).sum())} / "
        f"{int((validation_df['label'] == 0).sum())}"
    )
    print(
        f"Prompt-check pool : {len(prompt_check_df)}"
    )
    print(
        "  positive / negative: "
        f"{int((prompt_check_df['label'] == 1).sum())} / "
        f"{int((prompt_check_df['label'] == 0).sum())}"
    )
    print(
        f"Error batch size  : {args.error_batch_size}"
    )
    print(
        f"Multi-start       : {args.multistart_candidates}"
    )
    print(
        f"Branch factor     : {args.branch_factor}"
    )
    print(
        f"Beam width        : {args.beam_width}"
    )
    print(
        f"Depth             : {args.depth}"
    )
    print(
        "Accuracy stop    : disabled (always search to max depth)"
    )
    print(
        f"Evaluation engine : {args.evaluation_engine}"
    )
    print(
        f"Seed              : {args.seed}"
    )

    # --------------------------------------------------------
    # Engines
    # --------------------------------------------------------
    local_engine = HuggingFaceLlamaEngine(
        model_name=MODEL_NAME,
        batch_size=args.local_batch_size,
        max_length=args.max_length,
        max_new_tokens=8,
    )

    raw_api_engine = tg.get_engine(
        engine_name=
            args.evaluation_engine
    )

    api_engine = LoggedEngine(
        base_engine=raw_api_engine,
        csv_path=API_USAGE_PATH,
        input_price_per_million=
            args.api_input_price_per_million,
        output_price_per_million=
            args.api_output_price_per_million,
        max_cost_usd=
            args.max_api_cost_usd,
    )

    tg.set_backward_engine(
        api_engine,
        override=True,
    )

    evaluator = make_textgrad_evaluator(
        api_engine
    )

    # --------------------------------------------------------
    # Root
    # --------------------------------------------------------
    root_val_acc, root_val_df = (
        evaluate_prompt(
            dataframe=validation_df,
            engine=local_engine,
            system_prompt=
                INITIAL_SYSTEM_PROMPT,
            user_instruction=
                INITIAL_USER_INSTRUCTION,
        )
    )

    root_length = get_prompt_length_metrics(
        system_prompt=
            INITIAL_SYSTEM_PROMPT,
        user_instruction=
            INITIAL_USER_INSTRUCTION,
        length_check_df=optimization_df,
        engine=local_engine,
        percentile=
            args.length_percentile,
    )

    root = PromptNode(
        prompt_id="P000",
        parent_id=None,
        depth=0,
        branch_index=0,
        system_prompt=
            INITIAL_SYSTEM_PROMPT,
        user_instruction=
            INITIAL_USER_INSTRUCTION,
        validation_accuracy=
            root_val_acc,
        validation_correct_count=
            int(
                root_val_df[
                    "correct"
                ].sum()
            ),
        system_word_count=
            root_length[
                "system_word_count"
            ],
        user_instruction_word_count=
            root_length[
                "user_instruction_word_count"
            ],
        design_token_count=
            root_length[
                "design_token_count"
            ],
        combined_token_p95=
            root_length[
                "combined_token_p95"
            ],
        combined_token_max=
            root_length[
                "combined_token_max"
            ],
        over_limit_count=
            root_length[
                "over_limit_count"
            ],
        over_limit_rate=
            root_length[
                "over_limit_rate"
            ],
        length_valid=(
            root_length[
                "combined_token_p95"
            ]
            <= args.max_length
        ),
        search_valid=(
            root_length[
                "combined_token_p95"
            ]
            <= args.max_length
        ),
        unknown_count=int(
            (
                root_val_df[
                    "prediction"
                ]
                == "unknown"
            ).sum()
        ),
        unknown_rate=float(
            (
                root_val_df[
                    "prediction"
                ]
                == "unknown"
            ).mean()
        ),
        rejection_reason="",
        length_repair_count=0,
    )

    print(
        "\n===== Root Prompt ====="
    )
    print(
        make_design_text(
            root.system_prompt,
            root.user_instruction,
        )
    )
    print(
        f"Validation Accuracy: "
        f"{root.validation_accuracy * 100:.2f}% "
        f"({root.validation_correct_count}/"
        f"{len(validation_df)})"
    )

    prompt_tree_records = [
        node_to_record(
            root,
            generation_status="initial",
            selected_for_next_depth=True,
        )
    ]

    batch_feedback_logs = []
    synthesis_logs = []
    delta_logs = []
    gradient_logs = []

    all_nodes = [root]
    beam = [root]
    node_counter = 1

    existing_designs = {
        normalize_design(
            INITIAL_SYSTEM_PROMPT,
            INITIAL_USER_INSTRUCTION,
        )
    }
    design_history = [make_design_text(INITIAL_SYSTEM_PROMPT, INITIAL_USER_INSTRUCTION)]

    # optimization prediction cache:
    # Promptが次のDepthでも残った場合、ローカル推論は再利用する。
    optimization_cache = {}

    stopped_due_cost = False

    # --------------------------------------------------------
    # Tree Search
    # --------------------------------------------------------
    try:
        for depth in range(
            1,
            args.depth + 1,
        ):
            print(
                "\n"
                + "=" * 72
            )
            print(
                f"Depth {depth}/{args.depth}"
            )
            print(
                "=" * 72
            )

            depth_candidates = []

            for parent_order, parent in enumerate(
                beam,
                start=1,
            ):
                print(
                    "\n"
                    + "-" * 72
                )
                print(
                    f"Parent {parent_order}/{len(beam)}: "
                    f"{parent.prompt_id} "
                    f"val={parent.validation_accuracy * 100:.2f}%"
                )

                # --------------------------------------------
                # A. Optimization poolでParentを評価
                # --------------------------------------------
                if (
                    parent.prompt_id
                    in optimization_cache
                ):
                    (
                        optimization_acc,
                        optimization_result,
                    ) = optimization_cache[
                        parent.prompt_id
                    ]
                else:
                    (
                        optimization_acc,
                        optimization_result,
                    ) = evaluate_prompt(
                        dataframe=
                            optimization_df,
                        engine=
                            local_engine,
                        system_prompt=
                            parent.system_prompt,
                        user_instruction=
                            parent.user_instruction,
                    )

                    optimization_cache[
                        parent.prompt_id
                    ] = (
                        optimization_acc,
                        optimization_result,
                    )

                errors_df = (
                    optimization_result[
                        ~optimization_result[
                            "correct"
                        ]
                    ]
                    .copy()
                    .reset_index(drop=True)
                )

                error_count = len(
                    errors_df
                )

                print(
                    f"Optimization Accuracy: "
                    f"{optimization_acc * 100:.2f}% "
                    f"({len(optimization_df) - error_count}/"
                    f"{len(optimization_df)})"
                )
                print(
                    f"Errors: {error_count} "
                    f"-> 約"
                    f"{math.ceil(error_count / args.error_batch_size)} "
                    f"batch"
                )

                error_distribution = (
                    errors_df[
                        "true_label"
                    ]
                    .value_counts()
                    .to_dict()
                )

                print(
                    "Error gold distribution: "
                    f"{error_distribution}"
                )

                # --------------------------------------------
                # B. Parent->Child差分
                # --------------------------------------------
                delta_feedback = ""
                delta_summary = None

                if (
                    parent.parent_id
                    is not None
                    and parent.parent_id
                    in optimization_cache
                ):
                    parent_of_parent = next(
                        (
                            node
                            for node in all_nodes
                            if node.prompt_id
                            == parent.parent_id
                        ),
                        None,
                    )

                    if (
                        parent_of_parent
                        is not None
                    ):
                        (
                            _,
                            previous_result,
                        ) = optimization_cache[
                            parent.parent_id
                        ]

                        (
                            delta_feedback,
                            delta_summary,
                        ) = make_delta_feedback(
                            depth=depth,
                            parent_node_id=
                                parent.parent_id,
                            parent_system=
                                parent_of_parent.system_prompt,
                            parent_user=
                                parent_of_parent.user_instruction,
                            child_node_id=
                                parent.prompt_id,
                            child_system=
                                parent.system_prompt,
                            child_user=
                                parent.user_instruction,
                            parent_result_df=
                                previous_result,
                            child_result_df=
                                optimization_result,
                            api_engine=
                                api_engine,
                            max_examples=
                                args.delta_max_examples,
                            seed=
                                args.seed
                                + depth * 100
                                + parent_order,
                        )

                        delta_logs.append(
                            delta_summary
                        )

                        print(
                            "Delta: "
                            f"+{delta_summary['improved_count']} fixed "
                            f"(u→correct {delta_summary['fixed_from_unknown']}, "
                            f"wrong→correct {delta_summary['fixed_from_binary_wrong']}) / "
                            f"-{delta_summary['regressed_count']} regressed "
                            f"(correct→u {delta_summary['regressed_to_unknown']}, "
                            f"correct→wrong {delta_summary['regressed_to_binary_wrong']})"
                        )

                # --------------------------------------------
                # C. 全誤分類を50件程度ずつ分析
                #    ここではbackwardしない
                # --------------------------------------------
                (
                    prompt_design_var,
                    parent_batch_logs,
                ) = collect_batch_feedback_from_all_errors(
                    depth=depth,
                    parent_id=
                        parent.prompt_id,
                    system_prompt=
                        parent.system_prompt,
                    user_instruction=
                        parent.user_instruction,
                    prediction_df=
                        optimization_result,
                    evaluator=
                        evaluator,
                    api_engine=
                        api_engine,
                    error_batch_size=
                        args.error_batch_size,
                )

                batch_feedback_logs.extend(
                    parent_batch_logs
                )

                # --------------------------------------------
                # D. Batch分析を全体統合
                # --------------------------------------------
                global_synthesis = (
                    synthesize_batch_feedback(
                        depth=depth,
                        parent_id=
                            parent.prompt_id,
                        system_prompt=
                            parent.system_prompt,
                        user_instruction=
                            parent.user_instruction,
                        errors_df=
                            errors_df,
                        batch_feedback_logs=
                            parent_batch_logs,
                        api_engine=
                            api_engine,
                        representative_count=
                            args.representative_errors,
                        seed=
                            args.seed
                            + depth * 1000
                            + parent_order,
                    )
                )

                synthesis_logs.append(
                    {
                        "depth":
                            depth,
                        "parent_id":
                            parent.prompt_id,
                        "optimization_accuracy":
                            optimization_acc,
                        "error_count":
                            error_count,
                        "global_synthesis":
                            global_synthesis,
                    }
                )

                # --------------------------------------------
                # E. 圧縮情報だけでTextGrad backwardを1回実行
                # --------------------------------------------
                (
                    aggregated_gradient,
                    compact_loss_feedback,
                ) = apply_compact_textgrad_backward(
                    depth=depth,
                    parent_id=
                        parent.prompt_id,
                    prompt_design=
                        prompt_design_var,
                    global_synthesis=
                        global_synthesis,
                    delta_feedback=
                        delta_feedback,
                    api_engine=
                        api_engine,
                )

                gradient_logs.append(
                    {
                        "depth":
                            depth,
                        "parent_id":
                            parent.prompt_id,
                        "optimization_accuracy":
                            optimization_acc,
                        "error_count":
                            error_count,
                        "num_error_batches":
                            len(parent_batch_logs),
                        "compact_loss_feedback":
                            compact_loss_feedback,
                        "aggregated_gradient":
                            aggregated_gradient,
                        "system_prompt":
                            parent.system_prompt,
                        "user_instruction":
                            parent.user_instruction,
                    }
                )

                # --------------------------------------------
                # F. Candidate生成
                # Depth1 Rootのみ Multi-start 10案
                # --------------------------------------------
                if (
                    depth == 1
                    and parent.prompt_id
                    == "P000"
                ):
                    current_branch_factor = (
                        args.multistart_candidates
                    )
                else:
                    current_branch_factor = (
                        args.branch_factor
                    )

                candidates = (
                    generate_branch_candidates(
                        prompt_design=
                            prompt_design_var,
                        textual_gradient=
                            aggregated_gradient,
                        api_engine=
                            api_engine,
                        global_synthesis=
                            global_synthesis,
                        delta_feedback=
                            delta_feedback,
                        branch_factor=
                            current_branch_factor,
                        existing_designs=
                            existing_designs,
                        design_history=
                            design_history,
                        max_retries=
                            args.candidate_retries,
                        depth=
                            depth,
                        parent_id=
                            parent.prompt_id,
                    )
                )

                # --------------------------------------------
                # G. 各CandidateをGemmaで実測
                # --------------------------------------------
                for candidate in candidates:
                    prompt_id = (
                        f"P{node_counter:03d}"
                    )
                    node_counter += 1

                    branch_index = candidate[
                        "branch_index"
                    ]

                    if not candidate.get(
                        "structured_design"
                    ):
                        prompt_tree_records.append(
                            {
                                "prompt_id":
                                    prompt_id,
                                "parent_id":
                                    parent.prompt_id,
                                "depth":
                                    depth,
                                "branch_index":
                                    branch_index,
                                "validation_accuracy":
                                    0.0,
                                "validation_correct_count":
                                    0,
                                "system_word_count":
                                    0,
                                "user_instruction_word_count":
                                    0,
                                "design_token_count":
                                    0,
                                "combined_token_p95":
                                    None,
                                "combined_token_max":
                                    None,
                                "over_limit_count":
                                    None,
                                "over_limit_rate":
                                    None,
                                "length_valid":
                                    False,
                                "search_valid":
                                    False,
                                "unknown_count":
                                    None,
                                "unknown_rate":
                                    None,
                                "rejection_reason":
                                    candidate[
                                        "rejection_reason"
                                    ],
                                "length_repair_count":
                                    0,
                                "generation_status":
                                    candidate[
                                        "generation_status"
                                    ],
                                "selected_for_next_depth":
                                    False,
                                "parent_optimization_accuracy":
                                    optimization_acc,
                                "parent_error_count":
                                    error_count,
                                "system_prompt":
                                    "",
                                "user_instruction":
                                    "",
                                "structured_design":
                                    "",
                            }
                        )
                        continue

                    candidate_system = candidate[
                        "system_prompt"
                    ]
                    candidate_user = candidate[
                        "user_instruction"
                    ]

                    # 長さ確認・修復
                    (
                        candidate_system,
                        candidate_user,
                        length_metrics,
                        repair_count,
                        _,
                    ) = repair_design_length(
                        system_prompt=
                            candidate_system,
                        user_instruction=
                            candidate_user,
                        api_engine=
                            api_engine,
                        length_check_df=
                            optimization_df,
                        local_engine=
                            local_engine,
                        percentile=
                            args.length_percentile,
                        max_length=
                            args.max_length,
                        max_retries=
                            args.length_repair_retries,
                        context_prefix=
                            f"depth{depth}:{prompt_id}",
                    )

                    length_valid = (
                        length_metrics[
                            "combined_token_p95"
                        ]
                        <= args.max_length
                    )

                    if not length_valid:
                        node = PromptNode(
                            prompt_id=
                                prompt_id,
                            parent_id=
                                parent.prompt_id,
                            depth=
                                depth,
                            branch_index=
                                branch_index,
                            system_prompt=
                                candidate_system,
                            user_instruction=
                                candidate_user,
                            validation_accuracy=
                                0.0,
                            validation_correct_count=
                                0,
                            system_word_count=
                                length_metrics[
                                    "system_word_count"
                                ],
                            user_instruction_word_count=
                                length_metrics[
                                    "user_instruction_word_count"
                                ],
                            design_token_count=
                                length_metrics[
                                    "design_token_count"
                                ],
                            combined_token_p95=
                                length_metrics[
                                    "combined_token_p95"
                                ],
                            combined_token_max=
                                length_metrics[
                                    "combined_token_max"
                                ],
                            over_limit_count=
                                length_metrics[
                                    "over_limit_count"
                                ],
                            over_limit_rate=
                                length_metrics[
                                    "over_limit_rate"
                                ],
                            length_valid=False,
                            search_valid=False,
                            unknown_count=0,
                            unknown_rate=0.0,
                            rejection_reason=
                                "length_limit",
                            length_repair_count=
                                repair_count,
                        )

                        all_nodes.append(
                            node
                        )
                        prompt_tree_records.append(
                            node_to_record(
                                node,
                                generation_status=
                                    "length_rejected",
                                parent_optimization_accuracy=
                                    optimization_acc,
                                parent_error_count=
                                    error_count,
                            )
                        )
                        continue

                    # 修復後のDesignがタスク条件を保持しているか
                    valid_design, reason = (
                        validate_design(
                            candidate_system,
                            candidate_user,
                        )
                    )

                    if not valid_design:
                        node = PromptNode(
                            prompt_id=
                                prompt_id,
                            parent_id=
                                parent.prompt_id,
                            depth=
                                depth,
                            branch_index=
                                branch_index,
                            system_prompt=
                                candidate_system,
                            user_instruction=
                                candidate_user,
                            validation_accuracy=
                                0.0,
                            validation_correct_count=
                                0,
                            system_word_count=
                                length_metrics[
                                    "system_word_count"
                                ],
                            user_instruction_word_count=
                                length_metrics[
                                    "user_instruction_word_count"
                                ],
                            design_token_count=
                                length_metrics[
                                    "design_token_count"
                                ],
                            combined_token_p95=
                                length_metrics[
                                    "combined_token_p95"
                                ],
                            combined_token_max=
                                length_metrics[
                                    "combined_token_max"
                                ],
                            over_limit_count=
                                length_metrics[
                                    "over_limit_count"
                                ],
                            over_limit_rate=
                                length_metrics[
                                    "over_limit_rate"
                                ],
                            length_valid=True,
                            search_valid=False,
                            unknown_count=0,
                            unknown_rate=0.0,
                            rejection_reason=
                                reason,
                            length_repair_count=
                                repair_count,
                        )

                        all_nodes.append(
                            node
                        )
                        prompt_tree_records.append(
                            node_to_record(
                                node,
                                generation_status=
                                    "contract_rejected",
                                parent_optimization_accuracy=
                                    optimization_acc,
                                parent_error_count=
                                    error_count,
                            )
                        )
                        continue

                    # Smoke test
                    (
                        smoke_acc,
                        smoke_result,
                    ) = evaluate_prompt(
                        dataframe=
                            smoke_df,
                        engine=
                            local_engine,
                        system_prompt=
                            candidate_system,
                        user_instruction=
                            candidate_user,
                    )

                    smoke_unknown_rate = float(
                        (
                            smoke_result[
                                "prediction"
                            ]
                            == "unknown"
                        ).mean()
                    )

                    smoke_reject_reasons = []
                    if smoke_unknown_rate > args.smoke_max_unknown_rate:
                        smoke_reject_reasons.append("unknown_rate")
                    if smoke_acc < args.smoke_min_accuracy:
                        smoke_reject_reasons.append("low_accuracy")

                    if smoke_reject_reasons:
                        node = PromptNode(
                            prompt_id=
                                prompt_id,
                            parent_id=
                                parent.prompt_id,
                            depth=
                                depth,
                            branch_index=
                                branch_index,
                            system_prompt=
                                candidate_system,
                            user_instruction=
                                candidate_user,
                            validation_accuracy=
                                0.0,
                            validation_correct_count=
                                0,
                            system_word_count=
                                length_metrics[
                                    "system_word_count"
                                ],
                            user_instruction_word_count=
                                length_metrics[
                                    "user_instruction_word_count"
                                ],
                            design_token_count=
                                length_metrics[
                                    "design_token_count"
                                ],
                            combined_token_p95=
                                length_metrics[
                                    "combined_token_p95"
                                ],
                            combined_token_max=
                                length_metrics[
                                    "combined_token_max"
                                ],
                            over_limit_count=
                                length_metrics[
                                    "over_limit_count"
                                ],
                            over_limit_rate=
                                length_metrics[
                                    "over_limit_rate"
                                ],
                            length_valid=True,
                            search_valid=False,
                            unknown_count=int(
                                (
                                    smoke_result[
                                        "prediction"
                                    ]
                                    == "unknown"
                                ).sum()
                            ),
                            unknown_rate=
                                smoke_unknown_rate,
                            rejection_reason=
                                "smoke_rejected:" + ",".join(smoke_reject_reasons),
                            length_repair_count=
                                repair_count,
                        )

                        all_nodes.append(
                            node
                        )
                        prompt_tree_records.append(
                            node_to_record(
                                node,
                                generation_status=
                                    "smoke_rejected",
                                parent_optimization_accuracy=
                                    optimization_acc,
                                parent_error_count=
                                    error_count,
                            )
                        )

                        print(
                            f"{prompt_id}: Smoke NG "
                            f"(acc={smoke_acc * 100:.2f}%, "
                            f"unknown={smoke_unknown_rate:.1%})"
                        )
                        continue

                    # Full Validation
                    (
                        val_acc,
                        val_result,
                    ) = evaluate_prompt(
                        dataframe=
                            validation_df,
                        engine=
                            local_engine,
                        system_prompt=
                            candidate_system,
                        user_instruction=
                            candidate_user,
                    )

                    val_correct = int(
                        val_result[
                            "correct"
                        ].sum()
                    )

                    val_unknown_count = int(
                        (
                            val_result[
                                "prediction"
                            ]
                            == "unknown"
                        ).sum()
                    )

                    val_unknown_rate = float(
                        val_unknown_count
                        / len(val_result)
                    )

                    node = PromptNode(
                        prompt_id=
                            prompt_id,
                        parent_id=
                            parent.prompt_id,
                        depth=
                            depth,
                        branch_index=
                            branch_index,
                        system_prompt=
                            candidate_system,
                        user_instruction=
                            candidate_user,
                        validation_accuracy=
                            val_acc,
                        validation_correct_count=
                            val_correct,
                        system_word_count=
                            length_metrics[
                                "system_word_count"
                            ],
                        user_instruction_word_count=
                            length_metrics[
                                "user_instruction_word_count"
                            ],
                        design_token_count=
                            length_metrics[
                                "design_token_count"
                            ],
                        combined_token_p95=
                            length_metrics[
                                "combined_token_p95"
                            ],
                        combined_token_max=
                            length_metrics[
                                "combined_token_max"
                            ],
                        over_limit_count=
                            length_metrics[
                                "over_limit_count"
                            ],
                        over_limit_rate=
                            length_metrics[
                                "over_limit_rate"
                            ],
                        length_valid=True,
                        search_valid=True,
                        unknown_count=
                            val_unknown_count,
                        unknown_rate=
                            val_unknown_rate,
                        rejection_reason="",
                        length_repair_count=
                            repair_count,
                    )

                    all_nodes.append(node)
                    depth_candidates.append(
                        node
                    )

                    prompt_tree_records.append(
                        node_to_record(
                            node,
                            generation_status=
                                "generated",
                            parent_optimization_accuracy=
                                optimization_acc,
                            parent_error_count=
                                error_count,
                        )
                    )

                    print(
                        f"{prompt_id}: "
                        f"val={val_acc * 100:.2f}% "
                        f"({val_correct}/{len(validation_df)}), "
                        f"unknown={val_unknown_count}, "
                        f"tokens={node.design_token_count}"
                    )

                # Parent単位で途中保存
                save_records(
                    prompt_tree_records=
                        prompt_tree_records,
                    batch_feedback_logs=
                        batch_feedback_logs,
                    synthesis_logs=
                        synthesis_logs,
                    delta_logs=
                        delta_logs,
                    gradient_logs=
                        gradient_logs,
                )

            # --------------------------------------------
            # G. Elitism:
            # 現BeamのParent + 新Candidateから次Beam
            # --------------------------------------------
            next_pool = (
                beam
                + [
                    node
                    for node in depth_candidates
                    if node.search_valid
                ]
            )

            beam = choose_beam(
                nodes=next_pool,
                beam_width=
                    args.beam_width,
                near_tie_examples=
                    args.near_tie_examples,
            )

            selected_ids = {
                node.prompt_id
                for node in beam
            }

            for record in prompt_tree_records:
                if (
                    record.get("prompt_id")
                    in selected_ids
                ):
                    record[
                        "selected_for_next_depth"
                    ] = True

            print(
                "\n===== Next Beam ====="
            )

            for rank, node in enumerate(
                beam,
                start=1,
            ):
                print(
                    f"{rank}. {node.prompt_id}: "
                    f"{node.validation_accuracy * 100:.2f}% "
                    f"({node.validation_correct_count}/"
                    f"{len(validation_df)}), "
                    f"design_tokens={node.design_token_count}"
                )

            save_records(
                prompt_tree_records=
                    prompt_tree_records,
                batch_feedback_logs=
                    batch_feedback_logs,
                synthesis_logs=
                    synthesis_logs,
                delta_logs=
                    delta_logs,
                gradient_logs=
                    gradient_logs,
            )

            valid_now = [
                node
                for node in all_nodes
                if node.search_valid
            ]

            best_now = choose_beam(
                nodes=valid_now,
                beam_width=1,
                near_tie_examples=
                    args.near_tie_examples,
            )[0]

            # Full-depth版ではAccuracyによる早期終了を行わない。
            # best_now は記録・確認用であり、Accuracy が高くても
            # 指定した最大Depthまで探索を継続する。
            print(
                f"Current global best after Depth {depth}: "
                f"{best_now.prompt_id} "
                f"{best_now.validation_accuracy * 100:.2f}% "
                f"({best_now.validation_correct_count}/{len(validation_df)})"
            )

    except APICostLimitExceeded as exc:
        stopped_due_cost = True
        print(
            "\n===== API Cost Safety Stop ====="
        )
        print(str(exc))

    # --------------------------------------------------------
    # Independent Prompt-check: Validation上位候補 + Root
    # --------------------------------------------------------
    valid_all_nodes = [node for node in all_nodes if node.search_valid]
    validation_ranked = choose_beam(
        nodes=valid_all_nodes,
        beam_width=min(args.prompt_check_top_k, len(valid_all_nodes)),
        near_tie_examples=args.near_tie_examples,
    )
    check_candidates = {node.prompt_id: node for node in validation_ranked}
    check_candidates[root.prompt_id] = root
    prompt_check_records = []
    print("\n===== Independent Prompt-check =====")
    for node in check_candidates.values():
        check_acc, check_df = evaluate_prompt(
            dataframe=prompt_check_df, engine=local_engine,
            system_prompt=node.system_prompt, user_instruction=node.user_instruction,
        )
        check_correct = int(check_df["correct"].sum())
        check_unknown = int((check_df["prediction"] == "unknown").sum())
        check_binary_wrong = int(len(check_df) - check_correct - check_unknown)
        prompt_check_records.append({
            "prompt_id": node.prompt_id, "depth": node.depth,
            "validation_accuracy": node.validation_accuracy,
            "validation_correct_count": node.validation_correct_count,
            "validation_unknown_count": node.unknown_count,
            "prompt_check_accuracy": check_acc,
            "prompt_check_correct_count": check_correct,
            "prompt_check_unknown_count": check_unknown,
            "prompt_check_binary_wrong_count": check_binary_wrong,
            "design_token_count": node.design_token_count,
            "system_prompt": node.system_prompt, "user_instruction": node.user_instruction,
        })
        print(
            f"{node.prompt_id}: check={check_acc * 100:.2f}% "
            f"({check_correct}/{len(prompt_check_df)}), unknown={check_unknown}, "
            f"binary_wrong={check_binary_wrong}"
        )
    pd.DataFrame(prompt_check_records).to_csv(PROMPT_CHECK_PATH, index=False)
    best_check_correct = max(r["prompt_check_correct_count"] for r in prompt_check_records)
    near_check = [r for r in prompt_check_records
                  if best_check_correct-r["prompt_check_correct_count"] <= args.near_tie_examples]
    selected_check = min(near_check, key=lambda r: (
        r["prompt_check_unknown_count"], r["design_token_count"],
        -r["prompt_check_correct_count"], r["prompt_id"]
    ))
    best_node = next(node for node in valid_all_nodes
                     if node.prompt_id == selected_check["prompt_id"])
    validation_best_node = choose_beam(
        nodes=valid_all_nodes, beam_width=1, near_tie_examples=args.near_tie_examples
    )[0]

    # --------------------------------------------------------
    # Final Best
    # --------------------------------------------------------
    valid_all_nodes = [
        node
        for node in all_nodes
        if node.search_valid
    ]

    best_design = make_design_text(
        best_node.system_prompt,
        best_node.user_instruction,
    )

    best_prompt_data = {
        "problem": 98,
        "method":
            "Llama Global TextGrad Prompt Design Search",
        "base_model":
            MODEL_NAME,
        "initial_condition":
            "system_user",
        "initial_system_prompt":
            INITIAL_SYSTEM_PROMPT,
        "initial_user_instruction":
            INITIAL_USER_INSTRUCTION,
        "best_prompt_id":
            best_node.prompt_id,
        "best_depth":
            best_node.depth,
        "best_system_prompt":
            best_node.system_prompt,
        "best_user_instruction":
            best_node.user_instruction,
        "best_structured_design":
            best_design,
        "fixed_sentence_template":
            "Sentence: {sentence}",
        "fixed_labels":
            ["negative", "positive"],
        "fixed_output_contract":
            "exactly one label: positive or negative",
        "best_validation_accuracy":
            best_node.validation_accuracy,
        "best_validation_correct_count":
            best_node.validation_correct_count,
        "best_prompt_check_accuracy":
            selected_check["prompt_check_accuracy"],
        "best_prompt_check_correct_count":
            selected_check["prompt_check_correct_count"],
        "best_prompt_check_unknown_count":
            selected_check["prompt_check_unknown_count"],
        "validation_only_best_prompt_id":
            validation_best_node.prompt_id,
        "best_design_token_count":
            best_node.design_token_count,
        "seed":
            args.seed,
        "search": {
            "optimization_size":
                len(optimization_df),
            "validation_size":
                len(validation_df),
            "prompt_check_size":
                len(prompt_check_df),
            "error_batch_size":
                args.error_batch_size,
            "representative_errors":
                args.representative_errors,
            "multistart_candidates":
                args.multistart_candidates,
            "branch_factor":
                args.branch_factor,
            "beam_width":
                args.beam_width,
            "depth":
                args.depth,
            "near_tie_examples":
                args.near_tie_examples,
            "delta_max_examples":
                args.delta_max_examples,
            "length_percentile":
                args.length_percentile,
            "max_length":
                args.max_length,
            "smoke_size":
                args.smoke_size,
            "smoke_max_unknown_rate":
                args.smoke_max_unknown_rate,
            "smoke_min_accuracy":
                args.smoke_min_accuracy,
            "prompt_check_top_k":
                args.prompt_check_top_k,
            "accuracy_early_stopping":
                False,
            "dev_used_during_search":
                False,
        },
    }

    with open(
        BEST_PROMPT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            best_prompt_data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        BEST_PROMPT_TXT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "===== SYSTEM PROMPT =====\n"
        )
        f.write(
            best_node.system_prompt
        )
        f.write(
            "\n\n===== USER INSTRUCTION =====\n"
        )
        f.write(
            best_node.user_instruction
            if best_node.user_instruction
            else "(empty)"
        )
        f.write(
            "\n\n===== USER MESSAGE TEMPLATE =====\n"
        )

        if best_node.user_instruction:
            f.write(
                best_node.user_instruction
                + "\n\nSentence: {sentence}\n"
            )
        else:
            f.write(
                "Sentence: {sentence}\n"
            )

    api_engine.save()

    result = {
        "problem": 98,
        "experiment":
            "textgrad_prompt_llama32_1b_root_only",
        "base_model":
            MODEL_NAME,
        "evaluation_engine":
            args.evaluation_engine,
        "seed":
            args.seed,
        "data": {
            "source":
                "SST original train root sentences only",
            "optimization_size":
                len(optimization_df),
            "validation_size":
                len(validation_df),
            "prompt_check_size":
                len(prompt_check_df),
            "dev_used_during_search":
                False,
        },
        "initial": {
            "prompt_id":
                root.prompt_id,
            "validation_accuracy":
                root.validation_accuracy,
            "validation_correct_count":
                root.validation_correct_count,
            "system_prompt":
                root.system_prompt,
            "user_instruction":
                root.user_instruction,
        },
        "best": {
            "prompt_id":
                best_node.prompt_id,
            "depth":
                best_node.depth,
            "validation_accuracy":
                best_node.validation_accuracy,
            "validation_correct_count":
                best_node.validation_correct_count,
            "system_prompt":
                best_node.system_prompt,
            "user_instruction":
                best_node.user_instruction,
            "design_token_count":
                best_node.design_token_count,
            "prompt_check_accuracy": selected_check["prompt_check_accuracy"],
            "prompt_check_correct_count": selected_check["prompt_check_correct_count"],
            "prompt_check_unknown_count": selected_check["prompt_check_unknown_count"],
        },
        "validation_only_best": {
            "prompt_id": validation_best_node.prompt_id,
            "validation_accuracy": validation_best_node.validation_accuracy,
            "validation_correct_count": validation_best_node.validation_correct_count,
            "unknown_count": validation_best_node.unknown_count,
        },
        "search": best_prompt_data[
            "search"
        ],
        "candidate_attempt_count":
            max(
                0,
                len(
                    prompt_tree_records
                )
                - 1,
            ),
        "valid_node_count":
            len(valid_all_nodes),
        "accuracy_early_stopping":
            False,
        "stopped_due_api_cost":
            stopped_due_cost,
        "api_usage": {
            "logged_calls":
                api_engine.call_index,
            "estimated_cost_usd":
                api_engine.cumulative_cost_usd,
            "input_price_per_million":
                args.api_input_price_per_million,
            "output_price_per_million":
                args.api_output_price_per_million,
            "token_counter": (
                "tiktoken"
                if api_engine._encoding
                is not None
                else "chars_div_4_fallback"
            ),
            "note": (
                "Cost is estimated from the configured token prices. "
                "Check the provider dashboard for the authoritative billed amount."
            ),
        },
    }

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    save_records(
        prompt_tree_records=
            prompt_tree_records,
        batch_feedback_logs=
            batch_feedback_logs,
        synthesis_logs=
            synthesis_logs,
        delta_logs=
            delta_logs,
        gradient_logs=
            gradient_logs,
    )

    print(
        "\n"
        + "=" * 72
    )
    print(
        "===== Search Complete ====="
    )
    print(
        "=" * 72
    )
    print(
        f"Initial: "
        f"{root.validation_accuracy * 100:.2f}% "
        f"({root.validation_correct_count}/"
        f"{len(validation_df)})"
    )
    print(
        f"Best   : "
        f"{best_node.validation_accuracy * 100:.2f}% "
        f"({best_node.validation_correct_count}/"
        f"{len(validation_df)})"
    )
    print(
        f"Best ID: {best_node.prompt_id}, "
        f"Depth={best_node.depth}"
    )
    print(
        f"Search policy: accuracy early-stop disabled; "
        f"configured max depth={args.depth}"
    )
    print(
        f"API calls logged: "
        f"{api_engine.call_index}"
    )
    print(
        f"Estimated API cost: "
        f"${api_engine.cumulative_cost_usd:.4f}"
    )

    print(
        "\n===== Best System Prompt ====="
    )
    print(
        best_node.system_prompt
    )

    print(
        "\n===== Best User Instruction ====="
    )
    print(
        best_node.user_instruction
        if best_node.user_instruction
        else "(empty)"
    )

    print(
        "\n===== 保存先 ====="
    )
    print(
        f"Best Prompt : {BEST_PROMPT_PATH}"
    )
    print(
        f"Tree        : {PROMPT_TREE_PATH}"
    )
    print(
        f"Batch FB    : {BATCH_FEEDBACK_PATH}"
    )
    print(
        f"Global FB   : {GLOBAL_SYNTHESIS_PATH}"
    )
    print(
        f"Delta FB    : {DELTA_FEEDBACK_PATH}"
    )
    print(
        f"API usage   : {API_USAGE_PATH}"
    )
    print(
        f"Prompt-check: {PROMPT_CHECK_PATH}"
    )
    print(
        f"Result      : {RESULT_PATH}"
    )


if __name__ == "__main__":
    main()