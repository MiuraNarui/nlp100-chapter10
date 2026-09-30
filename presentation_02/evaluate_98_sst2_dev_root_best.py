#!/usr/bin/env python3
"""Evaluate the initial and optimized prompts on the held-out SST-2 dev set.

This script performs *evaluation only*; it does not update model weights and it
does not call the OpenAI API.  It reads the original Stanford Sentiment
Treebank files in ``data/SST-2/original`` and evaluates the root sentences with
``splitset_label == 3``.  Neutral sentences are removed with the same binary
label rule used by the prompt-optimization scripts:

* sentiment_value <= 0.4 -> negative
* sentiment_value > 0.6  -> positive

After neutral examples are removed, the expected dev-set size is 872.

The script supports both local environments used by this project:

* ``--model llama``: meta-llama/Llama-3.2-1B-Instruct
* ``--model gemma``: google/gemma-4-E2B-it

Run the two models separately because Gemma uses ``.venv-gemma4``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np
import pandas as pd
import torch


SEED = 42

INITIAL_SYSTEM_PROMPT = (
    "You are a sentiment classification model for movie reviews. "
    "Classify each review as positive or negative. "
    "Return only one word: positive or negative."
)
INITIAL_USER_INSTRUCTION = ""

LABEL_TEXT = {0: "negative", 1: "positive"}

MODEL_CONFIGS: dict[str, dict[str, str]] = {
    "llama": {
        "model_name": "meta-llama/Llama-3.2-1B-Instruct",
        "best_prompt_relative": (
            "presentation_02/result/"
            "textgrad_prompt_llama32_1b_root_only/best_prompt.json"
        ),
        "output_relative": (
            "presentation_02/result/"
            "final_sst2_dev_llama32_1b_root_best"
        ),
    },
    "gemma": {
        "model_name": "google/gemma-4-E2B-it",
        "best_prompt_relative": (
            "presentation_02/result/"
            "textgrad_prompt_gemma4_e2b_llama_matched_root_only/"
            "best_prompt.json"
        ),
        "output_relative": (
            "presentation_02/result/"
            "final_sst2_dev_gemma4_e2b_root_best"
        ),
    },
}


class BatchEngine(Protocol):
    model_name: str

    def generate_batch(
        self,
        prompts: list[str],
        system_prompt: Optional[str] = None,
    ) -> list[str]: ...


@dataclass(frozen=True)
class PromptPair:
    initial_system_prompt: str
    initial_user_instruction: str
    best_system_prompt: str
    best_user_instruction: str
    best_prompt_id: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_sst_phrase(text: str) -> str:
    """Normalize only known notation differences between SST source files."""
    normalized = str(text)
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


def load_sst2_dev(data_dir: Path, expected_size: int = 872) -> pd.DataFrame:
    """Load binary root sentences from the official SST dev split (label 3)."""
    sentences_path = data_dir / "datasetSentences.txt"
    split_path = data_dir / "datasetSplit.txt"
    dictionary_path = data_dir / "dictionary.txt"
    labels_path = data_dir / "sentiment_labels.txt"

    required = [sentences_path, split_path, dictionary_path, labels_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "SST originalの必要ファイルが見つかりません:\n- "
            + "\n- ".join(missing)
        )

    sentences = pd.read_csv(
        sentences_path,
        sep="\t",
        keep_default_na=False,
    )
    sentences.columns = ["sentence_index", "sentence"]

    split_info = pd.read_csv(split_path)
    split_info.columns = ["sentence_index", "splitset_label"]

    dev_roots = sentences.merge(
        split_info,
        on="sentence_index",
        how="inner",
        validate="one_to_one",
    )
    dev_roots = dev_roots[dev_roots["splitset_label"] == 3].copy()

    exact_phrase_to_id: dict[str, int] = {}
    normalized_phrase_to_id: dict[str, int] = {}
    ambiguous_normalized: set[str] = set()

    with open(dictionary_path, "r", encoding="utf-8") as file:
        for raw_line in file:
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

    dev_roots["phrase_id"] = dev_roots["sentence"].map(find_phrase_id)
    unmatched = dev_roots[dev_roots["phrase_id"].isna()]
    if len(unmatched):
        examples = unmatched["sentence"].head(5).tolist()
        raise ValueError(
            f"dev root文 {len(unmatched)}件をdictionary.txtに照合できません。"
            f" 例: {examples}"
        )

    dev_roots["phrase_id"] = dev_roots["phrase_id"].astype(int)
    dev_roots["sentence_raw"] = dev_roots["sentence"]
    dev_roots["sentence"] = dev_roots["sentence"].map(normalize_sst_phrase)

    labels = pd.read_csv(labels_path, sep="|")
    labels.columns = ["phrase_id", "sentiment_value"]
    dev_roots = dev_roots.merge(
        labels,
        on="phrase_id",
        how="left",
        validate="many_to_one",
    )
    if dev_roots["sentiment_value"].isna().any():
        raise ValueError("dev root文のsentiment valueに欠損があります。")

    binary = dev_roots[
        (dev_roots["sentiment_value"] <= 0.4)
        | (dev_roots["sentiment_value"] > 0.6)
    ].copy()
    binary["label"] = (binary["sentiment_value"] > 0.6).astype(int)
    binary["true_label"] = binary["label"].map(LABEL_TEXT)
    binary["source_index"] = binary["sentence_index"].astype(int)
    binary["official_split"] = "dev"
    binary = binary.sort_values("sentence_index").reset_index(drop=True)

    if expected_size > 0 and len(binary) != expected_size:
        raise ValueError(
            f"SST-2 devの件数が期待値と一致しません: "
            f"actual={len(binary)}, expected={expected_size}。"
            "異なるデータ版を使用する場合は--expected-size 0を指定してください。"
        )

    return binary[
        [
            "source_index",
            "sentence_index",
            "phrase_id",
            "sentence",
            "sentence_raw",
            "sentiment_value",
            "label",
            "true_label",
            "splitset_label",
            "official_split",
        ]
    ]


def make_user_prompt(sentence: str, user_instruction: str) -> str:
    if user_instruction.strip():
        return f"{user_instruction.strip()}\n\nSentence: {sentence}"
    return f"Sentence: {sentence}"


def parse_prediction(response: str) -> str:
    match = re.search(r"\b(positive|negative)\b", response.lower())
    if match is None:
        return "unknown"
    return match.group(1)


def load_prompt_pair(path: Path) -> PromptPair:
    if not path.exists():
        raise FileNotFoundError(f"best_prompt.jsonが見つかりません: {path}")

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    required = [
        "best_system_prompt",
        "best_user_instruction",
        "best_prompt_id",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(
            f"best_prompt.jsonに必要なキーがありません: {missing}"
        )

    return PromptPair(
        initial_system_prompt=data.get(
            "initial_system_prompt", INITIAL_SYSTEM_PROMPT
        ),
        initial_user_instruction=data.get(
            "initial_user_instruction", INITIAL_USER_INSTRUCTION
        ),
        best_system_prompt=str(data["best_system_prompt"]),
        best_user_instruction=str(data["best_user_instruction"]),
        best_prompt_id=str(data["best_prompt_id"]),
    )


class LlamaEngine:
    def __init__(
        self,
        model_name: str,
        batch_size: int,
        max_length: int,
        max_new_tokens: int,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPUが見つかりません。")

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

        print(f"分類モデルを読み込みます: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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
        system_prompt: Optional[str],
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
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
        responses: list[str] = []
        for start in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[start : start + self.batch_size]
            chat_texts = [
                self._make_chat_text(prompt, system_prompt)
                for prompt in batch_prompts
            ]
            inputs = self.tokenizer(
                chat_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            device = next(self.model.parameters()).device
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
            input_length = inputs["input_ids"].shape[1]
            decoded = self.tokenizer.batch_decode(
                generated[:, input_length:],
                skip_special_tokens=True,
            )
            responses.extend(response.strip() for response in decoded)
        return responses


class GemmaEngine:
    def __init__(
        self,
        model_name: str,
        batch_size: int,
        max_length: int,
        max_new_tokens: int,
    ) -> None:
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPUが見つかりません。")

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

        print(f"分類モデルを読み込みます: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        print(
            "Gemma loaded: "
            f"device={next(self.model.parameters()).device}, "
            f"dtype={next(self.model.parameters()).dtype}"
        )

    def _make_chat_text(
        self,
        user_prompt: str,
        system_prompt: Optional[str],
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: list[str],
        system_prompt: Optional[str] = None,
    ) -> list[str]:
        responses: list[str] = []
        for start in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[start : start + self.batch_size]
            chat_texts = [
                self._make_chat_text(prompt, system_prompt)
                for prompt in batch_prompts
            ]
            inputs = self.processor(
                text=chat_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            device = next(self.model.parameters()).device
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
            )
            input_length = inputs["input_ids"].shape[1]
            decoded = self.processor.batch_decode(
                generated[:, input_length:],
                skip_special_tokens=True,
            )
            responses.extend(response.strip() for response in decoded)
        return responses


def build_engine(
    model_kind: str,
    model_name: str,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
) -> BatchEngine:
    engine_class = LlamaEngine if model_kind == "llama" else GemmaEngine
    return engine_class(
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )


def evaluate_prompt(
    dataframe: pd.DataFrame,
    engine: BatchEngine,
    system_prompt: str,
    user_instruction: str,
    prompt_name: str,
) -> pd.DataFrame:
    user_prompts = [
        make_user_prompt(sentence, user_instruction)
        for sentence in dataframe["sentence"]
    ]
    raw_responses = engine.generate_batch(
        user_prompts,
        system_prompt=system_prompt,
    )
    if len(raw_responses) != len(dataframe):
        raise RuntimeError(
            f"応答数が一致しません: responses={len(raw_responses)}, "
            f"examples={len(dataframe)}"
        )

    result = dataframe.copy()
    result["prompt_name"] = prompt_name
    result["raw_output"] = raw_responses
    result["prediction"] = result["raw_output"].map(parse_prediction)
    result["correct"] = result["prediction"] == result["true_label"]
    result["unknown"] = result["prediction"] == "unknown"
    result["error_type"] = np.select(
        [
            result["unknown"],
            (~result["unknown"])
            & (result["prediction"] != result["true_label"]),
        ],
        [
            "unknown",
            result["true_label"] + "_to_" + result["prediction"],
        ],
        default="correct",
    )
    return result


def combine_results(
    initial: pd.DataFrame,
    best: pd.DataFrame,
) -> pd.DataFrame:
    key_columns = [
        "source_index",
        "sentence_index",
        "phrase_id",
        "sentence",
        "sentence_raw",
        "sentiment_value",
        "label",
        "true_label",
        "splitset_label",
        "official_split",
    ]
    initial_columns = key_columns + [
        "raw_output",
        "prediction",
        "correct",
        "unknown",
        "error_type",
    ]
    best_columns = [
        "source_index",
        "raw_output",
        "prediction",
        "correct",
        "unknown",
        "error_type",
    ]

    combined = initial[initial_columns].rename(
        columns={
            "raw_output": "initial_raw_output",
            "prediction": "initial_prediction",
            "correct": "initial_correct",
            "unknown": "initial_unknown",
            "error_type": "initial_error_type",
        }
    )
    best_part = best[best_columns].rename(
        columns={
            "raw_output": "best_raw_output",
            "prediction": "best_prediction",
            "correct": "best_correct",
            "unknown": "best_unknown",
            "error_type": "best_error_type",
        }
    )
    combined = combined.merge(
        best_part,
        on="source_index",
        how="inner",
        validate="one_to_one",
    )

    conditions = [
        combined["initial_correct"] & combined["best_correct"],
        (~combined["initial_correct"]) & combined["best_correct"],
        combined["initial_correct"] & (~combined["best_correct"]),
        (~combined["initial_correct"])
        & (~combined["best_correct"])
        & (
            combined["initial_prediction"]
            == combined["best_prediction"]
        ),
    ]
    choices = [
        "stable_correct",
        "fixed",
        "regressed",
        "persistent_wrong_same",
    ]
    combined["transition"] = np.select(
        conditions,
        choices,
        default="persistent_wrong_changed",
    )
    combined["prediction_changed"] = (
        combined["initial_prediction"] != combined["best_prediction"]
    )
    return combined


def metrics(result: pd.DataFrame) -> dict[str, Any]:
    correct = int(result["correct"].sum())
    unknown = int(result["unknown"].sum())
    total = int(len(result))
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "unknown": unknown,
        "binary_wrong": total - correct - unknown,
        "gold_positive": int((result["true_label"] == "positive").sum()),
        "gold_negative": int((result["true_label"] == "negative").sum()),
        "predicted_positive": int(
            (result["prediction"] == "positive").sum()
        ),
        "predicted_negative": int(
            (result["prediction"] == "negative").sum()
        ),
    }


def save_outputs(
    output_dir: Path,
    dev_df: pd.DataFrame,
    initial: pd.DataFrame,
    best: pd.DataFrame,
    combined: pd.DataFrame,
    prompt_pair: PromptPair,
    model_kind: str,
    model_name: str,
    best_prompt_path: Path,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    dev_df.to_csv(output_dir / "sst2_dev_manifest.csv", index=False)
    initial.to_csv(output_dir / "initial_predictions.csv", index=False)
    best.to_csv(output_dir / "best_predictions.csv", index=False)
    combined.to_csv(output_dir / "root_best_predictions.csv", index=False)

    transition_summary = (
        combined.groupby("transition", dropna=False)
        .agg(
            count=("source_index", "size"),
            gold_positive=(
                "true_label",
                lambda values: int((values == "positive").sum()),
            ),
            gold_negative=(
                "true_label",
                lambda values: int((values == "negative").sum()),
            ),
        )
        .reset_index()
    )
    transition_summary["rate"] = transition_summary["count"] / len(combined)
    transition_summary.to_csv(
        output_dir / "transition_summary.csv",
        index=False,
    )

    initial_metrics = metrics(initial)
    best_metrics = metrics(best)
    summary = {
        "task": "SST-2 held-out dev root-sentence binary classification",
        "official_split": "dev",
        "splitset_label": 3,
        "model_kind": model_kind,
        "model_name": model_name,
        "best_prompt_source": str(best_prompt_path),
        "best_prompt_id": prompt_pair.best_prompt_id,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "initial": initial_metrics,
        "best": best_metrics,
        "accuracy_delta": (
            best_metrics["accuracy"] - initial_metrics["accuracy"]
        ),
        "correct_count_delta": (
            best_metrics["correct"] - initial_metrics["correct"]
        ),
        "transition_counts": {
            str(key): int(value)
            for key, value in combined["transition"].value_counts().items()
        },
    }
    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    prompts_used = {
        "initial": {
            "system_prompt": prompt_pair.initial_system_prompt,
            "user_instruction": prompt_pair.initial_user_instruction,
        },
        "best": {
            "prompt_id": prompt_pair.best_prompt_id,
            "system_prompt": prompt_pair.best_system_prompt,
            "user_instruction": prompt_pair.best_user_instruction,
        },
    }
    with open(output_dir / "prompts_used.json", "w", encoding="utf-8") as file:
        json.dump(prompts_used, file, ensure_ascii=False, indent=2)

    print("\n===== SST-2 Dev Final Evaluation =====")
    print(f"Model   : {model_name}")
    print(f"Examples: {len(dev_df)}")
    print(
        "Initial : "
        f"{initial_metrics['accuracy']:.2%} "
        f"({initial_metrics['correct']}/{initial_metrics['total']}), "
        f"unknown={initial_metrics['unknown']}"
    )
    print(
        "Best    : "
        f"{best_metrics['accuracy']:.2%} "
        f"({best_metrics['correct']}/{best_metrics['total']}), "
        f"unknown={best_metrics['unknown']}"
    )
    print(f"Delta   : {summary['accuracy_delta']:+.2%}")
    print("Transitions:")
    for name, count in summary["transition_counts"].items():
        print(f"  {name}: {count}")
    print(f"保存先  : {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SST-2 dev 872件を初期プロンプトとBestプロンプトで評価し、"
            "文章単位の変化を保存します。"
        )
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CONFIGS),
        required=True,
        help="評価するローカルモデル。",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Hugging Faceモデル名。省略時は実験と同じモデルを使用。",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="SST originalの4ファイルがあるディレクトリ。",
    )
    parser.add_argument(
        "--best-prompt-json",
        type=Path,
        default=None,
        help="最適化で保存したbest_prompt.json。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="評価結果の保存先。",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--expected-size", type=int, default=872)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="動作確認用。0は全872件、正の値なら先頭N件のみ評価。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="データとプロンプトだけ確認し、モデル推論は行わない。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存の出力CSV/JSONがある場合に上書きを許可。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-sizeは1以上にしてください。")
    if args.max_length <= 0 or args.max_new_tokens <= 0:
        raise ValueError("--max-lengthと--max-new-tokensは1以上にしてください。")
    if args.limit < 0:
        raise ValueError("--limitは0以上にしてください。")

    project_root = Path(__file__).resolve().parents[1]
    config = MODEL_CONFIGS[args.model]
    model_name = args.model_name or config["model_name"]
    data_dir = args.data_dir or project_root / "data" / "SST-2" / "original"
    best_prompt_path = args.best_prompt_json or (
        project_root / config["best_prompt_relative"]
    )
    output_dir = args.output_dir or (
        project_root / config["output_relative"]
    )
    if args.limit > 0 and args.output_dir is None:
        output_dir = Path(f"{output_dir}_limit{args.limit}")

    expected_outputs = [
        output_dir / "sst2_dev_manifest.csv",
        output_dir / "initial_predictions.csv",
        output_dir / "best_predictions.csv",
        output_dir / "root_best_predictions.csv",
        output_dir / "transition_summary.csv",
        output_dir / "evaluation_summary.json",
        output_dir / "prompts_used.json",
    ]
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not args.overwrite and not args.dry_run:
        raise FileExistsError(
            "出力ファイルが既に存在します。上書きする場合は--overwriteを指定してください:\n- "
            + "\n- ".join(str(path) for path in existing)
        )

    set_seed(args.seed)
    dev_df = load_sst2_dev(data_dir, expected_size=args.expected_size)
    prompt_pair = load_prompt_pair(best_prompt_path)

    print("===== SST-2 Dev Evaluation Setup =====")
    print(f"Model kind       : {args.model}")
    print(f"Model name       : {model_name}")
    print(f"Data directory   : {data_dir}")
    print(f"Best prompt JSON : {best_prompt_path}")
    print(f"Dev examples     : {len(dev_df)}")
    print(
        "  positive / negative: "
        f"{int((dev_df['label'] == 1).sum())} / "
        f"{int((dev_df['label'] == 0).sum())}"
    )
    print(f"Best prompt ID   : {prompt_pair.best_prompt_id}")

    if args.dry_run:
        print("dry-runのため、モデル推論とファイル保存は行いません。")
        return

    if args.limit > 0:
        dev_df = dev_df.head(args.limit).copy()
        print(f"動作確認のため先頭{len(dev_df)}件だけ評価します。")

    engine = build_engine(
        model_kind=args.model,
        model_name=model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n初期プロンプトを評価します。")
    initial_result = evaluate_prompt(
        dataframe=dev_df,
        engine=engine,
        system_prompt=prompt_pair.initial_system_prompt,
        user_instruction=prompt_pair.initial_user_instruction,
        prompt_name="initial",
    )

    print("Bestプロンプトを評価します。")
    best_result = evaluate_prompt(
        dataframe=dev_df,
        engine=engine,
        system_prompt=prompt_pair.best_system_prompt,
        user_instruction=prompt_pair.best_user_instruction,
        prompt_name=prompt_pair.best_prompt_id,
    )

    combined = combine_results(initial_result, best_result)
    save_outputs(
        output_dir=output_dir,
        dev_df=dev_df,
        initial=initial_result,
        best=best_result,
        combined=combined,
        prompt_pair=prompt_pair,
        model_kind=args.model,
        model_name=model_name,
        best_prompt_path=best_prompt_path,
        args=args,
    )


if __name__ == "__main__":
    main()