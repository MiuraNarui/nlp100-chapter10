#!/usr/bin/env python3
"""TextGrad optimization for SST-2 rationale-first label generation.

The default experiment directly reads ``sst2_train2000_distill.jsonl``, keeps
only rows whose ``text`` exactly matches a full sentence in
``source_sentences``, excludes rows without original-SST continuous polarity,
and uses all selected roots as one pool without splitting. The student reasons
first and then emits the label. TextGrad may revise only the analysis strategy;
the task, exact JSON contract, output order, grounding rules, and faithful
Japanese translation requirement are immutable. Format and rationale metrics
are recorded but are not hard gates. Prompts are
ranked first by gold-label accuracy; rationale quality, strict JSON, translation
quality, and prompt length are tie-breakers in that order.

The initial visible prompt intentionally matches the earlier successful P000.
Internally, only its first analysis-strategy paragraph is editable. The program
appends the fixed output contract after every proposed strategy, so TextGrad
cannot rewrite the JSON schema or Japanese-translation requirement.

The script uses one in-sample pool of exact SST root sentences. The student sees
only each review. The evaluator additionally sees the gold label, Silver English
and Japanese rationales, original SST p_positive, and label_doubt. Those
evaluator-only fields are never inserted into a student prompt.

Expected repository layout (defaults can be overridden with CLI arguments):

    presentation_02/
      result/
      optimize_98_textgrad_rationale_root_separated_prompt.py
    presentation_02/data/sst2_train2000_distill.jsonl
    data/SST-2/original/
      datasetSentences.txt
      datasetSplit.txt
      SOStr.txt
      dictionary.txt
      sentiment_labels.txt

The exact paths used in a run are written to config.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_RESULT_ROOT = SCRIPT_DIR / "result"
DEFAULT_DISTILL_FILE = SCRIPT_DIR / "data" / "sst2_train2000_distill.jsonl"

SST_NEGATIVE_MAX = 0.4
SST_POSITIVE_MIN = 0.6

MODEL_NAMES = {
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "gemma": "google/gemma-4-E2B-it",
}

LABEL_FIRST = "label_first"
RATIONALE_FIRST = "rationale_first"
OUTPUT_ORDERS = (LABEL_FIRST, RATIONALE_FIRST)

DEFAULT_SYSTEM_PROMPT = (
    "You are a sentiment classification model for movie reviews."
)

PROMPT_TEMPLATE_VERSION = "separated_prompt_label_primary_v3"

# Only this paragraph is revised by TextGrad.  Its initial value intentionally
# matches the earlier P000 prompt that obtained 175/198 correct labels.
INITIAL_ANALYSIS_STRATEGY = """Analyze the language of the review and briefly explain what overall sentiment it expresses. Then classify the review as positive or negative based on that explanation."""

# This block is appended by the program and is never returned by the proposer.
# Keeping the visible student prompt identical to the earlier successful P000
# avoids changing the baseline merely because the implementation separates the
# optimizable and fixed components.
FIXED_RATIONALE_FIRST_CONTRACT = """Return only one valid JSON object with exactly these fields in this order:
- "rationale": a concise English explanation based on specific language in the review
- "label": either "positive" or "negative", consistent with the rationale
- "rationale_ja": a faithful Japanese translation of "rationale"

Base the rationale on the review text.
Do not add unsupported claims or specific facts about the movie that are not stated or clearly implied by the review.
The Japanese translation must not add, remove, or change the reasoning.
Do not include any text outside the JSON object."""


def compose_rationale_first_instruction(analysis_strategy: str) -> str:
    strategy = str(analysis_strategy).strip()
    if not strategy:
        raise ValueError("analysis_strategy must not be empty")
    return f"{strategy}\n\n{FIXED_RATIONALE_FIRST_CONTRACT}"


def extract_analysis_strategy(user_instruction: str) -> str:
    text = str(user_instruction).strip()
    suffix = FIXED_RATIONALE_FIRST_CONTRACT.strip()
    if not text.endswith(suffix):
        raise ValueError("Prompt does not end with the immutable output contract")
    strategy = text[: -len(suffix)].strip()
    if not strategy:
        raise ValueError("Prompt contains an empty analysis strategy")
    return strategy


def analysis_strategy_is_safe(strategy: str) -> bool:
    """Reject candidate text that attempts to rewrite immutable output rules."""
    normalized = normalize_text(strategy)
    forbidden = (
        "json",
        "rationale_ja",
        "japanese translation",
        "translate into japanese",
        "output format",
        "field order",
        "key order",
        "markdown",
        "outside the json",
    )
    return not any(fragment in normalized for fragment in forbidden)

LABEL_FIRST_INSTRUCTION = """Classify the overall sentiment of the review as positive or negative. Then briefly explain the classification.

Return only one valid JSON object with exactly these fields in this order:
- "label": either "positive" or "negative"
- "rationale": a concise English explanation based on specific language in the review
- "rationale_ja": a faithful Japanese translation of "rationale"

Base the rationale on the review text.
Do not add unsupported claims or specific facts about the movie that are not stated or clearly implied by the review.
The Japanese translation must not add, remove, or change the reasoning.
Do not include any text outside the JSON object."""

RATIONALE_FIRST_INSTRUCTION = compose_rationale_first_instruction(
    INITIAL_ANALYSIS_STRATEGY
)


@dataclass(frozen=True)
class Example:
    example_id: str
    source_index: int
    review: str
    gold_label: str
    sentiment_value: float
    reference_rationale: str
    reference_rationale_ja: str
    label_doubt: bool


@dataclass(frozen=True)
class PromptDesign:
    prompt_id: str
    parent_id: str | None
    depth: int
    output_order: str
    system_prompt: str
    user_instruction: str
    hypothesis: str = ""


@dataclass
class Prediction:
    example_id: str
    prompt_id: str
    raw_output: str
    strict_json_success: bool
    parse_success: bool
    predicted_label: str | None
    rationale: str
    rationale_ja: str
    label_correct: bool
    generation_truncated: bool


@dataclass
class Judgment:
    example_id: str
    prompt_id: str
    label_consistency: str
    evidence_support: str
    faithfulness: str
    context_understanding: str
    overall_sentiment: str
    conciseness: str
    sentiment_intensity_alignment: str
    translation_faithfulness: str
    review_difficulty: str
    judge_confidence: str
    strengths: list[str]
    problems: list[str]
    review_evidence: list[str]
    actionable_feedback: str
    reference_rationale_questionable: bool
    judge_parse_success: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize the reasoning block of an SST-2 rationale-first prompt, "
            "selecting candidates primarily by label accuracy."
        )
    )
    parser.add_argument(
        "--stage", choices=("compare", "optimize"), default="optimize"
    )
    parser.add_argument("--model", choices=tuple(MODEL_NAMES), default="llama")
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--data-mode",
        choices=("root_original", "distill", "distill_root_single_pool"),
        default="distill_root_single_pool",
        help=(
            "root_original extracts balanced root sentences from SST original; "
            "distill directly uses every row in sst2_train2000_distill.jsonl; "
            "distill_root_single_pool keeps only exact root sentences with original "
            "SST polarity and uses them without train/validation/check splitting."
        ),
    )
    parser.add_argument(
        "--distill-file",
        type=Path,
        default=DEFAULT_DISTILL_FILE,
        help="JSONL with text, label, rationale, rationale_ja, and label_doubt.",
    )
    parser.add_argument(
        "--output-order",
        choices=OUTPUT_ORDERS,
        default=RATIONALE_FIRST,
        help="Required for --stage optimize; selected after compare.",
    )
    parser.add_argument(
        "--root-train-file",
        type=Path,
        default=None,
        help=(
            "Path to original/datasetSentences.txt. Its sibling SST files are "
            "auto-detected unless their paths are given explicitly."
        ),
    )
    parser.add_argument(
        "--dataset-split-file",
        type=Path,
        default=None,
        help=(
            "datasetSplit.txt corresponding to datasetSentences.txt."
        ),
    )
    parser.add_argument(
        "--sostr-file",
        type=Path,
        default=None,
        help="Original SOStr.txt used to recover the exact root phrase.",
    )
    parser.add_argument("--dictionary-file", type=Path, default=None)
    parser.add_argument("--sentiment-labels-file", type=Path, default=None)
    parser.add_argument("--positive-size", type=int, default=1000)
    parser.add_argument("--negative-size", type=int, default=1000)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--optimization-size", type=int, default=1000)
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--prompt-check-size", type=int, default=500)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")

    parser.add_argument("--evaluation-model", default="gpt-4o")
    parser.add_argument("--proposal-model", default=None)
    parser.add_argument("--judge-batch-size", type=int, default=8)
    parser.add_argument(
        "--validation-judge-size",
        type=int,
        default=100,
        help="Fixed validation subset judged by the API for every prompt; labels use all examples.",
    )
    parser.add_argument(
        "--prompt-check-judge-size",
        type=int,
        default=100,
        help="Fixed prompt-check subset judged by the API; labels use all examples.",
    )
    parser.add_argument("--feedback-sample-size", type=int, default=198)
    parser.add_argument(
        "--feedback-synthesis-batch-size",
        type=int,
        default=40,
        help=(
            "Number of evaluated examples included in one feedback-synthesis API "
            "request. Chunked summaries are merged hierarchically so all selected "
            "examples can be used without exceeding the model TPM limit."
        ),
    )
    parser.add_argument("--max-api-cost-usd", type=float, default=15.0)
    parser.add_argument("--api-input-usd-per-million", type=float, default=2.50)
    parser.add_argument("--api-output-usd-per-million", type=float, default=10.00)
    parser.add_argument("--api-max-retries", type=int, default=4)

    parser.add_argument("--multi-start", type=int, default=4)
    parser.add_argument("--branch-factor", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--label-tolerance-count", type=int, default=1)
    parser.add_argument(
        "--min-strict-json-rate",
        type=float,
        default=0.95,
        help="Legacy compatibility option; reported metrics are not used as a hard gate.",
    )
    parser.add_argument(
        "--rationale-success-threshold",
        type=float,
        default=0.75,
        help="Minimum graded rationale quality for Soft Joint success.",
    )
    parser.add_argument(
        "--max-rationale-quality-drop",
        type=float,
        default=0.03,
        help="Legacy compatibility option; no rationale-quality gate is applied.",
    )
    parser.add_argument(
        "--max-translation-quality-drop",
        type=float,
        default=0.03,
        help="Legacy compatibility option; no translation-quality gate is applied.",
    )
    parser.add_argument("--disable-early-stop", action="store_true")

    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--data-check-only",
        action="store_true",
        help="Validate exact-root single-pool extraction, then exit before loading a model.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Run student inference and label/JSON metrics without API judging.",
    )
    args = parser.parse_args()

    if args.data_mode != "distill_root_single_pool":
        parser.error(
            "This separated-prompt script is dedicated to --data-mode "
            "distill_root_single_pool. Use the earlier script for other modes."
        )

    if args.stage == "optimize" and args.output_order is None:
        parser.error("--output-order is required for --stage optimize")
    if args.stage == "compare" and args.output_order is not None:
        parser.error("--output-order is used only with --stage optimize")
    if args.data_mode == "distill_root_single_pool":
        if args.stage != "optimize":
            parser.error("distill_root_single_pool requires --stage optimize")
        if args.output_order != RATIONALE_FIRST:
            parser.error(
                "distill_root_single_pool requires --output-order rationale_first"
            )
    if args.depth < 1:
        parser.error("--depth must be at least 1")
    if args.multi_start < 1 or args.branch_factor < 1 or args.beam_width < 1:
        parser.error("Search width arguments must be positive")
    if not 0.0 <= args.rationale_success_threshold <= 1.0:
        parser.error("--rationale-success-threshold must be between 0 and 1")
    if args.max_rationale_quality_drop < 0.0:
        parser.error("--max-rationale-quality-drop must be non-negative")
    if args.max_translation_quality_drop < 0.0:
        parser.error("--max-translation-quality-drop must be non-negative")
    if args.positive_size < 1 or args.negative_size < 1:
        parser.error("--positive-size and --negative-size must be positive")
    requested = args.optimization_size + args.validation_size + args.prompt_check_size
    available = args.positive_size + args.negative_size
    if args.data_mode == "root_original" and requested > available:
        parser.error("Requested split sizes exceed --positive-size + --negative-size")
    if min(
        args.optimization_size,
        args.validation_size,
        args.prompt_check_size,
        args.validation_judge_size,
        args.prompt_check_judge_size,
    ) < 0:
        parser.error("Split and judge sizes must be non-negative")
    return args


def normalize_text(text: str) -> str:
    text = str(text).replace("\u00a0", " ").strip().lower()
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "1.0", "positive", "pos"}:
        return "positive"
    if text in {"0", "0.0", "negative", "neg"}:
        return "negative"
    return None


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def append_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def detect_delimiter(path: Path) -> str:
    if path.suffix.lower() == ".tsv":
        return "\t"
    sample = path.read_text(encoding="utf-8", errors="replace")[:8192]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t").delimiter
    except csv.Error:
        return ","


def read_table(path: Path) -> tuple[list[str], list[list[str]]]:
    delimiter = detect_delimiter(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.reader(handle, delimiter=delimiter))
    if not raw:
        raise ValueError(f"Empty table: {path}")

    known_headers = {
        "sentence", "text", "phrase", "review", "label", "sentiment", "target",
        "is_root", "root", "id", "index", "sentence_index", "splitset_label",
    }
    first = [cell.strip().lower() for cell in raw[0]]
    has_header = bool(set(first) & known_headers)
    if has_header:
        return first, raw[1:]

    width = max(len(row) for row in raw)
    return [f"column_{i}" for i in range(width)], raw


def locate_column(headers: Sequence[str], candidates: Sequence[str]) -> int | None:
    normalized = [header.strip().lower() for header in headers]
    for candidate in candidates:
        if candidate in normalized:
            return normalized.index(candidate)
    return None


def auto_find_file(explicit: Path | None, candidates: Sequence[Path], description: str) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{description} not found: {path}")
        return path
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    formatted = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        f"Could not locate {description}. Pass its path explicitly. Tried:\n{formatted}"
    )


def load_sst_split_map(path: Path) -> dict[int, int]:
    headers, rows = read_table(path)
    index_col = locate_column(headers, ("sentence_index", "index", "id"))
    split_col = locate_column(headers, ("splitset_label", "split", "label"))
    if index_col is None or split_col is None:
        if not rows or len(rows[0]) < 2:
            raise ValueError(f"Could not infer sentence/split columns in {path}")
        index_col, split_col = 0, 1
    result: dict[int, int] = {}
    for row in rows:
        if len(row) <= max(index_col, split_col):
            continue
        result[int(row[index_col])] = int(row[split_col])
    return result


def load_dataset_sentences(path: Path) -> dict[int, str]:
    headers, rows = read_table(path)
    text_col = locate_column(headers, ("sentence", "text", "phrase", "review"))
    index_col = locate_column(headers, ("sentence_index", "index", "id"))
    if text_col is None or index_col is None:
        raise ValueError(f"Could not find sentence_index/sentence columns in {path}")
    result: dict[int, str] = {}
    for row in rows:
        if len(row) <= max(text_col, index_col):
            continue
        sentence_index = int(row[index_col])
        sentence = row[text_col].strip()
        if sentence_index in result:
            raise ValueError(f"Duplicate sentence_index={sentence_index} in {path}")
        result[sentence_index] = sentence
    if not result:
        raise ValueError(f"No sentences found in {path}")
    return result


def load_sostr_roots(path: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for sentence_index, line in enumerate(handle, 1):
            result[sentence_index] = " ".join(line.rstrip("\n").split("|"))
    if not result:
        raise ValueError(f"No tokenized sentences found in {path}")
    return result


def load_dictionary(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.rstrip("\n")
            try:
                phrase, phrase_id_text = stripped.rsplit("|", 1)
                phrase_id = int(phrase_id_text)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid dictionary row at {path}:{line_no}") from exc
            if phrase in result and result[phrase] != phrase_id:
                raise ValueError(f"Duplicate dictionary phrase at {path}:{line_no}")
            result[phrase] = phrase_id
    if not result:
        raise ValueError(f"Empty dictionary: {path}")
    return result


def load_sentiment_labels(path: Path) -> dict[int, float]:
    result: dict[int, float] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        header = next(handle, None)
        if header is None:
            raise ValueError(f"Empty sentiment label file: {path}")
        for line_no, line in enumerate(handle, 2):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                phrase_id_text, value_text = stripped.split("|", 1)
                phrase_id = int(phrase_id_text)
                value = float(value_text)
            except ValueError as exc:
                raise ValueError(f"Invalid sentiment row at {path}:{line_no}") from exc
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Sentiment value outside [0, 1] at {path}:{line_no}")
            result[phrase_id] = value
    if not result:
        raise ValueError(f"No sentiment labels found in {path}")
    return result


def class_from_sentiment(value: float) -> str | None:
    if value <= SST_NEGATIVE_MAX:
        return "negative"
    if value > SST_POSITIVE_MIN:
        return "positive"
    return None


def resolve_original_file(
    explicit: Path | None,
    root_path: Path,
    filename: str,
) -> Path:
    return auto_find_file(
        explicit,
        (
            root_path.with_name(filename),
            PROJECT_ROOT / "data" / "SST-2" / "original" / filename,
            SCRIPT_DIR / "data" / "SST-2" / "original" / filename,
            SCRIPT_DIR / "data" / filename,
        ),
        filename,
    )


def build_root_examples(args: argparse.Namespace) -> tuple[list[Example], dict[str, str]]:
    root_path = auto_find_file(
        args.root_train_file,
        (
            PROJECT_ROOT / "data" / "SST-2" / "original" / "datasetSentences.txt",
            SCRIPT_DIR / "data" / "SST-2" / "original" / "datasetSentences.txt",
            SCRIPT_DIR / "data" / "datasetSentences.txt",
        ),
        "datasetSentences.txt",
    )
    split_path = resolve_original_file(args.dataset_split_file, root_path, "datasetSplit.txt")
    sostr_path = resolve_original_file(args.sostr_file, root_path, "SOStr.txt")
    dictionary_path = resolve_original_file(args.dictionary_file, root_path, "dictionary.txt")
    labels_path = resolve_original_file(
        args.sentiment_labels_file, root_path, "sentiment_labels.txt"
    )

    sentences = load_dataset_sentences(root_path)
    split_map = load_sst_split_map(split_path)
    sostr_roots = load_sostr_roots(sostr_path)
    phrase_dictionary = load_dictionary(dictionary_path)
    sentiment_labels = load_sentiment_labels(labels_path)

    all_nontrain_texts = {
        normalize_text(sentences[index])
        for index, split in split_map.items()
        if split != 1 and index in sentences
    }
    eligible: dict[str, list[Example]] = {"positive": [], "negative": []}
    seen_train_texts: set[str] = set()
    skipped_neutral = 0
    skipped_duplicate = 0
    skipped_cross_split_text = 0
    for sentence_index in sorted(sentences):
        if split_map.get(sentence_index) != 1:
            continue
        if sentence_index not in sostr_roots:
            raise ValueError(f"SOStr.txt is missing sentence_index={sentence_index}")
        root_phrase = sostr_roots[sentence_index]
        phrase_id = phrase_dictionary.get(root_phrase)
        if phrase_id is None:
            raise ValueError(
                f"Root phrase for sentence_index={sentence_index} was not found in dictionary.txt"
            )
        if phrase_id not in sentiment_labels:
            raise ValueError(f"phrase_id={phrase_id} is missing from sentiment_labels.txt")
        sentiment_value = sentiment_labels[phrase_id]
        label = class_from_sentiment(sentiment_value)
        if label is None:
            skipped_neutral += 1
            continue
        review = sentences[sentence_index].strip()
        normalized = normalize_text(review)
        if normalized in all_nontrain_texts:
            skipped_cross_split_text += 1
            continue
        if normalized in seen_train_texts:
            skipped_duplicate += 1
            continue
        seen_train_texts.add(normalized)
        eligible[label].append(
            Example(
                example_id=f"root-{sentence_index}",
                source_index=sentence_index,
                review=review,
                gold_label=label,
                sentiment_value=sentiment_value,
                reference_rationale="",
                reference_rationale_ja="",
                label_doubt=False,
            )
        )

    requested = {"positive": args.positive_size, "negative": args.negative_size}
    for label, size in requested.items():
        if len(eligible[label]) < size:
            raise ValueError(
                f"Requested {size} {label} roots, but only {len(eligible[label])} "
                "eligible unique original-train roots are available."
            )
    rng = random.Random(args.split_seed)
    sampled: list[Example] = []
    for label in ("positive", "negative"):
        values = list(eligible[label])
        rng.shuffle(values)
        sampled.extend(values[: requested[label]])
    rng.shuffle(sampled)

    path_info = {
        "dataset_sentences": str(root_path),
        "dataset_split": str(split_path),
        "sostr": str(sostr_path),
        "dictionary": str(dictionary_path),
        "sentiment_labels": str(labels_path),
        "label_rule": (
            f"negative <= {SST_NEGATIVE_MAX}; neutral ({SST_NEGATIVE_MAX}, "
            f"{SST_POSITIVE_MIN}]; positive > {SST_POSITIVE_MIN}"
        ),
        "eligible_positive": str(len(eligible["positive"])),
        "eligible_negative": str(len(eligible["negative"])),
        "skipped_neutral": str(skipped_neutral),
        "skipped_duplicate_train_text": str(skipped_duplicate),
        "skipped_cross_split_text": str(skipped_cross_split_text),
    }
    return sampled, path_info


def build_distill_examples(args: argparse.Namespace) -> tuple[list[Example], dict[str, str]]:
    path = args.distill_file.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Distill JSONL not found: {path}\n"
            "Place sst2_train2000_distill.jsonl in presentation_02/data/ "
            "or pass --distill-file explicitly."
        )
    rows = read_jsonl(path)
    required = {"id", "text", "label", "rationale", "rationale_ja", "label_doubt"}
    examples: list[Example] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    for row_no, row in enumerate(rows, 1):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Missing fields at {path}:{row_no}: {', '.join(missing)}")
        raw_id = str(row["id"]).strip()
        if not raw_id:
            raise ValueError(f"Empty id at {path}:{row_no}")
        if raw_id in seen_ids:
            raise ValueError(f"Duplicate id={raw_id!r} at {path}:{row_no}")
        seen_ids.add(raw_id)

        review = str(row["text"]).strip()
        if not review:
            raise ValueError(f"Empty text at {path}:{row_no}")
        normalized = normalize_text(review)
        if normalized in seen_texts:
            raise ValueError(f"Duplicate normalized text at {path}:{row_no}: {review!r}")
        seen_texts.add(normalized)

        label = normalize_label(row["label"])
        if label is None:
            raise ValueError(f"Invalid binary label at {path}:{row_no}: {row['label']!r}")
        rationale = str(row["rationale"]).strip()
        rationale_ja = str(row["rationale_ja"]).strip()
        if not rationale or not rationale_ja:
            raise ValueError(f"Empty reference rationale at {path}:{row_no}")
        try:
            source_index = int(row["id"])
        except (TypeError, ValueError):
            source_index = row_no - 1
        try:
            sentiment_value = float(row.get("p_positive", "nan"))
        except (TypeError, ValueError):
            sentiment_value = float("nan")
        examples.append(
            Example(
                example_id=f"distill-{raw_id}",
                source_index=source_index,
                review=review,
                gold_label=label,
                sentiment_value=sentiment_value,
                reference_rationale=rationale,
                reference_rationale_ja=rationale_ja,
                label_doubt=normalize_bool(row["label_doubt"]),
            )
        )

    requested = args.optimization_size + args.validation_size + args.prompt_check_size
    if requested > len(examples):
        raise ValueError(
            f"Requested {requested} split examples, but {path} contains only {len(examples)} rows."
        )
    return examples, {
        "distill_file": str(path),
        "rows": str(len(examples)),
        "student_input_fields": "text",
        "evaluator_reference_fields": "label,rationale,rationale_ja,label_doubt",
        "ignored_fields": "source_sentences,p_positive,p_source",
        "reference_status": "Silver LLM-generated rationales; not human Gold",
    }


def build_distill_root_single_pool_examples(
    args: argparse.Namespace,
) -> tuple[list[Example], dict[str, str]]:
    path = args.distill_file.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Distill JSONL not found: {path}\n"
            "Place sst2_train2000_distill.jsonl in presentation_02/data/ "
            "or pass --distill-file explicitly."
        )
    rows = read_jsonl(path)
    required = {
        "id", "text", "source_sentences", "label", "p_positive", "p_source",
        "rationale", "rationale_ja", "label_doubt",
    }
    examples: list[Example] = []
    excluded_non_sst_polarity = 0
    excluded_subtree = 0
    excluded_invalid_polarity = 0
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    for row_no, row in enumerate(rows, 1):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Missing fields at {path}:{row_no}: {', '.join(missing)}")

        # The eight spelling-mismatch records use fallback hard labels rather
        # than a continuous value recovered from the original SST. They must
        # never enter this experiment.
        if str(row["p_source"]).strip().lower() != "sst":
            excluded_non_sst_polarity += 1
            continue
        try:
            sentiment_value = float(row["p_positive"])
        except (TypeError, ValueError):
            excluded_invalid_polarity += 1
            continue
        if not math.isfinite(sentiment_value) or not 0.0 <= sentiment_value <= 1.0:
            excluded_invalid_polarity += 1
            continue

        review = str(row["text"]).strip()
        sources = row["source_sentences"]
        if not isinstance(sources, list):
            raise ValueError(f"source_sentences must be an array at {path}:{row_no}")
        normalized_review = normalize_text(review)
        is_root = any(
            normalized_review == normalize_text(source)
            for source in sources
            if str(source).strip()
        )
        if not is_root:
            excluded_subtree += 1
            continue

        raw_id = str(row["id"]).strip()
        if not raw_id:
            raise ValueError(f"Empty id at {path}:{row_no}")
        if raw_id in seen_ids:
            raise ValueError(f"Duplicate selected id={raw_id!r} at {path}:{row_no}")
        if normalized_review in seen_texts:
            raise ValueError(f"Duplicate selected root text at {path}:{row_no}")
        seen_ids.add(raw_id)
        seen_texts.add(normalized_review)

        label = normalize_label(row["label"])
        if label is None:
            raise ValueError(f"Invalid binary label at {path}:{row_no}: {row['label']!r}")
        rationale = str(row["rationale"]).strip()
        rationale_ja = str(row["rationale_ja"]).strip()
        if not rationale or not rationale_ja:
            raise ValueError(f"Empty reference rationale at {path}:{row_no}")
        try:
            source_index = int(row["id"])
        except (TypeError, ValueError):
            source_index = row_no - 1
        examples.append(
            Example(
                example_id=f"distill-root-{raw_id}",
                source_index=source_index,
                review=review,
                gold_label=label,
                sentiment_value=sentiment_value,
                reference_rationale=rationale,
                reference_rationale_ja=rationale_ja,
                label_doubt=normalize_bool(row["label_doubt"]),
            )
        )

    if not examples:
        raise ValueError("No exact root sentences with original SST polarity were found")
    return examples, {
        "distill_file": str(path),
        "input_rows": str(len(rows)),
        "selected_exact_root_rows": str(len(examples)),
        "excluded_non_sst_polarity": str(excluded_non_sst_polarity),
        "excluded_subtree": str(excluded_subtree),
        "excluded_invalid_polarity": str(excluded_invalid_polarity),
        "root_rule": "normalized text exactly equals one source_sentences entry",
        "student_input_fields": "text",
        "evaluator_reference_fields": (
            "label,rationale,rationale_ja,p_positive,label_doubt"
        ),
        "reference_status": "Silver LLM-generated rationales; not human Gold",
        "selection_scope": "single in-sample root pool; no split",
    }


def build_examples(args: argparse.Namespace) -> tuple[list[Example], dict[str, str]]:
    if args.data_mode == "distill_root_single_pool":
        return build_distill_root_single_pool_examples(args)
    if args.data_mode == "distill":
        return build_distill_examples(args)
    return build_root_examples(args)


def stratified_split(
    examples: Sequence[Example],
    optimization_size: int,
    validation_size: int,
    prompt_check_size: int,
    seed: int,
) -> tuple[list[Example], list[Example], list[Example]]:
    by_label: dict[str, list[Example]] = {"positive": [], "negative": []}
    for example in examples:
        by_label[example.gold_label].append(example)
    rng = random.Random(seed)
    for values in by_label.values():
        rng.shuffle(values)

    total_requested = optimization_size + validation_size + prompt_check_size
    positive_total = len(by_label["positive"])
    negative_total = len(by_label["negative"])
    if total_requested > positive_total + negative_total:
        raise ValueError("Not enough root examples for the requested split")

    split_sizes = (optimization_size, validation_size, prompt_check_size)
    total_available = positive_total + negative_total
    positive_targets: list[int] = []
    cumulative_size = 0
    cumulative_positive = 0
    for size in split_sizes:
        cumulative_size += size
        target_cumulative = round(cumulative_size * positive_total / total_available)
        positive_targets.append(target_cumulative - cumulative_positive)
        cumulative_positive = target_cumulative
    negative_targets = [size - pos for size, pos in zip(split_sizes, positive_targets)]
    if sum(positive_targets) > positive_total or sum(negative_targets) > negative_total:
        raise ValueError("Could not construct balanced requested splits")

    result: list[list[Example]] = []
    positive_start = 0
    negative_start = 0
    for size, pos_count, neg_count in zip(
        split_sizes, positive_targets, negative_targets, strict=True
    ):
        values = (
            by_label["positive"][positive_start : positive_start + pos_count]
            + by_label["negative"][negative_start : negative_start + neg_count]
        )
        if len(values) != size:
            raise ValueError("Could not construct the requested stratified split")
        rng.shuffle(values)
        result.append(values)
        positive_start += pos_count
        negative_start += neg_count
    optimization, validation, prompt_check = result
    return optimization, validation, prompt_check


def count_labels(examples: Sequence[Example]) -> dict[str, int]:
    return {
        "positive": sum(e.gold_label == "positive" for e in examples),
        "negative": sum(e.gold_label == "negative" for e in examples),
        "label_doubt_true": sum(e.label_doubt for e in examples),
    }


def initial_prompt(output_order: str) -> PromptDesign:
    instruction = (
        LABEL_FIRST_INSTRUCTION if output_order == LABEL_FIRST else RATIONALE_FIRST_INSTRUCTION
    )
    return PromptDesign(
        prompt_id="P000",
        parent_id=None,
        depth=0,
        output_order=output_order,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        user_instruction=instruction,
        hypothesis="Initial minimal prompt.",
    )


class LocalStudentModel:
    def __init__(self, args: argparse.Namespace):
        import torch

        self.torch = torch
        self.args = args
        self.model_kind = args.model
        self.model_name = args.model_name or MODEL_NAMES[args.model]
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[args.dtype]

        print(f"Loading student model: {self.model_name}")
        if self.model_kind == "llama":
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.processor = AutoTokenizer.from_pretrained(self.model_name)
            if self.processor.pad_token_id is None:
                self.processor.pad_token = self.processor.eos_token
            self.processor.padding_side = "left"
            self.processor.truncation_side = "left"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto",
            )
        else:
            try:
                from transformers import AutoModelForMultimodalLM, AutoProcessor
            except ImportError as exc:
                raise RuntimeError(
                    "Gemma 4 requires a recent Transformers build providing "
                    "AutoModelForMultimodalLM. Run this script with the Gemma environment."
                ) from exc
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer is not None:
                tokenizer.padding_side = "left"
                tokenizer.truncation_side = "left"
            self.model = AutoModelForMultimodalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto",
            )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"Student loaded: device={self.device}, dtype={dtype}")

    def _messages(self, prompt: PromptDesign, review: str) -> list[dict[str, Any]]:
        user_text = (
            f"{prompt.user_instruction}\n\n"
            "<REVIEW>\n"
            f"{review}\n"
            "</REVIEW>"
        )
        if self.model_kind == "gemma":
            return [
                {"role": "system", "content": [{"type": "text", "text": prompt.system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": user_text}]},
            ]
        return [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": user_text},
        ]

    def _render(self, prompt: PromptDesign, review: str) -> str:
        messages = self._messages(prompt, review)
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, prompt: PromptDesign, examples: Sequence[Example]) -> list[str]:
        outputs: list[str] = []
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        for start in range(0, len(examples), self.args.batch_size):
            batch = examples[start : start + self.args.batch_size]
            rendered = [self._render(prompt, example.review) for example in batch]
            encoded = self.processor(
                text=rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.args.max_input_tokens,
            )
            encoded = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
            input_length = encoded["input_ids"].shape[1]
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=self.args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated_only = generated[:, input_length:]
            decoded = tokenizer.batch_decode(generated_only, skip_special_tokens=True)
            outputs.extend(text.strip() for text in decoded)
            print(
                f"  inference {min(start + len(batch), len(examples))}/{len(examples)}",
                flush=True,
            )
        return outputs


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, bool]:
    stripped = text.strip()
    try:
        value = json.loads(stripped)
        return (value if isinstance(value, dict) else None), isinstance(value, dict)
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
        return (value if isinstance(value, dict) else None), False
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for position, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value, False
    return None, False


def rescue_label(text: str) -> str | None:
    matches = re.findall(r"\b(positive|negative)\b", text.lower())
    unique = list(dict.fromkeys(matches))
    return unique[0] if len(unique) == 1 else None


def parse_prediction(
    example: Example,
    prompt: PromptDesign,
    raw_output: str,
    max_new_tokens: int,
) -> Prediction:
    value, strict = extract_json_object(raw_output)
    parse_success = value is not None
    label: str | None = None
    rationale = ""
    rationale_ja = ""
    if value is not None:
        label = normalize_label(value.get("label"))
        rationale = str(value.get("rationale", "")).strip()
        rationale_ja = str(value.get("rationale_ja", "")).strip()
    if label is None:
        label = rescue_label(raw_output)
    # Exact token truncation is model-specific; this conservative flag catches
    # visibly unfinished JSON and unusually long outputs.
    truncated = bool(raw_output.strip() and not raw_output.rstrip().endswith("}"))
    expected_keys = (
        ["label", "rationale", "rationale_ja"]
        if prompt.output_order == LABEL_FIRST
        else ["rationale", "label", "rationale_ja"]
    )
    return Prediction(
        example_id=example.example_id,
        prompt_id=prompt.prompt_id,
        raw_output=raw_output,
        strict_json_success=(
            strict
            and value is not None
            and list(value.keys()) == expected_keys
            and label in {"positive", "negative"}
            and bool(rationale)
            and bool(rationale_ja)
        ),
        parse_success=parse_success,
        predicted_label=label,
        rationale=rationale,
        rationale_ja=rationale_ja,
        label_correct=(label == example.gold_label),
        generation_truncated=truncated,
    )


class ApiBudgetExceeded(RuntimeError):
    pass


class OpenAIEvaluator:
    def __init__(self, args: argparse.Namespace, output_dir: Path):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the openai package to use API judging") from exc
        self.client = OpenAI()
        self.args = args
        self.output_dir = output_dir
        self.usage_path = output_dir / "api_usage.csv"
        self.total_cost = 0.0
        self._load_existing_cost()

    def _load_existing_cost(self) -> None:
        if not self.usage_path.exists():
            return
        with self.usage_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    self.total_cost += float(row["cost_usd"])
                except (KeyError, ValueError):
                    continue

    def _record_usage(
        self,
        purpose: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        cost = (
            prompt_tokens * self.args.api_input_usd_per_million
            + completion_tokens * self.args.api_output_usd_per_million
        ) / 1_000_000
        self.total_cost += cost
        new_file = not self.usage_path.exists()
        with self.usage_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "timestamp", "purpose", "model", "prompt_tokens",
                    "completion_tokens", "cost_usd", "cumulative_cost_usd",
                ),
            )
            if new_file:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "purpose": purpose,
                    "model": model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost_usd": f"{cost:.8f}",
                    "cumulative_cost_usd": f"{self.total_cost:.8f}",
                }
            )
        print(f"  API cost: ${self.total_cost:.4f} / ${self.args.max_api_cost_usd:.2f}")

    def _call_json(self, purpose: str, model: str, system: str, user: str) -> dict[str, Any]:
        if self.total_cost >= self.args.max_api_cost_usd:
            raise ApiBudgetExceeded(
                f"API budget reached: ${self.total_cost:.4f} >= ${self.args.max_api_cost_usd:.2f}"
            )
        last_error: Exception | None = None
        for attempt in range(self.args.api_max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                usage = response.usage
                self._record_usage(
                    purpose,
                    model,
                    int(getattr(usage, "prompt_tokens", 0) or 0),
                    int(getattr(usage, "completion_tokens", 0) or 0),
                )
                content = response.choices[0].message.content or "{}"
                append_jsonl(
                    self.output_dir / "api_responses.jsonl",
                    [
                        {
                            "timestamp": time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                            ),
                            "purpose": purpose,
                            "model": model,
                            "raw_response": content,
                        }
                    ],
                )
                return json.loads(content)
            except ApiBudgetExceeded:
                raise
            except Exception as exc:  # API/network/JSON errors are retried.
                last_error = exc
                wait_seconds = min(2**attempt, 20)
                print(f"  API attempt {attempt + 1} failed: {exc}; retrying in {wait_seconds}s")
                time.sleep(wait_seconds)
        raise RuntimeError(f"API call failed after retries: {last_error}")

    def judge_batch(
        self,
        prompt: PromptDesign,
        examples: Sequence[Example],
        predictions: Sequence[Prediction],
    ) -> list[Judgment]:
        system = """You are a strict evaluator of movie-review sentiment classifications and rationales.

The gold label is authoritative. Judge from the review text, gold label, student's output, and—when supplied—the Silver reference rationale and original SST p_positive value. A Silver reference is an auxiliary comparison point generated by another LLM, not a unique human Gold answer. Do not demand wording overlap, and do not copy a possible error from the reference. The p_positive value is a human-annotation-derived sentiment polarity on [0, 1], not the student's confidence: values near 0 are more negative, values near 1 are more positive, and values near the middle are weaker or more ambiguous. Use it only as evaluator-side diagnostic context. If label_doubt is true, scrutinize the reference especially carefully. No source sentence or subtree evidence is available.

For each item, evaluate the generated English rationale using pass, partial, fail, or not_applicable for:
- label_consistency: supports the model's own predicted label
- evidence_support: uses concrete language from the review
- faithfulness: adds no unsupported movie-specific claims
- context_understanding: handles relevant negation, contrast, sarcasm, idioms, metaphor, or rhetorical questions
- overall_sentiment: explains the review's overall rather than merely local sentiment
- conciseness: concise and clear
- sentiment_intensity_alignment: describes the strength, mixedness, or ambiguity of sentiment consistently with the review and the reference p_positive value; do not require numeric wording
- translation_faithfulness: Japanese faithfully preserves the English rationale

Also assign:
- review_difficulty: one of clear, mixed, implicit, highly_ambiguous
- judge_confidence: one of high, medium, low

General linguistic knowledge needed to interpret the review is allowed, but unsupported movie-specific facts are not. A fluent rationale cannot compensate for a wrong label. Partial is a meaningful intermediate rating and must not be collapsed into fail. For an intrinsically ambiguous review, reward a rationale that explicitly recognizes the ambiguity and gives a defensible account of the dominant sentiment. Also report whether the Silver reference rationale itself appears questionable given the review and gold label. Give prompt-level actionable feedback that generalizes beyond the individual review. Return JSON only."""

        items = []
        for example, prediction in zip(examples, predictions, strict=True):
            items.append(
                {
                    "example_id": example.example_id,
                    "review": example.review,
                    "gold_label": example.gold_label,
                    "predicted_label": prediction.predicted_label,
                    "generated_rationale": prediction.rationale,
                    "generated_rationale_ja": prediction.rationale_ja,
                    "reference_rationale": example.reference_rationale or None,
                    "reference_rationale_ja": example.reference_rationale_ja or None,
                    "reference_p_positive": (
                        example.sentiment_value
                        if math.isfinite(example.sentiment_value)
                        else None
                    ),
                    "label_doubt": example.label_doubt,
                    "strict_json_success": prediction.strict_json_success,
                }
            )
        user = json.dumps(
            {
                "instruction": (
                    "Evaluate every item. Return {\"evaluations\": [...]} with exactly one "
                    "entry per example_id. Each entry must contain example_id, all eight "
                    "rubric fields, review_difficulty, judge_confidence, strengths (array), problems (array), review_evidence "
                    "(array of exact short spans), actionable_feedback, and "
                    "reference_rationale_questionable (boolean)."
                ),
                "items": items,
            },
            ensure_ascii=False,
        )
        payload = self._call_json("judge", self.args.evaluation_model, system, user)
        raw_evaluations = payload.get("evaluations", [])
        by_id = {
            str(item.get("example_id")): item
            for item in raw_evaluations
            if isinstance(item, dict)
        }
        judgments: list[Judgment] = []
        allowed = {"pass", "partial", "fail", "not_applicable"}
        for example in examples:
            item = by_id.get(example.example_id)
            if item is None:
                judgments.append(failed_judgment(example.example_id, prompt.prompt_id, "Missing judge result"))
                continue

            def rating(name: str) -> str:
                value = str(item.get(name, "fail")).strip().lower()
                return value if value in allowed else "fail"

            judgments.append(
                Judgment(
                    example_id=example.example_id,
                    prompt_id=prompt.prompt_id,
                    label_consistency=rating("label_consistency"),
                    evidence_support=rating("evidence_support"),
                    faithfulness=rating("faithfulness"),
                    context_understanding=rating("context_understanding"),
                    overall_sentiment=rating("overall_sentiment"),
                    conciseness=rating("conciseness"),
                    sentiment_intensity_alignment=rating(
                        "sentiment_intensity_alignment"
                    ),
                    translation_faithfulness=rating("translation_faithfulness"),
                    review_difficulty=(
                        str(item.get("review_difficulty", "")).strip().lower()
                        if str(item.get("review_difficulty", "")).strip().lower()
                        in {"clear", "mixed", "implicit", "highly_ambiguous"}
                        else "highly_ambiguous"
                    ),
                    judge_confidence=(
                        str(item.get("judge_confidence", "")).strip().lower()
                        if str(item.get("judge_confidence", "")).strip().lower()
                        in {"high", "medium", "low"}
                        else "low"
                    ),
                    strengths=[str(x) for x in item.get("strengths", [])],
                    problems=[str(x) for x in item.get("problems", [])],
                    review_evidence=[str(x) for x in item.get("review_evidence", [])],
                    actionable_feedback=str(item.get("actionable_feedback", "")),
                    reference_rationale_questionable=bool(
                        item.get("reference_rationale_questionable", False)
                    ),
                )
            )
        return judgments

    def synthesize_feedback(
        self,
        prompt: PromptDesign,
        examples: Sequence[Example],
        predictions: Sequence[Prediction],
        judgments: Sequence[Judgment],
    ) -> str:
        system = """You synthesize textual gradients for the optimizable analysis-strategy section of a reusable sentiment-classification prompt. The primary optimization target is gold-label accuracy. Use rationale ratings and written evaluator comments as diagnostic evidence about why a label was correct or incorrect; do not improve fluency at the expense of label accuracy. Identify recurring reasoning weaknesses jointly across examples. Do not propose memorizing individual answers. Preserve successful behavior. The task, review-only student input, JSON schema, output order, grounding rules, and faithful Japanese translation requirement are immutable. Report format or translation failures, but do not recommend changing the fixed contract. Return JSON only."""
        records = []
        for example, prediction, judgment in zip(examples, predictions, judgments, strict=True):
            records.append(
                {
                    "example_id": example.example_id,
                    "review": example.review,
                    "gold_label": example.gold_label,
                    "predicted_label": prediction.predicted_label,
                    "label_correct": prediction.label_correct,
                    "strict_json_success": prediction.strict_json_success,
                    "generated_rationale": prediction.rationale,
                    "generated_rationale_ja": prediction.rationale_ja,
                    "reference_rationale": example.reference_rationale or None,
                    "reference_rationale_ja": example.reference_rationale_ja or None,
                    "reference_p_positive": (
                        example.sentiment_value
                        if math.isfinite(example.sentiment_value)
                        else None
                    ),
                    "label_doubt": example.label_doubt,
                    "ratings": {
                        "label_consistency": judgment.label_consistency,
                        "evidence_support": judgment.evidence_support,
                        "faithfulness": judgment.faithfulness,
                        "context_understanding": judgment.context_understanding,
                        "overall_sentiment": judgment.overall_sentiment,
                        "sentiment_intensity_alignment": (
                            judgment.sentiment_intensity_alignment
                        ),
                        "translation_faithfulness": judgment.translation_faithfulness,
                        "review_difficulty": judgment.review_difficulty,
                        "judge_confidence": judgment.judge_confidence,
                    },
                    "graded_rationale_quality": rationale_quality_score(judgment),
                    "critical_rationale_error": judgment_has_critical_error(
                        judgment
                    ),
                    "strengths": judgment.strengths,
                    "problems": judgment.problems,
                    "review_evidence": judgment.review_evidence,
                    "actionable_feedback": judgment.actionable_feedback,
                }
            )
        batch_size = max(1, int(self.args.feedback_synthesis_batch_size))
        instruction = (
            "Return an object with recurring_weaknesses (array), strengths_to_preserve "
            "(array), and textual_gradient (a concise actionable paragraph). Prioritize "
            "changes likely to correct wrong labels. Treat rationale and translation "
            "scores as secondary diagnostic evidence, not hard acceptance criteria."
        )

        # A single synthesis over all 198 rationale records can exceed the API's
        # tokens-per-minute request limit. Summarize bounded chunks first, then
        # merge those summaries into one prompt-level textual gradient. This still
        # uses every selected example while keeping each request comfortably small.
        if len(records) <= batch_size:
            user = json.dumps(
                {
                    "current_analysis_strategy": extract_analysis_strategy(
                        prompt.user_instruction
                    ),
                    "immutable_contract_summary": (
                        "review-only input; rationale->label->rationale_ja; exact JSON; "
                        "grounded English rationale; faithful complete Japanese translation"
                    ),
                    "records": records,
                    "instruction": instruction,
                },
                ensure_ascii=False,
            )
            payload = self._call_json(
                "feedback_synthesis", self.args.evaluation_model, system, user
            )
        else:
            partial_summaries: list[dict[str, Any]] = []
            chunks = [
                records[start : start + batch_size]
                for start in range(0, len(records), batch_size)
            ]
            print(
                f"[{prompt.prompt_id}] Feedback synthesis: {len(records)} records "
                f"in {len(chunks)} chunks (batch_size={batch_size})"
            )
            for chunk_index, chunk in enumerate(chunks, start=1):
                chunk_user = json.dumps(
                    {
                        "current_analysis_strategy": extract_analysis_strategy(
                            prompt.user_instruction
                        ),
                        "immutable_contract_summary": (
                            "review-only input; rationale->label->rationale_ja; exact JSON; "
                            "grounded rationale; faithful Japanese translation"
                        ),
                        "chunk_index": chunk_index,
                        "chunk_count": len(chunks),
                        "records": chunk,
                        "instruction": (
                            instruction
                            + " Focus on patterns supported by this chunk; do not infer "
                            "their global frequency yet."
                        ),
                    },
                    ensure_ascii=False,
                )
                chunk_payload = self._call_json(
                    "feedback_synthesis_chunk",
                    self.args.evaluation_model,
                    system,
                    chunk_user,
                )
                partial_summaries.append(
                    {
                        "chunk_index": chunk_index,
                        "example_count": len(chunk),
                        "summary": chunk_payload,
                    }
                )
                print(f"  synthesized chunk {chunk_index}/{len(chunks)}")

            merge_user = json.dumps(
                {
                    "current_analysis_strategy": extract_analysis_strategy(
                        prompt.user_instruction
                    ),
                    "immutable_contract_summary": (
                        "review-only input; rationale->label->rationale_ja; exact JSON; "
                        "grounded rationale; faithful Japanese translation"
                    ),
                    "total_example_count": len(records),
                    "chunk_summaries": partial_summaries,
                    "instruction": (
                        "Merge the chunk summaries into one global prompt-level analysis. "
                        "Prioritize weaknesses recurring across chunks, retain important "
                        "minority failure modes, remove duplicates, and do not introduce "
                        "example-specific rules. "
                        + instruction
                    ),
                },
                ensure_ascii=False,
            )
            payload = self._call_json(
                "feedback_synthesis_merge",
                self.args.evaluation_model,
                system,
                merge_user,
            )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def propose_prompts(
        self,
        parent: PromptDesign,
        feedback: str,
        count: int,
        next_id: int,
    ) -> list[PromptDesign]:
        model = self.args.proposal_model or self.args.evaluation_model
        system = """You revise only the analysis-strategy section of a reusable sentiment prompt using textual gradients. The primary goal is higher sentiment-label accuracy; rationale quality is secondary diagnostic evidence. Produce meaningfully different, reusable reasoning strategies rather than example-specific rules. Never rewrite or repeat the system prompt or fixed output contract. Do not insert gold labels, reference rationales, dataset examples, polarity scores, rubric scores, or evaluator-only information into the strategy. Return JSON only."""
        user = json.dumps(
            {
                "fixed_system_prompt": DEFAULT_SYSTEM_PROMPT,
                "parent_analysis_strategy": extract_analysis_strategy(
                    parent.user_instruction
                ),
                "immutable_output_contract": FIXED_RATIONALE_FIRST_CONTRACT,
                "textual_gradient": feedback,
                "required_count": count,
                "required_output_order": parent.output_order,
                "instruction": (
                    "Return {\"candidates\": [...]} with exactly required_count entries. "
                    "Each entry must contain analysis_strategy and hypothesis. "
                    "Candidates must address different recurring weaknesses and not differ only "
                    "in wording. Keep each analysis strategy compact. Do not include the fixed "
                    "output contract in analysis_strategy because the program appends it."
                ),
            },
            ensure_ascii=False,
        )
        payload = self._call_json("prompt_proposal", model, system, user)
        raw = payload.get("candidates", [])
        candidates: list[PromptDesign] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            analysis_strategy = str(item.get("analysis_strategy", "")).strip()
            if not analysis_strategy:
                continue
            if not analysis_strategy_is_safe(analysis_strategy):
                continue
            system_prompt = DEFAULT_SYSTEM_PROMPT
            user_instruction = compose_rationale_first_instruction(
                analysis_strategy
            )
            fingerprint = prompt_fingerprint(system_prompt, user_instruction)
            if fingerprint in seen or fingerprint == prompt_fingerprint(
                parent.system_prompt, parent.user_instruction
            ):
                continue
            seen.add(fingerprint)
            prompt_id = f"P{next_id + len(candidates):03d}"
            candidates.append(
                PromptDesign(
                    prompt_id=prompt_id,
                    parent_id=parent.prompt_id,
                    depth=parent.depth + 1,
                    output_order=parent.output_order,
                    system_prompt=system_prompt,
                    user_instruction=user_instruction,
                    hypothesis=str(item.get("hypothesis", "")).strip(),
                )
            )
            if len(candidates) == count:
                break
        if len(candidates) != count:
            raise RuntimeError(
                f"Proposal model returned {len(candidates)} usable candidates; expected {count}."
            )
        return candidates


def failed_judgment(example_id: str, prompt_id: str, message: str) -> Judgment:
    return Judgment(
        example_id=example_id,
        prompt_id=prompt_id,
        label_consistency="fail",
        evidence_support="fail",
        faithfulness="fail",
        context_understanding="not_applicable",
        overall_sentiment="fail",
        conciseness="fail",
        sentiment_intensity_alignment="fail",
        translation_faithfulness="fail",
        review_difficulty="highly_ambiguous",
        judge_confidence="low",
        strengths=[],
        problems=[message],
        review_evidence=[],
        actionable_feedback=message,
        reference_rationale_questionable=False,
        judge_parse_success=False,
    )


def prompt_fingerprint(system_prompt: str, user_instruction: str) -> str:
    canonical = normalize_text(system_prompt) + "\n" + normalize_text(user_instruction)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


RATING_POINTS = {
    "fail": 0.0,
    "partial": 0.5,
    "pass": 1.0,
    "not_applicable": None,
}

RATIONALE_QUALITY_FIELDS = (
    "label_consistency",
    "evidence_support",
    "faithfulness",
    "context_understanding",
    "overall_sentiment",
    "conciseness",
    "sentiment_intensity_alignment",
)

CRITICAL_RATIONALE_FIELDS = (
    "label_consistency",
    "faithfulness",
    "overall_sentiment",
)


def rationale_quality_score(judgment: Judgment) -> float:
    scores = [
        RATING_POINTS.get(getattr(judgment, field), 0.0)
        for field in RATIONALE_QUALITY_FIELDS
    ]
    applicable = [score for score in scores if score is not None]
    return sum(applicable) / len(applicable) if applicable else 0.0


def translation_quality_score(judgment: Judgment) -> float:
    # Translation is mandatory, so not_applicable must not improve the mean.
    score = RATING_POINTS.get(judgment.translation_faithfulness, 0.0)
    return 0.0 if score is None else score


def judgment_has_critical_error(judgment: Judgment) -> bool:
    return any(
        getattr(judgment, field) == "fail"
        for field in CRITICAL_RATIONALE_FIELDS
    )


def judgment_passes_rationale(
    judgment: Judgment,
    threshold: float = 0.75,
) -> bool:
    """Soft rationale success: graded quality with non-negotiable safety gates."""
    return (
        not judgment_has_critical_error(judgment)
        and judgment.evidence_support != "fail"
        and rationale_quality_score(judgment) >= threshold
    )


def judgment_passes_strict_rationale(judgment: Judgment) -> bool:
    """All applicable rationale dimensions must receive pass."""
    return all(
        getattr(judgment, field) in {"pass", "not_applicable"}
        for field in RATIONALE_QUALITY_FIELDS
    )


def compute_metrics(
    examples: Sequence[Example],
    predictions: Sequence[Prediction],
    judged_examples: Sequence[Example] | None,
    judged_predictions: Sequence[Prediction] | None,
    judgments: Sequence[Judgment] | None,
    rationale_success_threshold: float = 0.75,
) -> dict[str, Any]:
    n = len(examples)
    label_correct = sum(prediction.label_correct for prediction in predictions)
    strict_json = sum(prediction.strict_json_success for prediction in predictions)
    label_extract = sum(prediction.predicted_label is not None for prediction in predictions)
    metrics: dict[str, Any] = {
        "n": n,
        "label_correct_count": label_correct,
        "label_accuracy": label_correct / n if n else 0.0,
        "strict_json_count": strict_json,
        "strict_json_rate": strict_json / n if n else 0.0,
        "label_extraction_count": label_extract,
        "label_extraction_rate": label_extract / n if n else 0.0,
        "unknown_count": n - label_extract,
    }
    if judgments is None or judged_examples is None or judged_predictions is None:
        return metrics
    if not (
        len(judged_examples) == len(judged_predictions) == len(judgments)
    ):
        raise ValueError("Judged examples, predictions, and judgments must have equal lengths")

    soft_joint_count = 0
    strict_joint_count = 0
    complete_success_count = 0
    rationale_success_among_correct = 0
    translation_success_among_joint = 0
    judged_label_correct = 0
    quality_total = 0.0
    translation_total = 0.0
    translation_n = 0
    critical_error_count = 0
    difficulty_counts = {
        "clear": 0,
        "mixed": 0,
        "implicit": 0,
        "highly_ambiguous": 0,
    }
    confidence_counts = {"high": 0, "medium": 0, "low": 0}
    for example, prediction, judgment in zip(
        judged_examples, judged_predictions, judgments, strict=True
    ):
        rationale_score = rationale_quality_score(judgment)
        rationale_ok = judgment_passes_rationale(
            judgment, rationale_success_threshold
        )
        strict_rationale_ok = judgment_passes_strict_rationale(judgment)
        soft_joint = prediction.label_correct and rationale_ok
        strict_joint = prediction.label_correct and strict_rationale_ok
        complete_success = (
            soft_joint
            and prediction.strict_json_success
            and judgment.translation_faithfulness == "pass"
        )
        soft_joint_count += int(soft_joint)
        strict_joint_count += int(strict_joint)
        complete_success_count += int(complete_success)
        judged_label_correct += int(prediction.label_correct)
        rationale_success_among_correct += int(prediction.label_correct and rationale_ok)
        translation_success_among_joint += int(
            soft_joint and judgment.translation_faithfulness == "pass"
        )
        quality_total += rationale_score
        translation_score = translation_quality_score(judgment)
        translation_total += translation_score
        translation_n += 1
        critical_error_count += int(judgment_has_critical_error(judgment))
        difficulty_counts[judgment.review_difficulty] = (
            difficulty_counts.get(judgment.review_difficulty, 0) + 1
        )
        confidence_counts[judgment.judge_confidence] = (
            confidence_counts.get(judgment.judge_confidence, 0) + 1
        )

    judge_n = len(judged_examples)
    metrics.update(
        {
            "judge_n": judge_n,
            "judged_label_correct_count": judged_label_correct,
            "judged_label_accuracy": (
                judged_label_correct / judge_n if judge_n else 0.0
            ),
            # joint_success_* aliases Soft Joint for backwards-compatible logs.
            "joint_success_count": soft_joint_count,
            "joint_success_rate": soft_joint_count / judge_n if judge_n else 0.0,
            "soft_joint_success_count": soft_joint_count,
            "soft_joint_success_rate": (
                soft_joint_count / judge_n if judge_n else 0.0
            ),
            "strict_joint_success_count": strict_joint_count,
            "strict_joint_success_rate": (
                strict_joint_count / judge_n if judge_n else 0.0
            ),
            "complete_success_count": complete_success_count,
            "complete_success_rate": (
                complete_success_count / judge_n if judge_n else 0.0
            ),
            "conditional_rationale_success_rate": (
                rationale_success_among_correct / judged_label_correct
                if judged_label_correct else 0.0
            ),
            "translation_success_among_joint_rate": (
                translation_success_among_joint / soft_joint_count
                if soft_joint_count else 0.0
            ),
            "mean_rationale_quality": quality_total / judge_n if judge_n else 0.0,
            "mean_translation_quality": (
                translation_total / translation_n if translation_n else 0.0
            ),
            "critical_rationale_error_count": critical_error_count,
            "critical_rationale_error_rate": (
                critical_error_count / judge_n if judge_n else 0.0
            ),
            "rationale_success_threshold": rationale_success_threshold,
            "review_difficulty_counts": difficulty_counts,
            "judge_confidence_counts": confidence_counts,
        }
    )
    return metrics


class ArtifactStore:
    def __init__(self, output_dir: Path, overwrite: bool):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_path = output_dir / "predictions.jsonl"
        self.judgments_path = output_dir / "judgments.jsonl"
        self.prompt_metrics_path = output_dir / "prompt_metrics.jsonl"
        self.prompts_path = output_dir / "prompts.jsonl"
        self.textual_gradients_path = output_dir / "textual_gradients.jsonl"
        if overwrite:
            for path in (
                self.predictions_path,
                self.judgments_path,
                self.prompt_metrics_path,
                self.prompts_path,
                self.textual_gradients_path,
                output_dir / "api_usage.csv",
                output_dir / "api_responses.jsonl",
                output_dir / "run_state.json",
                output_dir / "best_prompt.json",
            ):
                if path.exists():
                    path.unlink()
            feedback_dir = output_dir / "feedback"
            if feedback_dir.exists():
                shutil.rmtree(feedback_dir)
        self.predictions = self._load_keyed(self.predictions_path)
        self.judgments = self._load_keyed(self.judgments_path)
        self.prompt_metrics = self._load_metrics()
        self.prompts = self._load_prompts()
        self.textual_gradients = self._load_textual_gradients()

    @staticmethod
    def _load_keyed(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
        result: dict[tuple[str, str, str], dict[str, Any]] = {}
        if not path.exists():
            return result
        for row in read_jsonl(path):
            split = str(row.get("split", ""))
            result[(str(row["prompt_id"]), str(row["example_id"]), split)] = row
        return result

    def _load_metrics(self) -> dict[tuple[str, str], dict[str, Any]]:
        result: dict[tuple[str, str], dict[str, Any]] = {}
        if self.prompt_metrics_path.exists():
            for row in read_jsonl(self.prompt_metrics_path):
                result[(str(row["prompt_id"]), str(row["split"]))] = row
        return result

    def _load_prompts(self) -> dict[str, PromptDesign]:
        result: dict[str, PromptDesign] = {}
        if self.prompts_path.exists():
            for row in read_jsonl(self.prompts_path):
                result[str(row["prompt_id"])] = PromptDesign(**row)
        return result

    def _load_textual_gradients(self) -> dict[tuple[int, str], dict[str, Any]]:
        result: dict[tuple[int, str], dict[str, Any]] = {}
        if self.textual_gradients_path.exists():
            for row in read_jsonl(self.textual_gradients_path):
                result[(int(row["depth"]), str(row["parent_id"]))] = row
        return result

    def save_prompt(self, prompt: PromptDesign) -> None:
        if prompt.prompt_id in self.prompts:
            return
        append_jsonl(self.prompts_path, [asdict(prompt)])
        self.prompts[prompt.prompt_id] = prompt

    def save_predictions(
        self,
        split: str,
        examples: Sequence[Example],
        predictions: Sequence[Prediction],
    ) -> None:
        if len(examples) != len(predictions):
            raise ValueError("Examples and predictions must have equal lengths")
        rows = []
        for example, prediction in zip(examples, predictions, strict=True):
            key = (prediction.prompt_id, prediction.example_id, split)
            if key in self.predictions:
                continue
            row = {
                "split": split,
                **asdict(prediction),
                "review": example.review,
                "gold_label": example.gold_label,
                "reference_rationale": example.reference_rationale,
                "reference_rationale_ja": example.reference_rationale_ja,
                "reference_p_positive": (
                    example.sentiment_value
                    if math.isfinite(example.sentiment_value)
                    else None
                ),
                "label_doubt": example.label_doubt,
            }
            rows.append(row)
            self.predictions[key] = row
        append_jsonl(self.predictions_path, rows)

    def save_judgments(self, split: str, judgments: Sequence[Judgment]) -> None:
        rows = []
        for judgment in judgments:
            key = (judgment.prompt_id, judgment.example_id, split)
            if key in self.judgments:
                continue
            row = {
                "split": split,
                **asdict(judgment),
                "rationale_quality_score": rationale_quality_score(judgment),
                "critical_rationale_error": judgment_has_critical_error(judgment),
                "strict_rationale_success": judgment_passes_strict_rationale(
                    judgment
                ),
                "translation_quality_score": translation_quality_score(judgment),
            }
            rows.append(row)
            self.judgments[key] = row
        append_jsonl(self.judgments_path, rows)

    def save_metrics(self, prompt: PromptDesign, split: str, metrics: dict[str, Any]) -> None:
        key = (prompt.prompt_id, split)
        row = {
            "prompt_id": prompt.prompt_id,
            "parent_id": prompt.parent_id,
            "depth": prompt.depth,
            "output_order": prompt.output_order,
            "split": split,
            **metrics,
        }
        if self.prompt_metrics.get(key) == row:
            return
        append_jsonl(self.prompt_metrics_path, [row])
        self.prompt_metrics[key] = row

    def save_textual_gradient(
        self,
        depth: int,
        parent_id: str,
        selected_example_ids: Sequence[str],
        synthesis: dict[str, Any],
    ) -> None:
        """Save one prompt-level textual gradient in an analysis-friendly file."""
        key = (int(depth), str(parent_id))
        row = {
            "depth": int(depth),
            "parent_id": str(parent_id),
            "selected_example_ids": list(selected_example_ids),
            "synthesis": synthesis,
        }
        if self.textual_gradients.get(key) == row:
            return
        if key in self.textual_gradients:
            raise RuntimeError(
                "A different textual gradient already exists for "
                f"depth={depth}, parent_id={parent_id}. Use --overwrite."
            )
        append_jsonl(self.textual_gradients_path, [row])
        self.textual_gradients[key] = row

    def get_predictions(
        self, prompt_id: str, split: str, examples: Sequence[Example]
    ) -> list[Prediction] | None:
        prediction_fields = set(Prediction.__dataclass_fields__)
        rows = []
        for example in examples:
            row = self.predictions.get((prompt_id, example.example_id, split))
            if row is None:
                return None
            rows.append(
                Prediction(**{k: v for k, v in row.items() if k in prediction_fields})
            )
        return rows

    def get_judgments(
        self, prompt_id: str, split: str, examples: Sequence[Example]
    ) -> list[Judgment] | None:
        judgment_fields = set(Judgment.__dataclass_fields__)
        rows = []
        for example in examples:
            row = self.judgments.get((prompt_id, example.example_id, split))
            if row is None:
                return None
            values = {k: v for k, v in row.items() if k in judgment_fields}
            values.setdefault("reference_rationale_questionable", False)
            values.setdefault("sentiment_intensity_alignment", "not_applicable")
            values.setdefault("review_difficulty", "highly_ambiguous")
            values.setdefault("judge_confidence", "low")
            rows.append(Judgment(**values))
        return rows


def fixed_judge_indices(
    examples: Sequence[Example], limit: int, seed: int, split: str
) -> list[int]:
    if limit <= 0:
        return []
    if limit >= len(examples):
        return list(range(len(examples)))
    by_label: dict[str, list[int]] = {"positive": [], "negative": []}
    for index, example in enumerate(examples):
        by_label[example.gold_label].append(index)

    def stable_key(index: int) -> str:
        value = f"{seed}:{split}:{examples[index].example_id}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    for values in by_label.values():
        values.sort(key=stable_key)
    positive_target = min((limit + 1) // 2, len(by_label["positive"]))
    negative_target = min(limit - positive_target, len(by_label["negative"]))
    if positive_target + negative_target < limit:
        positive_target = min(
            limit - negative_target, len(by_label["positive"])
        )
    selected = (
        by_label["positive"][:positive_target]
        + by_label["negative"][:negative_target]
    )
    selected.sort()
    if len(selected) != limit:
        raise ValueError(f"Could not select {limit} fixed judge examples for {split}")
    return selected


def evaluate_prompt(
    prompt: PromptDesign,
    split: str,
    examples: Sequence[Example],
    student: LocalStudentModel,
    evaluator: OpenAIEvaluator | None,
    store: ArtifactStore,
    args: argparse.Namespace,
) -> tuple[list[Prediction], list[Judgment] | None, dict[str, Any]]:
    cached_predictions = store.get_predictions(prompt.prompt_id, split, examples)
    if cached_predictions is None:
        print(f"\n[{prompt.prompt_id}] Student inference on {split}: {len(examples)} examples")
        raw_outputs = student.generate(prompt, examples)
        predictions = [
            parse_prediction(example, prompt, raw, args.max_new_tokens)
            for example, raw in zip(examples, raw_outputs, strict=True)
        ]
        store.save_predictions(split, examples, predictions)
    else:
        predictions = cached_predictions
        print(f"\n[{prompt.prompt_id}] Reusing cached {split} predictions")

    judged_examples: list[Example] | None = None
    judged_predictions: list[Prediction] | None = None
    judgments: list[Judgment] | None = None
    if evaluator is not None:
        if split == "validation":
            judge_limit = min(args.validation_judge_size, len(examples))
        elif split == "prompt_check":
            judge_limit = min(args.prompt_check_judge_size, len(examples))
        else:
            judge_limit = len(examples)
        judged_indices = fixed_judge_indices(
            examples, judge_limit, args.split_seed, split
        )
        judged_examples = [examples[index] for index in judged_indices]
        judged_predictions = [predictions[index] for index in judged_indices]
        cached_judgments = store.get_judgments(
            prompt.prompt_id, split, judged_examples
        )
        if cached_judgments is not None:
            judgments = cached_judgments
            print(
                f"[{prompt.prompt_id}] Reusing cached {split} judgments "
                f"({len(judged_examples)}/{len(examples)})"
            )
        else:
            judgments = []
            print(
                f"[{prompt.prompt_id}] API judging on {split}: "
                f"{len(judged_examples)}/{len(examples)} fixed examples"
            )
            for start in range(0, len(judged_examples), args.judge_batch_size):
                example_batch = judged_examples[start : start + args.judge_batch_size]
                prediction_batch = judged_predictions[
                    start : start + args.judge_batch_size
                ]
                batch_judgments = store.get_judgments(
                    prompt.prompt_id, split, example_batch
                )
                if batch_judgments is None:
                    batch_judgments = evaluator.judge_batch(
                        prompt, example_batch, prediction_batch
                    )
                    store.save_judgments(split, batch_judgments)
                judgments.extend(batch_judgments)
                print(
                    f"  judged {min(start + len(example_batch), len(judged_examples))}/"
                    f"{len(judged_examples)}"
                )

    metrics = compute_metrics(
        examples,
        predictions,
        judged_examples,
        judged_predictions,
        judgments,
        args.rationale_success_threshold,
    )
    store.save_metrics(prompt, split, metrics)
    print_metrics(prompt, split, metrics)
    return predictions, judgments, metrics


def print_metrics(prompt: PromptDesign, split: str, metrics: dict[str, Any]) -> None:
    print(
        f"[{prompt.prompt_id}] {split}: "
        f"label={metrics['label_correct_count']}/{metrics['n']} "
        f"({metrics['label_accuracy']:.2%}), "
        f"strict_json={metrics['strict_json_count']}/{metrics['n']} "
        f"({metrics['strict_json_rate']:.2%})",
        end="",
    )
    if "joint_success_count" in metrics:
        print(
            f", soft_joint={metrics['soft_joint_success_count']}/"
            f"{metrics['judge_n']} judged "
            f"({metrics['soft_joint_success_rate']:.2%}), "
            f"complete={metrics['complete_success_count']}/"
            f"{metrics['judge_n']} "
            f"({metrics['complete_success_rate']:.2%})"
        )
    else:
        print()


def select_feedback_examples(
    examples: Sequence[Example], predictions: Sequence[Prediction], size: int, seed: int
) -> list[int]:
    wrong = [i for i, prediction in enumerate(predictions) if not prediction.label_correct]
    format_errors = [
        i for i, prediction in enumerate(predictions)
        if not prediction.strict_json_success and i not in wrong
    ]
    correct = [
        i for i, prediction in enumerate(predictions)
        if prediction.label_correct and prediction.strict_json_success
    ]
    rng = random.Random(seed)
    rng.shuffle(wrong)
    rng.shuffle(format_errors)
    rng.shuffle(correct)

    chosen: list[int] = []
    # Prioritize errors, while retaining at least one quarter successful cases
    # so the gradient can state what should be preserved.
    success_target = max(1, size // 4)
    error_target = max(0, size - success_target)
    chosen.extend((wrong + format_errors)[:error_target])
    chosen.extend(correct[:success_target])
    if len(chosen) < size:
        remaining = [i for i in range(len(examples)) if i not in set(chosen)]
        rng.shuffle(remaining)
        chosen.extend(remaining[: size - len(chosen)])
    return chosen[:size]


def rank_prompts(
    prompt_metrics: Sequence[tuple[PromptDesign, dict[str, Any]]],
    args: argparse.Namespace,
) -> list[tuple[PromptDesign, dict[str, Any]]]:
    if not prompt_metrics:
        return []
    eligible_format = [
        item for item in prompt_metrics
        if item[1]["strict_json_rate"] >= args.min_strict_json_rate
        and item[1]["label_extraction_rate"] == 1.0
    ]
    pool = eligible_format or list(prompt_metrics)
    best_label_count = max(metrics["label_correct_count"] for _, metrics in pool)
    label_pool = [
        item for item in pool
        if item[1]["label_correct_count"] >= best_label_count - args.label_tolerance_count
    ]

    def key(item: tuple[PromptDesign, dict[str, Any]]) -> tuple[float, ...]:
        prompt, metrics = item
        return (
            float(metrics.get("joint_success_count", -1)),
            float(metrics["label_correct_count"]),
            float(metrics.get("mean_rationale_quality", -1)),
            float(metrics["strict_json_count"]),
            -float(len(prompt.system_prompt) + len(prompt.user_instruction)),
        )

    ranked_label_pool = sorted(label_pool, key=key, reverse=True)
    remainder = [item for item in pool if item not in label_pool]
    remainder.sort(key=key, reverse=True)
    ineligible = [item for item in prompt_metrics if item not in pool]
    ineligible.sort(key=key, reverse=True)
    return ranked_label_pool + remainder + ineligible


def rank_prompts_by_label_accuracy(
    prompt_metrics: Sequence[tuple[PromptDesign, dict[str, Any]]],
    args: argparse.Namespace,
    baseline_metrics: dict[str, Any] | None = None,
) -> list[tuple[PromptDesign, dict[str, Any]]]:
    """Rank every prompt lexicographically, with label accuracy strictly first."""
    if not prompt_metrics:
        return []

    def key(item: tuple[PromptDesign, dict[str, Any]]) -> tuple[float, ...]:
        prompt, metrics = item
        return (
            float(metrics["label_correct_count"]),
            float(metrics.get("mean_rationale_quality", -1)),
            float(metrics.get("strict_json_count", -1)),
            float(metrics.get("mean_translation_quality", -1)),
            -float(len(prompt.system_prompt) + len(prompt.user_instruction)),
        )
    return sorted(prompt_metrics, key=key, reverse=True)


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Return Pearson's r, or None when fewer than two varying pairs exist."""
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys, strict=True)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return None
    x_mean = sum(x for x, _ in pairs) / len(pairs)
    y_mean = sum(y for _, y in pairs) / len(pairs)
    x_delta = [x - x_mean for x, _ in pairs]
    y_delta = [y - y_mean for _, y in pairs]
    denominator = math.sqrt(
        sum(value * value for value in x_delta)
        * sum(value * value for value in y_delta)
    )
    if denominator == 0.0:
        return None
    return sum(x * y for x, y in zip(x_delta, y_delta, strict=True)) / denominator


def select_single_pool_feedback_examples(
    examples: Sequence[Example],
    predictions: Sequence[Prediction],
    judgments: Sequence[Judgment],
    size: int,
    seed: int,
    rationale_success_threshold: float = 0.75,
) -> list[int]:
    """Prioritize label errors, then rationale defects, while preserving successes."""
    wrong = [i for i, prediction in enumerate(predictions) if not prediction.label_correct]
    rationale_errors = [
        i
        for i, (prediction, judgment) in enumerate(
            zip(predictions, judgments, strict=True)
        )
        if prediction.label_correct
        and (
            not prediction.strict_json_success
            or not judgment_passes_rationale(
                judgment, rationale_success_threshold
            )
            or judgment.translation_faithfulness != "pass"
        )
    ]
    successful = [
        i
        for i, (prediction, judgment) in enumerate(
            zip(predictions, judgments, strict=True)
        )
        if prediction.label_correct
        and prediction.strict_json_success
        and judgment_passes_rationale(
            judgment, rationale_success_threshold
        )
        and judgment.translation_faithfulness == "pass"
    ]
    rng = random.Random(seed)
    for values in (wrong, rationale_errors, successful):
        rng.shuffle(values)
    success_target = max(1, size // 5)
    chosen = (wrong + rationale_errors)[: max(0, size - success_target)]
    chosen.extend(successful[:success_target])
    if len(chosen) < size:
        chosen_set = set(chosen)
        remaining = [i for i in range(len(examples)) if i not in chosen_set]
        rng.shuffle(remaining)
        chosen.extend(remaining[: size - len(chosen)])
    return chosen[:size]


def run_compare(
    args: argparse.Namespace,
    validation: Sequence[Example],
    prompt_check: Sequence[Example],
    student: LocalStudentModel,
    evaluator: OpenAIEvaluator | None,
    store: ArtifactStore,
) -> None:
    results: list[tuple[PromptDesign, dict[str, Any]]] = []
    for index, order in enumerate(OUTPUT_ORDERS):
        prompt = initial_prompt(order)
        prompt = PromptDesign(**{**asdict(prompt), "prompt_id": f"C{index:03d}"})
        store.save_prompt(prompt)
        _, _, metrics = evaluate_prompt(
            prompt, "validation", validation, student, evaluator, store, args
        )
        results.append((prompt, metrics))

    ranked = rank_prompts(results, args)
    recommended_prompt = ranked[0][0]
    _, _, prompt_check_metrics = evaluate_prompt(
        recommended_prompt,
        "prompt_check",
        prompt_check,
        student,
        evaluator,
        store,
        args,
    )
    summary = {
        "ranking": [
            {"rank": rank, "prompt": asdict(prompt), "metrics": metrics}
            for rank, (prompt, metrics) in enumerate(ranked, 1)
        ],
        "recommended_output_order": recommended_prompt.output_order,
        "recommended_prompt_check_metrics": prompt_check_metrics,
        "note": (
            "The output order is selected on validation. Prompt-check is confirmation only "
            "and is not used to change the selection."
        ),
    }
    write_json(store.output_dir / "comparison_result.json", summary)
    print("\n===== Output-order comparison =====")
    for item in summary["ranking"]:
        print(
            f"{item['rank']}. {item['prompt']['output_order']}: "
            f"label={item['metrics']['label_accuracy']:.2%}, "
            f"joint={item['metrics'].get('joint_success_rate', float('nan')):.2%}, "
            f"strict_json={item['metrics']['strict_json_rate']:.2%}"
        )
    print(f"Recommended: {summary['recommended_output_order']}")
    print(
        "Prompt-check confirmation: "
        f"label={prompt_check_metrics['label_accuracy']:.2%}, "
        f"joint={prompt_check_metrics.get('joint_success_rate', float('nan')):.2%}"
    )


def run_optimize(
    args: argparse.Namespace,
    optimization: Sequence[Example],
    validation: Sequence[Example],
    prompt_check: Sequence[Example],
    student: LocalStudentModel,
    evaluator: OpenAIEvaluator,
    store: ArtifactStore,
) -> None:
    root = initial_prompt(args.output_order)
    store.save_prompt(root)
    _, _, root_metrics = evaluate_prompt(
        root, "validation", validation, student, evaluator, store, args
    )
    all_results: list[tuple[PromptDesign, dict[str, Any]]] = [(root, root_metrics)]
    beam = [root]
    next_id = 1
    best_joint = root_metrics.get("joint_success_count", -1)
    no_improvement_depths = 0

    for depth in range(1, args.depth + 1):
        print("\n" + "=" * 72)
        print(f"Depth {depth}/{args.depth}")
        print("=" * 72)
        children: list[PromptDesign] = []
        for parent_index, parent in enumerate(beam):
            parent_predictions, _, _ = evaluate_prompt(
                parent, "optimization", optimization, student, None, store, args
            )
            candidate_count = args.multi_start if depth == 1 else args.branch_factor
            existing_children = sorted(
                (
                    prompt for prompt in store.prompts.values()
                    if prompt.parent_id == parent.prompt_id and prompt.depth == depth
                ),
                key=lambda prompt: prompt.prompt_id,
            )
            if len(existing_children) >= candidate_count:
                proposed = existing_children[:candidate_count]
                print(
                    f"[{parent.prompt_id}] Reusing {len(proposed)} saved prompt proposals"
                )
                next_id = max(
                    (int(match.group(1)) + 1 for prompt in store.prompts.values()
                     if (match := re.fullmatch(r"P(\d+)", prompt.prompt_id))),
                    default=next_id,
                )
                children.extend(proposed)
                continue
            if existing_children:
                raise RuntimeError(
                    f"Found only {len(existing_children)}/{candidate_count} saved children "
                    f"for {parent.prompt_id}. Re-run with --overwrite to avoid mixing runs."
                )

            selected_indices = select_feedback_examples(
                optimization,
                parent_predictions,
                min(args.feedback_sample_size, len(optimization)),
                args.split_seed + depth * 100 + parent_index,
            )
            selected_examples = [optimization[i] for i in selected_indices]
            selected_predictions = [parent_predictions[i] for i in selected_indices]

            # Store optimization judgments under a distinct split name that
            # identifies the fixed feedback subset for this parent.
            feedback_split = f"optimization_feedback_d{depth}_{parent.prompt_id}"
            feedback_judgments = store.get_judgments(
                parent.prompt_id, feedback_split, selected_examples
            )
            if feedback_judgments is None:
                feedback_judgments = []
                for start in range(0, len(selected_examples), args.judge_batch_size):
                    batch_examples = selected_examples[start : start + args.judge_batch_size]
                    batch_predictions = selected_predictions[start : start + args.judge_batch_size]
                    batch_judgments = store.get_judgments(
                        parent.prompt_id, feedback_split, batch_examples
                    )
                    if batch_judgments is None:
                        batch_judgments = evaluator.judge_batch(
                            parent, batch_examples, batch_predictions
                        )
                        store.save_judgments(feedback_split, batch_judgments)
                    feedback_judgments.extend(batch_judgments)

            feedback_path = store.output_dir / "feedback" / f"depth_{depth}_{parent.prompt_id}.json"
            feedback_path.parent.mkdir(parents=True, exist_ok=True)
            if feedback_path.exists():
                saved_feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
                synthesis = json.dumps(
                    saved_feedback["synthesis"], ensure_ascii=False, indent=2
                )
                print(f"[{parent.prompt_id}] Reusing saved feedback synthesis")
            else:
                synthesis = evaluator.synthesize_feedback(
                    parent, selected_examples, selected_predictions, feedback_judgments
                )
                write_json(
                    feedback_path,
                    {
                        "depth": depth,
                        "parent_id": parent.prompt_id,
                        "selected_example_ids": [e.example_id for e in selected_examples],
                        "synthesis": json.loads(synthesis),
                    },
                )
            proposed = evaluator.propose_prompts(
                parent, synthesis, candidate_count, next_id
            )
            for candidate in proposed:
                store.save_prompt(candidate)
            next_id = max(
                (int(match.group(1)) + 1 for prompt in store.prompts.values()
                 if (match := re.fullmatch(r"P(\d+)", prompt.prompt_id))),
                default=next_id + len(proposed),
            )
            children.extend(proposed)

        depth_results: list[tuple[PromptDesign, dict[str, Any]]] = []
        for child in children:
            _, _, metrics = evaluate_prompt(
                child, "validation", validation, student, evaluator, store, args
            )
            depth_results.append((child, metrics))
            all_results.append((child, metrics))

        ranked_depth = rank_prompts(depth_results, args)
        beam = [prompt for prompt, _ in ranked_depth[: args.beam_width]]
        print("\nNext beam:")
        for rank, (prompt, metrics) in enumerate(ranked_depth[: args.beam_width], 1):
            print(
                f"  {rank}. {prompt.prompt_id}: label={metrics['label_accuracy']:.2%}, "
                f"joint={metrics.get('joint_success_rate', float('nan')):.2%}, "
                f"json={metrics['strict_json_rate']:.2%}"
            )

        depth_best_joint = max(
            (metrics.get("joint_success_count", -1) for _, metrics in depth_results),
            default=-1,
        )
        if depth_best_joint > best_joint:
            best_joint = depth_best_joint
            no_improvement_depths = 0
        else:
            no_improvement_depths += 1
        if not args.disable_early_stop and no_improvement_depths >= 2:
            print("Early stop: joint success did not improve for two consecutive depths.")
            break

    ranked_all = rank_prompts(all_results, args)
    best_prompt, best_metrics = ranked_all[0]
    _, _, root_prompt_check_metrics = evaluate_prompt(
        root, "prompt_check", prompt_check, student, evaluator, store, args
    )
    _, _, best_prompt_check_metrics = evaluate_prompt(
        best_prompt, "prompt_check", prompt_check, student, evaluator, store, args
    )
    write_json(
        store.output_dir / "best_prompt.json",
        {
            "prompt": asdict(best_prompt),
            "optimized_analysis_strategy": extract_analysis_strategy(
                best_prompt.user_instruction
            ),
            "immutable_output_contract": FIXED_RATIONALE_FIRST_CONTRACT,
            "validation_metrics": best_metrics,
            "prompt_check": {
                "initial_prompt_id": root.prompt_id,
                "initial_metrics": root_prompt_check_metrics,
                "best_prompt_id": best_prompt.prompt_id,
                "best_metrics": best_prompt_check_metrics,
                "selection_note": (
                    "Prompt-check was evaluated only after selection and was not used "
                    "to choose the best prompt."
                ),
            },
            "selection_rule": {
                "format_requirement": (
                    f"strict_json_rate >= {args.min_strict_json_rate} and "
                    "label_extraction_rate == 1.0"
                ),
                "label_constraint": (
                    f"within {args.label_tolerance_count} correct example(s) of the "
                    "best eligible label count"
                ),
                "primary_tiebreaker": "joint_success_count",
                "secondary_tiebreakers": [
                    "label_correct_count", "mean_rationale_quality",
                    "strict_json_count", "shorter_prompt",
                ],
            },
            "all_ranking": [
                {
                    "rank": rank,
                    "prompt_id": prompt.prompt_id,
                    "parent_id": prompt.parent_id,
                    "depth": prompt.depth,
                    "metrics": metrics,
                }
                for rank, (prompt, metrics) in enumerate(ranked_all, 1)
            ],
        },
    )
    print("\n===== Best prompt =====")
    print(f"ID: {best_prompt.prompt_id}")
    print(f"Label accuracy: {best_metrics['label_accuracy']:.2%}")
    print(f"Joint success: {best_metrics.get('joint_success_rate', float('nan')):.2%}")
    print(
        "Prompt-check label: "
        f"{root_prompt_check_metrics['label_accuracy']:.2%} -> "
        f"{best_prompt_check_metrics['label_accuracy']:.2%}"
    )
    print(
        "Prompt-check joint: "
        f"{root_prompt_check_metrics.get('joint_success_rate', float('nan')):.2%} -> "
        f"{best_prompt_check_metrics.get('joint_success_rate', float('nan')):.2%}"
    )
    print(best_prompt.system_prompt)
    print()
    print(best_prompt.user_instruction)


def run_optimize_single_pool(
    args: argparse.Namespace,
    examples: Sequence[Example],
    student: LocalStudentModel,
    evaluator: OpenAIEvaluator,
    store: ArtifactStore,
) -> None:
    """Optimize on one exact-root pool and select strictly by label accuracy."""
    pool_name = "root_single_pool"
    root = initial_prompt(RATIONALE_FIRST)
    store.save_prompt(root)
    root_predictions, root_judgments, root_metrics = evaluate_prompt(
        root, pool_name, examples, student, evaluator, store, args
    )
    if root_judgments is None:
        raise RuntimeError("Single-pool optimization requires rationale judgments")
    all_results: list[tuple[PromptDesign, dict[str, Any]]] = [(root, root_metrics)]
    beam = [root]
    next_id = 1
    best_label_count = root_metrics["label_correct_count"]
    no_improvement_depths = 0

    for depth in range(1, args.depth + 1):
        print("\n" + "=" * 72)
        print(f"Depth {depth}/{args.depth}")
        print("=" * 72)
        children: list[PromptDesign] = []
        for parent_index, parent in enumerate(beam):
            parent_predictions, parent_judgments, _ = evaluate_prompt(
                parent, pool_name, examples, student, evaluator, store, args
            )
            if parent_judgments is None:
                raise RuntimeError("Missing single-pool judgments")
            candidate_count = args.multi_start if depth == 1 else args.branch_factor
            existing_children = sorted(
                (
                    prompt for prompt in store.prompts.values()
                    if prompt.parent_id == parent.prompt_id and prompt.depth == depth
                ),
                key=lambda prompt: prompt.prompt_id,
            )
            if len(existing_children) >= candidate_count:
                proposed = existing_children[:candidate_count]
                print(f"[{parent.prompt_id}] Reusing {len(proposed)} saved proposals")
                next_id = max(
                    (
                        int(match.group(1)) + 1
                        for prompt in store.prompts.values()
                        if (match := re.fullmatch(r"P(\d+)", prompt.prompt_id))
                    ),
                    default=next_id,
                )
                children.extend(proposed)
                continue
            if existing_children:
                raise RuntimeError(
                    f"Found only {len(existing_children)}/{candidate_count} saved children "
                    f"for {parent.prompt_id}. Re-run with --overwrite to avoid mixing runs."
                )

            selected_indices = select_single_pool_feedback_examples(
                examples,
                parent_predictions,
                parent_judgments,
                min(args.feedback_sample_size, len(examples)),
                args.split_seed + depth * 100 + parent_index,
                args.rationale_success_threshold,
            )
            selected_examples = [examples[i] for i in selected_indices]
            selected_predictions = [parent_predictions[i] for i in selected_indices]
            selected_judgments = [parent_judgments[i] for i in selected_indices]

            feedback_path = (
                store.output_dir / "feedback" / f"depth_{depth}_{parent.prompt_id}.json"
            )
            feedback_path.parent.mkdir(parents=True, exist_ok=True)
            if feedback_path.exists():
                saved_feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
                synthesis_payload = saved_feedback["synthesis"]
                synthesis = json.dumps(synthesis_payload, ensure_ascii=False, indent=2)
                print(f"[{parent.prompt_id}] Reusing saved feedback synthesis")
            else:
                synthesis = evaluator.synthesize_feedback(
                    parent,
                    selected_examples,
                    selected_predictions,
                    selected_judgments,
                )
                synthesis_payload = json.loads(synthesis)
                write_json(
                    feedback_path,
                    {
                        "depth": depth,
                        "parent_id": parent.prompt_id,
                        "selected_example_ids": [e.example_id for e in selected_examples],
                        "selection_priority": (
                            "wrong label, then rationale/format defect, then successful cases"
                        ),
                        "synthesis": synthesis_payload,
                    },
                )
            store.save_textual_gradient(
                depth,
                parent.prompt_id,
                [example.example_id for example in selected_examples],
                synthesis_payload,
            )
            proposed = evaluator.propose_prompts(
                parent, synthesis, candidate_count, next_id
            )
            for candidate in proposed:
                store.save_prompt(candidate)
            next_id = max(
                (
                    int(match.group(1)) + 1
                    for prompt in store.prompts.values()
                    if (match := re.fullmatch(r"P(\d+)", prompt.prompt_id))
                ),
                default=next_id + len(proposed),
            )
            children.extend(proposed)

        depth_results: list[tuple[PromptDesign, dict[str, Any]]] = []
        for child in children:
            _, _, metrics = evaluate_prompt(
                child, pool_name, examples, student, evaluator, store, args
            )
            depth_results.append((child, metrics))
            all_results.append((child, metrics))

        ranked_depth = rank_prompts_by_label_accuracy(depth_results, args)
        beam = [prompt for prompt, _ in ranked_depth[: args.beam_width]]
        print("\nNext beam (label accuracy first; rationale quality as tie-breaker):")
        for rank, (prompt, metrics) in enumerate(
            ranked_depth[: args.beam_width], 1
        ):
            print(
                f"  {rank}. {prompt.prompt_id}: "
                f"label={metrics['label_accuracy']:.2%}, "
                f"soft_joint={metrics.get('soft_joint_success_rate', float('nan')):.2%}, "
                f"complete={metrics.get('complete_success_rate', float('nan')):.2%}, "
                f"json={metrics['strict_json_rate']:.2%}, "
                f"rationale={metrics.get('mean_rationale_quality', float('nan')):.3f}"
            )

        depth_best_label = max(
            (metrics["label_correct_count"] for _, metrics in depth_results),
            default=-1,
        )
        if depth_best_label > best_label_count:
            best_label_count = depth_best_label
            no_improvement_depths = 0
        else:
            no_improvement_depths += 1
        if not args.disable_early_stop and no_improvement_depths >= 2:
            print("Early stop: label accuracy did not improve for two depths.")
            break

    ranked_all = rank_prompts_by_label_accuracy(all_results, args)
    best_prompt, best_metrics = ranked_all[0]
    label_rationale_correlation = pearson_correlation(
        [metrics["label_accuracy"] for _, metrics in all_results],
        [metrics.get("mean_rationale_quality", float("nan")) for _, metrics in all_results],
    )
    write_json(
        store.output_dir / "best_prompt.json",
        {
            "prompt": asdict(best_prompt),
            "optimized_analysis_strategy": extract_analysis_strategy(
                best_prompt.user_instruction
            ),
            "immutable_output_contract": FIXED_RATIONALE_FIRST_CONTRACT,
            "single_pool_metrics": best_metrics,
            "initial_prompt_metrics": root_metrics,
            "selection_rule": {
                "hard_gates": None,
                "primary": "label_correct_count across all evaluated prompts",
                "tiebreakers": [
                    "mean_rationale_quality",
                    "strict_json_count",
                    "mean_translation_quality",
                    "shorter_prompt",
                ],
                "rationale_metrics_are_diagnostic": True,
                "warning": (
                    "The same pool is used for feedback and prompt selection; "
                    "this is in-sample accuracy, not held-out performance."
                ),
            },
            "analysis": {
                "label_accuracy_vs_mean_rationale_quality_pearson_r": (
                    label_rationale_correlation
                ),
                "note": (
                    "Correlation is descriptive across evaluated prompt candidates and "
                    "does not establish that better rationales cause higher accuracy."
                ),
            },
            "all_ranking": [
                {
                    "rank": rank,
                    "prompt_id": prompt.prompt_id,
                    "parent_id": prompt.parent_id,
                    "depth": prompt.depth,
                    "metrics": metrics,
                }
                for rank, (prompt, metrics) in enumerate(ranked_all, 1)
            ],
        },
    )
    print("\n===== Best prompt (single in-sample root pool) =====")
    print(f"ID: {best_prompt.prompt_id}")
    print(
        f"Label accuracy: {root_metrics['label_accuracy']:.2%} -> "
        f"{best_metrics['label_accuracy']:.2%}"
    )
    print(
        "Soft Joint success: "
        f"{best_metrics.get('soft_joint_success_rate', float('nan')):.2%}"
    )
    print(
        "Complete success  : "
        f"{best_metrics.get('complete_success_rate', float('nan')):.2%}"
    )
    print(
        "Label/rationale r : "
        + (
            f"{label_rationale_correlation:.3f}"
            if label_rationale_correlation is not None
            else "undefined"
        )
    )
    print(best_prompt.system_prompt)
    print()
    print(best_prompt.user_instruction)


def make_output_dir(args: argparse.Namespace) -> Path:
    model_slug = "llama32_1b" if args.model == "llama" else "gemma4_e2b"
    if args.data_mode == "distill":
        data_slug = "distill_train2000"
    elif args.data_mode == "distill_root_single_pool":
        data_slug = "distill_exact_root_single_pool"
    else:
        data_slug = "root_only_no_phrase"
    if args.stage == "compare":
        name = f"rationale_order_comparison_{model_slug}_{data_slug}"
    else:
        name = (
            f"textgrad_rationale_{model_slug}_{args.output_order}_{data_slug}"
            "_separated_prompt_label_primary_v3"
        )
    return args.result_root.expanduser().resolve() / name


def save_run_configuration(
    args: argparse.Namespace,
    output_dir: Path,
    path_info: dict[str, str],
    optimization: Sequence[Example],
    validation: Sequence[Example],
    prompt_check: Sequence[Example],
) -> None:
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value.expanduser().resolve())
    config.update(
        {
            "resolved_model_name": args.model_name or MODEL_NAMES[args.model],
            "resolved_paths": path_info,
            "output_dir": str(output_dir),
            "optimization_distribution": count_labels(optimization),
            "validation_distribution": count_labels(validation),
            "prompt_check_distribution": count_labels(prompt_check),
        }
    )
    signature_keys = (
        "stage", "model", "resolved_model_name", "data_mode", "distill_file",
        "output_order", "split_seed",
        "positive_size", "negative_size", "optimization_size", "validation_size",
        "prompt_check_size",
        "max_input_tokens", "max_new_tokens", "evaluation_model", "proposal_model",
        "judge_batch_size", "validation_judge_size", "prompt_check_judge_size",
        "feedback_sample_size", "multi_start", "branch_factor", "beam_width",
        "depth", "label_tolerance_count", "min_strict_json_rate",
        "rationale_success_threshold", "max_rationale_quality_drop",
        "max_translation_quality_drop",
        "resolved_paths",
    )
    config["run_signature"] = {key: config.get(key) for key in signature_keys}
    config_path = output_dir / "config.json"
    if config_path.exists() and not args.overwrite:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("run_signature") != config["run_signature"]:
            raise RuntimeError(
                "The existing result directory was created with different run settings. "
                "Use the original settings, choose another --result-root, or intentionally "
                "start over with --overwrite."
            )
    write_json(config_path, config)
    split_rows = []
    for split, values in (
        ("optimization", optimization),
        ("validation", validation),
        ("prompt_check", prompt_check),
    ):
        for example in values:
            split_rows.append({"split": split, **asdict(example)})
    split_path = output_dir / "data_split.jsonl"
    if not split_path.exists() or args.overwrite:
        if split_path.exists():
            split_path.unlink()
        append_jsonl(split_path, split_rows)


def save_single_pool_configuration(
    args: argparse.Namespace,
    output_dir: Path,
    path_info: dict[str, str],
    examples: Sequence[Example],
) -> None:
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value.expanduser().resolve())
    config.update(
        {
            "resolved_model_name": args.model_name or MODEL_NAMES[args.model],
            "resolved_paths": path_info,
            "output_dir": str(output_dir),
            "single_pool_distribution": count_labels(examples),
            "single_pool_size": len(examples),
            "selection_is_in_sample": True,
            "student_visible_fields": ["text"],
            "evaluator_only_fields": [
                "label", "rationale", "rationale_ja", "p_positive", "label_doubt"
            ],
            "fixed_system_prompt": DEFAULT_SYSTEM_PROMPT,
            "initial_analysis_strategy": INITIAL_ANALYSIS_STRATEGY,
            "immutable_output_contract": FIXED_RATIONALE_FIRST_CONTRACT,
            "textgrad_optimizable_fields": ["analysis_strategy"],
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "selection_rule": (
                "label_correct_count, then mean_rationale_quality, then "
                "strict_json_count, then mean_translation_quality, then shorter_prompt"
            ),
            "hard_quality_gates": False,
        }
    )
    signature_keys = (
        "stage", "model", "resolved_model_name", "data_mode", "distill_file",
        "output_order", "split_seed", "max_input_tokens", "max_new_tokens",
        "evaluation_model", "proposal_model", "judge_batch_size",
        "feedback_sample_size", "multi_start", "branch_factor", "beam_width",
        "depth", "rationale_success_threshold", "prompt_template_version",
        "resolved_paths",
    )
    config["run_signature"] = {key: config.get(key) for key in signature_keys}
    config_path = output_dir / "config.json"
    if config_path.exists() and not args.overwrite:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("run_signature") != config["run_signature"]:
            raise RuntimeError(
                "The existing result directory uses different settings. "
                "Choose another --result-root or intentionally use --overwrite."
            )
    write_json(config_path, config)
    pool_path = output_dir / "data_pool.jsonl"
    if not pool_path.exists() or args.overwrite:
        if pool_path.exists():
            pool_path.unlink()
        append_jsonl(
            pool_path,
            ({"pool": "root_single_pool", **asdict(example)} for example in examples),
        )


def main() -> int:
    args = parse_args()
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        # The OpenAI client can still read OPENAI_API_KEY directly from the
        # environment when python-dotenv is not installed.
        pass
    output_dir = make_output_dir(args)
    store = ArtifactStore(output_dir, overwrite=args.overwrite)

    examples, path_info = build_examples(args)
    if args.data_mode == "distill_root_single_pool":
        save_single_pool_configuration(args, output_dir, path_info, examples)
        print("===== Rationale Prompt Optimization Setup =====")
        print(f"Stage             : {args.stage}")
        print(f"Model             : {args.model_name or MODEL_NAMES[args.model]}")
        print(f"Data mode         : {args.data_mode}")
        print(f"Single root pool  : {len(examples)} {count_labels(examples)}")
        print("Split             : disabled (same pool for feedback and selection)")
        print("Output order      : rationale -> label -> rationale_ja")
        print("Student input     : text only")
        print(
            "Evaluator refs    : label, rationale, rationale_ja, p_positive, "
            "label_doubt"
        )
        print(
            f"Excluded fallback polarity: "
            f"{path_info['excluded_non_sst_polarity']}"
        )
        print(f"Excluded subtrees : {path_info['excluded_subtree']}")
        print("Optimizable block : analysis strategy only")
        print("Fixed contract    : JSON schema, order, grounding, Japanese fidelity")
        print("Quality gates     : disabled (all evaluated prompts remain eligible)")
        print(
            "Best criterion    : label -> rationale score -> strict JSON -> "
            "translation -> shorter prompt"
        )
        print(f"Result directory  : {output_dir}")
        max_candidates = 1 + args.multi_start
        if args.depth >= 2:
            max_candidates += (args.depth - 1) * args.beam_width * args.branch_factor
        print(
            f"Search            : starts={args.multi_start}, branch={args.branch_factor}, "
            f"beam={args.beam_width}, depth={args.depth}, max_prompts={max_candidates}"
        )
        print(
            f"Feedback examples : {min(args.feedback_sample_size, len(examples))}/"
            f"{len(examples)} per parent prompt"
        )
        print(f"API budget        : ${args.max_api_cost_usd:.2f}")
        print(
            "Warning           : reported pool accuracy is in-sample; "
            "evaluate the final prompt on SST-2 dev separately."
        )
        if args.data_check_only:
            print("Data check completed successfully; model and API were not used.")
            return 0
        if args.local_only:
            raise ValueError("Single-pool TextGrad optimization requires API judging")
        student = LocalStudentModel(args)
        evaluator = OpenAIEvaluator(args, output_dir)
        run_optimize_single_pool(args, examples, student, evaluator, store)
        print("\n===== Saved artifacts =====")
        for path in sorted(output_dir.iterdir()):
            print(path)
        return 0

    optimization, validation, prompt_check = stratified_split(
        examples,
        args.optimization_size,
        args.validation_size,
        args.prompt_check_size,
        args.split_seed,
    )
    save_run_configuration(
        args, output_dir, path_info, optimization, validation, prompt_check
    )

    print("===== Rationale Prompt Optimization Setup =====")
    print(f"Stage             : {args.stage}")
    print(f"Model             : {args.model_name or MODEL_NAMES[args.model]}")
    print(f"Data mode         : {args.data_mode}")
    print(f"Examples          : {len(examples)} {count_labels(examples)}")
    print(f"Optimization      : {len(optimization)} {count_labels(optimization)}")
    print(f"Validation        : {len(validation)} {count_labels(validation)}")
    print(f"Prompt-check      : {len(prompt_check)} {count_labels(prompt_check)}")
    print(
        "API judge samples : "
        f"validation={min(args.validation_judge_size, len(validation))}, "
        f"prompt-check={min(args.prompt_check_judge_size, len(prompt_check))}"
    )
    if args.data_mode == "distill":
        print("Student input      : text only")
        print("Evaluator reference: label, rationale, rationale_ja, label_doubt")
        print("Ignored fields     : source_sentences, p_positive, p_source")
        print("Reference status   : Silver (LLM-generated), not human Gold")
    else:
        print("Subtree information: disabled")
    print(f"Result directory  : {output_dir}")
    if args.stage == "optimize":
        max_candidates = 1 + args.multi_start
        if args.depth >= 2:
            max_candidates += (args.depth - 1) * args.beam_width * args.branch_factor
        print(f"Output order      : {args.output_order}")
        print(
            f"Search            : starts={args.multi_start}, branch={args.branch_factor}, "
            f"beam={args.beam_width}, depth={args.depth}, max_prompts={max_candidates}"
        )
    print(f"API budget        : ${args.max_api_cost_usd:.2f}")

    if args.data_check_only:
        print("Data check completed successfully; model and API were not used.")
        return 0

    student = LocalStudentModel(args)
    evaluator = None if args.local_only else OpenAIEvaluator(args, output_dir)
    if args.stage == "compare":
        run_compare(args, validation, prompt_check, student, evaluator, store)
    else:
        if evaluator is None:
            raise ValueError("--stage optimize requires API judging; remove --local-only")
        run_optimize(
            args, optimization, validation, prompt_check, student, evaluator, store
        )

    print("\n===== Saved artifacts =====")
    for path in sorted(output_dir.iterdir()):
        print(path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ApiBudgetExceeded as exc:
        print(f"\nStopped safely: {exc}", file=sys.stderr)
        raise SystemExit(2)
