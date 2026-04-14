from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vkr_stage1.llm.openai_client import OpenAILLMClient


def _load_rows(parquet_path: str, limit: int) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required for Spider-1C eval") from exc

    table = pq.read_table(parquet_path, columns=["schema", "question", "query"])
    rows = table.to_pylist()
    return rows[:limit]


def _norm_sql(text: str) -> str:
    return " ".join(text.strip().lower().split())


def run_eval(parquet_path: str, api_base: str, api_key: str, model: str, limit: int) -> dict[str, Any]:
    llm = OpenAILLMClient(api_base=api_base, api_key=api_key, model=model)
    rows = _load_rows(parquet_path, limit)

    exact_match = 0
    results: list[dict[str, Any]] = []
    for row in rows:
        predicted = llm.generate_sql(question=row["question"], schema=row["schema"])
        is_match = _norm_sql(predicted) == _norm_sql(row["query"])
        exact_match += int(is_match)
        results.append(
            {
                "question": row["question"],
                "gold_sql": row["query"],
                "predicted_sql": predicted,
                "exact_match": is_match,
            }
        )

    total = len(rows)
    return {
        "total": total,
        "exact_match_count": exact_match,
        "exact_match_rate": (exact_match / total) if total else 0.0,
        "samples": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sample eval on Spider-1C")
    parser.add_argument(
        "--parquet",
        default="data/Spider-1C/data/train-00000-of-00001.parquet",
        help="Path to Spider-1C parquet",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="data/spider1c_eval_sample.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.parquet).exists():
        raise FileNotFoundError(f"Dataset file not found: {args.parquet}")

    report = run_eval(
        parquet_path=args.parquet,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        limit=args.limit,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"saved_to": args.out, "exact_match_rate": report["exact_match_rate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
