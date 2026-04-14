from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from vkr_stage1.connectors.onec_connector import OneCConnector
from vkr_stage1.core.config import get_settings
from vkr_stage1.llm.openai_client import OpenAILLMClient


@dataclass
class SampleResult:
    idx: int
    question: str
    schema: str
    predicted_query: str | None
    ground_truth_query: str
    normalized_predicted_query: str | None
    normalized_ground_truth_query: str
    exact_match: bool
    cosine_similarity: float | None
    similarity_error: str | None
    gen_latency_ms: float
    exec_latency_ms: float
    total_latency_ms: float
    error: str | None


def _norm_query(text: str) -> str:
    return " ".join(text.replace(";", " ; ").lower().split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LLM->1C on Spider-1C")
    parser.add_argument("--parquet", default="data/Spider-1C/data/train-00000-of-00001.parquet")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out", default="data/bench_spider1c.json")
    parser.add_argument("--exec-query", action="store_true", help="Execute predicted query via OneC connector")
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress every N samples")
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="Embedding model for cosine similarity between prediction and ground truth",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()

    if not Path(args.parquet).exists():
        raise FileNotFoundError(args.parquet)

    table = pq.read_table(args.parquet, columns=["schema", "question", "query"])
    rows = table.to_pylist()[args.offset : args.offset + args.limit]

    llm = OpenAILLMClient(
        api_base=settings.llm_api_base,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        embedding_model=args.embedding_model,
    )
    onec = OneCConnector(
        base_url=settings.onec_base_url,
        query_path=settings.onec_query_path,
        schema_path=settings.onec_schema_path,
        username=settings.onec_username,
        password=settings.onec_password,
        mock_mode=settings.onec_mock,
    )

    results: list[SampleResult] = []
    for idx, row in enumerate(rows, start=1):
        t0 = time.perf_counter()
        error: str | None = None
        exec_ms = 0.0
        pred: str | None = None
        norm_pred: str | None = None
        cosine_similarity: float | None = None
        similarity_error: str | None = None
        ground_truth = row["query"]
        norm_ground_truth = _norm_query(ground_truth)
        try:
            g0 = time.perf_counter()
            pred = llm.generate_1c_query(question=row["question"], schema=row["schema"])
            g1 = time.perf_counter()
            gen_ms = (g1 - g0) * 1000
            norm_pred = _norm_query(pred)
            try:
                pred_embedding = llm.get_embedding(norm_pred)
                gt_embedding = llm.get_embedding(norm_ground_truth)
                cosine_similarity = llm.cosine_similarity(pred_embedding, gt_embedding)
            except Exception as exc:
                similarity_error = str(exc)

            if args.exec_query:
                e0 = time.perf_counter()
                onec.execute_query(pred)
                e1 = time.perf_counter()
                exec_ms = (e1 - e0) * 1000

            em = norm_pred == norm_ground_truth
        except Exception as exc:
            gen_ms = 0.0
            em = False
            error = str(exc)

        total_ms = (time.perf_counter() - t0) * 1000
        results.append(
            SampleResult(
                idx=idx,
                question=row["question"],
                schema=row["schema"],
                predicted_query=pred,
                ground_truth_query=ground_truth,
                normalized_predicted_query=norm_pred,
                normalized_ground_truth_query=norm_ground_truth,
                exact_match=em,
                cosine_similarity=cosine_similarity,
                similarity_error=similarity_error,
                gen_latency_ms=gen_ms,
                exec_latency_ms=exec_ms,
                total_latency_ms=total_ms,
                error=error,
            )
        )
        if args.progress_every > 0 and (idx % args.progress_every == 0 or idx == len(rows)):
            done = len(results)
            done_em = sum(1 for r in results if r.exact_match)
            done_ok = sum(1 for r in results if r.error is None)
            done_lat = [r.total_latency_ms for r in results if r.error is None]
            done_cos = [r.cosine_similarity for r in results if r.cosine_similarity is not None]
            avg_ms = statistics.mean(done_lat) if done_lat else 0.0
            avg_cos = statistics.mean(done_cos) if done_cos else 0.0
            pct = (done / len(rows)) * 100 if rows else 100.0
            print(
                f"[{done}/{len(rows)} | {pct:.1f}%] "
                f"success={done_ok/done:.3f} exact={done_em/done:.3f} cos={avg_cos:.3f} avg_ms={avg_ms:.1f}",
                flush=True,
            )

    total = len(results)
    success = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]
    em_count = sum(1 for r in results if r.exact_match)
    cosine_values = [r.cosine_similarity for r in results if r.cosine_similarity is not None]

    latencies = [r.total_latency_ms for r in success]
    report: dict[str, Any] = {
        "total": total,
        "success_count": len(success),
        "error_count": len(failed),
        "success_rate": (len(success) / total) if total else 0.0,
        "exact_match_count": em_count,
        "exact_match_rate": (em_count / total) if total else 0.0,
        "cosine_similarity": {
            "avg": statistics.mean(cosine_values) if cosine_values else None,
            "p50": statistics.median(cosine_values) if cosine_values else None,
            "min": min(cosine_values) if cosine_values else None,
            "max": max(cosine_values) if cosine_values else None,
        },
        "latency_ms": {
            "avg": statistics.mean(latencies) if latencies else None,
            "p50": statistics.median(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "settings": {
            "model": settings.llm_model,
            "embedding_model": args.embedding_model,
            "api_base": settings.llm_api_base,
            "onec_mock": settings.onec_mock,
            "exec_query": args.exec_query,
            "limit": args.limit,
            "offset": args.offset,
        },
        "samples": [r.__dict__ for r in results],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "saved_to": str(out_path),
        "success_rate": report["success_rate"],
        "exact_match_rate": report["exact_match_rate"],
        "avg_cosine_similarity": report["cosine_similarity"]["avg"],
        "avg_latency_ms": report["latency_ms"]["avg"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
