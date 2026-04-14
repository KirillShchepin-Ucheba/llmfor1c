from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


def _get_nested(payload: dict[str, Any], path: str, default: Any = None) -> Any:
    node: Any = payload
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", value).strip("._")
    return cleaned or "model"


def _error_signature(error_text: str) -> str:
    normalized = " ".join((error_text or "").split())
    lowered = normalized.lower()
    signatures = [
        ("поле не найдено", "field_not_found"),
        ("синтаксическая ошибка", "syntax_error"),
        ("неверные параметры", "invalid_params"),
        ("нельзя сравнивать", "invalid_comparison"),
        ("unauthorized", "http_unauthorized"),
        ("forbidden", "http_forbidden"),
        ("not found", "http_not_found"),
        ("internal server error", "http_internal_error"),
        ("timeout", "timeout"),
    ]
    for token, label in signatures:
        if token in lowered:
            return label
    return normalized[:120] if normalized else "unknown"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _common_rate(values: list[float], threshold: float) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if value >= threshold) / len(values)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    k = (len(ordered) - 1) * (percentile / 100.0)
    left = math.floor(k)
    right = math.ceil(k)
    if left == right:
        return ordered[int(k)]
    return ordered[left] * (right - k) + ordered[right] * (k - left)


CLAUSE_PATTERNS: dict[str, str] = {
    "select": r"\b(?:select|\u0432\u044b\u0431\u0440\u0430\u0442\u044c)\b",
    "top": r"\b(?:top|\u043f\u0435\u0440\u0432\u044b\u0435)\b",
    "distinct": r"\b(?:distinct|\u0440\u0430\u0437\u043b\u0438\u0447\u043d\u044b\u0435)\b",
    "from": r"\b(?:from|\u0438\u0437)\b",
    "join": r"\b(?:join|\u0441\u043e\u0435\u0434\u0438\u043d\u0435\u043d\u0438\u0435)\b",
    "where": r"\b(?:where|\u0433\u0434\u0435)\b",
    "group_by": r"(?:\bgroup\s+by\b|\u0441\u0433\u0440\u0443\u043f\u043f\u0438\u0440\u043e\u0432\u0430\u0442\u044c\s+\u043f\u043e)",
    "having": r"\b(?:having|\u0438\u043c\u0435\u044e\u0449\w*)\b",
    "order_by": r"(?:\border\s+by\b|\u0443\u043f\u043e\u0440\u044f\u0434\u043e\u0447\u0438\u0442\u044c\s+\u043f\u043e)",
}


def _norm_query(text: str) -> str:
    return " ".join(text.replace(";", " ; ").lower().split())


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\.]+|[,;()=*<>+\-]", text.lower(), flags=re.UNICODE)


def _sequence_similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, left, right).ratio()


def _token_jaccard(left: str, right: str) -> float:
    left_set = set(_tokenize(left))
    right_set = set(_tokenize(right))
    union = left_set | right_set
    if not union:
        return 1.0
    return len(left_set & right_set) / len(union)


def _extract_clauses(norm_query: str) -> list[str]:
    found: list[str] = []
    for clause, pattern in CLAUSE_PATTERNS.items():
        if re.search(pattern, norm_query, flags=re.IGNORECASE):
            found.append(clause)
    return found


def _clause_f1(predicted: list[str], ground_truth: list[str]) -> float:
    pred_set = set(predicted)
    gt_set = set(ground_truth)
    if not pred_set and not gt_set:
        return 1.0
    if not pred_set or not gt_set:
        return 0.0
    inter = len(pred_set & gt_set)
    precision = inter / len(pred_set)
    recall = inter / len(gt_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _tool_success(sample: dict[str, Any]) -> bool | None:
    exec_success = _safe_bool(sample.get("execution_success"))
    if exec_success is not None:
        return exec_success
    gen_success = _safe_bool(sample.get("generation_success"))
    if gen_success is not None:
        return gen_success
    err = sample.get("error")
    if err is None:
        return True
    return False


def _enrich_sample(sample: dict[str, Any]) -> dict[str, Any]:
    predicted = sample.get("normalized_predicted_query") or sample.get("predicted_query")
    gold = sample.get("normalized_ground_truth_query") or sample.get("ground_truth_query")
    norm_pred = _norm_query(str(predicted)) if isinstance(predicted, str) and predicted.strip() else None
    norm_gold = _norm_query(str(gold)) if isinstance(gold, str) and gold.strip() else None

    if norm_pred and not sample.get("normalized_predicted_query"):
        sample["normalized_predicted_query"] = norm_pred
    if norm_gold and not sample.get("normalized_ground_truth_query"):
        sample["normalized_ground_truth_query"] = norm_gold

    if sample.get("predicted_clauses") is None:
        sample["predicted_clauses"] = _extract_clauses(norm_pred) if norm_pred else []
    if sample.get("ground_truth_clauses") is None:
        sample["ground_truth_clauses"] = _extract_clauses(norm_gold) if norm_gold else []

    if sample.get("sequence_similarity") is None and norm_pred and norm_gold:
        sample["sequence_similarity"] = _sequence_similarity(norm_pred, norm_gold)
    if sample.get("token_jaccard") is None and norm_pred and norm_gold:
        sample["token_jaccard"] = _token_jaccard(norm_pred, norm_gold)
    if sample.get("clause_f1") is None:
        sample["clause_f1"] = _clause_f1(
            list(sample.get("predicted_clauses") or []),
            list(sample.get("ground_truth_clauses") or []),
        )

    if sample.get("generation_success") is None:
        sample["generation_success"] = (sample.get("predicted_query") is not None) and (sample.get("error") is None)

    return sample


def _normalize_weights(components: list[tuple[float, float | None]]) -> float | None:
    available = [(weight, value) for weight, value in components if value is not None]
    if not available:
        return None
    total_weight = sum(weight for weight, _ in available)
    if total_weight <= 0:
        return None
    return sum(weight * value for weight, value in available) / total_weight


def _sample_quality_score(sample: dict[str, Any]) -> float | None:
    exact = 1.0 if sample.get("exact_match") is True else 0.0 if sample.get("exact_match") is False else None
    exec_success = sample.get("execution_success")
    exec_score = 1.0 if exec_success is True else 0.0 if exec_success is False else None
    cosine = _safe_float(sample.get("cosine_similarity"))
    seq = _safe_float(sample.get("sequence_similarity"))
    clause = _safe_float(sample.get("clause_f1"))
    return _normalize_weights(
        [
            (0.45, exact),
            (0.20, exec_score),
            (0.15, cosine),
            (0.10, seq),
            (0.10, clause),
        ]
    )


def _sample_key(sample: dict[str, Any], index: int) -> str:
    idx = sample.get("idx")
    question = sample.get("question")
    if idx is not None and question:
        return f"{idx}|{question}"
    if idx is not None:
        return f"{idx}"
    if question:
        return f"q:{question}"
    return f"pos:{index}"


@dataclass
class ModelRun:
    label: str
    report_path: Path
    report: dict[str, Any]
    sample_by_key: dict[str, dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple benchmark_spider1c reports")
    parser.add_argument("--reports", nargs="+", required=True, help="List of benchmark report JSON files")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional labels for reports")
    parser.add_argument("--out-dir", default="data/bench_compare_viz", help="Output directory")
    parser.add_argument("--max-error-bars", type=int, default=12, help="Top error signatures for charts")
    parser.add_argument("--no-png", action="store_true", help="Disable PNG rendering")
    return parser.parse_args()


def _load_runs(report_paths: list[str], labels: list[str] | None) -> list[ModelRun]:
    runs: list[ModelRun] = []
    for idx, raw_path in enumerate(report_paths):
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        model_name = _get_nested(report, "settings.model", path.stem)
        label = labels[idx] if labels and idx < len(labels) else str(model_name)
        samples = report.get("samples") or []
        sample_by_key: dict[str, dict[str, Any]] = {}
        for sample_idx, sample in enumerate(samples, start=1):
            sample = _enrich_sample(dict(sample))
            key = _sample_key(sample, sample_idx)
            sample_by_key[key] = sample
        runs.append(ModelRun(label=label, report_path=path, report=report, sample_by_key=sample_by_key))
    return runs


def _compute_summary_rows(runs: list[ModelRun]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        report = run.report
        samples = list(run.sample_by_key.values())
        cosine_values = [_safe_float(sample.get("cosine_similarity")) for sample in samples]
        cosine_values = [value for value in cosine_values if value is not None]
        seq_values = [_safe_float(sample.get("sequence_similarity")) for sample in samples]
        seq_values = [value for value in seq_values if value is not None]
        jacc_values = [_safe_float(sample.get("token_jaccard")) for sample in samples]
        jacc_values = [value for value in jacc_values if value is not None]
        clause_values = [_safe_float(sample.get("clause_f1")) for sample in samples]
        clause_values = [value for value in clause_values if value is not None]
        qualities = [_sample_quality_score(sample) for sample in samples]
        qualities = [value for value in qualities if value is not None]
        gen_success_values = [_tool_success(sample) for sample in samples]
        gen_success_true = sum(1 for value in gen_success_values if value is True)
        exec_success_values = [_safe_bool(sample.get("execution_success")) for sample in samples]
        exec_known = [value for value in exec_success_values if value is not None]

        lat_total_samples = [_safe_float(sample.get("total_latency_ms")) for sample in samples]
        lat_total_samples = [value for value in lat_total_samples if value is not None]
        lat_gen_samples = [_safe_float(sample.get("gen_latency_ms")) for sample in samples]
        lat_gen_samples = [value for value in lat_gen_samples if value is not None]
        lat_exec_samples = [_safe_float(sample.get("exec_latency_ms")) for sample in samples]
        lat_exec_samples = [value for value in lat_exec_samples if value is not None]

        lat_total_avg = _safe_float(
            _get_nested(report, "latency_ms.total.avg", _get_nested(report, "latency_ms.avg", _mean(lat_total_samples)))
        )
        lat_total_p95 = _safe_float(_get_nested(report, "latency_ms.total.p95", _percentile(lat_total_samples, 95)))
        lat_gen_avg = _safe_float(_get_nested(report, "latency_ms.gen.avg", _mean(lat_gen_samples)))
        lat_exec_avg = _safe_float(_get_nested(report, "latency_ms.exec.avg", _mean(lat_exec_samples)))

        error_signatures = Counter()
        for sample in samples:
            if sample.get("error"):
                error_signatures[_error_signature(str(sample.get("error")))] += 1

        rows.append(
            {
                "label": run.label,
                "report_path": str(run.report_path),
                "model": _get_nested(report, "settings.model", run.label),
                "total": report.get("total"),
                "success_rate": _safe_float(report.get("success_rate")),
                "generation_success_rate": _safe_float(
                    report.get(
                        "generation_success_rate",
                        (gen_success_true / len(samples)) if samples else None,
                    )
                ),
                "execution_success_rate": _safe_float(
                    report.get(
                        "execution_success_rate",
                        (sum(1 for value in exec_known if value is True) / len(exec_known))
                        if exec_known
                        else ((gen_success_true / len(samples)) if samples else None),
                    )
                ),
                "exact_match_rate": _safe_float(report.get("exact_match_rate")),
                "cosine_avg": _safe_float(_get_nested(report, "cosine_similarity.avg", _mean(cosine_values))),
                "cosine_p90": _safe_float(_get_nested(report, "cosine_similarity.p90", _percentile(cosine_values, 90))),
                "cosine_ge_0_8_rate": _safe_float(
                    _get_nested(report, "cosine_similarity.ge_0_80_rate", _common_rate(cosine_values, 0.8))
                ),
                "sequence_avg": _mean(seq_values),
                "token_jaccard_avg": _mean(jacc_values),
                "clause_f1_avg": _mean(clause_values),
                "quality_composite_avg": _mean(qualities),
                "latency_total_avg_ms": lat_total_avg,
                "latency_total_p95_ms": lat_total_p95,
                "latency_gen_avg_ms": lat_gen_avg,
                "latency_exec_avg_ms": lat_exec_avg,
                "error_count": report.get("error_count"),
                "top_error_signature": (error_signatures.most_common(1)[0][0] if error_signatures else None),
                "top_error_count": (error_signatures.most_common(1)[0][1] if error_signatures else 0),
            }
        )
    return rows


def _pairwise_matrix(
    runs: list[ModelRun],
    metric_getter: Callable[[dict[str, Any]], float | bool | None],
    higher_is_better: bool = True,
    tolerance: float = 1e-12,
) -> tuple[list[str], list[list[float | None]], list[list[int]]]:
    labels = [run.label for run in runs]
    n = len(runs)
    matrix: list[list[float | None]] = [[None for _ in range(n)] for _ in range(n)]
    supports: list[list[int]] = [[0 for _ in range(n)] for _ in range(n)]

    sample_sets = [set(run.sample_by_key.keys()) for run in runs]
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 0.5
                supports[i][j] = len(sample_sets[i])
                continue
            keys = sample_sets[i] & sample_sets[j]
            wins_i = 0
            ties = 0
            comparable = 0
            for key in keys:
                left_raw = metric_getter(runs[i].sample_by_key[key])
                right_raw = metric_getter(runs[j].sample_by_key[key])
                if isinstance(left_raw, bool):
                    left = 1.0 if left_raw else 0.0
                else:
                    left = _safe_float(left_raw)
                if isinstance(right_raw, bool):
                    right = 1.0 if right_raw else 0.0
                else:
                    right = _safe_float(right_raw)
                if left is None or right is None:
                    continue
                comparable += 1
                delta = left - right if higher_is_better else right - left
                if abs(delta) <= tolerance:
                    ties += 1
                elif delta > 0:
                    wins_i += 1
            supports[i][j] = comparable
            if comparable == 0:
                matrix[i][j] = None
            else:
                matrix[i][j] = (wins_i + 0.5 * ties) / comparable
    return labels, matrix, supports


def _matrix_to_rows(labels: list[str], matrix: list[list[float | None]], supports: list[list[int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, row_label in enumerate(labels):
        for j, col_label in enumerate(labels):
            rows.append(
                {
                    "row_model": row_label,
                    "col_model": col_label,
                    "value": matrix[i][j],
                    "support": supports[i][j],
                }
            )
    return rows


def _clause_confusion_rows(runs: list[ModelRun]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        clause_names: set[str] = set()
        for sample in run.sample_by_key.values():
            clause_names.update(sample.get("predicted_clauses") or [])
            clause_names.update(sample.get("ground_truth_clauses") or [])
        for clause in sorted(clause_names):
            tp = fp = fn = tn = 0
            for sample in run.sample_by_key.values():
                pred = clause in set(sample.get("predicted_clauses") or [])
                gt = clause in set(sample.get("ground_truth_clauses") or [])
                if pred and gt:
                    tp += 1
                elif pred and not gt:
                    fp += 1
                elif (not pred) and gt:
                    fn += 1
                else:
                    tn += 1
            total = tp + fp + fn + tn
            precision = tp / (tp + fp) if (tp + fp) else None
            recall = tp / (tp + fn) if (tp + fn) else None
            f1 = (
                (2 * precision * recall / (precision + recall))
                if precision is not None and recall is not None and (precision + recall) > 0
                else None
            )
            accuracy = (tp + tn) / total if total else None
            rows.append(
                {
                    "label": run.label,
                    "clause": clause,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "accuracy": accuracy,
                    "support_positive_gt": tp + fn,
                    "support_positive_pred": tp + fp,
                }
            )
    return rows


def _error_rows(runs: list[ModelRun]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        counter: Counter[str] = Counter()
        for sample in run.sample_by_key.values():
            if sample.get("error"):
                counter[_error_signature(str(sample["error"]))] += 1
        for signature, count in counter.most_common():
            rows.append({"label": run.label, "signature": signature, "count": count})
    return rows


def _similarity_bin_rows(runs: list[ModelRun]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        bins = [0.0 + i * 0.1 for i in range(11)]
        bin_map = {(bins[i], bins[i + 1]): {"total": 0, "exact_true": 0, "exec_true": 0, "exec_false": 0} for i in range(10)}
        for sample in run.sample_by_key.values():
            sim = _safe_float(sample.get("cosine_similarity"))
            if sim is None:
                sim = _safe_float(sample.get("sequence_similarity"))
            if sim is None:
                continue
            for i in range(10):
                lo = bins[i]
                hi = bins[i + 1]
                if sim >= lo and (sim < hi or (i == 9 and sim <= hi)):
                    bucket = bin_map[(lo, hi)]
                    bucket["total"] += 1
                    if sample.get("exact_match") is True:
                        bucket["exact_true"] += 1
                    tool_ok = _tool_success(sample)
                    if tool_ok is True:
                        bucket["exec_true"] += 1
                    elif tool_ok is False:
                        bucket["exec_false"] += 1
                    break
        for (lo, hi), values in bin_map.items():
            total = values["total"]
            rows.append(
                {
                    "label": run.label,
                    "bin_start": lo,
                    "bin_end": hi,
                    "total": total,
                    "exact_true": values["exact_true"],
                    "exact_true_rate": (values["exact_true"] / total) if total else None,
                    "exec_true": values["exec_true"],
                    "exec_false": values["exec_false"],
                    "exec_true_rate": (values["exec_true"] / (values["exec_true"] + values["exec_false"]))
                    if (values["exec_true"] + values["exec_false"])
                    else None,
                }
            )
    return rows


def _write_markdown_summary(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    if not summary_rows:
        path.write_text("# Benchmark comparison\n\nNo data.", encoding="utf-8")
        return
    rows_by_quality = sorted(
        summary_rows,
        key=lambda row: (
            row.get("quality_composite_avg") if row.get("quality_composite_avg") is not None else -1,
            row.get("exact_match_rate") if row.get("exact_match_rate") is not None else -1,
        ),
        reverse=True,
    )
    rows_by_latency = sorted(
        summary_rows,
        key=lambda row: row.get("latency_total_avg_ms") if row.get("latency_total_avg_ms") is not None else float("inf"),
    )
    best_quality = rows_by_quality[0]
    fastest = rows_by_latency[0]
    lines = [
        "# Benchmark comparison summary",
        "",
        f"- Best composite quality: **{best_quality['label']}** ({best_quality.get('quality_composite_avg')})",
        f"- Fastest by total latency avg: **{fastest['label']}** ({fastest.get('latency_total_avg_ms')} ms)",
        "",
        "| model | exact | success | exec_success | cosine_avg | seq_avg | clause_f1 | quality_composite | lat_total_avg_ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows_by_quality:
        lines.append(
            f"| {row['label']} | {row.get('exact_match_rate')} | {row.get('success_rate')} | "
            f"{row.get('execution_success_rate')} | {row.get('cosine_avg')} | {row.get('sequence_avg')} | "
            f"{row.get('clause_f1_avg')} | {row.get('quality_composite_avg')} | {row.get('latency_total_avg_ms')} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _render_png(
    out_dir: Path,
    summary_rows: list[dict[str, Any]],
    pairwise_exact: tuple[list[str], list[list[float | None]], list[list[int]]],
    pairwise_cosine: tuple[list[str], list[list[float | None]], list[list[int]]],
    pairwise_quality: tuple[list[str], list[list[float | None]], list[list[int]]],
    pairwise_exec: tuple[list[str], list[list[float | None]], list[list[int]]],
    pairwise_latency: tuple[list[str], list[list[float | None]], list[list[int]]],
    clause_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    similarity_rows: list[dict[str, Any]],
    max_error_bars: int,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    produced: list[Path] = []

    def save_current(path: Path) -> None:
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        produced.append(path)

    labels = [row["label"] for row in summary_rows]

    # 1) quality leaderboard
    fig = plt.figure(figsize=(12, 5))
    x = range(len(summary_rows))
    width = 0.16
    exact = [row.get("exact_match_rate") or 0.0 for row in summary_rows]
    cosine = [row.get("cosine_avg") or 0.0 for row in summary_rows]
    seq = [row.get("sequence_avg") or 0.0 for row in summary_rows]
    clause = [row.get("clause_f1_avg") or 0.0 for row in summary_rows]
    quality = [row.get("quality_composite_avg") or 0.0 for row in summary_rows]
    plt.bar([idx - 2 * width for idx in x], exact, width=width, label="exact")
    plt.bar([idx - width for idx in x], cosine, width=width, label="cosine")
    plt.bar([idx for idx in x], seq, width=width, label="sequence")
    plt.bar([idx + width for idx in x], clause, width=width, label="clause_f1")
    plt.bar([idx + 2 * width for idx in x], quality, width=width, label="composite")
    plt.xticks(list(x), labels, rotation=20, ha="right")
    plt.ylim(0, 1.02)
    plt.title("Quality leaderboard")
    plt.ylabel("score")
    plt.legend()
    save_current(out_dir / "leaderboard_quality.png")

    # 2) latency leaderboard
    fig = plt.figure(figsize=(10, 5))
    lat_avg = [row.get("latency_total_avg_ms") or 0.0 for row in summary_rows]
    lat_p95 = [row.get("latency_total_p95_ms") or 0.0 for row in summary_rows]
    x = range(len(summary_rows))
    width = 0.35
    plt.bar([idx - width / 2 for idx in x], lat_avg, width=width, label="avg")
    plt.bar([idx + width / 2 for idx in x], lat_p95, width=width, label="p95")
    plt.xticks(list(x), labels, rotation=20, ha="right")
    plt.title("Latency leaderboard (total ms)")
    plt.ylabel("ms")
    plt.legend()
    save_current(out_dir / "leaderboard_latency.png")

    # 3) Pareto chart (with collision-safe labels)
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    pareto_points: list[dict[str, float | str]] = []
    for row in summary_rows:
        x_val = row.get("latency_total_avg_ms")
        y_val = row.get("quality_composite_avg")
        if x_val is None or y_val is None:
            continue
        pareto_points.append({"x": float(x_val), "y": float(y_val), "label": str(row["label"])})

    if pareto_points:
        xs = [float(point["x"]) for point in pareto_points]
        ys = [float(point["y"]) for point in pareto_points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_range = max(x_max - x_min, 1.0)
        y_range = max(y_max - y_min, 1e-4)

        for point in pareto_points:
            plt.scatter(
                float(point["x"]),
                float(point["y"]),
                s=95,
                zorder=3,
                edgecolors="black",
                linewidths=0.6,
            )

        plt.xlim(x_min - x_range * 0.05, x_max + x_range * 0.15)
        plt.ylim(y_min - y_range * 0.09, y_max + y_range * 0.16)

        # Greedy offset selection in pixel coordinates to avoid label overlaps.
        fig.canvas.draw()
        ax_box = ax.get_window_extent()
        px_per_point = fig.dpi / 72.0
        candidates_pt: list[tuple[float, float]] = [
            (36, 26), (38, -26), (-40, 26), (-42, -26),
            (58, 6), (58, -6), (-60, 6), (-60, -6),
            (26, 46), (-26, 46), (26, -46), (-26, -46),
            (76, 24), (-78, 24), (76, -24), (-78, -24),
        ]
        point_pixels = [ax.transData.transform((float(point["x"]), float(point["y"]))) for point in pareto_points]
        used_boxes: list[tuple[float, float, float, float]] = []
        placed: list[dict[str, float | str]] = []

        for point in sorted(pareto_points, key=lambda item: (float(item["x"]), -float(item["y"]))):
            px, py = ax.transData.transform((float(point["x"]), float(point["y"])))
            label = str(point["label"])
            text_w = max(70.0, 6.8 * len(label) + 18.0)
            text_h = 22.0

            best_score = None
            best_off = (24.0, 18.0)

            for off_x_pt, off_y_pt in candidates_pt:
                cx = px + off_x_pt * px_per_point
                cy = py + off_y_pt * px_per_point
                x1 = cx - text_w / 2.0
                y1 = cy - text_h / 2.0
                x2 = cx + text_w / 2.0
                y2 = cy + text_h / 2.0

                overlap = 0.0
                for ux1, uy1, ux2, uy2 in used_boxes:
                    ix = max(0.0, min(x2, ux2) - max(x1, ux1))
                    iy = max(0.0, min(y2, uy2) - max(y1, uy1))
                    overlap += ix * iy

                # Penalize label boxes that cover or touch any data point.
                point_penalty = 0.0
                min_gap_px = 14.0
                for pp_x, pp_y in point_pixels:
                    dx = 0.0
                    if pp_x < x1:
                        dx = x1 - pp_x
                    elif pp_x > x2:
                        dx = pp_x - x2
                    dy = 0.0
                    if pp_y < y1:
                        dy = y1 - pp_y
                    elif pp_y > y2:
                        dy = pp_y - y2
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist < min_gap_px:
                        point_penalty += (min_gap_px - dist) * 300.0

                out_penalty = 0.0
                if x1 < ax_box.x0:
                    out_penalty += (ax_box.x0 - x1) * 200.0
                if y1 < ax_box.y0:
                    out_penalty += (ax_box.y0 - y1) * 200.0
                if x2 > ax_box.x1:
                    out_penalty += (x2 - ax_box.x1) * 200.0
                if y2 > ax_box.y1:
                    out_penalty += (y2 - ax_box.y1) * 200.0

                dist_penalty = (off_x_pt * off_x_pt + off_y_pt * off_y_pt) * 0.1
                score = overlap * 60.0 + point_penalty + out_penalty + dist_penalty

                if best_score is None or score < best_score:
                    best_score = score
                    best_off = (float(off_x_pt), float(off_y_pt))

            bx = px + best_off[0] * px_per_point
            by = py + best_off[1] * px_per_point
            used_boxes.append((bx - text_w / 2.0, by - text_h / 2.0, bx + text_w / 2.0, by + text_h / 2.0))
            placed.append(
                {
                    "label": label,
                    "x": float(point["x"]),
                    "y": float(point["y"]),
                    "ox": best_off[0],
                    "oy": best_off[1],
                }
            )

        for item in placed:
            plt.annotate(
                str(item["label"]),
                xy=(float(item["x"]), float(item["y"])),
                xytext=(float(item["ox"]), float(item["oy"])),
                textcoords="offset points",
                fontsize=10,
                ha="center",
                va="center",
                zorder=4,
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.5", "alpha": 0.92},
                arrowprops={"arrowstyle": "-", "color": "0.4", "lw": 0.8, "shrinkA": 2, "shrinkB": 4},
            )

    plt.title("Pareto: quality vs latency")
    plt.xlabel("total latency avg (ms)")
    plt.ylabel("quality composite")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.35, zorder=0)
    save_current(out_dir / "pareto_quality_latency.png")

    def plot_heatmap(matrix_bundle: tuple[list[str], list[list[float | None]], list[list[int]]], title: str, filename: str) -> None:
        names, matrix, _ = matrix_bundle
        data: list[list[float]] = []
        for row in matrix:
            data.append([value if value is not None else float("nan") for value in row])
        fig = plt.figure(figsize=(7, 6))
        im = plt.imshow(data, vmin=0.0, vmax=1.0, cmap="viridis")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(names)), names, rotation=35, ha="right")
        plt.yticks(range(len(names)), names)
        plt.title(title)
        for i in range(len(names)):
            for j in range(len(names)):
                value = matrix[i][j]
                label = "NA" if value is None else f"{value:.2f}"
                plt.text(j, i, label, ha="center", va="center", fontsize=8, color="white")
        save_current(out_dir / filename)

    # 4-8) pairwise heatmaps
    plot_heatmap(pairwise_exact, "Pairwise win score (exact match)", "pairwise_exact_win.png")
    plot_heatmap(pairwise_cosine, "Pairwise win score (cosine similarity)", "pairwise_cosine_win.png")
    plot_heatmap(pairwise_quality, "Pairwise win score (composite quality)", "pairwise_quality_win.png")
    plot_heatmap(pairwise_exec, "Pairwise win score (execution success)", "pairwise_exec_win.png")
    plot_heatmap(pairwise_latency, "Pairwise win score (lower total latency)", "pairwise_latency_win.png")

    # 9) clause f1 heatmap (model x clause)
    clauses = sorted({row["clause"] for row in clause_rows})
    if clauses:
        by_model: dict[str, dict[str, float]] = {label: {} for label in labels}
        for row in clause_rows:
            f1 = row.get("f1")
            if f1 is not None:
                by_model[row["label"]][row["clause"]] = f1
        data = [[by_model[label].get(clause, float("nan")) for clause in clauses] for label in labels]
        fig = plt.figure(figsize=(max(10, len(clauses) * 0.55), max(4, len(labels) * 0.55)))
        im = plt.imshow(data, vmin=0.0, vmax=1.0, cmap="magma")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(clauses)), clauses, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)
        plt.title("Clause-level F1 heatmap")
        save_current(out_dir / "clause_f1_heatmap.png")

    # 10) top errors stacked
    if error_rows:
        total_by_signature = Counter()
        for row in error_rows:
            total_by_signature[row["signature"]] += int(row["count"])
        top_signatures = [signature for signature, _ in total_by_signature.most_common(max_error_bars)]
        model_index = {label: idx for idx, label in enumerate(labels)}
        base = [0] * len(labels)
        fig = plt.figure(figsize=(11, 5))
        for signature in top_signatures:
            values = [0] * len(labels)
            for row in error_rows:
                if row["signature"] == signature:
                    values[model_index[row["label"]]] += int(row["count"])
            plt.bar(labels, values, bottom=base, label=signature)
            base = [base[idx] + values[idx] for idx in range(len(base))]
        plt.xticks(rotation=20, ha="right")
        plt.title("Top error signatures by model")
        plt.ylabel("count")
        plt.legend(fontsize=8)
        save_current(out_dir / "errors_stacked.png")

    # 11+) similarity-vs-success by model
    rows_by_model: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
    for row in similarity_rows:
        rows_by_model[row["label"]].append(row)
    for label, rows in rows_by_model.items():
        rows_sorted = sorted(rows, key=lambda row: float(row["bin_start"]))
        x_labels = [f"{row['bin_start']:.1f}-{row['bin_end']:.1f}" for row in rows_sorted]
        exact_rate = [row.get("exact_true_rate") or 0.0 for row in rows_sorted]
        fig = plt.figure(figsize=(10, 4))
        plt.bar(x_labels, exact_rate)
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Exact-match rate by similarity bin: {label}")
        plt.ylabel("exact_true_rate")
        save_current(out_dir / f"similarity_exact_rate_{_sanitize_filename(label)}.png")

    return produced


def main() -> None:
    args = parse_args()
    runs = _load_runs(args.reports, args.labels)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = _compute_summary_rows(runs)
    summary_rows = sorted(
        summary_rows,
        key=lambda row: (
            row.get("quality_composite_avg") if row.get("quality_composite_avg") is not None else -1,
            row.get("exact_match_rate") if row.get("exact_match_rate") is not None else -1,
        ),
        reverse=True,
    )

    pairwise_exact = _pairwise_matrix(runs, lambda sample: _safe_bool(sample.get("exact_match")), higher_is_better=True)
    pairwise_cosine = _pairwise_matrix(
        runs, lambda sample: _safe_float(sample.get("cosine_similarity")), higher_is_better=True
    )
    pairwise_quality = _pairwise_matrix(runs, _sample_quality_score, higher_is_better=True)
    pairwise_exec = _pairwise_matrix(runs, _tool_success, higher_is_better=True)
    pairwise_latency = _pairwise_matrix(
        runs, lambda sample: _safe_float(sample.get("total_latency_ms")), higher_is_better=False
    )

    clause_rows = _clause_confusion_rows(runs)
    error_rows = _error_rows(runs)
    similarity_rows = _similarity_bin_rows(runs)

    files: list[Path] = []
    summary_csv = out_dir / "model_summary.csv"
    _write_csv(summary_csv, list(summary_rows[0].keys()) if summary_rows else ["label"], summary_rows)
    files.append(summary_csv)

    for name, bundle in [
        ("pairwise_exact_win.csv", pairwise_exact),
        ("pairwise_cosine_win.csv", pairwise_cosine),
        ("pairwise_quality_win.csv", pairwise_quality),
        ("pairwise_exec_win.csv", pairwise_exec),
        ("pairwise_latency_win.csv", pairwise_latency),
    ]:
        path = out_dir / name
        _write_csv(path, ["row_model", "col_model", "value", "support"], _matrix_to_rows(*bundle))
        files.append(path)

    clause_csv = out_dir / "clause_confusion.csv"
    _write_csv(
        clause_csv,
        [
            "label",
            "clause",
            "tp",
            "fp",
            "fn",
            "tn",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "support_positive_gt",
            "support_positive_pred",
        ],
        clause_rows,
    )
    files.append(clause_csv)

    error_csv = out_dir / "error_signatures_by_model.csv"
    _write_csv(error_csv, ["label", "signature", "count"], error_rows)
    files.append(error_csv)

    similarity_csv = out_dir / "similarity_success_bins.csv"
    _write_csv(
        similarity_csv,
        [
            "label",
            "bin_start",
            "bin_end",
            "total",
            "exact_true",
            "exact_true_rate",
            "exec_true",
            "exec_false",
            "exec_true_rate",
        ],
        similarity_rows,
    )
    files.append(similarity_csv)

    summary_md = out_dir / "summary.md"
    _write_markdown_summary(summary_md, summary_rows)
    files.append(summary_md)

    if not args.no_png:
        files.extend(
            _render_png(
                out_dir=out_dir,
                summary_rows=summary_rows,
                pairwise_exact=pairwise_exact,
                pairwise_cosine=pairwise_cosine,
                pairwise_quality=pairwise_quality,
                pairwise_exec=pairwise_exec,
                pairwise_latency=pairwise_latency,
                clause_rows=clause_rows,
                error_rows=error_rows,
                similarity_rows=similarity_rows,
                max_error_bars=args.max_error_bars,
            )
        )

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "models": [run.label for run in runs],
                "generated_files": [str(path) for path in files],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
