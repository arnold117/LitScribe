#!/usr/bin/env python3
"""Run ablation experiments for LitScribe.

Automatically runs the full pipeline and variants with individual
components disabled, then produces a comparison table.

Usage:
    python scripts/run_ablation.py --query benchmarks/queries/bio_alkaloid.json
    python scripts/run_ablation.py --query benchmarks/queries/cs_transformer.json --skip-full
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# Ablation configurations: name -> extra CLI flags
ABLATION_CONFIGS = {
    "full": [],
    "no_graphrag": ["--disable-graphrag"],
    "no_self_review": ["--disable-self-review"],
    "no_domain_filter": ["--disable-domain-filter"],
    "no_snowball": ["--disable-snowball"],
}


def run_single_experiment(
    query: Dict[str, Any],
    config_name: str,
    extra_flags: List[str],
    output_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Run a single experiment configuration.

    Args:
        query: Benchmark query config
        config_name: Name of this configuration
        extra_flags: Additional CLI flags
        output_dir: Where to save output

    Returns:
        The output JSON, or None on failure
    """
    research_question = query["research_question"]
    max_papers = query.get("max_papers", 10)
    sources = ",".join(query.get("sources", ["arxiv", "semantic_scholar", "pubmed"]))
    language = query.get("language", "en")

    output_path = output_dir / f"ablation_{config_name}"

    cmd = [
        sys.executable, "-m", "cli.litscribe_cli",
        "review", research_question,
        "-p", str(max_papers),
        "-s", sources,
        "--lang", language,
        "-o", str(output_path),
        "--auto",  # Skip interactive prompts
    ] + extra_flags

    print(f"\n{'='*60}")
    print(f"Running: {config_name}")
    print(f"Flags: {extra_flags or '(none)'}")
    print(f"{'='*60}")

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=str(Path(__file__).parent.parent / "src"),
        )
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.1f}s (exit code: {result.returncode})")

        if result.returncode != 0:
            print(f"STDERR: {result.stderr[:500]}")
            return None

        # Load the output JSON
        json_path = output_path.with_suffix(".json")
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_ablation_config"] = config_name
            data["_ablation_elapsed"] = round(elapsed, 1)
            return data
        else:
            print(f"Output JSON not found: {json_path}")
            return None

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after 600s")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def evaluate_result(
    output: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate a single result using the evaluation framework."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        # Import evaluate script as module
        sys.path.insert(0, str(Path(__file__).parent))
        from evaluate import ReviewEvaluator
        evaluator = ReviewEvaluator()
        return evaluator.evaluate(output, ground_truth)
    except ImportError:
        return {"error": "evaluate.py not found"}


def format_comparison_table(
    results: Dict[str, Dict[str, Any]],
) -> str:
    """Format results as a markdown comparison table."""
    lines = []
    lines.append("# Ablation Experiment Results\n")

    # Header
    configs = list(results.keys())
    lines.append("| Metric | " + " | ".join(configs) + " |")
    lines.append("|--------|" + "|".join(["--------"] * len(configs)) + "|")

    # Metrics to compare
    metrics = [
        ("Overall Score", lambda r: f"{r.get('overall_score', 0):.2%}"),
        ("Papers Found", lambda r: str(r.get("search", {}).get("paper_count", 0))),
        ("Keyword Hit Rate", lambda r: f"{r.get('search', {}).get('keyword_hit_rate', 0):.0%}"),
        ("Theme Coverage", lambda r: f"{r.get('theme_coverage', {}).get('coverage_rate', 0):.0%}"),
        ("Domain Purity", lambda r: f"{r.get('domain_purity', {}).get('purity_rate', 0):.0%}"),
        ("Citation Grounding", lambda r: f"{r.get('citation_grounding', {}).get('grounding_rate', 0):.0%}" if r.get('citation_grounding', {}).get('grounding_rate') is not None else "N/A"),
        ("Self-Review Score", lambda r: f"{r.get('self_review', {}).get('overall_score', 0):.2f}" if r.get('self_review', {}).get('available') else "N/A"),
        ("Total Tokens", lambda r: f"{r.get('efficiency', {}).get('total_tokens', 0):,}" if r.get('efficiency', {}).get('available') else "N/A"),
        ("Cost (USD)", lambda r: f"${r.get('efficiency', {}).get('estimated_cost_usd', 0):.4f}" if r.get('efficiency', {}).get('available') else "N/A"),
        ("Failures", lambda r: str(len(r.get("failure_modes", [])))),
    ]

    for label, extractor in metrics:
        row = f"| {label} | "
        row += " | ".join(extractor(results.get(c, {})) for c in configs)
        row += " |"
        lines.append(row)

    # Failure details
    lines.append("\n## Failure Details\n")
    for config, result in results.items():
        failures = result.get("failure_modes", [])
        if failures:
            lines.append(f"### {config}")
            for f in failures:
                lines.append(f"- **{f['mode']}**: {f['detail']}")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run LitScribe ablation experiments")
    parser.add_argument("--query", "-q", required=True, help="Path to benchmark query JSON")
    parser.add_argument("--ground-truth", "-g", help="Path to ground truth JSON (auto-detected if not provided)")
    parser.add_argument("--output-dir", "-o", default="benchmarks/results", help="Output directory")
    parser.add_argument("--configs", nargs="+", default=list(ABLATION_CONFIGS.keys()),
                        help="Configurations to run (default: all)")
    parser.add_argument("--skip-run", action="store_true", help="Skip running, just evaluate existing outputs")
    args = parser.parse_args()

    # Load query
    query_path = Path(args.query)
    with open(query_path, "r", encoding="utf-8") as f:
        query = json.load(f)

    query_id = query.get("id", query_path.stem)

    # Auto-detect ground truth
    if args.ground_truth:
        gt_path = Path(args.ground_truth)
    else:
        gt_path = query_path.parent.parent / "ground_truth" / f"{query_id}_expected.json"

    if gt_path.exists():
        with open(gt_path, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)
    else:
        print(f"Warning: Ground truth not found at {gt_path}, using empty")
        ground_truth = {}

    output_dir = Path(args.output_dir) / query_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiments
    outputs = {}
    if not args.skip_run:
        for config_name in args.configs:
            if config_name not in ABLATION_CONFIGS:
                print(f"Unknown config: {config_name}, skipping")
                continue
            output = run_single_experiment(
                query, config_name, ABLATION_CONFIGS[config_name], output_dir
            )
            if output:
                outputs[config_name] = output
                # Save individual result
                with open(output_dir / f"{config_name}.json", "w") as f:
                    json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    else:
        # Load existing outputs
        for config_name in args.configs:
            json_path = output_dir / f"{config_name}.json"
            if json_path.exists():
                with open(json_path, "r") as f:
                    outputs[config_name] = json.load(f)

    if not outputs:
        print("No results to evaluate")
        return

    # Evaluate all results
    eval_results = {}
    for config_name, output in outputs.items():
        eval_results[config_name] = evaluate_result(output, ground_truth)

    # Generate comparison table
    report = format_comparison_table(eval_results)
    print(f"\n{report}")

    # Save report
    report_path = output_dir / "ablation_comparison.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # Save raw evaluation results
    eval_path = output_dir / "ablation_eval.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
