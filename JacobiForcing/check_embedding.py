import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether per-position next-token top-k embeddings are similar "
            "for accepted_prefix_global_raw_tokens."
        )
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=(
            "/home/szf/profiling/jacobi_debug_logits_s0_c42_g42_itr1.json"
        ),
        help="Path to JSON or JSONL file containing accepted_prefix_global_raw_tokens",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/szf/huggingface/JacobiForcing_Coder_7B_v1",
        help="Model path for forward pass",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="/home/szf/huggingface/Qwen2.5-Coder-7B-Instruct",
        help="Tokenizer path for token decode in debug output",
    )
    parser.add_argument("--topk", type=int, default=10, help="Top-k size per position")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="How many prefixes to analyze (for speed)",
    )
    parser.add_argument(
        "--max-prefix-len",
        type=int,
        default=2048,
        help="Truncate each prefix to this length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run analysis on",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="embedding_similarity_report.json",
        help="Output JSON path for analysis results",
    )
    return parser.parse_args()


def load_jsonl(path):
    with open(path, "r") as f:
        text = f.read()

    # 1) Try whole-file JSON first (single object or list of objects).
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
        raise ValueError(f"Unsupported JSON root type: {type(obj)}")
    except json.JSONDecodeError:
        pass

    # 2) Fallback to JSONL (one object per non-empty line).
    rows = []
    bad_lines = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            bad_lines.append((line_no, str(exc)))

    if rows and not bad_lines:
        return rows

    if bad_lines:
        first_line, first_err = bad_lines[0]
        raise ValueError(
            f"Failed to parse file as JSON or JSONL. First bad line {first_line}: {first_err}"
        )
    raise ValueError("Input file is empty or has no valid JSON content.")


def to_int_list(values):
    out = []
    for v in values:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            return None
    return out


def tokens_to_ids(tokens, tokenizer):
    ids = []
    for tok in tokens:
        if isinstance(tok, int):
            ids.append(int(tok))
            continue

        if isinstance(tok, str):
            # Raw pieces like " elements" / "\n" come from single-token decode.
            # Encode each piece without special tokens; this is robust even if a
            # piece maps to multiple IDs in edge cases.
            piece_ids = tokenizer(tok, add_special_tokens=False).input_ids
            if len(piece_ids) == 0:
                continue
            ids.extend(piece_ids)
            continue

        return None
    return ids


def convert_raw_tokens_to_ids(raw_tokens, tokenizer):
    if not isinstance(raw_tokens, list) or len(raw_tokens) < 1:
        return None
    casted = to_int_list(raw_tokens)
    if casted is not None:
        return casted
    return tokens_to_ids(raw_tokens, tokenizer)


def collect_noisy_input_tokens(item):
    per_token = item.get("per_token")
    if not isinstance(per_token, list):
        return []

    noisy_tokens = []
    for t in per_token:
        if not isinstance(t, dict):
            continue
        tok = t.get("input_token")
        if tok is None:
            continue
        noisy_tokens.append(tok)
    return noisy_tokens


def collect_context_pairs(records, max_samples, max_prefix_len, tokenizer):
    pairs = []
    for idx, item in enumerate(records):
        clean_raw = item.get("accepted_prefix_global_raw_tokens")
        clean_ids = convert_raw_tokens_to_ids(clean_raw, tokenizer)
        if clean_ids is None or len(clean_ids) < 2:
            continue

        noisy_raw = collect_noisy_input_tokens(item)
        noisy_ids = convert_raw_tokens_to_ids(noisy_raw, tokenizer) if len(noisy_raw) > 0 else []
        if noisy_ids is None:
            noisy_ids = []

        clean_context_ids = clean_ids[:max_prefix_len]
        noisy_context_ids = (clean_ids + noisy_ids)[:max_prefix_len]

        if len(noisy_context_ids) < 2:
            continue

        pairs.append(
            {
                "record_index": idx,
                "task_id": item.get("task_id", f"idx_{idx}"),
                "clean_context": clean_context_ids,
                "noisy_context": noisy_context_ids,
                "noisy_suffix_len": int(len(noisy_ids)),
            }
        )
        if len(pairs) >= max_samples:
            break
    return pairs


def pairwise_upper_values(sim_matrix):
    n = sim_matrix.shape[0]
    if n < 2:
        return torch.tensor([], device=sim_matrix.device)
    idx = torch.triu_indices(n, n, offset=1, device=sim_matrix.device)
    return sim_matrix[idx[0], idx[1]]


def analyze_one_prefix(model, tokenizer, prefix_tokens, topk, device):
    input_ids = torch.tensor([prefix_tokens], dtype=torch.long, device=device)

    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits[0]  # [seq_len, vocab]

    k = min(topk, logits.shape[-1])
    topk_ids = torch.topk(logits, k=k, dim=-1).indices  # [seq_len, k]
    emb_weight = model.get_input_embeddings().weight

    # top-k token embeddings for each position: [seq_len, k, hidden]
    topk_emb = emb_weight[topk_ids]

    # Use centroid embedding of top-k at each position as summary.
    centroids = F.normalize(topk_emb.mean(dim=1), p=2, dim=-1)
    sim_matrix = centroids @ centroids.transpose(0, 1)

    upper_vals = pairwise_upper_values(sim_matrix)
    if sim_matrix.shape[0] > 1:
        adjacent_vals = sim_matrix.diagonal(offset=1)
    else:
        adjacent_vals = torch.tensor([], device=sim_matrix.device)

    # Jaccard overlap between adjacent top-k token-id sets.
    adjacent_jaccard = []
    for pos in range(topk_ids.shape[0] - 1):
        a = set(topk_ids[pos].tolist())
        b = set(topk_ids[pos + 1].tolist())
        inter = len(a & b)
        union = len(a | b)
        adjacent_jaccard.append((inter / union) if union > 0 else 0.0)

    preview = []
    for pos in range(topk_ids.shape[0]):
        token_ids = topk_ids[pos].tolist()
        preview.append(
            {
                "position": pos,
                "topk_ids": token_ids,
                "topk_tokens": tokenizer.convert_ids_to_tokens(token_ids),
            }
        )

    return {
        "prefix_len": int(len(prefix_tokens)),
        "mean_pairwise_cosine": float(upper_vals.mean().item()) if upper_vals.numel() > 0 else float("nan"),
        "mean_adjacent_cosine": float(adjacent_vals.mean().item()) if adjacent_vals.numel() > 0 else float("nan"),
        "min_adjacent_cosine": float(adjacent_vals.min().item()) if adjacent_vals.numel() > 0 else float("nan"),
        "max_adjacent_cosine": float(adjacent_vals.max().item()) if adjacent_vals.numel() > 0 else float("nan"),
        "mean_adjacent_jaccard_topk": (
            float(sum(adjacent_jaccard) / len(adjacent_jaccard)) if len(adjacent_jaccard) > 0 else float("nan")
        ),
        "topk_preview": preview,
    }


def safe_mean(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    vals = [v for v in vals if v == v]  # filter NaN
    return (sum(vals) / len(vals)) if vals else float("nan")


def main():
    args = parse_args()

    print(f"Loading JSON/JSONL from: {args.input_path}")
    records = load_jsonl(args.input_path)
    print(f"Loaded records: {len(records)}")

    print(f"Loading tokenizer: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    context_pairs = collect_context_pairs(records, args.max_samples, args.max_prefix_len, tokenizer)
    print(f"Found valid context pairs: {len(context_pairs)}")

    if len(context_pairs) == 0:
        print("No valid clean/noisy context pairs found. Nothing to analyze.")
        return

    print(f"Loading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
    ).to(args.device)
    model.eval()

    report_rows = []
    for i, entry in enumerate(context_pairs, start=1):
        print(
            f"[{i}/{len(context_pairs)}] analyzing task_id={entry['task_id']} "
            f"record_index={entry['record_index']} clean_len={len(entry['clean_context'])} "
            f"noisy_len={len(entry['noisy_context'])} noisy_suffix_len={entry['noisy_suffix_len']}"
        )
        clean_metrics = analyze_one_prefix(
            model=model,
            tokenizer=tokenizer,
            prefix_tokens=entry["clean_context"],
            topk=args.topk,
            device=args.device,
        )
        noisy_metrics = analyze_one_prefix(
            model=model,
            tokenizer=tokenizer,
            prefix_tokens=entry["noisy_context"],
            topk=args.topk,
            device=args.device,
        )

        delta_pairwise = clean_metrics["mean_pairwise_cosine"]
        delta_adjacent = clean_metrics["mean_adjacent_cosine"]
        if noisy_metrics["mean_pairwise_cosine"] == noisy_metrics["mean_pairwise_cosine"] and delta_pairwise == delta_pairwise:
            delta_pairwise = noisy_metrics["mean_pairwise_cosine"] - delta_pairwise
        else:
            delta_pairwise = float("nan")
        if noisy_metrics["mean_adjacent_cosine"] == noisy_metrics["mean_adjacent_cosine"] and delta_adjacent == delta_adjacent:
            delta_adjacent = noisy_metrics["mean_adjacent_cosine"] - delta_adjacent
        else:
            delta_adjacent = float("nan")

        row = {
            "task_id": entry["task_id"],
            "record_index": entry["record_index"],
            "noisy_suffix_len": entry["noisy_suffix_len"],
            "clean": clean_metrics,
            "noisy": noisy_metrics,
            "delta_noisy_minus_clean": {
                "mean_pairwise_cosine": delta_pairwise,
                "mean_adjacent_cosine": delta_adjacent,
            },
        }
        report_rows.append(row)

        print(
            "  clean(pairwise={:.4f}, adjacent={:.4f}) | "
            "noisy(pairwise={:.4f}, adjacent={:.4f}) | "
            "delta(noisy-clean): pairwise={:.4f}, adjacent={:.4f}".format(
                row["clean"]["mean_pairwise_cosine"],
                row["clean"]["mean_adjacent_cosine"],
                row["noisy"]["mean_pairwise_cosine"],
                row["noisy"]["mean_adjacent_cosine"],
                row["delta_noisy_minus_clean"]["mean_pairwise_cosine"],
                row["delta_noisy_minus_clean"]["mean_adjacent_cosine"],
            )
        )

    summary = {
        "num_prefixes": len(report_rows),
        "mean_clean_prefix_len": safe_mean([r["clean"]["prefix_len"] for r in report_rows]),
        "mean_noisy_prefix_len": safe_mean([r["noisy"]["prefix_len"] for r in report_rows]),
        "mean_clean_pairwise_cosine": safe_mean([r["clean"]["mean_pairwise_cosine"] for r in report_rows]),
        "mean_noisy_pairwise_cosine": safe_mean([r["noisy"]["mean_pairwise_cosine"] for r in report_rows]),
        "mean_clean_adjacent_cosine": safe_mean([r["clean"]["mean_adjacent_cosine"] for r in report_rows]),
        "mean_noisy_adjacent_cosine": safe_mean([r["noisy"]["mean_adjacent_cosine"] for r in report_rows]),
        "mean_delta_pairwise_noisy_minus_clean": safe_mean([
            r["delta_noisy_minus_clean"]["mean_pairwise_cosine"] for r in report_rows
        ]),
        "mean_delta_adjacent_noisy_minus_clean": safe_mean([
            r["delta_noisy_minus_clean"]["mean_adjacent_cosine"] for r in report_rows
        ]),
        "mean_clean_adjacent_jaccard_topk": safe_mean([r["clean"]["mean_adjacent_jaccard_topk"] for r in report_rows]),
        "mean_noisy_adjacent_jaccard_topk": safe_mean([r["noisy"]["mean_adjacent_jaccard_topk"] for r in report_rows]),
    }

    output = {
        "config": {
            "input_path": args.input_path,
            "model_path": args.model_path,
            "tokenizer_path": args.tokenizer_path,
            "topk": args.topk,
            "max_samples": args.max_samples,
            "max_prefix_len": args.max_prefix_len,
            "device": args.device,
        },
        "summary": summary,
        "rows": report_rows,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, ensure_ascii=True, indent=2)

    print("\n=== Summary ===")
    print(f"num_prefixes: {summary['num_prefixes']}")
    print(f"mean_clean_prefix_len: {summary['mean_clean_prefix_len']:.4f}")
    print(f"mean_noisy_prefix_len: {summary['mean_noisy_prefix_len']:.4f}")
    print(f"mean_clean_pairwise_cosine: {summary['mean_clean_pairwise_cosine']:.4f}")
    print(f"mean_noisy_pairwise_cosine: {summary['mean_noisy_pairwise_cosine']:.4f}")
    print(f"mean_clean_adjacent_cosine: {summary['mean_clean_adjacent_cosine']:.4f}")
    print(f"mean_noisy_adjacent_cosine: {summary['mean_noisy_adjacent_cosine']:.4f}")
    print(f"mean_delta_pairwise_noisy_minus_clean: {summary['mean_delta_pairwise_noisy_minus_clean']:.4f}")
    print(f"mean_delta_adjacent_noisy_minus_clean: {summary['mean_delta_adjacent_noisy_minus_clean']:.4f}")
    print(f"mean_clean_adjacent_jaccard_topk: {summary['mean_clean_adjacent_jaccard_topk']:.4f}")
    print(f"mean_noisy_adjacent_jaccard_topk: {summary['mean_noisy_adjacent_jaccard_topk']:.4f}")
    print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()