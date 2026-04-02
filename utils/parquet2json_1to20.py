#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import pandas as pd


def parquet_to_records(parquet_path: Path):
    df = pd.read_parquet(parquet_path)
    return df.to_dict(orient="records")


def write_jsonl(records, output_path: Path):
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/mnt/szf_temp/datasets/OpenCodeInstruct/data",
        help="Directory containing parquet files",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="train-*.parquet",
        help="Glob pattern for parquet files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save per-file jsonl files; default is input_dir",
    )
    parser.add_argument(
        "--merged_jsonl",
        type=str,
        default="all.jsonl",
        help="Filename for merged jsonl",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir = Path(args.output_dir) if args.output_dir is not None else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(input_dir.glob(args.pattern))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files matched pattern {args.pattern} in {input_dir}")

    merged_jsonl_path = output_dir / args.merged_jsonl
    total_count = 0

    with merged_jsonl_path.open("w", encoding="utf-8") as merged_f:
        for parquet_file in parquet_files:
            print(f"Processing: {parquet_file}")
            records = parquet_to_records(parquet_file)

            # write per-file jsonl
            per_file_jsonl = output_dir / (parquet_file.stem + ".jsonl")
            write_jsonl(records, per_file_jsonl)

            # append to merged jsonl
            for record in records:
                merged_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_count += len(records)
            print(f"  -> wrote {per_file_jsonl} ({len(records)} records)")

    print(f"\nMerged jsonl saved to: {merged_jsonl_path}")
    print(f"Total records: {total_count}")


if __name__ == "__main__":
    main()