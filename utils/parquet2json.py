#!/usr/bin/env python3
# transfer /mnt/szf_temp/datasets/OpenCodeInstruct/data/train-00011-of-00050_first1000.parquet to jsonl

import argparse
import json
from pathlib import Path

import pandas as pd


def parquet_to_jsonl(input_path: str, output_path: str | None = None) -> Path:
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if output_path is None:
        output_file = input_file.with_suffix(".jsonl")
    else:
        output_file = Path(output_path)

    df = pd.read_parquet(input_file)

    with output_file.open("w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="/mnt/szf_temp/datasets/OpenCodeInstruct/data/train-00011-of-00050.parquet",
        help="Path to input parquet file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output jsonl file",
    )
    args = parser.parse_args()

    output_file = parquet_to_jsonl(args.input, args.output)
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()