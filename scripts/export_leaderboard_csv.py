import json
import csv
import sys
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Convert pathologic leaderboard.json to CSV")
    parser.add_argument("input_path", type=Path, help="Path to leaderboard.json file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output CSV path. Defaults to same directory as input.")
    return parser.parse_args()

def main():
    args = parse_args()

    if not args.input_path.exists():
        print(f"Error: {args.input_path} does not exist", file=sys.stderr)
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        output_path = args.input_path.with_name("leaderboard_summary.csv")

    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "rows" not in data:
        print("Error: Invalid leaderboard format. Missing 'rows' key.", file=sys.stderr)
        sys.exit(1)

    rows = data["rows"]

    # Extract headers based on the first row's structure, standardizing common keys
    csv_rows = []

    for row in rows:
        flat_row = {
            "candidate": row.get("candidate", ""),
            "kind": row.get("kind", ""),
            "status": row.get("status", ""),
        }

        # HPO / NAS best score
        kind = row.get("kind", "")
        if "hpo" in row and row["hpo"].get("best_score") is not None:
            flat_row["validation_score"] = row["hpo"].get("best_score", "")
        elif "nas" in row and row["nas"].get("best_score") is not None:
            flat_row["validation_score"] = row["nas"].get("best_score", "")
        elif "ensemble" in row and row["ensemble"].get("validation_score") is not None:
            flat_row["validation_score"] = row["ensemble"].get("validation_score", "")
        else:
            flat_row["validation_score"] = ""

        # Test metrics
        metrics = row.get("test_metrics", {})
        if metrics:
            for k, v in metrics.items():
                flat_row[f"test_{k}"] = v

        csv_rows.append(flat_row)

    if not csv_rows:
        print("No rows found to export.", file=sys.stderr)
        sys.exit(0)

    # Gather all keys to form headers
    headers = []
    for row in csv_rows:
        for k in row.keys():
            if k not in headers:
                headers.append(k)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"CSV exported successfully to: {output_path}")

if __name__ == "__main__":
    main()
