from pathlib import Path
import csv
import argparse

from metrics.evaluation_metrics import EvaluationMetrics


def get_args():
    parser = argparse.ArgumentParser(
        description="Calculate mean of evaluation metrics."
    )
    parser.add_argument(
        "--table",
        "-t",
        type=str,
        nargs="+",
        help="File(s)/dir(s) where the evaluation result table(s) are saved.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Directory where to save the calculated mean of evaluation metrics.",
    )
    return parser.parse_args()


def read_csv(file: Path):
    with open(file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [row for row in reader]
    return header, data


def main():
    args = get_args()

    all_data = []
    for table in args.table:
        table_path = Path(table)
        if table_path.is_file():
            header, data = read_csv(table_path)
            all_data.append((header, data))
        else:
            table_files = list(table_path.glob("*.csv"))
            for table_file in table_files:
                header, data = read_csv(table_file)
                all_data.append((header, data))

    # Calculate mean of evaluation metrics
    mean_metrics = {}
    for header, data in all_data:
        for d in data:
            id = d[0].split(":")[0]
            if id not in mean_metrics:
                mean_metrics[id] = {
                    h: [float(d[i + 1])] for i, h in enumerate(header[1:])
                }
            else:
                for i, h in enumerate(header[1:]):
                    mean_metrics[id][h].append(float(d[i + 1]))

    for id, metrics in mean_metrics.items():
        for metric, values in metrics.items():
            mean_metrics[id][metric] = sum(values) / len(values)

    # Save the calculated mean of evaluation metrics
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = output_path / f"{EvaluationMetrics._get_current_timestamp()}.csv"
    with open(filename, "w") as f:
        writer = csv.writer(f)
        header = ["id"]
        header.extend(metrics.keys())
        writer.writerow(header)
        for id, metrics in mean_metrics.items():
            writer.writerow([id] + list(metrics.values()))


if __name__ == "__main__":
    main()
