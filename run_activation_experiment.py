"""Compare DoG activation times under a shared adaptive pruning policy."""
import argparse
import csv
import filecmp
import json
import subprocess
import sys
from pathlib import Path


def run(script, *args):
    command = [sys.executable, str(Path(__file__).with_name(script)), *map(str, args)]
    subprocess.run(command, check=True)


def read(path):
    return json.loads(path.read_text())


def training_result(path, common, switch, expected):
    summary = path / "training_summary.json"
    if not summary.exists():
        if path.exists():
            raise ValueError(f"Incomplete run in {path}. Move it aside before retrying.")
        extra = [] if switch is None else ["--dog_start_iter", switch]
        run("train_prune_DoG.py", *common, "-m", path, *extra)
    stats = read(summary)
    options = stats["options"]
    for key, value in expected.items():
        actual = options[key]
        if key == "source_path":
            actual = Path(actual).resolve()
        if actual != value:
            raise ValueError(f"{path}: expected {key}={value}, found {actual}")
    wanted = options["pruning_stop_iter"] if switch is None else switch
    if stats["dog_start_iter"] != wanted or stats["final_gaussians"] != expected["pruning_target"]:
        raise ValueError(f"Activation time or final budget mismatch in {path}")
    ply = path / "point_cloud" / f"iteration_{options['iterations']}" / "point_cloud.ply"
    if ply.stat().st_size != stats["model_ply_bytes"]:
        raise ValueError(f"Final model size mismatch in {path}")
    return stats


def result_row(path, stats):
    iteration = stats["options"]["iterations"]
    method = f"ours_{iteration}"
    timing = path / "test" / method / "render_timing.json"
    if not timing.exists():
        run("render_DoG.py", "-m", path, "--iteration", iteration, "--skip_train")
    results = path / "results.json"
    if not results.exists() or method not in read(results):
        run("metrics.py", "-m", path)
    row = {"condition": path.name, "seed": stats["options"]["seed"]}
    for key in ("dog_start_iter", "final_gaussians", "model_ply_bytes", "training_seconds",
                "setup_and_training_seconds"):
        row[key] = stats[key]
    row.update(read(results)[method])
    row.update(read(timing))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--source_path", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--pruning_target", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-r", "--resolution", type=int, default=-1)
    parser.add_argument("--white_background", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--early_iter", type=int, help="Single early activation iteration")
    group.add_argument("--early_iters", type=int, nargs="+", help="Explicit early activation iterations")
    group.add_argument("--sweep", action="store_true",
                       help="One early condition per gap between successive baseline pruning events")
    args = parser.parse_args()
    if args.pruning_target <= 0:
        parser.error("pruning_target must be positive")
    multi = args.sweep or args.early_iters is not None
    args.output.mkdir(parents=True, exist_ok=multi)
    expected = dict(source_path=args.source_path.resolve(), pruning_target=args.pruning_target,
                    seed=args.seed, resolution=args.resolution, white_background=args.white_background,
                    iterations=40000)
    common = ["-s", expected["source_path"], "--pruning_target", args.pruning_target,
              "--seed", args.seed, "-r", args.resolution, "--disable_viewer"]
    if args.white_background:
        common.append("--white_background")
    baseline = args.output / "post_pruning"
    reference = training_result(baseline, common, None, expected)
    events = [event["iteration"] for event in reference["pruning_events"]]
    if len(events) < 2:
        raise ValueError("At least two baseline pruning events are needed")
    midpoints = [(a + b) // 2 for a, b in zip(events, events[1:])]
    if args.sweep:
        switches = midpoints
    elif args.early_iters is not None:
        switches = args.early_iters
    else:
        switches = [midpoints[-1] if args.early_iter is None else args.early_iter]
    if len(set(switches)) != len(switches) or not all(events[0] < s < events[-1] for s in switches):
        raise ValueError(f"Choose distinct early iterations strictly between {events[0]} and {events[-1]}")
    print("Baseline pruning events:", events, "Early activation iterations:", switches, flush=True)

    legacy = args.output / "late_pruning"
    legacy_summary = legacy / "training_summary.json"
    legacy_switch = read(legacy_summary)["dog_start_iter"] if legacy_summary.exists() else None
    rows = [result_row(baseline, reference)]
    for switch in switches:
        path = legacy if not multi or switch == legacy_switch else args.output / f"early_{switch}"
        stats = training_result(path, common, switch, expected)
        for key, value in reference["options"].items():
            if key not in {"dog_start_iter", "source_path", "model_path"} and stats["options"][key] != value:
                raise ValueError(f"Training option differs in {path}: {key}")
        for name in ("input.ply", "cameras.json"):
            if not filecmp.cmp(baseline / name, path / name, shallow=False):
                raise ValueError(f"Paired inputs differ in {path}: {name}")
        if not any(e["iteration"] > switch for e in stats["pruning_events"]):
            raise ValueError(f"No actual pruning occurred after DoG activation in {path}")
        row = result_row(path, stats)
        if row["views"] != rows[0]["views"]:
            raise ValueError(f"Test view count differs in {path}")
        rows.append(row)

    dest = args.output / ("comparison_sweep.csv" if multi else "comparison.csv")
    temp = dest.with_suffix(".csv.tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(dest)
    print(dest)


if __name__ == "__main__":
    main()
