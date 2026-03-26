"""
Batch GOF evaluation across all checkpoints for all seeds.

Usage:
  python batch_gof_eval.py --variant baseline --seed 42
  python batch_gof_eval.py --variant conserve --seed 43
  python batch_gof_eval.py --variant all

Runs GOF (goodness-of-fit) test on every saved checkpoint for the specified
variant/seed combination. Results are written to a CSV file for easy analysis.

Designed to be parallelized on AutoDL: run one instance per seed.
"""

import os
import sys
import csv
import glob
import re
import argparse
import time

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Use non-interactive matplotlib backend (headless server)
import matplotlib
matplotlib.use('Agg')

from critical_nca import CriticalNCA
import utils
from evaluate_criticality import evaluate_nca
import numpy as np
import inspect

# Phase 2 training run timestamps -> (variant, seed)
BASELINE_RUNS = {
    "20260306-184855": 42,
    "20260308-172413": 43,
    "20260310-115208": 44,
}

CONSERVE_RUNS = {
    "20260306-184946": 42,
    "20260308-103239": 43,
    "20260309-234527": 44,
}


def get_logdir(variant, timestamp):
    """Return the log directory for a given variant and timestamp."""
    if variant == "baseline":
        return os.path.join("logs", "train_nca", timestamp)
    else:
        return os.path.join("logs", "train_nca_conserve", timestamp)


def list_checkpoints(logdir):
    """List all checkpoint files in a logdir, sorted by generation."""
    pattern = os.path.join(logdir, "*.ckpt.index")
    ckpt_files = glob.glob(pattern)
    # Extract (gen, fitness, filename) tuples
    results = []
    for f in ckpt_files:
        basename = os.path.basename(f).replace(".ckpt.index", "")
        match = re.match(r"(\d+)_([\d.]+)", basename)
        if match:
            gen = int(match.group(1))
            fitness = float(match.group(2))
            ckpt_name = basename + ".ckpt"
            results.append((gen, fitness, ckpt_name))
    results.sort(key=lambda x: x[0])
    return results


def run_gof_for_checkpoint(args, ckpt_filename, logdir):
    """
    Run GOF evaluation for a single checkpoint.
    Returns dict with gen, fitness, and 6 GOF p-values.
    """
    valid_keys = set(inspect.signature(CriticalNCA.__init__).parameters.keys())
    valid_keys.discard("self")
    model_kwargs = {k: v for k, v in args.nca_model.items() if k in valid_keys}

    nca = CriticalNCA(**model_kwargs)

    # Load weights
    ckpt_path = os.path.join(logdir, ckpt_filename)
    nca.load_weights(ckpt_path)
    s = utils.get_flat_weights(nca.weights)

    # Redirect log_dir to a temp dir to avoid cluttering the training logdir
    # with hundreds of test images and plots
    import tempfile
    original_log_dir = args.log_dir
    tmpdir = tempfile.mkdtemp(prefix="gof_eval_")
    args.log_dir = tmpdir

    # Run evaluation with test=1 to trigger GOF computation
    # Redirect stdout to capture the printed GOF values
    import io
    from contextlib import redirect_stdout

    captured = io.StringIO()
    with redirect_stdout(captured):
        fit, val_dict = evaluate_nca(s, args, test=1)

    # Restore original log_dir
    args.log_dir = original_log_dir

    # Clean up temp dir (remove generated plots)
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    output = captured.getvalue()

    # Parse GOF p-values from output
    # Format: "goodness_of_fit(fit, data) 1.0" appears 6 times
    gof_values = []
    for line in output.split('\n'):
        if line.startswith("goodness_of_fit(fit, data)"):
            try:
                p_val = float(line.split()[-1])
                gof_values.append(p_val)
            except (ValueError, IndexError):
                gof_values.append(-1.0)

    # Pad if fewer than 6 (some distributions may be skipped)
    while len(gof_values) < 6:
        gof_values.append(-1.0)

    passes = sum(1 for p in gof_values[:6] if p >= 0.1)

    return {
        'gof_values': gof_values[:6],
        'passes': passes,
        'fitness_eval': -1 * fit,  # evaluate_nca returns negative fitness
        'val_dict': val_dict,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch GOF evaluation")
    parser.add_argument("--variant", required=True, choices=["baseline", "conserve", "all"],
                        help="Which variant to evaluate")
    parser.add_argument("--seed", type=int, default=None,
                        help="Specific seed to evaluate (42, 43, or 44). If omitted, runs all seeds for the variant.")
    parser.add_argument("--width", type=int, default=1000,
                        help="CA width for evaluation (default: 1000)")
    parser.add_argument("--outdir", default="gof_results",
                        help="Output directory for CSV results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Build list of (variant, timestamp, seed) to process
    jobs = []
    if args.variant in ("baseline", "all"):
        for ts, seed in BASELINE_RUNS.items():
            if args.seed is None or args.seed == seed:
                jobs.append(("baseline", ts, seed))
    if args.variant in ("conserve", "all"):
        for ts, seed in CONSERVE_RUNS.items():
            if args.seed is None or args.seed == seed:
                jobs.append(("conserve", ts, seed))

    for variant, timestamp, seed in jobs:
        logdir = get_logdir(variant, timestamp)
        if not os.path.isdir(logdir):
            print(f"WARNING: {logdir} not found, skipping")
            continue

        checkpoints = list_checkpoints(logdir)
        print(f"\n{'='*60}")
        print(f"Variant: {variant}, Seed: {seed}, Logdir: {logdir}")
        print(f"Found {len(checkpoints)} checkpoints")
        print(f"{'='*60}")

        # Load args from the training run
        args_filename = os.path.join(logdir, "args.json")
        argsio = utils.ArgsIO(args_filename)
        train_args = argsio.load_json()
        train_args.ca_width = args.width
        train_args.log_dir = logdir

        # Output CSV
        csv_path = os.path.join(args.outdir, f"gof_{variant}_seed{seed}.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'variant', 'seed', 'gen', 'ckpt_fitness', 'eval_fitness',
                'p1_s0', 'p2_d0', 'p3_t0', 'p4_s1', 'p5_d1', 'p6_t1',
                'gof_passes'
            ])

            for i, (gen, ckpt_fitness, ckpt_name) in enumerate(checkpoints):
                print(f"\n[{i+1}/{len(checkpoints)}] {variant} seed {seed}: "
                      f"gen {gen}, fitness {ckpt_fitness:.4f}")
                t0 = time.time()

                try:
                    result = run_gof_for_checkpoint(train_args, ckpt_name, logdir)
                    elapsed = time.time() - t0
                    gof = result['gof_values']
                    print(f"  GOF: {gof} -> {result['passes']}/6 passes "
                          f"({elapsed:.1f}s)")

                    writer.writerow([
                        variant, seed, gen, f"{ckpt_fitness:.7f}",
                        f"{result['fitness_eval']:.7f}",
                        *[f"{p:.4f}" for p in gof],
                        result['passes']
                    ])
                    csvfile.flush()

                except Exception as e:
                    print(f"  ERROR: {e}")
                    writer.writerow([
                        variant, seed, gen, f"{ckpt_fitness:.7f}",
                        "ERROR", *["ERROR"]*6, "ERROR"
                    ])
                    csvfile.flush()

        print(f"\nResults saved to: {csv_path}")

    # After all jobs, print summary
    print(f"\n{'='*60}")
    print("BATCH GOF EVALUATION COMPLETE")
    print(f"Results in: {args.outdir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
