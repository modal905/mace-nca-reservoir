"""
Summarize batch GOF evaluation results.

Usage:
  python summarize_gof.py --indir gof_results

Reads all CSV files from batch_gof_eval.py and produces:
1. Best-GOF checkpoint per seed (most passes, then highest min p-value)
2. Best-fitness checkpoint GOF for comparison
3. Final-gen (499) checkpoint GOF for comparison
"""

import os
import csv
import argparse
from collections import defaultdict


def load_csv(filepath):
    """Load a GOF results CSV into a list of dicts."""
    rows = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row['gen'] = int(row['gen'])
                row['ckpt_fitness'] = float(row['ckpt_fitness'])
                row['eval_fitness'] = float(row['eval_fitness'])
                row['gof_passes'] = int(row['gof_passes'])
                for k in ['p1_s0', 'p2_d0', 'p3_t0', 'p4_s1', 'p5_d1', 'p6_t1']:
                    row[k] = float(row[k])
                rows.append(row)
            except (ValueError, KeyError):
                continue  # skip error rows
    return rows


def best_gof(rows):
    """Select the best-GOF checkpoint: most passes, then highest min p-value, then highest fitness."""
    p_keys = ['p1_s0', 'p2_d0', 'p3_t0', 'p4_s1', 'p5_d1', 'p6_t1']
    def sort_key(r):
        return (r['gof_passes'], min(r[k] for k in p_keys), r['ckpt_fitness'])
    return max(rows, key=sort_key)


def best_fitness(rows):
    """Select the checkpoint with highest training fitness."""
    return max(rows, key=lambda r: r['ckpt_fitness'])


def final_gen(rows):
    """Select the gen 499 checkpoint."""
    gen499 = [r for r in rows if r['gen'] == 499]
    if gen499:
        return gen499[0]
    return max(rows, key=lambda r: r['gen'])  # fallback to highest gen


def format_gof(row):
    """Format a row for display."""
    p_keys = ['p1_s0', 'p2_d0', 'p3_t0', 'p4_s1', 'p5_d1', 'p6_t1']
    p_vals = [row[k] for k in p_keys]
    p_str = ", ".join(f"{p:.2f}" for p in p_vals)
    return (f"Gen {row['gen']:>3d} | Fitness {row['ckpt_fitness']:.4f} | "
            f"GOF [{p_str}] | Passes {row['gof_passes']}/6")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", default="gof_results")
    args = parser.parse_args()

    csv_files = sorted([f for f in os.listdir(args.indir) if f.endswith('.csv')])

    if not csv_files:
        print(f"No CSV files found in {args.indir}")
        return

    print("=" * 80)
    print("BATCH GOF EVALUATION SUMMARY")
    print("=" * 80)

    all_results = {}
    for csv_file in csv_files:
        rows = load_csv(os.path.join(args.indir, csv_file))
        if not rows:
            print(f"\nWARNING: {csv_file} has no valid rows")
            continue

        variant = rows[0]['variant']
        seed = rows[0]['seed']
        key = f"{variant}_seed{seed}"
        all_results[key] = rows

        bg = best_gof(rows)
        bf = best_fitness(rows)
        fg = final_gen(rows)

        print(f"\n{'─' * 80}")
        print(f" {variant.upper()} SEED {seed}  ({len(rows)} checkpoints)")
        print(f"{'─' * 80}")
        print(f"  Best GOF:     {format_gof(bg)}")
        print(f"  Best Fitness: {format_gof(bf)}")
        print(f"  Final Gen:    {format_gof(fg)}")

        if bg['gen'] == bf['gen'] == fg['gen']:
            print(f"  → All three are the same checkpoint (gen {bg['gen']})")
        else:
            if bg['gen'] != fg['gen']:
                print(f"  → Best-GOF differs from final-gen: "
                      f"gen {bg['gen']} vs gen {fg['gen']}")
            if bg['gen'] != bf['gen']:
                print(f"  → Best-GOF differs from best-fitness: "
                      f"gen {bg['gen']} vs gen {bf['gen']}")

    # Print comparison table
    print(f"\n{'=' * 80}")
    print("COMPARISON TABLE: Best-GOF vs Final-Gen checkpoint")
    print(f"{'=' * 80}")
    print(f"{'Variant':<10} {'Seed':<6} {'Best-GOF Gen':<14} {'Best-GOF Passes':<18} "
          f"{'Final-Gen Passes':<18} {'Change'}")
    print("-" * 80)

    for key in sorted(all_results.keys()):
        rows = all_results[key]
        variant = rows[0]['variant']
        seed = rows[0]['seed']
        bg = best_gof(rows)
        fg = final_gen(rows)
        change = bg['gof_passes'] - fg['gof_passes']
        change_str = f"+{change}" if change > 0 else str(change)
        print(f"{variant:<10} {seed:<6} {bg['gen']:<14d} {bg['gof_passes']:<18d} "
              f"{fg['gof_passes']:<18d} {change_str}")


if __name__ == "__main__":
    main()
