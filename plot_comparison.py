"""
Side-by-side training curve comparison: Baseline vs Conserving NCA.
Usage:
    python plot_comparison.py
Outputs:
    results/comparison_baseline_vs_conserving.png
    results/comparison_baseline_vs_conserving_bestever.png
"""
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

BASELINE_CSV = "baseline_run_2/logs/train_nca/20260222-130628/loss_history.csv"
CONSERVING_CSV = "conserving_run_2/logs/train_nca_conserve/20260222-130839/loss_history.csv"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

def load(path):
    data = np.genfromtxt(path, delimiter=';') * -1
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data  # shape: (generations, popsize)

def bestever(fitness_matrix):
    """Running best-ever (max over all generations so far)."""
    max_per_gen = np.max(fitness_matrix, axis=1)
    return np.maximum.accumulate(max_per_gen)

# ── Load ──────────────────────────────────────────────────────────────────────
baseline = load(BASELINE_CSV)
conserving = load(CONSERVING_CSV)
gens_b = np.arange(baseline.shape[0])
gens_c = np.arange(conserving.shape[0])

avg_b = np.mean(baseline, axis=1)
max_b = np.max(baseline, axis=1)
best_b = bestever(baseline)

avg_c = np.mean(conserving, axis=1)
max_c = np.max(conserving, axis=1)
best_c = bestever(conserving)

# ── Palette ───────────────────────────────────────────────────────────────────
BLUE       = '#4477AA'
BLUE_LIGHT = '#99BBDD'
ORANGE     = '#EE7733'
ORANGE_LT  = '#FFBB77'
GREEN      = '#228833'

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: per-generation max + average (population scatter omitted for clarity)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
fig.suptitle('Training Curves — Baseline vs Mass-Conserving NCA\n'
             '(width=100, timesteps=100, 500 gens, seed=671052, popsize=96)',
             fontsize=11)

for ax, gens, avg, mx, be, color, lcolor, label, best_val in [
    (axes[0], gens_b, avg_b, max_b, best_b, BLUE, BLUE_LIGHT,
     'Baseline', max(best_b)),
    (axes[1], gens_c, avg_c, max_c, best_c, ORANGE, ORANGE_LT,
     'Conserving', max(best_c)),
]:
    ax.fill_between(gens, avg, mx, color=lcolor, alpha=0.35, label='avg–max band')
    ax.plot(gens, avg, color=color, lw=1.2, alpha=0.8, label='Average fitness')
    ax.plot(gens, mx,  color=color, lw=1.5, label='Generation best')
    ax.plot(gens, be,  color=GREEN, lw=2.0, ls='--', label='Best-ever')
    ax.axhline(best_val, color=GREEN, lw=0.8, ls=':', alpha=0.6)
    ax.text(len(gens) * 0.98, best_val + 0.03,
            f'bestever={best_val:.4f}', ha='right', fontsize=8, color=GREEN)
    ax.set_title(f'{label}  (bestever={best_val:.4f})')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_xlim(0, max(len(gens_b), len(gens_c)))
    ax.set_ylim(-0.1, 4.5)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "comparison_baseline_vs_conserving.png")
plt.savefig(out1, dpi=200)
plt.close()
print(f"Saved: {out1}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: best-ever overlay on single axes
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_title('Best-Ever Fitness — Baseline vs Conserving NCA\n'
             '(width=100, timesteps=100, 500 gens, seed=671052)',
             fontsize=11)

ax.plot(gens_b, best_b, color=BLUE,   lw=2.0, label=f'Baseline     (bestever={max(best_b):.4f})')
ax.plot(gens_c, best_c, color=ORANGE, lw=2.0, label=f'Conserving  (bestever={max(best_c):.4f})')
ax.axhline(3.0, color='gray', ls=':', lw=1.0, alpha=0.7, label='fitness=3.0 threshold')

ax.set_xlabel('Generation')
ax.set_ylabel('Best-ever fitness')
ax.set_xlim(0, 500)
ax.set_ylim(-0.1, 4.5)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Annotate convergence generation (first time best-ever > 3.0)
thresh = 3.0
for gens, be, color, label in [(gens_b, best_b, BLUE, 'Baseline'),
                                (gens_c, best_c, ORANGE, 'Conserving')]:
    cross = np.where(be > thresh)[0]
    if len(cross):
        g = cross[0]
        ax.axvline(g, color=color, lw=0.8, ls='--', alpha=0.5)
        ax.text(g + 3, 3.05, f'{label}\ncross gen {g}',
                fontsize=7.5, color=color, va='bottom')

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "comparison_baseline_vs_conserving_bestever.png")
plt.savefig(out2, dpi=200)
plt.close()
print(f"Saved: {out2}")

# ══════════════════════════════════════════════════════════════════════════════
# Text summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Summary ──────────────────────────────────────────────")
print(f"{'Metric':<32} {'Baseline':>12} {'Conserving':>12}")
print("-" * 58)
print(f"{'Bestever fitness':<32} {max(best_b):>12.4f} {max(best_c):>12.4f}")
print(f"{'Final-gen average':<32} {avg_b[-1]:>12.4f} {avg_c[-1]:>12.4f}")
print(f"{'Final-gen best':<32} {max_b[-1]:>12.4f} {max_c[-1]:>12.4f}")
cross_b = np.where(best_b > 3.0)[0]
cross_c = np.where(best_c > 3.0)[0]
cb_str = str(cross_b[0]) if len(cross_b) else "never"
cc_str = str(cross_c[0]) if len(cross_c) else "never"
print(f"{'First gen > 3.0':<32} {cb_str:>12} {cc_str:>12}")
print(f"{'Generations to bestever':<32} {np.argmax(best_b):>12} {np.argmax(best_c):>12}")
print("-" * 58)
