"""
Compose SOC comparison figure panels: Baseline vs Conserving NCA.

Outputs (in results/):
  soc_comparison_avalanches.png  — 3x2 grid: (size/duration/lifetime) x (baseline/conserving)
  soc_comparison_ca_state.png    — side-by-side CA spacetime diagrams
  soc_comparison_full.png        — combined figure: CA states + all 3 avalanche distributions
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import os
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
BASELINE_DIR    = "baseline_run_2/logs/train_nca/20260222-130628"
CONSERVING_DIR  = "conserving_run_2/logs/train_nca_conserve/20260222-130839"
BASELINE_GEN    = "000483"
CONSERVING_GEN  = "000312"
OUT_DIR         = "results"
os.makedirs(OUT_DIR, exist_ok=True)

def img(directory, gen, suffix):
    path = os.path.join(directory, f"ca_{gen}{suffix}.png")
    return mpimg.imread(path)

def add_img(ax, directory, gen, suffix, title=None):
    ax.imshow(img(directory, gen, suffix))
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=10, pad=4)

BASELINE_LABEL    = f"Baseline NCA\n(gen {int(BASELINE_GEN)}, fitness=3.5675)"
CONSERVING_LABEL  = f"Conserving NCA\n(gen {int(CONSERVING_GEN)}, fitness=3.9958)"

ROW_LABELS = [
    "Avalanche size\n$P(s) \\propto s^{-\\alpha}$",
    "Avalanche duration\n$P(d) \\propto d^{-\\alpha}$",
    "Avalanche lifetime\n$P(t) \\propto t^{-\\alpha}$",
]
SUFFIXES = ["_s_0", "_d_0", "_t_0"]

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: 3×2 avalanche distribution grid
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 2, figsize=(10, 13))
fig.suptitle(
    'Avalanche Statistics: Baseline vs Mass-Conserving NCA\n'
    '(width=100, timesteps=100, channel 0)',
    fontsize=12, y=0.98
)

for row, (suffix, row_label) in enumerate(zip(SUFFIXES, ROW_LABELS)):
    for col, (directory, gen, col_label) in enumerate([
        (BASELINE_DIR,   BASELINE_GEN,   BASELINE_LABEL),
        (CONSERVING_DIR, CONSERVING_GEN, CONSERVING_LABEL),
    ]):
        ax = axes[row, col]
        add_img(ax, directory, gen, suffix)
        if row == 0:
            ax.set_title(col_label, fontsize=10, pad=6)
        if col == 0:
            ax.set_ylabel(row_label, fontsize=9, labelpad=6)
            ax.yaxis.set_label_position('left')

plt.tight_layout(rect=[0, 0, 1, 0.97])
out1 = os.path.join(OUT_DIR, "soc_comparison_avalanches.png")
plt.savefig(out1, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out1}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: CA spacetime diagrams side-by-side
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('CA Spacetime Diagrams at Best Checkpoint', fontsize=12)

for ax, directory, gen, label in [
    (axes[0], BASELINE_DIR,   BASELINE_GEN,   BASELINE_LABEL),
    (axes[1], CONSERVING_DIR, CONSERVING_GEN, CONSERVING_LABEL),
]:
    add_img(ax, directory, gen, "", title=label)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "soc_comparison_ca_state.png")
plt.savefig(out2, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out2}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Full panel — CA states (top) + 3 avalanche rows (below)
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(11, 18))
fig.suptitle(
    'SOC Characterization: Baseline vs Mass-Conserving NCA\n'
    '(width=100, timesteps=100, seed=671052)',
    fontsize=13, y=0.99
)

gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.08, wspace=0.04,
                       height_ratios=[1.2, 1, 1, 1])

# Row 0: CA states
for col, (directory, gen, label) in enumerate([
    (BASELINE_DIR,   BASELINE_GEN,   BASELINE_LABEL),
    (CONSERVING_DIR, CONSERVING_GEN, CONSERVING_LABEL),
]):
    ax = fig.add_subplot(gs[0, col])
    add_img(ax, directory, gen, "", title=label)

# Rows 1-3: avalanche distributions
for row, (suffix, row_label) in enumerate(zip(SUFFIXES, ROW_LABELS)):
    for col, (directory, gen) in enumerate([
        (BASELINE_DIR,   BASELINE_GEN),
        (CONSERVING_DIR, CONSERVING_GEN),
    ]):
        ax = fig.add_subplot(gs[row + 1, col])
        add_img(ax, directory, gen, suffix)
        if col == 0:
            ax.set_ylabel(row_label, fontsize=9)
            ax.yaxis.set_label_position('left')

# Row labels on left margin
row_labels_y = [0.76, 0.52, 0.27]
row_header   = ['CA State', 'Size $P(s)$', 'Duration $P(d)$', 'Lifetime $P(t)$']
for i, (ypos, label) in enumerate(zip([0.91, 0.74, 0.50, 0.26], row_header)):
    fig.text(0.01, ypos, label, va='center', ha='left', fontsize=9,
             rotation=90, color='#444444')

out3 = os.path.join(OUT_DIR, "soc_comparison_full.png")
plt.savefig(out3, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out3}")

# ══════════════════════════════════════════════════════════════════════════════
# Metrics table (text)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("── SOC Metrics Table (from test_nca.py, 3-repeat mean) ─────────────────")
print(f"{'Metric':<30} {'Baseline':>12} {'Conserving':>12} {'Δ':>10}")
print("-" * 68)

metrics = [
    ("Fitness",              3.5675,  3.9958),
    ("norm_ksdist_res",      0.9422,  0.9498),
    ("norm_coef_res",        1.4463,  1.5531),
    ("norm_unique_states",   1.0000,  1.0000),
    ("norm_avalanche_pdf",   0.7291,  0.7621),
    ("norm_linscore_res",    0.6669,  0.9120),
    ("norm_R_res",           0.5060,  0.4997),
]
for name, b, c in metrics:
    delta = c - b
    sign  = "+" if delta >= 0 else ""
    print(f"{name:<30} {b:>12.4f} {c:>12.4f} {sign}{delta:>9.4f}")
print("-" * 68)
print("norm_linscore_res measures log-log linearity of the avalanche distribution")
print("(higher = closer to ideal power-law = better SOC)")
