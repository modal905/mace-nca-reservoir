# MaCE-NCA Reservoir

**Mass-Conserving Neural Cellular Automata for Reservoir Computing**

This project extends [Sidney et al.'s critical-nca-reservoir](https://github.com/bioAI-Oslo/critical-nca-reservoir) by introducing a **MaCE-style mass-conserving NCA variant**  a softmax-weighted mass redistribution mechanism that preserves total cell activation across timesteps. The conservation constraint acts as an inductive bias toward the edge of chaos, achieving faster criticality convergence and improved self-organised criticality (SOC) signatures.

---

## Key Results

| Metric | Baseline NCA | Mass-Conserving NCA |
|--------|-------------|---------------------|
| Training fitness (best-ever) | 3.5675 | **3.9958** |
| First generation to cross fitness 3.0 | ~gen 470 | **gen 53** (8.9x faster) |
| SOC score (`norm_linscore_res`) | 0.667 | **0.912** (+36.7%) |
| MNIST reservoir accuracy | 93.47%* | **93.72%** |

*Baseline LinearSVC not fully converged (lower bound).

---

## Mass-Conservation Mechanism

At each timestep, cell activations are redistributed via a two-step donate-then-collect operation. Each cell donates its activation weighted by softmax affinity to its neighbours, then collects from neighbours proportionally. Total mass is conserved across the lattice at every step.

---

## Project Structure

```
critical_nca.py                  # NCA model definition
train_nca.py                     # Baseline NCA training (CMA-ES)
train_nca_conserve.py            # Mass-conserving NCA training
evaluate_criticality.py          # SOC evaluation (avalanche statistics)
test_nca.py                      # Checkpoint evaluation
reservoir_mnist_make_dataset.py  # Generate NCA reservoir features for MNIST
reservoir_mnist_classify.py      # LinearSVC readout on NCA features
make_animation.py                # Generate CA dynamics animations
plot_comparison.py               # Training curve comparison figure
plot_soc_comparison.py           # SOC / avalanche comparison figure
reproduction_guide.md            # Step-by-step reproduction instructions
results/                         # Output figures
```

---

## Reproduction

See [reproduction_guide.md](reproduction_guide.md) for full step-by-step instructions.

**Train mass-conserving NCA:**
```bash
python train_nca_conserve.py --ca_width 100 --ca_timesteps 100 --num_generations 500
```

**Generate animations:**
```bash
python make_animation.py \
  --baseline_logdir   <path_to_baseline_logdir> \
  --baseline_ckpt     <best_checkpoint.ckpt> \
  --conserving_logdir <path_to_conserving_logdir> \
  --conserving_ckpt   <best_checkpoint.ckpt> \
  --width 100 --timesteps 100 --fps 10 --scale 6
```

---

## Dependencies

```
tensorflow>=2.7
numpy
scipy
scikit-learn
matplotlib
imageio
Pillow
cma
powerlaw
```

---

## Credits

Built on top of [bioAI-Oslo/critical-nca-reservoir](https://github.com/bioAI-Oslo/critical-nca-reservoir) by Sidney Pontes-Filho et al.

The mass-conservation mechanism is inspired by [MaCE](https://arxiv.org/html/2507.12306v1) (Mass-Conserving Lenia) by Etienne Musa et al. ([github.com/frotaur/MaceLenia](https://github.com/frotaur/MaceLenia)). MaCE applies conservation laws to 2D continuous Lenia systems; here we adapt the principle to 1D discrete NCA for reservoir computing.
