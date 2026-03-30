# MaCE-NCA Reservoir

**Mass-Conserving Neural Cellular Automata for Reservoir Computing**

This project extends [Sidney et al.'s critical-nca-reservoir](https://github.com/bioAI-Oslo/critical-nca-reservoir) by introducing a **MaCE-style mass-conserving NCA variant**  a softmax-weighted mass redistribution mechanism that preserves total cell activation across timesteps. The conservation constraint acts as an inductive bias toward the edge of chaos, achieving faster criticality convergence and improved self-organised criticality (SOC) signatures.

---

## Key Results

All results from 3 independent seeds (42, 43, 44), W=1000, T=1000, 500 generations.

| Metric | Baseline NCA | Mass-Conserving NCA | Δ |
|--------|-------------|---------------------|---|
| Mean training fitness | 3.82 | **4.10** | +7.3% |
| Mean GOF passes (out of 6) | 2.3 | **5.0** | +117% |
| Perfect criticality (6/6 GOF) | 0/3 seeds | **2/3 seeds** | — |
| Training time | 47.2h | **37.1h** | 1.27× faster |
| 5-bit memory accuracy | 1.0 | 1.0 | Tie |
| MNIST accuracy | **93.81%** | 93.54% | −0.27 pp |
| CartPole mean reward | 88.6 | **120.6** | +36% |

Conservation produces stronger criticality and faster convergence. Downstream task performance is at parity or better — the conserving variant leads on temporal control (CartPole) after best-GOF checkpoint selection.

---

## Mass-Conservation Mechanism

At each timestep, cell activations are redistributed via a two-step donate-then-collect operation. Each cell donates its activation weighted by softmax affinity to its neighbours, then collects from neighbours proportionally. Total mass is conserved across the lattice at every step.

---

## Project Structure

```
critical_nca.py                       # NCA model definition
train_nca.py                          # Baseline NCA training (CMA-ES)
train_nca_conserve.py                 # Mass-conserving NCA training
evaluate_criticality.py               # SOC evaluation & apply_conservation()
test_nca.py                           # Checkpoint evaluation
helper.py / utils.py                  # Shared utilities
reservoir_mnist_make_dataset.py       # Generate NCA reservoir features for MNIST
reservoir_mnist_classify.py           # LinearSVC readout on NCA features
reservoir_X-bit_make_dataset.py       # 5-bit memory dataset preparation
ReCA_X-bit_memory_NCA.py              # 5-bit memory task (NCA variant)
reservoir_cartpole_train_qlearning.py # CartPole Q-learning with NCA reservoir
reservoir_cartpole_evaluate_rl.py     # CartPole evaluation (100 episodes)
batch_gof_eval.py                     # GOF sweep across checkpoints
summarize_gof.py                      # Aggregate GOF results
checkpoint/                           # Pre-trained NCA weights
```

---

## Reproduction

**Train mass-conserving NCA (author scale):**
```bash
python train_nca_conserve.py --maxgen 500 --popsize 96 --threads 30 \
  --width 1000 --timesteps 1000 --seed 42
```

**Evaluate downstream tasks:**
```bash
# 5-bit memory
python reservoir_X-bit_make_dataset.py --logdir <logdir> --ckpt <ckpt>
python ReCA_X-bit_memory_NCA.py

# MNIST classification
python reservoir_mnist_make_dataset.py --logdir <logdir>
python reservoir_mnist_classify.py

# CartPole reinforcement learning
python reservoir_cartpole_train_qlearning.py --logdir <logdir> --ckpt <ckpt>
python reservoir_cartpole_evaluate_rl.py --logdir <logdir> --model <model.pkl>
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

The mass-conservation mechanism is inspired by MaCE (Mass-Conserving Lenia) by Etienne Guichard et al. MaCE applies conservation laws to 2D continuous Lenia systems; here we adapt the principle to 1D discrete NCA for reservoir computing.
