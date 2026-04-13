# GAINS: Gated Arbitration of Individual and Social Learning Signals

Implementation of **GAINS**, a meta-reinforcement learning architecture for adaptive social learning, 
as described in the paper:

> **GAINS: A Meta-Reinforcement Learning Model for Gated Arbitration of Individual and Social Learning Signals**  
> Ashwin Moongathottathil James, Eitan Hemed, Julia M. Rodriguez Buritica, Marcel Brass, Verena V. Hafner
---

## Overview

GAINS learns to adaptively arbitrate between self-experienced and socially observed evidence 
in a two-choice visual bandit task. It supports three learning regimes:

- **IL (Individual Learning)** — relies solely on self-experience
- **AO (Action-Only)** — augments self-experience with observed demonstrator actions
- **OL (Observational Learning)** — augments self-experience with demonstrator actions *and* outcomes

Key features:
- Causal episodic memory with condition-specific retrieval via multi-head attention
- Separate gating networks (`g_AO`, `g_OL`) for adaptive social arbitration
- Hybrid training: PPO-based RL + supervised belief learning
- Memory decay parameter (`λ_mem`) to study developmental differences in social learning

---

## Repository Structure

| File | Description |
|------|-------------|
| `visual_bandit_env2.py` | Two-choice visual bandit environment |
| `var_bandit_learner.py` | GAINS model architecture |
| `var_bandit_learner2.py` | Extended variant of GAINS |
| `bandit_train.py` | Training script (single run) |
| `bandit_train_batch.py` | Batch training across multiple conditions |

---

## Installation

```bash
git clone https://github.com/mattapattu/VarMetaLearning.git
cd VarMetaLearning
pip install -r requirements.txt
```

---

## Usage

**Batch training across demonstrator regimes and difficulty levels:**
```bash
python bandit_train_batch.py
```

---

## Key Results

| Demonstrator | IL | AO | OL |
|---|---|---|---|
| Expert | 82.25% | 90.27% | 89.59% |
| Learning | 80.59% | 81.00% | 84.03% |
| Unreliable | 81.33% | 79.84% | 86.87% |

- OL consistently outperforms IL and AO across all conditions
- Adaptive gating outperforms fixed-gate baselines, especially under unreliable demonstrators
- Reduced memory access increases the relative benefit of social information

---

## Citation

```bibtex
@inproceedings{james2026gains,
  title={GAINS: A Meta-Reinforcement Learning Model for Gated Arbitration of Individual and Social Learning Signals},
  author={James, Ashwin Moongathottathil and Hemed, Eitan and Rodriguez Buritica, Julia M. and Brass, Marcel and Hafner, Verena V.},
  booktitle={ICDL 2026},
  year={2026}
}
```

---

## Acknowledgements

This study was funded by the Deutsche Forschungsgemeinschaft (DFG) under Germany's Excellence Strategy – EXC 2002/1 "Science of Intelligence" – project number 390523135.
