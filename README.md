Here is a clean, simplified version of the README. It focuses strictly on what the code does and how to run it.

```markdown
# Doom Hybrid Agent (IMPALA + LSTM + Attention)

A high-performance Reinforcement Learning agent designed to solve the **VizDoom Deadly Corridor** scenario.

This implementation uses **Stable-Baselines3 (Contrib)** to train a `RecurrentPPO` agent. It features a custom "Hybrid" architecture that combines:
1.  **IMPALA ResNet** for deep visual feature extraction.
2.  **Spatial Attention** to automatically focus on enemies.
3.  **LSTM** memory to handle partial observability (the corridor layout).

## 📋 Requirements

You need Python 3.8+ and a system with a CUDA-capable GPU.

```bash
# Install core dependencies
pip install gymnasium "gymnasium[box2d]" stable-baselines3 sb3-contrib shimmy

# Install VizDoom (Linux/WSL recommended)
pip install vizdoom

```

## 🧠 Architecture Overview

* **Policy:** `RecurrentPPO` (PPO with LSTM support).
* **Feature Extractor:** Custom `HybridCNN` with Orthogonal Initialization.
* **Reward Shaping:**
* **Nuclear:** Bonuses for kills.
* **Efficiency:** Penalties for wasted ammo.
* **Survival:** Small bonuses for staying alive and moving.


* **Normalization:** Uses `VecNormalize` to automatically scale rewards and observations.

## 🚀 How to Run

1. Make sure you have the `deadly_corridor.cfg` and `deadly_corridor.wad` files in the same directory as the script.
2. Run the training script:

```bash
python train_agent.py

```

The script will automatically create the following directories:

* `doom_hybrid_logs/`: Tensorboard training logs.
* `doom_hybrid_models/`: Saved models (`.zip`) and normalization stats (`.pkl`).

## 📈 Curriculum Training

The training process uses a curriculum to gradually increase difficulty:

1. **Easy** (Difficulty 2): 300k steps
2. **Medium** (Difficulty 3): 500k steps
3. **Hard** (Difficulty 4): 800k steps
4. **Nightmare** (Difficulty 5): 1.2M steps

## 📊 Monitoring

To view training metrics (kills, health, survival time) in real-time:

```bash
tensorboard --logdir=doom_hybrid_logs
```

```
