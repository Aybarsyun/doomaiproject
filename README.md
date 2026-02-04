
# Doom Hybrid Agent (IMPALA + LSTM + Attention)

This repository contains a high-performance Reinforcement Learning agent designed to solve the **VizDoom Deadly Corridor** scenario.

The agent uses a **Hybrid Architecture** combining:
1.  **IMPALA-style ResNet** for deep visual feature extraction.
2.  **Spatial Attention** to prioritize enemy locations.
3.  **LSTM (RecurrentPPO)** to handle memory and partial observability.
4.  **Nuclear Reward Shaping** to aggressively encourage combat.

## 📋 Requirements

**System:**
* Python 3.8+
* Linux/WSL (Required for VizDoom)
* CUDA-capable GPU

**Dependencies:**
```bash
pip install gymnasium "gymnasium[box2d]" stable-baselines3 sb3-contrib vizdoom

```

## 🧠 Architecture Details

The code implements a custom feature extractor (`HybridCNN`) and environment wrapper (`DoomHybridEnv`):

* **Visual Backbone:** 3-layer CNN with Orthogonal Initialization (`gain=sqrt(2)`).
* **Residual Blocks:** IMPALA-style residual connections to prevent vanishing gradients in deep layers.
* **Attention Mechanism:** A `SpatialAttention` module that applies a sigmoid gate to the feature map, allowing the agent to focus on specific regions (e.g., enemies at the end of the corridor).
* **Memory:** `RecurrentPPO` (LSTM) maintains a hidden state to remember map layout and enemy positions over time.

## ⚔️ Reward Shaping Logic

The environment uses a custom "Nuclear" reward function to enforce aggressive behavior:

* **Kill Bonus:** Base +10.0, scaling with difficulty.
* **Combo System:** +5.0 bonus for chaining kills quickly (<50 steps).
* **Ammo Efficiency:** +1.0 for high accuracy shots, penalty for wasting ammo.
* **Health Management:** Non-linear penalty for taking damage (taking 50+ dmg hurts significantly more than taking 5 dmg).
* **Victory Bonus:** Massive reward scaled by difficulty and kill count.

## 🚀 Training Curriculum

The agent trains through a 4-phase curriculum to gradually master the game:

1. **Easy** (Difficulty 2): 300,000 steps
2. **Medium** (Difficulty 3): 500,000 steps
3. **Hard** (Difficulty 4): 800,000 steps
4. **Nightmare** (Difficulty 5): 1,200,000 steps

**Total Training:** ~2.8 Million Steps

## 🏃 How to Run

1. Ensure `deadly_corridor.cfg` and `deadly_corridor.wad` are in the project root.
2. Run the script:

```bash
python train.py

```

The script will generate:

* `doom_hybrid_logs/`: Tensorboard metrics.
* `doom_hybrid_models/`: Saved agent checkpoints (`.zip`) and normalization stats (`.pkl`).

## 📊 Monitoring

Track Kills, Health, and Survival Time in real-time:

```bash
tensorboard --logdir=doom_hybrid_logs

```

```

```
