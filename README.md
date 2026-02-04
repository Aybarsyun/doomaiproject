Installation
Prerequisites
Python 3.8+

Linux/WSL (Required for VizDoom)

CUDA-capable GPU (Strongly Recommended)

Setup
Bash
# 1. Clone the repository
git clone https://github.com/your-username/doom-transformer-rl.git
cd doom-transformer-rl

# 2. Install dependencies (Virtual Environment recommended)
pip install gymnasium[box2d] stable-baselines3 torch torchvision opencv-python transformers shimmy

# 3. Install VizDoom (System dependencies may be required)
pip install vizdoom
