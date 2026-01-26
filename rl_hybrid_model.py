import os
import gymnasium as gym
from gymnasium import spaces
import vizdoom as vzd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO
from typing import Callable

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONF_NAME = os.path.join(CURRENT_DIR, "deadly_corridor.cfg")
SCENARIO_NAME = os.path.join(CURRENT_DIR, "deadly_corridor.wad")
LOG_DIR = os.path.join(CURRENT_DIR, "doom_hybrid_logs")
MODEL_DIR = os.path.join(CURRENT_DIR, "doom_hybrid_models")

HYPERPARAMS = {
    "n_steps": 1024,  # can increase or decrease depending of specs
    "batch_size": 128,
    "n_epochs": 8,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "vf_coef": 1.0,
    "max_grad_norm": 0.5,
    "normalize_advantage": True,
}

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def cosine_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return initial_value * 0.5 * (1 + np.cos(np.pi * (1 - progress_remaining)))

    return func


# Split IMPALA architecture
class ImpalaResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)
        nn.init.orthogonal_(self.conv1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.conv2.weight, gain=np.sqrt(2))

    def forward(self, x):
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return x + out


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        nn.init.orthogonal_(self.conv.weight)

    def forward(self, x):
        attention = torch.sigmoid(self.conv(x))
        return x * attention


class HybridCNN(BaseFeaturesExtractor):
    """
    - IMPALA residual blocks (resnet)
    - Spatial attention (focus on enemies)
    - Orthogonal init (better convergence/speed up the training)
    """

    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        depths = [32, 64, 64]
        layers = []
        curr_channel = n_input_channels

        for depth in depths:
            conv = nn.Conv2d(curr_channel, depth, 3, stride=1, padding=1)
            nn.init.orthogonal_(conv.weight, gain=np.sqrt(2))
            layers.append(conv)
            layers.append(nn.MaxPool2d(3, stride=2, padding=1))
            layers.append(ImpalaResidualBlock(depth))
            layers.append(ImpalaResidualBlock(depth))
            layers.append(SpatialAttention(depth))
            curr_channel = depth

        self.trunk = nn.Sequential(*layers)

        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, 84, 84)
            out = self.trunk(sample)
            n_flatten = out.flatten(start_dim=1).shape[1]

        self.linear = nn.Sequential(
            nn.Flatten(), nn.ReLU(), nn.Linear(n_flatten, features_dim), nn.ReLU()
        )

    def forward(self, observations):
        return self.linear(self.trunk(observations))


# Reward Shaping
class DoomHybridEnv(gym.Env):
    """
    Hybrid approach:
    - Balanced reward shaping
    - VecNormalize will still help, but rewards guide learning
    """

    def __init__(self, render=False, difficulty=1):
        super().__init__()
        self.game = vzd.DoomGame()
        self.game.load_config(str(CONF_NAME))
        self.game.set_doom_scenario_path(str(SCENARIO_NAME))
        self.game.set_doom_skill(difficulty)
        self.game.set_window_visible(render)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        self.game.set_screen_format(vzd.ScreenFormat.GRAY8)
        self.game.init()

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(1, 84, 84), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(7)
        self.difficulty = difficulty
        self.state_history = {"health": 100, "kills": 0, "ammo": 0}
        self.steps_alive = 0
        self.last_kill_step = 0

    def step(self, action):
        actions = np.identity(7, dtype=np.uint8)
        reward = self.game.make_action(actions[action], 4)
        done = self.game.is_episode_finished()
        state = np.zeros((1, 84, 84), dtype=np.uint8)
        info = {}

        if not done:
            gs = self.game.get_state()
            state[0] = cv2.resize(
                gs.screen_buffer, (84, 84), interpolation=cv2.INTER_NEAREST
            )

            gv = gs.game_variables
            curr_health = gv[0]
            curr_kills = gv[1] if len(gv) > 1 else 0
            curr_ammo = gv[2] if len(gv) > 2 else 0

            kill_delta = curr_kills - self.state_history["kills"]
            health_delta = curr_health - self.state_history["health"]
            ammo_delta = curr_ammo - self.state_history["ammo"]

            # Kill rewards
            if kill_delta > 0:
                base_reward = 10.0 * (1.0 + 0.2 * self.difficulty)

                # Combo bonus
                steps_since_kill = self.steps_alive - self.last_kill_step
                if steps_since_kill < 50:
                    base_reward += 5.0

                reward += base_reward * kill_delta
                self.last_kill_step = self.steps_alive

            # Health management
            if health_delta < 0:
                damage = abs(health_delta)
                if damage >= 50:
                    reward -= damage * 0.2
                elif damage >= 20:
                    reward -= damage * 0.1
                else:
                    reward -= damage * 0.03

            # Ammo efficiency
            if ammo_delta < 0:
                if kill_delta > 0:
                    accuracy = kill_delta / abs(ammo_delta)
                    if accuracy >= 0.5:
                        reward += 1.0
                else:
                    reward -= 0.2

            # Survival bonus
            reward += 0.01
            self.steps_alive += 1

            # Health bonuses
            if curr_health >= 90:
                reward += 0.05
            elif curr_health < 30:
                reward -= 0.05

            self.state_history.update(
                {"health": curr_health, "kills": curr_kills, "ammo": curr_ammo}
            )
            info["kills"] = curr_kills
            info["health"] = curr_health
            info["steps_alive"] = self.steps_alive

        else:
            if self.game.is_player_dead():
                # Death penalty depending on performance
                penalty = -50.0
                mitigation = min(self.state_history["kills"] * 5, 30)
                reward += penalty + mitigation
                info["health"] = 0
            else:
                # Victory bonus for performance
                base = 100.0
                mult = (
                    1.0 + (self.difficulty * 0.3) + (self.state_history["kills"] * 0.1)
                )
                reward += base * mult
                info["health"] = self.state_history["health"]

            info["kills"] = self.state_history["kills"]
            info["steps_alive"] = self.steps_alive

        return state, reward, done, False, info

    def reset(self, seed=None, options=None):
        if seed:
            self.game.set_seed(seed)
        self.game.new_episode()
        self.state_history = {"health": 100, "kills": 0, "ammo": 0}
        self.steps_alive = 0
        self.last_kill_step = 0
        state = np.zeros((1, 84, 84), dtype=np.uint8)
        state[0] = cv2.resize(
            self.game.get_state().screen_buffer,
            (84, 84),
            interpolation=cv2.INTER_NEAREST,
        )
        return state, {}

    def close(self):
        self.game.close()


# Metrics Callback
class MetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_kills = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        for idx, done in enumerate(dones):
            if done:
                kills = infos[idx].get("kills", 0)
                health = infos[idx].get("health", 0)
                steps = infos[idx].get("steps_alive", 0)

                self.logger.record("game/kills", kills)
                self.logger.record("game/final_health", health)
                self.logger.record("game/survival_steps", steps)

                self.episode_kills.append(kills)
                self.episode_lengths.append(steps)

                if len(self.episode_kills) >= 10:
                    self.logger.record(
                        "game/avg_kills_10ep", np.mean(self.episode_kills[-10:])
                    )
                    self.logger.record(
                        "game/avg_survival_10ep", np.mean(self.episode_lengths[-10:])
                    )

        return True


# training loop
def make_env(rank, difficulty):
    def _init():
        env = DoomHybridEnv(difficulty=difficulty)
        env = Monitor(env, os.path.join(LOG_DIR, str(rank)))
        return env

    return _init


if __name__ == "__main__":
    CURRICULUM = [
        # (1, 200_000, "gen1_tutorial"),
        (2, 300_000, "gen2_easy"),
        (3, 500_000, "gen3_medium"),
        (4, 800_000, "gen4_hard"),
        (5, 1_200_000, "gen5_nightmare"),
    ]

    num_cpu = 4
    current_model_path = os.path.join(MODEL_DIR, "gen1_tutorial.zip")
    vec_norm_path = os.path.join(MODEL_DIR, "gen1_tutorial_vecnorm.pkl")

    base_lr = 3e-4
    base_clip = 0.2

    print("Initilization")
    print("=" * 70)

    for difficulty, steps, gen_name in CURRICULUM:
        print(f"\n[PHASE] {gen_name} | Diff {difficulty} | Steps {steps:,}")

        # Create env with VecNormalize
        env = SubprocVecEnv([make_env(i, difficulty) for i in range(num_cpu)])
        env = VecFrameStack(env, n_stack=4)

        if vec_norm_path:
            env = VecNormalize.load(vec_norm_path, env)
            env.training = True
        else:
            env = VecNormalize(
                env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0
            )

        # Model PPO loading/creation
        if current_model_path:

            print(f"loading {current_model_path}")
            model = RecurrentPPO.load(current_model_path, env=env)
            model.learning_rate = cosine_schedule(base_lr * 0.7)
            model.clip_range = linear_schedule(base_clip * 0.7)
        else:
            print("creating new model")
            model = RecurrentPPO(
                "CnnLstmPolicy",
                env,
                verbose=1,
                tensorboard_log=LOG_DIR,
                learning_rate=cosine_schedule(base_lr),
                clip_range=linear_schedule(base_clip),
                ent_coef=0.01,
                **HYPERPARAMS,
                policy_kwargs=dict(
                    features_extractor_class=HybridCNN,
                    features_extractor_kwargs=dict(features_dim=512),
                    ortho_init=True,
                    net_arch=dict(pi=[256], vf=[256]),
                    enable_critic_lstm=True,
                ),
            )

        print(f"Starting Training")
        model.learn(
            total_timesteps=steps,
            callback=MetricsCallback(),
            progress_bar=True,
            tb_log_name=gen_name,
            reset_num_timesteps=False,
        )

        # Save model and normalization stats
        save_path = os.path.join(MODEL_DIR, gen_name)
        model.save(save_path)

        norm_path = os.path.join(MODEL_DIR, f"{gen_name}_vecnorm.pkl")
        env.save(norm_path)

        print(f"saving {save_path}")

        current_model_path = save_path
        vec_norm_path = norm_path
        env.close()

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Model:{current_model_path}")
    print(f"Vecpath: {vec_norm_path}")
    print("=" * 70)
