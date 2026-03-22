"""Lightweight reusable RL helpers for the official RL-oriented games."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


class RunningNormalizer:
    """Online mean/std normalization for vector observations."""

    def __init__(self, epsilon=1e-6):
        self.epsilon = float(epsilon)
        self.count = 0
        self.mean = None
        self.m2 = None

    def update(self, observation):
        obs = np.asarray(observation, dtype=np.float64)
        if self.mean is None:
            self.mean = np.zeros_like(obs)
            self.m2 = np.zeros_like(obs)
        self.count += 1
        delta = obs - self.mean
        self.mean += delta / self.count
        delta2 = obs - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self):
        if self.mean is None:
            return None
        divisor = max(self.count - 1, 1)
        return self.m2 / divisor

    def normalize(self, observation):
        obs = np.asarray(observation, dtype=np.float32)
        if self.mean is None:
            return obs
        std = np.sqrt(np.asarray(self.variance, dtype=np.float32) + self.epsilon)
        return (obs - self.mean.astype(np.float32)) / std

    def copy(self):
        instance = RunningNormalizer(epsilon=self.epsilon)
        instance.count = self.count
        instance.mean = None if self.mean is None else self.mean.copy()
        instance.m2 = None if self.m2 is None else self.m2.copy()
        return instance

    def state_dict(self):
        return {
            "count": self.count,
            "mean": None if self.mean is None else self.mean.tolist(),
            "m2": None if self.m2 is None else self.m2.tolist(),
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_state_dict(cls, state):
        instance = cls(epsilon=state.get("epsilon", 1e-6))
        instance.count = int(state.get("count", 0))
        if state.get("mean") is not None:
            instance.mean = np.asarray(state["mean"], dtype=np.float64)
            instance.m2 = np.asarray(state["m2"], dtype=np.float64)
        return instance


class LinearSoftmaxPolicy:
    """Small dependency-light policy suitable for CEM-style search."""

    def __init__(self, observation_size, action_size, weights=None):
        self.observation_size = int(observation_size)
        self.action_size = int(action_size)
        if weights is None:
            self.weights = np.zeros((self.action_size, self.observation_size + 1), dtype=np.float32)
        else:
            self.weights = np.asarray(weights, dtype=np.float32).reshape(
                self.action_size, self.observation_size + 1
            )

    def logits(self, observation):
        obs = np.asarray(observation, dtype=np.float32)
        with_bias = np.concatenate([obs, np.ones(1, dtype=np.float32)])
        return self.weights @ with_bias

    def act(self, observation):
        return int(np.argmax(self.logits(observation)))

    def clone(self):
        return LinearSoftmaxPolicy(self.observation_size, self.action_size, self.weights.copy())

    @property
    def parameter_count(self):
        return self.weights.size

    def export_checkpoint(self, path, *, metadata=None):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "policy_type": "linear",
            "observation_size": self.observation_size,
            "action_size": self.action_size,
            "weights": self.weights.tolist(),
            "metadata": metadata or {},
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load_checkpoint(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            observation_size=payload["observation_size"],
            action_size=payload["action_size"],
            weights=payload["weights"],
        )


class MLPSoftmaxPolicy:
    """Small tanh MLP policy stored as a flat parameter vector for CEM search."""

    def __init__(self, observation_size, action_size, hidden_size=32, parameters=None):
        self.observation_size = int(observation_size)
        self.action_size = int(action_size)
        self.hidden_size = int(hidden_size)
        expected = (self.hidden_size * self.observation_size) + self.hidden_size
        expected += (self.action_size * self.hidden_size) + self.action_size
        if parameters is None:
            parameters = np.zeros(expected, dtype=np.float32)
        self.parameters = np.asarray(parameters, dtype=np.float32).reshape(expected)
        index = 0
        end = self.hidden_size * self.observation_size
        self.w1 = self.parameters[index : index + end].reshape(self.hidden_size, self.observation_size)
        index += end
        self.b1 = self.parameters[index : index + self.hidden_size]
        index += self.hidden_size
        end = self.action_size * self.hidden_size
        self.w2 = self.parameters[index : index + end].reshape(self.action_size, self.hidden_size)
        index += end
        self.b2 = self.parameters[index : index + self.action_size]

    @property
    def parameter_count(self):
        return self.parameters.size

    def logits(self, observation):
        obs = np.asarray(observation, dtype=np.float32)
        hidden = np.tanh(self.w1 @ obs + self.b1)
        return self.w2 @ hidden + self.b2

    def act(self, observation):
        return int(np.argmax(self.logits(observation)))

    def clone(self):
        return MLPSoftmaxPolicy(
            self.observation_size,
            self.action_size,
            hidden_size=self.hidden_size,
            parameters=self.parameters.copy(),
        )

    def export_checkpoint(self, path, *, metadata=None):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "policy_type": "mlp",
            "observation_size": self.observation_size,
            "action_size": self.action_size,
            "hidden_size": self.hidden_size,
            "parameters": self.parameters.tolist(),
            "metadata": metadata or {},
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load_checkpoint(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            observation_size=payload["observation_size"],
            action_size=payload["action_size"],
            hidden_size=payload["hidden_size"],
            parameters=payload["parameters"],
        )


class RandomDiscretePolicy:
    """Uniform random policy used to fit observation normalizers."""

    def __init__(self, action_size, *, seed=0):
        self.action_size = int(action_size)
        self.rng = np.random.default_rng(seed)

    def act(self, observation):
        return int(self.rng.integers(self.action_size))


def fit_normalizer(env_factory, *, seeds, episodes_per_seed=2, max_steps=128):
    """Collect a fixed observation normalizer from short random rollouts."""

    probe_env = env_factory()
    policy = RandomDiscretePolicy(probe_env.action_space.n)
    normalizer = RunningNormalizer()
    for seed in seeds:
        for rollout_index in range(episodes_per_seed):
            env = env_factory()
            observation, _info = env.reset(seed=(seed * 31) + rollout_index)
            steps = 0
            while True:
                steps += 1
                normalizer.update(observation)
                action = policy.act(observation)
                observation, _reward, terminated, truncated, _info = env.step(action)
                if terminated or truncated or steps >= max_steps:
                    break
    return normalizer


def rollout_episode(env_factory, policy, *, seed, normalizer=None, max_steps=None, update_normalizer=False):
    env = env_factory()
    observation, _info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    truncated = False
    while True:
        steps += 1
        if normalizer is not None:
            if update_normalizer:
                normalizer.update(observation)
            policy_observation = normalizer.normalize(observation)
        else:
            policy_observation = observation
        action = policy.act(policy_observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated or (max_steps is not None and steps >= max_steps):
            return {
                "reward": total_reward,
                "steps": steps,
                "terminated": terminated,
                "truncated": truncated,
                "result": info.get("result"),
                "score_points": info.get("score_points", 0),
            }


def evaluate_policy(env_factory, policy, *, seeds, normalizer=None, update_normalizer=False):
    episodes = [
        rollout_episode(
            env_factory,
            policy,
            seed=seed,
            normalizer=normalizer,
            update_normalizer=update_normalizer,
        )
        for seed in seeds
    ]
    rewards = np.asarray([episode["reward"] for episode in episodes], dtype=np.float32)
    score_points = np.asarray([episode["score_points"] for episode in episodes], dtype=np.float32)
    return {
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)) if len(rewards) else 0.0,
        "mean_score_points": float(np.mean(score_points)) if len(score_points) else 0.0,
    }


@dataclass
class CrossEntropyConfig:
    population_size: int = 32
    elite_fraction: float = 0.25
    iterations: int = 8
    hidden_size: int = 32
    initial_std: float = 0.50
    min_std: float = 0.05
    normalizer_episodes_per_seed: int = 2
    normalizer_max_steps: int = 96


class CrossEntropyTrainer:
    """A simple shared baseline trainer that works on both official RL games."""

    def __init__(self, env_factory, observation_size, action_size, *, config=None):
        self.env_factory = env_factory
        self.observation_size = int(observation_size)
        self.action_size = int(action_size)
        self.config = config or CrossEntropyConfig()

    def _build_policy(self, parameters):
        if self.config.hidden_size <= 0:
            return LinearSoftmaxPolicy(self.observation_size, self.action_size, weights=parameters)
        return MLPSoftmaxPolicy(
            self.observation_size,
            self.action_size,
            hidden_size=self.config.hidden_size,
            parameters=parameters,
        )

    def _score_key(self, metrics):
        return (float(metrics["mean_score_points"]), float(metrics["mean_reward"]))

    def train(self, *, seeds):
        rng = np.random.default_rng(0)
        seeds = list(seeds)
        elite_count = max(1, int(self.config.population_size * self.config.elite_fraction))
        normalizer = fit_normalizer(
            self.env_factory,
            seeds=seeds,
            episodes_per_seed=self.config.normalizer_episodes_per_seed,
            max_steps=self.config.normalizer_max_steps,
        )
        seed_policy = self._build_policy(None)
        mean = np.zeros(seed_policy.parameter_count, dtype=np.float32)
        std = np.full(seed_policy.parameter_count, self.config.initial_std, dtype=np.float32)
        best_policy = seed_policy.clone()
        best_score = (float("-inf"), float("-inf"))

        for _iteration in range(self.config.iterations):
            population = []
            for _member in range(self.config.population_size):
                parameters = mean + (rng.standard_normal(mean.shape).astype(np.float32) * std)
                policy = self._build_policy(parameters)
                metrics = evaluate_policy(self.env_factory, policy, seeds=seeds, normalizer=normalizer)
                population.append((self._score_key(metrics), metrics, policy))

            population.sort(key=lambda item: item[0], reverse=True)
            elites = population[:elite_count]
            elite_parameters = np.stack([policy.parameters for _score, _metrics, policy in elites], axis=0)
            mean = elite_parameters.mean(axis=0)
            std = np.maximum(elite_parameters.std(axis=0), self.config.min_std)
            top_score, _top_metrics, top_policy = population[0]
            if top_score > best_score:
                best_score = top_score
                best_policy = top_policy.clone()

        return best_policy, normalizer.copy()
