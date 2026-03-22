"""Noisy Moon Lander Lite core simulator and local Gym-like environment."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .common import clamp, make_step_rng, system_seed
from .compat import BaseEnv, spaces


@dataclass(frozen=True)
class NoisyMoonLanderConfig:
    world_width: float = 100.0
    world_height: float = 80.0
    pad_width: float = 14.0
    initial_fuel: float = 100.0
    max_steps: int = 180


class NoisyMoonLanderCore:
    """State-based simulator with deterministic stochasticity from hidden seeds."""

    ACTIONS = {
        "idle": "Hold current trajectory",
        "main": "Fire the main thruster",
        "left": "Fire the left correction thruster",
        "right": "Fire the right correction thruster",
        "main_left": "Main thrust with left correction",
        "main_right": "Main thrust with right correction",
        "stabilize": "Spend fuel to damp tilt and drift",
    }
    OBSERVATION_SCHEMA = (
        "altitude",
        "vx",
        "vy",
        "dx_pad",
        "fuel_fraction",
        "sin_theta",
        "cos_theta",
        "omega",
        "leg_contact_left",
        "leg_contact_right",
    )

    def __init__(self, seed=None, config=None):
        self.default_seed = seed
        self.config = config or NoisyMoonLanderConfig()

    def get_actions(self):
        return dict(self.ACTIONS)

    def reset(self, seed=None):
        episode_seed = self.default_seed if seed is None else seed
        if episode_seed is None:
            episode_seed = system_seed()

        rng = make_step_rng(episode_seed, 0, "lander_reset")
        params = {
            "gravity": rng.uniform(0.045, 0.060),
            "main_power": rng.uniform(0.095, 0.125),
            "side_power": rng.uniform(0.020, 0.035),
            "torque_power": rng.uniform(0.010, 0.020),
            "wind_alpha": rng.uniform(0.88, 0.96),
            "wind_sigma": rng.uniform(0.0015, 0.0060),
            "command_smoothing": rng.uniform(0.30, 0.55),
            "fuel_burn_main": rng.uniform(1.20, 1.60),
            "fuel_burn_side": rng.uniform(0.35, 0.55),
            "fuel_burn_stabilize": rng.uniform(0.25, 0.45),
        }
        pad_center = rng.uniform(18.0, self.config.world_width - 18.0)
        x = rng.uniform(12.0, self.config.world_width - 12.0)
        if abs(x - pad_center) < 10.0:
            x = clamp(pad_center + (12.0 if x <= pad_center else -12.0), 8.0, self.config.world_width - 8.0)

        return {
            "seed": int(episode_seed),
            "step": 0,
            "x": x,
            "y": rng.uniform(55.0, 70.0),
            "vx": rng.uniform(-0.8, 0.8),
            "vy": rng.uniform(-0.4, 0.2),
            "theta": rng.uniform(-0.15, 0.15),
            "omega": rng.uniform(-0.03, 0.03),
            "fuel": self.config.initial_fuel,
            "wind_x": 0.0,
            "pad_center": pad_center,
            "pad_width": self.config.pad_width,
            "done": False,
            "result": None,
            "applied_main": 0.0,
            "applied_side": 0.0,
            "params": params,
        }

    def normalize_state(self, state):
        normalized = dict(state)
        normalized.setdefault("seed", self.default_seed or 0)
        normalized.setdefault("step", 0)
        normalized.setdefault("done", False)
        normalized.setdefault("result", None)
        normalized.setdefault("fuel", self.config.initial_fuel)
        normalized.setdefault("wind_x", 0.0)
        normalized.setdefault("applied_main", 0.0)
        normalized.setdefault("applied_side", 0.0)
        normalized.setdefault("pad_width", self.config.pad_width)
        normalized.setdefault("params", {})
        return normalized

    def is_valid_action(self, action, state):
        return action in self.ACTIONS and not state.get("done", False)

    def _decode_action(self, action):
        mapping = {
            "idle": (0.0, 0.0, False),
            "main": (1.0, 0.0, False),
            "left": (0.0, -1.0, False),
            "right": (0.0, 1.0, False),
            "main_left": (1.0, -1.0, False),
            "main_right": (1.0, 1.0, False),
            "stabilize": (0.0, 0.0, True),
        }
        return mapping[action]

    def _leg_contacts(self, state):
        pad_x1 = state["pad_center"] - state["pad_width"] / 2.0
        pad_x2 = state["pad_center"] + state["pad_width"] / 2.0
        left_x = state["x"] - 1.8
        right_x = state["x"] + 1.8
        near_ground = state["y"] <= 0.6
        return (
            int(near_ground and pad_x1 <= left_x <= pad_x2),
            int(near_ground and pad_x1 <= right_x <= pad_x2),
        )

    def _is_safe_landing(self, state):
        pad_x1 = state["pad_center"] - state["pad_width"] / 2.0
        pad_x2 = state["pad_center"] + state["pad_width"] / 2.0
        left_contact, right_contact = self._leg_contacts(state)
        return (
            pad_x1 <= state["x"] <= pad_x2
            and abs(state["vx"]) <= 0.95
            and abs(state["vy"]) <= 1.10
            and abs(state["theta"]) <= 0.18
            and (left_contact or right_contact)
        )

    def observe(self, state):
        normalized = self.normalize_state(state)
        rng = make_step_rng(normalized["seed"], normalized["step"], "lander_obs")
        left_contact, right_contact = self._leg_contacts(normalized)
        observation = [
            clamp(max(normalized["y"], 0.0) + rng.gauss(0.0, 0.35), 0.0, self.config.world_height),
            normalized["vx"] + rng.gauss(0.0, 0.12),
            normalized["vy"] + rng.gauss(0.0, 0.12),
            (normalized["pad_center"] - normalized["x"]) + rng.gauss(0.0, 0.30),
            clamp((normalized["fuel"] / self.config.initial_fuel) + rng.gauss(0.0, 0.01), 0.0, 1.0),
            clamp(math.sin(normalized["theta"]) + rng.gauss(0.0, 0.015), -1.0, 1.0),
            clamp(math.cos(normalized["theta"]) + rng.gauss(0.0, 0.015), -1.0, 1.0),
            normalized["omega"] + rng.gauss(0.0, 0.03),
            float(left_contact),
            float(right_contact),
        ]
        return [round(value, 6) for value in observation]

    def step(self, state, action):
        normalized = self.normalize_state(state)
        if not self.is_valid_action(action, normalized):
            return normalized, -1.0, False, {"error": "Invalid action"}

        params = normalized["params"]
        rng = make_step_rng(normalized["seed"], normalized["step"], "lander_step")
        desired_main, desired_side, stabilize = self._decode_action(action)
        if normalized["fuel"] <= 0.0:
            desired_main = 0.0
            desired_side = 0.0
            stabilize = False

        smoothing = params["command_smoothing"]
        applied_main = smoothing * normalized["applied_main"] + (1.0 - smoothing) * desired_main
        applied_side = smoothing * normalized["applied_side"] + (1.0 - smoothing) * desired_side
        fuel_cost = (
            applied_main * params["fuel_burn_main"]
            + abs(applied_side) * params["fuel_burn_side"]
            + (params["fuel_burn_stabilize"] if stabilize else 0.0)
        )
        if fuel_cost > normalized["fuel"] and fuel_cost > 0:
            scale = normalized["fuel"] / fuel_cost
            applied_main *= scale
            applied_side *= scale
            fuel_cost = normalized["fuel"]

        fuel = max(0.0, normalized["fuel"] - fuel_cost)
        wind_x = clamp(
            params["wind_alpha"] * normalized["wind_x"] + rng.gauss(0.0, params["wind_sigma"]),
            -0.050,
            0.050,
        )
        omega = normalized["omega"] + (applied_side * params["torque_power"]) + (wind_x * 0.08)
        theta = normalized["theta"] + omega
        if stabilize:
            omega *= 0.42
            theta *= 0.82
        theta = clamp(theta, -0.75, 0.75)

        ax = wind_x + (applied_side * params["side_power"]) + (math.sin(theta) * params["main_power"] * applied_main)
        ay = (math.cos(theta) * params["main_power"] * applied_main) - params["gravity"]
        vx = clamp((normalized["vx"] + ax) * 0.997, -3.5, 3.5)
        vy = clamp((normalized["vy"] + ay) * 0.997, -4.0, 3.0)
        if stabilize:
            vx *= 0.97
        x = normalized["x"] + vx
        y = normalized["y"] + vy

        new_state = {
            **normalized,
            "step": normalized["step"] + 1,
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "theta": theta,
            "omega": omega,
            "fuel": fuel,
            "wind_x": wind_x,
            "applied_main": applied_main,
            "applied_side": applied_side,
        }

        dx_pad = abs(new_state["pad_center"] - new_state["x"])
        reward = (
            -0.03
            - (0.004 * dx_pad)
            - (0.02 * abs(new_state["vx"]))
            - (0.03 * abs(min(new_state["vy"], 0.0)))
            - (0.03 * abs(new_state["theta"]))
        )
        done = False
        info = {}

        if new_state["y"] <= 0.0:
            done = True
            if self._is_safe_landing(new_state):
                new_state["result"] = "win"
                reward = 120.0 + (12.0 * (new_state["fuel"] / self.config.initial_fuel))
            else:
                new_state["result"] = "lose"
                reward = -120.0
        elif new_state["x"] < 0.0 or new_state["x"] > self.config.world_width or new_state["y"] > self.config.world_height:
            done = True
            new_state["result"] = "lose"
            reward = -120.0
        elif new_state["step"] >= self.config.max_steps:
            done = True
            new_state["result"] = "max_steps"
            reward = -10.0 + max(0.0, 6.0 - dx_pad * 0.2)

        new_state["done"] = done
        if done:
            info["result"] = new_state["result"]

        return new_state, float(round(reward, 6)), done, info

    def terminal_points(self, result, state):
        if result == "win":
            fuel_bonus = 1 if state["fuel"] >= (0.35 * self.config.initial_fuel) else 0
            return 10 + fuel_bonus
        if result == "lose":
            return -2
        if result == "max_steps":
            return 0
        return 0

    def summary(self, state):
        normalized = self.normalize_state(state)
        left_contact, right_contact = self._leg_contacts(normalized)
        return {
            "position": {
                "x": round(normalized["x"], 2),
                "altitude": round(max(normalized["y"], 0.0), 2),
            },
            "velocity": {
                "vx": round(normalized["vx"], 3),
                "vy": round(normalized["vy"], 3),
            },
            "tilt": round(normalized["theta"], 4),
            "angular_rate": round(normalized["omega"], 4),
            "fuel": round(normalized["fuel"], 2),
            "done": normalized["done"],
            "result": normalized.get("result"),
            "landing_pad": {
                "center_x": round(normalized["pad_center"], 2),
                "x1": round(normalized["pad_center"] - normalized["pad_width"] / 2.0, 2),
                "x2": round(normalized["pad_center"] + normalized["pad_width"] / 2.0, 2),
                "y": 0.0,
            },
            "leg_contact_left": left_contact,
            "leg_contact_right": right_contact,
            "observation": self.observe(normalized),
            "observation_schema": list(self.OBSERVATION_SCHEMA),
            "world_bounds": {
                "width": self.config.world_width,
                "height": self.config.world_height,
            },
        }


class NoisyMoonLanderLiteEnv(BaseEnv):
    """Gym-like environment mirroring the server-side Noisy Moon Lander Lite."""

    metadata = {"render_modes": []}

    def __init__(self, seed=None):
        self.core = NoisyMoonLanderCore(seed=seed)
        self.action_names = list(self.core.get_actions().keys())
        self.observation_schema = list(self.core.OBSERVATION_SCHEMA)
        self.action_space = spaces.Discrete(len(self.action_names))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.observation_schema),),
            dtype=np.float32,
        )
        self.state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.state = self.core.reset(seed=seed)
        observation = np.asarray(self.core.observe(self.state), dtype=np.float32)
        return observation, {"action_names": list(self.action_names)}

    def step(self, action):
        action_name = action if isinstance(action, str) else self.action_names[int(action)]
        self.state, reward, done, info = self.core.step(self.state, action_name)
        observation = np.asarray(self.core.observe(self.state), dtype=np.float32)
        truncated = info.get("result") == "max_steps"
        terminated = done and not truncated
        if done:
            info = {
                **info,
                "score_points": self.core.terminal_points(info["result"], self.state),
            }
        return observation, float(reward), terminated, truncated, info

    def render(self):
        return self.core.summary(self.state or self.core.reset())
