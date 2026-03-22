"""Adaptive Traffic Racing core simulator and local Gym-like environment."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .common import clamp, make_step_rng, system_seed
from .compat import BaseEnv, spaces


@dataclass(frozen=True)
class AdaptiveTrafficConfig:
    lanes: int = 3
    track_length: float = 130.0
    max_steps: int = 82
    min_speed: float = 1.5
    max_speed: float = 5.0
    start_speed: float = 3.0
    cell_size: float = 4.0
    cell_offsets: tuple = (-2, -1, 0, 1, 2, 3, 4, 5, 6)
    spawn_horizon: float = 42.0
    despawn_margin: float = 12.0
    desired_ahead_per_lane: int = 2


class AdaptiveTrafficRacingCore:
    ACTIONS = {
        "left": "Change one lane to the left if safe",
        "right": "Change one lane to the right if safe",
        "keep": "Keep the current lane and speed",
        "accelerate": "Increase speed",
        "brake": "Reduce speed",
    }
    MODES = ("cautious", "aggressive", "erratic")
    TRAFFIC_PHASES = (
        "balanced",
        "compress",
        "release",
        "left_fast",
        "center_fast",
        "right_fast",
        "weave",
    )
    PHASE_TRANSITIONS = {
        "balanced": ("compress", "release", "left_fast", "center_fast", "right_fast", "weave"),
        "compress": ("balanced", "release", "center_fast", "weave"),
        "release": ("balanced", "left_fast", "center_fast", "right_fast"),
        "left_fast": ("balanced", "compress", "center_fast", "weave"),
        "center_fast": ("balanced", "release", "left_fast", "right_fast", "weave"),
        "right_fast": ("balanced", "compress", "center_fast", "weave"),
        "weave": ("balanced", "compress", "left_fast", "center_fast", "right_fast"),
    }

    def __init__(self, seed=None, config=None):
        self.default_seed = seed
        self.config = config or AdaptiveTrafficConfig()
        self.observation_schema = self._build_observation_schema()

    def get_actions(self):
        return dict(self.ACTIONS)

    def _build_observation_schema(self):
        schema = ["lane_norm", "speed_norm"]
        for prefix in ("occ_t0", "rel_t0", "occ_t1", "occ_t2"):
            for lane in range(self.config.lanes):
                for cell in self.config.cell_offsets:
                    schema.append(f"{prefix}_lane{lane}_cell{cell}")
        return schema

    def _mode_parameters(self, mode):
        if mode == "cautious":
            return {
                "base_speed": 2.7,
                "speed_jitter": 0.35,
                "lane_change": 0.16,
                "min_gap": 5.2,
                "safe_ahead": 2.8,
                "safe_behind": 2.2,
                "phase_steps": (8, 11),
            }
        if mode == "aggressive":
            return {
                "base_speed": 3.75,
                "speed_jitter": 0.55,
                "lane_change": 0.32,
                "min_gap": 4.0,
                "safe_ahead": 2.4,
                "safe_behind": 1.8,
                "phase_steps": (5, 8),
            }
        return {
            "base_speed": 3.2,
            "speed_jitter": 0.85,
            "lane_change": 0.24,
            "min_gap": 4.5,
            "safe_ahead": 2.4,
            "safe_behind": 1.9,
            "phase_steps": (4, 7),
        }

    def _phase_parameters(self, phase):
        if phase == "balanced":
            return {
                "speed_bias": (0.0, 0.1, 0.0),
                "gap_scale": 1.0,
                "lane_bias": (0.0, 0.15, 0.0),
                "lane_change_mult": 1.0,
                "opportunistic_change": 0.06,
                "spawn_lead_range": (11.0, 16.0),
                "second_vehicle_chance": 0.65,
                "extra_jitter": 0.0,
            }
        if phase == "compress":
            return {
                "speed_bias": (-0.25, -0.1, -0.25),
                "gap_scale": 0.95,
                "lane_bias": (0.0, 0.25, 0.0),
                "lane_change_mult": 0.8,
                "opportunistic_change": 0.04,
                "spawn_lead_range": (9.0, 14.0),
                "second_vehicle_chance": 0.9,
                "extra_jitter": 0.05,
            }
        if phase == "release":
            return {
                "speed_bias": (0.15, 0.05, 0.15),
                "gap_scale": 1.35,
                "lane_bias": (0.0, 0.1, 0.0),
                "lane_change_mult": 0.7,
                "opportunistic_change": 0.05,
                "spawn_lead_range": (12.0, 20.0),
                "second_vehicle_chance": 0.4,
                "extra_jitter": 0.0,
            }
        if phase == "left_fast":
            return {
                "speed_bias": (0.55, -0.15, -0.35),
                "gap_scale": 1.1,
                "lane_bias": (1.0, 0.15, -0.35),
                "lane_change_mult": 1.15,
                "opportunistic_change": 0.12,
                "spawn_lead_range": (10.0, 17.0),
                "second_vehicle_chance": 0.8,
                "extra_jitter": 0.1,
                "favored_lane": 0,
            }
        if phase == "center_fast":
            return {
                "speed_bias": (-0.15, 0.55, -0.15),
                "gap_scale": 1.05,
                "lane_bias": (-0.2, 0.9, -0.2),
                "lane_change_mult": 1.1,
                "opportunistic_change": 0.12,
                "spawn_lead_range": (10.0, 17.0),
                "second_vehicle_chance": 0.75,
                "extra_jitter": 0.1,
                "favored_lane": 1,
            }
        if phase == "right_fast":
            return {
                "speed_bias": (-0.35, -0.15, 0.55),
                "gap_scale": 1.1,
                "lane_bias": (-0.35, 0.15, 1.0),
                "lane_change_mult": 1.15,
                "opportunistic_change": 0.12,
                "spawn_lead_range": (10.0, 17.0),
                "second_vehicle_chance": 0.8,
                "extra_jitter": 0.1,
                "favored_lane": 2,
            }
        return {
            "speed_bias": (0.05, 0.0, 0.05),
            "gap_scale": 1.0,
            "lane_bias": (0.25, 0.0, 0.25),
            "lane_change_mult": 1.6,
            "opportunistic_change": 0.24,
            "spawn_lead_range": (10.0, 15.0),
            "second_vehicle_chance": 0.7,
            "extra_jitter": 0.35,
        }

    def _weighted_choice(self, rng, weighted_items):
        total = sum(weight for _item, weight in weighted_items)
        threshold = rng.random() * total
        cumulative = 0.0
        for item, weight in weighted_items:
            cumulative += weight
            if threshold <= cumulative:
                return item
        return weighted_items[-1][0]

    def _sample_phase(self, mode, rng, current=None):
        base_weights = {
            "cautious": {
                "balanced": 3.0,
                "compress": 2.2,
                "release": 2.0,
                "left_fast": 1.0,
                "center_fast": 1.2,
                "right_fast": 1.0,
                "weave": 0.4,
            },
            "aggressive": {
                "balanced": 1.0,
                "compress": 1.0,
                "release": 1.2,
                "left_fast": 2.0,
                "center_fast": 1.3,
                "right_fast": 2.0,
                "weave": 2.4,
            },
            "erratic": {
                "balanced": 0.9,
                "compress": 1.2,
                "release": 1.0,
                "left_fast": 1.7,
                "center_fast": 1.2,
                "right_fast": 1.7,
                "weave": 2.1,
            },
        }[mode]
        candidates = self.PHASE_TRANSITIONS.get(current, self.TRAFFIC_PHASES)
        weighted_items = [(phase, base_weights[phase]) for phase in candidates]
        return self._weighted_choice(rng, weighted_items)

    def _sample_phase_duration(self, mode, phase, rng):
        low, high = self._mode_parameters(mode)["phase_steps"]
        if phase == "weave":
            high = max(low, high - 1)
        elif phase == "release":
            high += 1
        return rng.randint(low, high)

    def _sample_gap(self, params, phase, rng, *, rank):
        phase_params = self._phase_parameters(phase)
        base_gap = params["min_gap"] * phase_params["gap_scale"]
        if rng.random() < 0.62:
            return rng.uniform(base_gap + 1.0 + (0.7 * rank), base_gap + 3.2 + (1.1 * rank))
        return rng.uniform(base_gap + 5.0, base_gap + 9.0)

    def _target_ahead_count(self, mode, phase, lane, rng):
        target = self.config.desired_ahead_per_lane
        phase_params = self._phase_parameters(phase)
        favored_lane = phase_params.get("favored_lane")
        if phase == "release":
            target = 1 if rng.random() < 0.7 else 2
        elif favored_lane is not None and lane == favored_lane:
            target = 1
        elif phase == "balanced":
            target = 2 if rng.random() < 0.55 else 1
        if mode == "cautious" and phase == "release" and lane == 1:
            target = 1
        return int(clamp(target, 1, self.config.desired_ahead_per_lane))

    def _sample_vehicle_speed(self, mode, phase, lane, rng):
        params = self._mode_parameters(mode)
        phase_params = self._phase_parameters(phase)
        jitter = params["speed_jitter"] + phase_params["extra_jitter"]
        target_speed = (
            params["base_speed"]
            + phase_params["speed_bias"][lane]
            + rng.uniform(-jitter, jitter)
        )
        favored_lane = phase_params.get("favored_lane")
        if favored_lane is not None and lane == favored_lane and rng.random() < 0.35:
            target_speed += 0.25
        if mode == "erratic" and rng.random() < 0.20:
            target_speed += rng.uniform(-0.8, 0.8)
        return clamp(target_speed, self.config.min_speed, self.config.max_speed)

    def _maybe_spawn_rear_chaser(self, behind, mode, phase, lane, player_progress, rng):
        phase_params = self._phase_parameters(phase)
        favored_lane = phase_params.get("favored_lane")
        behind = behind[-1:] if behind else []
        should_spawn = (
            phase == "weave"
            or lane == favored_lane
            or (favored_lane is not None and abs(lane - favored_lane) == 1 and rng.random() < 0.35)
        )
        if not should_spawn:
            return behind
        nearest = behind[-1] if behind else None
        if nearest is not None and nearest["s"] > player_progress - 8.0:
            return behind
        chaser_speed = clamp(
            self._sample_vehicle_speed(mode, phase, lane, rng) + rng.uniform(0.35, 0.75),
            self.config.min_speed,
            self.config.max_speed,
        )
        chaser = {
            "lane": lane,
            "s": round(player_progress - rng.uniform(4.5, 8.5), 4),
            "speed": round(chaser_speed, 4),
        }
        return [chaser]

    def _generate_vehicles(self, seed, mode, phase):
        rng = make_step_rng(seed, 0, "traffic_reset")
        phase_params = self._phase_parameters(phase)
        vehicles = []
        for lane in range(self.config.lanes):
            target_ahead = self._target_ahead_count(mode, phase, lane, rng)
            position = rng.uniform(*phase_params["spawn_lead_range"])
            for index in range(target_ahead):
                if index > 0:
                    position += self._sample_gap(self._mode_parameters(mode), phase, rng, rank=index - 1)
                if position >= self.config.spawn_horizon:
                    break
                vehicles.append(
                    {
                        "lane": lane,
                        "s": round(position, 4),
                        "speed": round(self._sample_vehicle_speed(mode, phase, lane, rng), 4),
                    }
                )
                if index == 0 and rng.random() > phase_params["second_vehicle_chance"]:
                    break
        vehicles.sort(key=lambda vehicle: (vehicle["s"], vehicle["lane"]))
        return vehicles

    def _lane_gaps(self, vehicles, lane, progress, *, exclude_s=None):
        ahead = math.inf
        behind = math.inf
        for vehicle in vehicles:
            if vehicle["lane"] != lane:
                continue
            if exclude_s is not None and abs(vehicle["s"] - exclude_s) < 1e-6:
                continue
            rel = vehicle["s"] - progress
            if rel >= 0:
                ahead = min(ahead, rel)
            else:
                behind = min(behind, abs(rel))
        return ahead, behind

    def _lane_context(self, vehicles, lane, progress, *, exclude_s=None):
        ahead_gap = math.inf
        behind_gap = math.inf
        ahead_speed = None
        behind_speed = None
        for vehicle in vehicles:
            if vehicle["lane"] != lane:
                continue
            if exclude_s is not None and abs(vehicle["s"] - exclude_s) < 1e-6:
                continue
            rel = vehicle["s"] - progress
            if rel >= 0 and rel < ahead_gap:
                ahead_gap = rel
                ahead_speed = vehicle["speed"]
            elif rel < 0 and abs(rel) < behind_gap:
                behind_gap = abs(rel)
                behind_speed = vehicle["speed"]
        return ahead_gap, behind_gap, ahead_speed, behind_speed

    def _lane_is_safe(
        self,
        vehicles,
        lane,
        progress,
        *,
        ahead_gap=2.0,
        behind_gap=1.5,
        ego_speed=None,
        exclude_s=None,
    ):
        ahead, behind, ahead_speed, behind_speed = self._lane_context(vehicles, lane, progress, exclude_s=exclude_s)
        if ego_speed is None:
            return ahead > ahead_gap and behind > behind_gap
        front_closure = 0.0 if ahead_speed is None else max(0.0, ego_speed - ahead_speed)
        rear_closure = 0.0 if behind_speed is None else max(0.0, behind_speed - ego_speed)
        dynamic_ahead_gap = ahead_gap + (0.7 * front_closure)
        dynamic_behind_gap = behind_gap + (1.1 * rear_closure)
        return ahead > dynamic_ahead_gap and behind > dynamic_behind_gap

    def _lane_score(self, vehicles, lane, position, phase, *, exclude_s=None):
        phase_params = self._phase_parameters(phase)
        ahead, behind = self._lane_gaps(vehicles, lane, position, exclude_s=exclude_s)
        ahead_score = 12.0 if math.isinf(ahead) else min(ahead, 12.0)
        behind_score = 3.0 if math.isinf(behind) else min(behind, 6.0) * 0.35
        return ahead_score + behind_score + phase_params["lane_bias"][lane]

    def _maybe_change_lane(self, vehicle, vehicles, mode, phase, rng):
        params = self._mode_parameters(mode)
        phase_params = self._phase_parameters(phase)
        current_lane = vehicle["lane"]
        ahead, _ = self._lane_gaps(vehicles, current_lane, vehicle["s"], exclude_s=vehicle["s"])
        pressure_gap = params["min_gap"] * phase_params["gap_scale"]
        under_pressure = ahead < pressure_gap
        if not under_pressure and rng.random() > phase_params["opportunistic_change"]:
            return current_lane

        current_score = self._lane_score(
            vehicles,
            current_lane,
            vehicle["s"],
            phase,
            exclude_s=vehicle["s"],
        )
        candidates = []
        for delta in (-1, 1):
            target_lane = current_lane + delta
            if not 0 <= target_lane < self.config.lanes:
                continue
            if self._lane_is_safe(
                vehicles,
                target_lane,
                vehicle["s"],
                ahead_gap=params["safe_ahead"],
                behind_gap=params["safe_behind"],
                ego_speed=vehicle["speed"],
            ):
                score = self._lane_score(vehicles, target_lane, vehicle["s"], phase)
                margin = 0.8 if under_pressure else 1.3
                if score > current_score + margin:
                    candidates.append((score + rng.uniform(0.0, 0.15), target_lane))
        if not candidates:
            return current_lane
        if under_pressure or rng.random() < (params["lane_change"] * phase_params["lane_change_mult"]):
            candidates.sort(reverse=True)
            return candidates[0][1]
        return current_lane

    def _step_vehicles(self, vehicles, mode, phase, player_progress, player_speed, seed, step):
        rng = make_step_rng(seed, step, "traffic_step")
        params = self._mode_parameters(mode)
        phase_params = self._phase_parameters(phase)
        updated = []
        occupied = [dict(vehicle) for vehicle in vehicles]
        for vehicle in sorted(occupied, key=lambda item: item["s"]):
            candidate_lane = self._maybe_change_lane(vehicle, occupied, mode, phase, rng)
            target_speed = self._sample_vehicle_speed(mode, phase, candidate_lane, rng)
            ahead_gap, _ = self._lane_gaps(
                occupied,
                candidate_lane,
                vehicle["s"],
                exclude_s=vehicle["s"] if candidate_lane == vehicle["lane"] else None,
            )
            follow_gap = params["min_gap"] * phase_params["gap_scale"]
            if ahead_gap < follow_gap:
                target_speed = min(target_speed, max(self.config.min_speed, ahead_gap - 0.7))
            elif ahead_gap > follow_gap + 4.0 and rng.random() < 0.35:
                target_speed += 0.25
            if phase == "compress" and ahead_gap < follow_gap + 1.0:
                target_speed -= 0.15
            if phase == "release" and ahead_gap > follow_gap + 2.0:
                target_speed += 0.2
            if phase == "weave" and rng.random() < 0.18:
                target_speed += rng.uniform(-0.7, 0.7)
            speed = clamp(target_speed, self.config.min_speed, self.config.max_speed)
            updated.append(
                {
                    "lane": candidate_lane,
                    "s": round(vehicle["s"] + speed, 4),
                    "speed": round(speed, 4),
                }
            )

        updated = [vehicle for vehicle in updated if vehicle["s"] >= player_progress - self.config.despawn_margin]
        updated = self._spawn_vehicles(updated, mode, phase, player_progress, player_speed, seed, step)
        updated.sort(key=lambda vehicle: (vehicle["s"], vehicle["lane"]))
        return updated

    def _spawn_vehicles(self, vehicles, mode, phase, player_progress, player_speed, seed, step):
        rng = make_step_rng(seed, step, "traffic_spawn")
        params = self._mode_parameters(mode)
        phase_params = self._phase_parameters(phase)
        by_lane = {lane: [] for lane in range(self.config.lanes)}
        for vehicle in vehicles:
            by_lane[vehicle["lane"]].append(vehicle)

        spawned = []
        for lane in range(self.config.lanes):
            lane_vehicles = sorted(by_lane[lane], key=lambda item: item["s"])
            behind = [vehicle for vehicle in lane_vehicles if vehicle["s"] < player_progress]
            target_ahead = self._target_ahead_count(mode, phase, lane, rng)
            ahead = [vehicle for vehicle in lane_vehicles if vehicle["s"] >= player_progress][:target_ahead]
            farthest = ahead[-1]["s"] if ahead else player_progress + rng.uniform(*phase_params["spawn_lead_range"])
            while len(ahead) < target_ahead and farthest < player_progress + self.config.spawn_horizon:
                if ahead:
                    farthest += self._sample_gap(params, phase, rng, rank=len(ahead) - 1)
                candidate = round(farthest, 4)
                if candidate >= player_progress + self.config.spawn_horizon:
                    break
                ahead.append(
                    {
                        "lane": lane,
                        "s": candidate,
                        "speed": round(self._sample_vehicle_speed(mode, phase, lane, rng), 4),
                    }
                )
            behind = self._maybe_spawn_rear_chaser(behind, mode, phase, lane, player_progress, rng)
            spawned.extend(behind + ahead)
        return spawned

    def _build_frame(self, lane, speed, progress, vehicles):
        occupancy = []
        relative_speed = []
        for target_lane in range(self.config.lanes):
            for cell in self.config.cell_offsets:
                center = cell * self.config.cell_size
                best = None
                best_distance = math.inf
                for vehicle in vehicles:
                    if vehicle["lane"] != target_lane:
                        continue
                    rel = vehicle["s"] - progress
                    if abs(rel - center) <= (self.config.cell_size / 2.0):
                        distance = abs(rel - center)
                        if distance < best_distance:
                            best = vehicle
                            best_distance = distance
                if best is None:
                    occupancy.append(0.0)
                    relative_speed.append(0.0)
                else:
                    occupancy.append(1.0)
                    relative_speed.append(
                        round(clamp((best["speed"] - speed) / 2.5, -1.0, 1.0), 6)
                    )
        return {"occupancy": occupancy, "relative_speed": relative_speed}

    def observe(self, state):
        current = state["frame_history"][-1]
        previous = state["frame_history"][-2]
        previous_previous = state["frame_history"][-3]
        observation = [
            round(state["lane"] / (self.config.lanes - 1), 6),
            round((state["speed"] - self.config.min_speed) / (self.config.max_speed - self.config.min_speed), 6),
        ]
        observation.extend(current["occupancy"])
        observation.extend(current["relative_speed"])
        observation.extend(previous["occupancy"])
        observation.extend(previous_previous["occupancy"])
        return observation

    def reset(self, seed=None):
        episode_seed = self.default_seed if seed is None else seed
        if episode_seed is None:
            episode_seed = system_seed()
        rng = make_step_rng(episode_seed, 0, "traffic_mode")
        mode = self.MODES[int(rng.random() * len(self.MODES)) % len(self.MODES)]
        phase_rng = make_step_rng(episode_seed, 0, "traffic_phase")
        traffic_phase = self._sample_phase(mode, phase_rng)
        phase_steps_remaining = self._sample_phase_duration(mode, traffic_phase, phase_rng)
        vehicles = self._generate_vehicles(episode_seed, mode, traffic_phase)
        frame = self._build_frame(1, self.config.start_speed, 0.0, vehicles)
        return {
            "seed": int(episode_seed),
            "mode": mode,
            "traffic_phase": traffic_phase,
            "phase_steps_remaining": phase_steps_remaining,
            "step": 0,
            "lane": 1,
            "speed": self.config.start_speed,
            "progress": 0.0,
            "vehicles": vehicles,
            "frame_history": [frame, frame, frame],
            "last_action": "keep",
            "done": False,
            "result": None,
        }

    def normalize_state(self, state):
        normalized = dict(state)
        normalized.setdefault("seed", self.default_seed or 0)
        normalized.setdefault("mode", "cautious")
        normalized.setdefault("traffic_phase", "balanced")
        normalized.setdefault("phase_steps_remaining", 1)
        normalized.setdefault("step", 0)
        normalized.setdefault("lane", 1)
        normalized.setdefault("speed", self.config.start_speed)
        normalized.setdefault("progress", 0.0)
        normalized.setdefault("vehicles", [])
        normalized.setdefault("frame_history", [])
        if not normalized["frame_history"]:
            frame = self._build_frame(
                normalized["lane"],
                normalized["speed"],
                normalized["progress"],
                normalized["vehicles"],
            )
            normalized["frame_history"] = [frame, frame, frame]
        normalized.setdefault("done", False)
        normalized.setdefault("result", None)
        return normalized

    def is_valid_action(self, action, state):
        return action in self.ACTIONS and not state.get("done", False)

    def step(self, state, action):
        normalized = self.normalize_state(state)
        if not self.is_valid_action(action, normalized):
            return normalized, -1.0, False, {"error": "Invalid action"}

        traffic_phase = normalized["traffic_phase"]
        phase_steps_remaining = int(normalized.get("phase_steps_remaining", 1))
        if phase_steps_remaining <= 0:
            phase_rng = make_step_rng(normalized["seed"], normalized["step"], "traffic_phase")
            traffic_phase = self._sample_phase(normalized["mode"], phase_rng, traffic_phase)
            phase_steps_remaining = self._sample_phase_duration(normalized["mode"], traffic_phase, phase_rng)

        lane = normalized["lane"]
        speed = normalized["speed"]
        if action == "left" and lane > 0 and self._lane_is_safe(
            normalized["vehicles"],
            lane - 1,
            normalized["progress"],
            ego_speed=speed,
        ):
            lane -= 1
        elif action == "right" and lane < self.config.lanes - 1 and self._lane_is_safe(
            normalized["vehicles"],
            lane + 1,
            normalized["progress"],
            ego_speed=speed,
        ):
            lane += 1
        elif action == "accelerate":
            speed = clamp(speed + 0.7, self.config.min_speed, self.config.max_speed)
        elif action == "brake":
            speed = clamp(speed - 0.8, self.config.min_speed, self.config.max_speed)

        vehicles = self._step_vehicles(
            normalized["vehicles"],
            normalized["mode"],
            traffic_phase,
            normalized["progress"],
            speed,
            normalized["seed"],
            normalized["step"],
        )
        progress = normalized["progress"] + speed
        collision = any(
            vehicle["lane"] == lane and abs(vehicle["s"] - progress) < 1.1
            for vehicle in vehicles
        )
        frame = self._build_frame(lane, speed, progress, vehicles)
        frame_history = [*normalized["frame_history"][-2:], frame]

        new_state = {
            **normalized,
            "step": normalized["step"] + 1,
            "traffic_phase": traffic_phase,
            "phase_steps_remaining": phase_steps_remaining - 1,
            "lane": lane,
            "speed": speed,
            "progress": progress,
            "vehicles": vehicles,
            "frame_history": frame_history,
            "last_action": action,
        }

        nearest_ahead, _ = self._lane_gaps(vehicles, lane, progress)
        reward = 0.15 + (0.35 * ((speed - self.config.min_speed) / (self.config.max_speed - self.config.min_speed)))
        if nearest_ahead < 3.0:
            reward -= 0.25

        done = False
        info = {}
        if collision:
            done = True
            new_state["result"] = "lose"
            reward = -40.0
        elif progress >= self.config.track_length:
            done = True
            new_state["result"] = "win"
            reward = 45.0
        elif new_state["step"] >= self.config.max_steps:
            done = True
            new_state["result"] = "max_steps"
            reward = 5.0 * (progress / self.config.track_length)

        new_state["done"] = done
        if done:
            info["result"] = new_state["result"]
        return new_state, float(round(reward, 6)), done, info

    def terminal_points(self, result, state):
        if result == "win":
            speed_bonus = 1 if state["speed"] >= 4.2 else 0
            return 6 + speed_bonus
        if result == "lose":
            return -2
        if result == "max_steps":
            progress_ratio = clamp(state["progress"] / self.config.track_length, 0.0, 1.0)
            return int(progress_ratio * 3)
        return 0

    def summary(self, state):
        normalized = self.normalize_state(state)
        nearby = []
        for vehicle in normalized["vehicles"]:
            distance = vehicle["s"] - normalized["progress"]
            if -8.0 <= distance <= 28.0:
                nearby.append(
                    {
                        "lane": vehicle["lane"],
                        "distance": round(distance, 2),
                        "relative_speed": round(vehicle["speed"] - normalized["speed"], 2),
                    }
                )
        nearby.sort(key=lambda vehicle: (vehicle["distance"], vehicle["lane"]))
        current_frame = normalized["frame_history"][-1]
        previous_frame = normalized["frame_history"][-2]
        oldest_frame = normalized["frame_history"][-3]
        window = []
        lane_count = self.config.lanes
        cell_count = len(self.config.cell_offsets)
        for lane in range(lane_count):
            lane_cells = []
            for index, offset in enumerate(self.config.cell_offsets):
                base_index = lane * cell_count + index
                lane_cells.append(
                    {
                        "offset": offset,
                        "occupied": bool(current_frame["occupancy"][base_index]),
                        "relative_speed": current_frame["relative_speed"][base_index],
                        "occupied_t1": bool(previous_frame["occupancy"][base_index]),
                        "occupied_t2": bool(oldest_frame["occupancy"][base_index]),
                    }
                )
            window.append({"lane": lane, "cells": lane_cells})

        lane_gaps = []
        for lane in range(lane_count):
            ahead, behind = self._lane_gaps(normalized["vehicles"], lane, normalized["progress"])
            lane_gaps.append(
                {
                    "lane": lane,
                    "ahead": None if math.isinf(ahead) else round(ahead, 2),
                    "behind": None if math.isinf(behind) else round(behind, 2),
                    "safe_now": self._lane_is_safe(
                        normalized["vehicles"],
                        lane,
                        normalized["progress"],
                        ego_speed=normalized["speed"],
                    ),
                }
            )
        return {
            "lane": normalized["lane"],
            "speed": round(normalized["speed"], 2),
            "progress": round(normalized["progress"], 2),
            "track_length": self.config.track_length,
            "step": normalized["step"],
            "max_steps": self.config.max_steps,
            "last_action": normalized.get("last_action"),
            "nearby_vehicles": nearby,
            "lane_gaps": lane_gaps,
            "sensor_window": window,
            "done": normalized["done"],
            "result": normalized.get("result"),
            "observation": self.observe(normalized),
            "observation_schema": list(self.observation_schema),
        }


class AdaptiveTrafficRacingEnv(BaseEnv):
    """Gym-like environment mirroring the server-side Adaptive Traffic Racing."""

    metadata = {"render_modes": []}

    def __init__(self, seed=None):
        self.core = AdaptiveTrafficRacingCore(seed=seed)
        self.action_names = list(self.core.get_actions().keys())
        self.action_space = spaces.Discrete(len(self.action_names))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.core.observation_schema),),
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
