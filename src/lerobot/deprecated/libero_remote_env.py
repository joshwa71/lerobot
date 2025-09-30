#!/usr/bin/env python

# /home/josh/phddev/lerobot-upstream/src/lerobot/envs/libero_remote_env.py

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from __future__ import annotations

import base64
import json
import socket
import struct
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np


def _encode_ndarray(arr: np.ndarray) -> Dict[str, Any]:
    return {
        "__ndarray__": True,
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes(order="C")).decode("ascii"),
    }


def _decode_ndarray(obj: Dict[str, Any]) -> np.ndarray:
    assert obj.get("__ndarray__", False), "Invalid ndarray encoding"
    dtype = np.dtype(obj["dtype"])  # type: ignore[arg-type]
    shape = tuple(obj["shape"])  # type: ignore[assignment]
    raw = base64.b64decode(obj["data"])  # type: ignore[arg-type]
    arr = np.frombuffer(raw, dtype=dtype)
    return arr.reshape(shape)


def _send_msg(sock: socket.socket, msg: Dict[str, Any]) -> None:
    data = json.dumps(msg).encode("utf-8")
    header = struct.pack(">I", len(data))
    sock.sendall(header + data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Remote closed the connection")
        buf.extend(chunk)
    return bytes(buf)


def _recv_msg(sock: socket.socket) -> Dict[str, Any]:
    header = _recv_exact(sock, 4)
    (size,) = struct.unpack(">I", header)
    payload = _recv_exact(sock, size)
    msg = json.loads(payload.decode("utf-8"))
    return msg


@dataclass
class _RemoteSpec:
    action_dim: int
    action_low: float | None
    action_high: float | None
    image_shapes: Dict[str, Tuple[int, int, int]]
    state_dim: int


class LiberoRemoteEnv(gym.Env):
    """
    Gymnasium environment client that connects to a LIBERO remote server process.

    The server handles the heavy dependencies (robosuite / mujoco / libero). This
    client exposes a minimal observation dict compatible with LeRobot wrappers:
    { "pixels": {cam: HxWxC uint8}, "agent_pos": float32[N] }.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout_s: float = 10.0):
        super().__init__()
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self._sock: socket.socket | None = None
        self._spec: _RemoteSpec | None = None

        # Connect and retrieve spec
        self._connect()
        self._handshake()
        self._build_spaces()

    # --- Connection and protocol ---
    def _connect(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout_s)
        sock.connect((self.host, self.port))
        self._sock = sock

    def _handshake(self) -> None:
        assert self._sock is not None
        _send_msg(self._sock, {"cmd": "hello"})
        reply = _recv_msg(self._sock)

        if reply.get("status") != "ok":
            raise RuntimeError(f"Remote handshake failed: {reply}")

        action_dim = int(reply["action_dim"])  # type: ignore[arg-type]
        action_low = reply.get("action_low")
        action_high = reply.get("action_high")
        state_dim = int(reply["state_dim"])  # type: ignore[arg-type]
        image_shapes = {k: tuple(v) for k, v in reply["image_shapes"].items()}  # type: ignore[arg-type]

        self._spec = _RemoteSpec(
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            image_shapes=image_shapes,
            state_dim=state_dim,
        )

    def _build_spaces(self) -> None:
        assert self._spec is not None
        # Action space: default to [-1,1] if bounds not provided
        low = -1.0 if self._spec.action_low is None else float(self._spec.action_low)
        high = 1.0 if self._spec.action_high is None else float(self._spec.action_high)
        self.action_space = gym.spaces.Box(low=low, high=high, shape=(self._spec.action_dim,), dtype=np.float32)

        # Observation space: Dict with nested pixels and agent_pos
        pixel_spaces: Dict[str, gym.Space] = {}
        for cam, shape in self._spec.image_shapes.items():
            h, w, c = shape
            pixel_spaces[cam] = gym.spaces.Box(low=0, high=255, shape=(h, w, c), dtype=np.uint8)

        self.observation_space = gym.spaces.Dict(
            {
                "pixels": gym.spaces.Dict(pixel_spaces),
                "agent_pos": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._spec.state_dim,), dtype=np.float32),
            }
        )

    # --- Gym API ---
    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        del seed, options
        assert self._sock is not None
        _send_msg(self._sock, {"cmd": "reset"})
        reply = _recv_msg(self._sock)
        if reply.get("status") != "ok":
            raise RuntimeError(f"Remote reset failed: {reply}")
        observation = self._decode_observation(reply["observation"])  # type: ignore[arg-type]
        info = reply.get("info", {})
        return observation, info

    def step(self, action):  # type: ignore[override]
        assert self._sock is not None
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        _send_msg(
            self._sock,
            {"cmd": "step", "action": _encode_ndarray(action)},
        )
        reply = _recv_msg(self._sock)
        if reply.get("status") != "ok":
            raise RuntimeError(f"Remote step failed: {reply}")

        observation = self._decode_observation(reply["observation"])  # type: ignore[arg-type]
        reward = float(reply.get("reward", 0.0))
        terminated = bool(reply.get("terminated", False))
        truncated = bool(reply.get("truncated", False))
        info = reply.get("info", {})
        return observation, reward, terminated, truncated, info

    def close(self):  # type: ignore[override]
        try:
            if self._sock is not None:
                try:
                    _send_msg(self._sock, {"cmd": "close"})
                except Exception:
                    pass
                self._sock.close()
        finally:
            self._sock = None

    # --- Helpers ---
    def _decode_observation(self, obs_msg: Dict[str, Any]) -> Dict[str, Any]:
        pixels_msg: Dict[str, Any] = obs_msg.get("pixels", {})
        pixels: Dict[str, np.ndarray] = {}
        for k, enc in pixels_msg.items():
            pixels[k] = _decode_ndarray(enc)

        state_enc = obs_msg.get("agent_pos")
        agent_pos = _decode_ndarray(state_enc)
        if agent_pos.dtype != np.float32:
            agent_pos = agent_pos.astype(np.float32, copy=False)

        return {"pixels": pixels, "agent_pos": agent_pos}


