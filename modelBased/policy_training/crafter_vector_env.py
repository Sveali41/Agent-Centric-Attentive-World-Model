"""Subprocess vector environment for the real Crafter PPO collector.

Workers deliberately do not auto-reset.  The parent can therefore bootstrap
from the terminal observation before resetting only the finished slots.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any

import numpy as np

from domain.crafter.crafter_custom_env import CustomCrafterEnv


def _worker(remote, layout_path: str, max_steps: int, seed: int) -> None:
    """Run one environment and communicate only through the pipe."""
    env = CustomCrafterEnv(
        txt_file_path=layout_path,
        max_steps=max_steps,
        seed=seed,
    )
    try:
        while True:
            command, payload = remote.recv()
            if command == "reset":
                observation, info = env.reset()
                remote.send((observation, info))
            elif command == "step":
                remote.send(env.step(int(payload)))
            elif command == "close":
                break
            else:
                raise RuntimeError(f"Unknown Crafter worker command: {command}")
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        env.close()
        remote.close()


class CrafterSubprocessVectorEnv:
    """Synchronous vectorized Crafter environment backed by subprocesses."""

    def __init__(
        self,
        layout_path: str,
        max_steps: int,
        num_envs: int,
        seed: int,
        start_method: str = "spawn",
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        if start_method not in mp.get_all_start_methods():
            raise ValueError(
                f"Unsupported multiprocessing start method: {start_method}"
            )

        self.num_envs = int(num_envs)
        self._closed = False
        context = mp.get_context(start_method)
        self._remotes, worker_remotes = zip(
            *[context.Pipe() for _ in range(self.num_envs)]
        )
        self._processes = []
        for index, (remote, worker_remote) in enumerate(
            zip(self._remotes, worker_remotes)
        ):
            process = context.Process(
                target=_worker,
                args=(worker_remote, layout_path, int(max_steps), int(seed) + index),
                daemon=True,
            )
            process.start()
            worker_remote.close()
            self._processes.append(process)

        self._image_shape: tuple[int, ...] | None = None
        self._inventory_shape: tuple[int, ...] | None = None

    @staticmethod
    def _split_observation(observation: dict[str, Any]):
        return (
            np.asarray(observation["image"], dtype=np.int32),
            np.asarray(observation["inventory"], dtype=np.float32),
        )

    def _batch_observations(self, observations):
        split = [self._split_observation(observation) for observation in observations]
        images = np.stack([item[0] for item in split], axis=0)
        inventories = np.stack([item[1] for item in split], axis=0)
        if self._image_shape is None:
            self._image_shape = images.shape[1:]
            self._inventory_shape = inventories.shape[1:]
        return {"image": images, "inventory": inventories}

    def reset(self, indices=None):
        """Reset all workers or only the specified worker indices."""
        if self._closed:
            raise RuntimeError("Cannot reset a closed vector environment")
        if indices is None:
            selected = list(range(self.num_envs))
        else:
            selected = [int(index) for index in np.asarray(indices).reshape(-1)]
            if any(index < 0 or index >= self.num_envs for index in selected):
                raise IndexError("Crafter vector reset index is out of range")

        for index in selected:
            self._remotes[index].send(("reset", None))
        results = {index: self._remotes[index].recv() for index in selected}

        if indices is None:
            observations = [results[index][0] for index in range(self.num_envs)]
            infos = [results[index][1] for index in range(self.num_envs)]
            return self._batch_observations(observations), infos

        return self._batch_observations(
            [results[index][0] for index in selected]
        ), [results[index][1] for index in selected]

    def reset_at(self, indices):
        """Reset selected workers and return observations in index order."""
        return self.reset(indices=indices)

    def step(self, actions: np.ndarray):
        if self._closed:
            raise RuntimeError("Cannot step a closed vector environment")
        actions = np.asarray(actions).reshape(-1)
        if actions.size != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} actions, got {actions.size}"
            )

        for index, action in enumerate(actions):
            self._remotes[index].send(("step", int(action)))
        results = [remote.recv() for remote in self._remotes]

        observations, rewards, terminated, truncated, infos = zip(*results)
        return (
            self._batch_observations(observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminated, dtype=np.bool_),
            np.asarray(truncated, dtype=np.bool_),
            list(infos),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for remote in self._remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for remote in self._remotes:
            remote.close()
        for process in self._processes:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
