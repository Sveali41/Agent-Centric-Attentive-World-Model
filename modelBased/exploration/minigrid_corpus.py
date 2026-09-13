"""Immutable MiniGrid corpus files and exact spatial reachability metrics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from minigrid.core.constants import IDX_TO_COLOR, OBJECT_TO_IDX, STATE_TO_IDX
from minigrid.wrappers import FullyObsWrapper

from domain.minigrid.minigrid_custom_env import CustomMiniGridEnv


CORPUS_VERSION = "minigrid_mac_explorer_ab_corpus_v1"


def map_hash(object_map, color_map, state_map, inventory_token, start_dir) -> str:
    digest = hashlib.sha256()
    for array in (object_map, color_map, state_map):
        value = np.ascontiguousarray(array, dtype=np.int16)
        digest.update(np.asarray(value.shape, dtype=np.int16).tobytes())
        digest.update(value.tobytes())
    digest.update(np.asarray([inventory_token, start_dir], dtype=np.int16).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class MiniGridCorpus:
    object_maps: np.ndarray
    color_maps: np.ndarray
    state_maps: np.ndarray
    inventory_tokens: np.ndarray
    start_positions: np.ndarray
    start_dirs: np.ndarray
    iterations: np.ndarray
    batch_indices: np.ndarray
    map_hashes: np.ndarray
    metadata: dict

    def __len__(self) -> int:
        return int(len(self.object_maps))


class MiniGridCorpusWriter:
    """Accumulate generated batches and atomically write one frozen corpus."""

    def __init__(
        self,
        path,
        expected_size: int,
        generation_seed: int | None = None,
        generation_metadata: dict | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.expected_size = int(expected_size)
        if self.expected_size <= 0:
            raise ValueError("expected corpus size must be positive")
        if self.path.exists():
            raise FileExistsError(f"Refusing to overwrite MiniGrid corpus: {self.path}")
        self.generation_seed = None if generation_seed is None else int(generation_seed)
        self.generation_metadata = dict(generation_metadata or {})
        self._records = []

    def append_batch(
        self,
        *,
        object_maps,
        color_maps,
        state_maps,
        inventory_tokens,
        iteration: int,
        start_dirs=None,
    ) -> None:
        arrays = [np.asarray(value) for value in (object_maps, color_maps, state_maps)]
        if not (arrays[0].shape == arrays[1].shape == arrays[2].shape):
            raise ValueError("object/color/state corpus batches must have identical shapes")
        if arrays[0].ndim != 3:
            raise ValueError(f"expected corpus batch [B,H,W], got {arrays[0].shape}")
        batch_size = len(arrays[0])
        inventory_tokens = np.asarray(inventory_tokens, dtype=np.int64).reshape(-1)
        if len(inventory_tokens) != batch_size:
            raise ValueError("inventory batch length does not match generated maps")
        if np.any((inventory_tokens < 0) | (inventory_tokens > len(IDX_TO_COLOR))):
            raise ValueError("inventory tokens must be empty=0 or colour+1")
        if start_dirs is None:
            start_dirs = np.zeros(batch_size, dtype=np.int64)
        start_dirs = np.asarray(start_dirs, dtype=np.int64).reshape(-1)
        if len(start_dirs) != batch_size or np.any((start_dirs < 0) | (start_dirs > 3)):
            raise ValueError("start directions must contain one value in 0..3 per map")
        for batch_index in range(batch_size):
            obj, color, state = (value[batch_index].astype(np.int64, copy=True) for value in arrays)
            if np.any((obj < 0) | (obj > max(OBJECT_TO_IDX.values()))):
                raise ValueError("generated map contains an invalid object id")
            if np.any((color < 0) | (color >= len(IDX_TO_COLOR))):
                raise ValueError("generated map contains an invalid colour id")
            door_mask = obj == OBJECT_TO_IDX["door"]
            if np.any(
                door_mask
                & ~np.isin(
                    state, (STATE_TO_IDX["closed"], STATE_TO_IDX["locked"])
                )
            ):
                raise ValueError("corpus doors must initially be closed or locked")
            starts = np.argwhere(obj == OBJECT_TO_IDX["agent"])
            if len(starts) != 1:
                raise ValueError(
                    f"generated map must have exactly one start marker, found {len(starts)}"
                )
            token = int(inventory_tokens[batch_index])
            direction = int(start_dirs[batch_index])
            self._records.append(
                (obj, color, state, token, direction, int(iteration), batch_index)
            )

    def finalize(self) -> Path:
        if len(self._records) != self.expected_size:
            raise ValueError(
                f"expected exactly {self.expected_size} maps, collected {len(self._records)}"
            )
        if self.path.exists():
            raise FileExistsError(f"Refusing to overwrite MiniGrid corpus: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        object_maps = np.stack([record[0] for record in self._records])
        color_maps = np.stack([record[1] for record in self._records])
        state_maps = np.stack([record[2] for record in self._records])
        inventory_tokens = np.asarray([record[3] for record in self._records], dtype=np.int64)
        start_dirs = np.asarray([record[4] for record in self._records], dtype=np.int64)
        start_positions = np.stack(
            [np.argwhere(record[0] == OBJECT_TO_IDX["agent"])[0] for record in self._records]
        ).astype(np.int64)
        iterations = np.asarray([record[5] for record in self._records], dtype=np.int64)
        batch_indices = np.asarray([record[6] for record in self._records], dtype=np.int64)
        hashes = np.asarray(
            [
                map_hash(obj, color, state, token, direction)
                for obj, color, state, token, direction, *_ in self._records
            ],
            dtype="<U64",
        )
        corpus_hash = hashlib.sha256("".join(hashes.tolist()).encode("ascii")).hexdigest()
        metadata = {
            "version": CORPUS_VERSION,
            "size": self.expected_size,
            "generation_exploration_policy": "random",
            "generation_seed": self.generation_seed,
            "start_direction_policy": "fixed_east",
            "corpus_hash": corpus_hash,
            **self.generation_metadata,
        }
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp.npz")
        np.savez_compressed(
            temporary_path,
            object_maps=object_maps,
            color_maps=color_maps,
            state_maps=state_maps,
            inventory_tokens=inventory_tokens,
            start_positions=start_positions,
            start_dirs=start_dirs,
            iterations=iterations,
            batch_indices=batch_indices,
            map_hashes=hashes,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        temporary_path.replace(self.path)
        return self.path


def load_corpus(path, expected_size: int | None = None) -> MiniGridCorpus:
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as data:
        required = {
            "object_maps", "color_maps", "state_maps", "inventory_tokens",
            "start_positions", "start_dirs", "iterations", "batch_indices",
            "map_hashes", "metadata",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"MiniGrid corpus is missing fields: {sorted(missing)}")
        metadata = json.loads(str(data["metadata"].item()))
        corpus = MiniGridCorpus(
            **{name: data[name].copy() for name in required if name != "metadata"},
            metadata=metadata,
        )
    if metadata.get("version") != CORPUS_VERSION:
        raise ValueError(f"unsupported MiniGrid corpus version: {metadata.get('version')!r}")
    lengths = {
        len(corpus.object_maps), len(corpus.color_maps), len(corpus.state_maps),
        len(corpus.inventory_tokens), len(corpus.start_positions), len(corpus.start_dirs), len(corpus.iterations),
        len(corpus.batch_indices), len(corpus.map_hashes),
    }
    if len(lengths) != 1:
        raise ValueError("MiniGrid corpus arrays have inconsistent lengths")
    if expected_size is not None and len(corpus) != int(expected_size):
        raise ValueError(f"expected {expected_size} corpus maps, found {len(corpus)}")
    for index in range(len(corpus)):
        actual = map_hash(
            corpus.object_maps[index], corpus.color_maps[index], corpus.state_maps[index],
            corpus.inventory_tokens[index], corpus.start_dirs[index],
        )
        if actual != str(corpus.map_hashes[index]):
            raise ValueError(f"corpus map hash mismatch at index {index}")
        actual_start = np.argwhere(corpus.object_maps[index] == OBJECT_TO_IDX["agent"])
        if len(actual_start) != 1 or not np.array_equal(actual_start[0], corpus.start_positions[index]):
            raise ValueError(f"corpus start position mismatch at index {index}")
    corpus_hash = hashlib.sha256(
        "".join(corpus.map_hashes.tolist()).encode("ascii")
    ).hexdigest()
    if corpus_hash != metadata.get("corpus_hash"):
        raise ValueError("corpus hash mismatch")
    return corpus


def map_strings(object_map, color_map, state_map) -> tuple[str, str]:
    object_map = np.asarray(object_map)
    color_map = np.asarray(color_map)
    state_map = np.asarray(state_map)
    if not (object_map.shape == color_map.shape == state_map.shape):
        raise ValueError("object/color/state map shapes must match")
    object_chars = {
        OBJECT_TO_IDX["unseen"]: "E", OBJECT_TO_IDX["empty"]: "E",
        OBJECT_TO_IDX["wall"]: "W", OBJECT_TO_IDX["floor"]: "F",
        OBJECT_TO_IDX["key"]: "K", OBJECT_TO_IDX["ball"]: "B",
        OBJECT_TO_IDX["box"]: "X", OBJECT_TO_IDX["goal"]: "G",
        OBJECT_TO_IDX["lava"]: "L", OBJECT_TO_IDX["agent"]: "S",
    }
    color_chars = {"red": "R", "green": "G", "blue": "B", "purple": "M", "yellow": "Y", "grey": "W"}
    layout_rows, color_rows = [], []
    for y in range(object_map.shape[0]):
        layout_row, color_row = [], []
        for x in range(object_map.shape[1]):
            object_id = int(object_map[y, x])
            if object_id == OBJECT_TO_IDX["door"]:
                layout_row.append("D" if int(state_map[y, x]) == STATE_TO_IDX["locked"] else "O")
            else:
                if object_id not in object_chars:
                    raise ValueError(f"unsupported MiniGrid object id {object_id}")
                layout_row.append(object_chars[object_id])
            color_name = IDX_TO_COLOR[int(color_map[y, x])]
            color_row.append(color_chars[color_name])
        layout_rows.append("".join(layout_row))
        color_rows.append("".join(color_row))
    return "\n".join(layout_rows), "\n".join(color_rows)


def make_env(corpus: MiniGridCorpus, index: int, max_steps: int):
    layout, colors = map_strings(
        corpus.object_maps[index], corpus.color_maps[index], corpus.state_maps[index]
    )
    token = int(corpus.inventory_tokens[index])
    initial_color = None if token == 0 else IDX_TO_COLOR[token - 1]
    return FullyObsWrapper(
        CustomMiniGridEnv(
            layout_str=layout,
            color_str=colors,
            agent_start_dir=int(corpus.start_dirs[index]),
            initial_carrying_key_color=initial_color,
            max_steps=int(max_steps),
        )
    )


def reachable_positions(object_map, color_map, state_map, inventory_token=0):
    """Return spatial cells reachable under MiniGrid key/door semantics.

    Normal doors can always be toggled. A locked door becomes traversable only
    after a reachable key of the matching colour is available. Goal and lava
    cells are included in coverage but are terminal and never expanded through.
    """
    object_map = np.asarray(object_map)
    color_map = np.asarray(color_map)
    state_map = np.asarray(state_map)
    starts = np.argwhere(object_map == OBJECT_TO_IDX["agent"])
    if len(starts) != 1:
        raise ValueError(f"expected exactly one start marker, found {len(starts)}")
    start = tuple(int(value) for value in starts[0])
    owned_colors = set()
    if int(inventory_token) > 0:
        owned_colors.add(int(inventory_token) - 1)
    terminal_ids = {OBJECT_TO_IDX["goal"], OBJECT_TO_IDX["lava"]}
    blocking_ids = {OBJECT_TO_IDX["wall"]}

    previous = None
    reached = set()
    while previous != (frozenset(reached), frozenset(owned_colors)):
        previous = (frozenset(reached), frozenset(owned_colors))
        queue = [start]
        reached = {start}
        for y, x in queue:
            if int(object_map[y, x]) in terminal_ids and (y, x) != start:
                continue
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if not (0 <= ny < object_map.shape[0] and 0 <= nx < object_map.shape[1]):
                    continue
                position = (ny, nx)
                if position in reached:
                    continue
                object_id = int(object_map[ny, nx])
                if object_id in blocking_ids:
                    continue
                if (
                    object_id == OBJECT_TO_IDX["door"]
                    and int(state_map[ny, nx]) == STATE_TO_IDX["locked"]
                    and int(color_map[ny, nx]) not in owned_colors
                ):
                    continue
                reached.add(position)
                queue.append(position)
        for y, x in reached:
            if int(object_map[y, x]) == OBJECT_TO_IDX["key"]:
                owned_colors.add(int(color_map[y, x]))
    return frozenset(reached)
