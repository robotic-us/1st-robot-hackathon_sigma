#!/usr/bin/env python3
"""The slot table: our description of what each pre-recorded motion does.

Split out of decider.py so the live pipeline (care.py) can read the table
without dragging in the circle-stimulus prototype behind it.

WHY THIS EXISTS: slot internals cannot be read back from the robot. ``phorce
list`` gives you a slot's id and name and nothing else -- not its direction, not
its size, not the pose it starts from. So we describe them here, in slots.json,
and the decision layer picks a slot by matching that description.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum

from phorce_iface import MAX_MOTION_ID, MIN_MOTION_ID

LOGGER = logging.getLogger("slot_table")


class Direction(str, Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class Amplitude(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    SETTLE = "settle"   # not produced by bucketing; only the wind-down uses it


# Where each amplitude bucket sits on the 0..1 intensity scale. Used when
# scoring "how well does this slot's size match what we're seeing".
AMPLITUDE_VALUE: dict[Amplitude, float] = {
    Amplitude.SMALL: 0.15,
    Amplitude.MEDIUM: 0.45,
    Amplitude.LARGE: 0.85,
    Amplitude.SETTLE: 0.0,
}


@dataclass(frozen=True)
class Slot:
    """Our description of one pre-recorded motion (see slots.json)."""

    slot_id: int
    desc: str
    direction: Direction
    amplitude: Amplitude
    start_pose: tuple[float, ...]
    end_pose: tuple[float, ...]
    placeholder: bool = False


@dataclass(frozen=True)
class SlotTable:
    axes: tuple[int, ...]
    slots: dict[int, Slot]

    def cell(self, direction: Direction, amplitude: Amplitude) -> list[Slot]:
        """Every slot in one grid cell (usually one, but more is fine)."""
        return [
            s for s in self.slots.values()
            if s.direction is direction and s.amplitude is amplitude
        ]

    def with_amplitude(self, amplitude: Amplitude) -> list[Slot]:
        return [s for s in self.slots.values() if s.amplitude is amplitude]


def load_slot_table(path: str) -> SlotTable:
    """Read and validate slots.json.

    Validation is strict and happens at startup: a typo here would otherwise
    surface as a mysterious "no candidates" at demo time.
    """
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)

    axes = tuple(int(a) for a in raw.get("axes", (0, 1, 2, 3)))
    slots: dict[int, Slot] = {}
    placeholders: list[int] = []

    for key, entry in raw.get("slots", {}).items():
        try:
            slot_id = int(key)
        except ValueError as exc:
            raise ValueError(f"slot key {key!r} is not an integer") from exc
        if not (MIN_MOTION_ID <= slot_id <= MAX_MOTION_ID):
            raise ValueError(
                f"slot {slot_id} is outside the contract range "
                f"{MIN_MOTION_ID}..{MAX_MOTION_ID} (0 is the no-motion sentinel)"
            )

        try:
            direction = Direction(entry["direction"])
            amplitude = Amplitude(entry["amplitude"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"slot {slot_id}: bad direction/amplitude ({exc})") from exc

        start = tuple(float(v) for v in entry.get("start_pose", ()))
        end = tuple(float(v) for v in entry.get("end_pose", ()))
        for name, pose in (("start_pose", start), ("end_pose", end)):
            if len(pose) != len(axes):
                raise ValueError(
                    f"slot {slot_id}: {name} has {len(pose)} values but "
                    f"axes has {len(axes)}"
                )

        slot = Slot(
            slot_id=slot_id,
            desc=str(entry.get("desc", "")),
            direction=direction,
            amplitude=amplitude,
            start_pose=start,
            end_pose=end,
            placeholder=bool(entry.get("placeholder", False)),
        )
        slots[slot_id] = slot
        if slot.placeholder:
            placeholders.append(slot_id)

    if not slots:
        raise ValueError(f"{path} defines no slots")

    LOGGER.info("loaded %d slots from %s (axes=%s)", len(slots), path, list(axes))
    if placeholders:
        LOGGER.warning(
            "slots %s still have PLACEHOLDER poses -- start_pose matching is "
            "guesswork until you teach the motions and fill them in",
            sorted(placeholders),
        )
    return SlotTable(axes=axes, slots=slots)
