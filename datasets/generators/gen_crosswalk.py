from __future__ import annotations

import json
import os
from pathlib import Path

SIGNALS = ("pressure", "temperature", "flow", "speed", "vibration", "current", "voltage", "state")


def _dataset_root() -> Path:
    return Path(os.getenv("DATASET_ROOT", "datasets/v1"))


def _count() -> int:
    return int(os.getenv("DATASET_SYNTHETIC_COUNT", "1200"))


def main() -> None:
    root = _dataset_root() / "crosswalk"
    root.mkdir(parents=True, exist_ok=True)
    gt = root / "gt_mappings.jsonl"
    lines = []
    for i in range(_count()):
        signal = SIGNALS[i % len(SIGNALS)]
        lines.append(
            json.dumps(
                {
                    "source_path": f"opcua://ns=2;s={signal.capitalize()}{i}",
                    "target_path": f"aas://asset-{i}/submodel/default/element/{signal}/value",
                    "mapping_type": "equivalent",
                    "transform": None,
                    "confidence": 1.0,
                    "rationale": "Deterministic synthetic ground-truth mapping with signal, unit, and instance alignment.",
                    "evidence": ["generator:gen_crosswalk", f"signal:{signal}", f"instance:{i}"],
                }
            )
        )
    gt.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
