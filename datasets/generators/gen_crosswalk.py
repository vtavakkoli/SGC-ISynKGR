from __future__ import annotations

import json
import os
from pathlib import Path

SIGNALS = ("pressure", "temperature", "flow", "speed", "vibration", "current", "voltage", "state")


def _count() -> int:
    return int(os.getenv("DATASET_SYNTHETIC_COUNT", "1200"))


def main() -> None:
    root = Path("datasets/v1/crosswalk")
    root.mkdir(parents=True, exist_ok=True)
    gt = root / "gt_mappings.jsonl"
    lines = []
    for i in range(_count()):
        signal = SIGNALS[i % len(SIGNALS)]
        lines.append(
            json.dumps(
                {
                    # Source/target are normalized later per directed pair.
                    # Keep the source at the variable node instead of the
                    # equipment object to avoid rewarding shortcut mappings.
                    "source_path": f"opcua://ns=2;i={2000+i}",
                    "target_path": f"aas://asset-{i}/submodel/default/element/{signal}/value",
                    "mapping_type": "equivalent",
                    "transform": None,
                    "confidence": 1.0,
                    "rationale": "Synthetic deterministic ground-truth mapping.",
                    "evidence": ["generator:gen_crosswalk"],
                }
            )
        )
    gt.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
