from __future__ import annotations

import json
import os
from pathlib import Path

SIGNALS = (
    ("pressure", "double", "bar"),
    ("temperature", "double", "C"),
    ("flow", "double", "l/s"),
    ("speed", "double", "rpm"),
    ("vibration", "double", "mm/s"),
    ("current", "double", "A"),
    ("voltage", "double", "V"),
    ("state", "string", ""),
)


def _dataset_root() -> Path:
    return Path(os.getenv("DATASET_ROOT", "datasets/v1"))


def _count() -> int:
    return int(os.getenv("DATASET_SYNTHETIC_COUNT", "1200"))


def _signal(i: int) -> tuple[str, str, str]:
    return SIGNALS[i % len(SIGNALS)]


def make_aas(i: int) -> dict:
    signal, value_type, unit = _signal(i)
    value = "ON" if signal == "state" else str(10 + (i % 90))
    elem = {
        "idShort": signal,
        "modelType": "Property",
        "valueType": value_type,
        "value": value,
        "description": f"{signal} measurement for asset-{i}",
    }
    if unit:
        elem["unit"] = unit
    return {
        "assetAdministrationShells": [
            {"id": f"aas-{i}", "idShort": f"AssetShell{i}", "submodels": [{"keys": [{"value": f"sm-{i}"}]}]}
        ],
        "submodels": [
            {"id": f"sm-{i}", "idShort": f"Telemetry{i}", "submodelElements": [elem]}
        ],
    }


def main() -> None:
    root = _dataset_root() / "aas" / "synthetic"
    root.mkdir(parents=True, exist_ok=True)
    for i in range(_count()):
        (root / f"aas_{i:03d}.json").write_text(json.dumps(make_aas(i), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
