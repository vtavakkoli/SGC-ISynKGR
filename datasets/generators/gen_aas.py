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


def _count() -> int:
    return int(os.getenv("DATASET_SYNTHETIC_COUNT", "1200"))


def _signal(i: int) -> tuple[str, str, str]:
    return SIGNALS[i % len(SIGNALS)]


def make_aas(i: int) -> dict:
    signal, value_type, unit = _signal(i)
    value = "RUNNING" if value_type == "string" else str(10 + i)
    element = {
        "idShort": signal,
        "modelType": "Property",
        "valueType": value_type,
        "value": value,
        "description": [{"language": "en", "text": f"{signal} measurement for asset-{i}"}],
    }
    if unit:
        element["unit"] = unit
    return {
        "assetAdministrationShells": [{"id": f"aas-{i}", "idShort": f"PumpAAS{i}", "submodels": [{"keys": [{"value": f"sm-{i}"}]}]}],
        "submodels": [{"id": f"sm-{i}", "idShort": f"{signal.capitalize()}Telemetry{i}", "submodelElements": [element]}],
    }


def main() -> None:
    root = Path("datasets/v1/aas/synthetic")
    root.mkdir(parents=True, exist_ok=True)
    for i in range(_count()):
        (root / f"aas_{i:03d}.json").write_text(json.dumps(make_aas(i), indent=2))


if __name__ == "__main__":
    main()
