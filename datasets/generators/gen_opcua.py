from __future__ import annotations

import os
from pathlib import Path

SIGNALS = (
    ("pressure", "FLOAT", "bar"),
    ("temperature", "FLOAT", "C"),
    ("flow", "FLOAT", "l/s"),
    ("speed", "FLOAT", "rpm"),
    ("vibration", "FLOAT", "mm/s"),
    ("current", "FLOAT", "A"),
    ("voltage", "FLOAT", "V"),
    ("state", "STRING", ""),
)


def _count() -> int:
    return int(os.getenv("DATASET_SYNTHETIC_COUNT", "1200"))


def _signal(i: int) -> tuple[str, str, str]:
    return SIGNALS[i % len(SIGNALS)]


def make_nodeset(i: int) -> str:
    signal, dtype, unit = _signal(i)
    signal_name = f"{signal.capitalize()}{i}"
    engineering_unit = f' Unit="{unit}"' if unit else ""
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<UANodeSet xmlns="http://opcfoundation.org/UA/2011/03/UANodeSet.xsd">\n'
        f'  <UAObjectType NodeId="ns=2;i={1000+i}" BrowseName="2:Pump{i}">\n'
        f'    <DisplayName>Pump{i}</DisplayName>\n'
        f'    <References><Reference ReferenceType="HasComponent">ns=2;i={2000+i}</Reference></References>\n'
        '  </UAObjectType>\n'
        f'  <UAVariable NodeId="ns=2;i={2000+i}" BrowseName="2:{signal_name}" DataType="{dtype}"{engineering_unit}>\n'
        f'    <DisplayName>{signal_name}</DisplayName>\n'
        f'    <Description>{signal} measurement for asset-{i}</Description>\n'
        '  </UAVariable>\n'
        '</UANodeSet>'
    )


def main() -> None:
    root = Path("datasets/v1/opcua/synthetic")
    root.mkdir(parents=True, exist_ok=True)
    for i in range(_count()):
        (root / f"opcua_{i:03d}.xml").write_text(make_nodeset(i))


if __name__ == "__main__":
    main()
