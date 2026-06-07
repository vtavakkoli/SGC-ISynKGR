from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetRequirements:
    opcua_synthetic: int = int(os.getenv("DATASET_SYNTHETIC_COUNT", "1200"))
    aas_synthetic: int = int(os.getenv("DATASET_SYNTHETIC_COUNT", "1200"))
    crosswalk_rows: int = int(os.getenv("DATASET_SYNTHETIC_COUNT", "1200"))
    opcua_semi_real: int = int(os.getenv("DATASET_SEMI_REAL_COUNT", "120"))
    aas_semi_real: int = int(os.getenv("DATASET_SEMI_REAL_COUNT", "120"))


def _default_root() -> Path:
    return Path(os.getenv("DATASET_ROOT", "datasets/v1"))


def _count_files(path: Path, suffix: str) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob(f"*.{suffix}")))


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _generator_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["DATASET_ROOT"] = str(root)
    return env


def validate_or_generate(root: Path | None = None, req: DatasetRequirements | None = None) -> dict[str, int]:
    root = root or _default_root()
    req = req or DatasetRequirements()
    print(f"[DATASET] stage=scan status=start root={root}", flush=True)
    counts = {
        "opcua_synthetic": _count_files(root / "opcua" / "synthetic", "xml"),
        "aas_synthetic": _count_files(root / "aas" / "synthetic", "json"),
        "crosswalk_rows": _count_jsonl_rows(root / "crosswalk" / "gt_mappings.jsonl"),
        "opcua_semi_real": _count_files(root / "opcua" / "semi_real", "xml"),
        "aas_semi_real": _count_files(root / "aas" / "semi_real", "json"),
    }
    print(f"[DATASET] stage=scan status=done counts={counts}", flush=True)

    needs_synthetic = counts["opcua_synthetic"] < req.opcua_synthetic or counts["aas_synthetic"] < req.aas_synthetic
    if needs_synthetic:
        print("[DATASET] stage=generate status=start targets=synthetic", flush=True)
        env = _generator_env(root)
        subprocess.run(["python", "datasets/generators/gen_opcua.py"], check=True, env=env)
        subprocess.run(["python", "datasets/generators/gen_aas.py"], check=True, env=env)
        counts["opcua_synthetic"] = _count_files(root / "opcua" / "synthetic", "xml")
        counts["aas_synthetic"] = _count_files(root / "aas" / "synthetic", "json")
        print(
            f"[DATASET] stage=generate status=done counts={{'opcua_synthetic': {counts['opcua_synthetic']}, 'aas_synthetic': {counts['aas_synthetic']}}}",
            flush=True,
        )

    if counts["crosswalk_rows"] < req.crosswalk_rows:
        print("[DATASET] stage=generate status=start targets=crosswalk", flush=True)
        subprocess.run(["python", "datasets/generators/gen_crosswalk.py"], check=True, env=_generator_env(root))
        counts["crosswalk_rows"] = _count_jsonl_rows(root / "crosswalk" / "gt_mappings.jsonl")
        print(f"[DATASET] stage=generate status=done counts={{'crosswalk_rows': {counts['crosswalk_rows']}}}", flush=True)

    if counts["opcua_semi_real"] < req.opcua_semi_real or counts["aas_semi_real"] < req.aas_semi_real:
        print("[DATASET] stage=generate status=start targets=semi_real", flush=True)
        opcua_semi = root / "opcua" / "semi_real"
        aas_semi = root / "aas" / "semi_real"
        opcua_semi.mkdir(parents=True, exist_ok=True)
        aas_semi.mkdir(parents=True, exist_ok=True)
        for i in range(req.opcua_semi_real):
            src = root / "opcua" / "synthetic" / f"opcua_{i:03d}.xml"
            dst = opcua_semi / f"example_{i:03d}.xml"
            if src.exists() and not dst.exists():
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        for i in range(req.aas_semi_real):
            src = root / "aas" / "synthetic" / f"aas_{i:03d}.json"
            dst = aas_semi / f"example_{i:03d}.json"
            if src.exists() and not dst.exists():
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        counts["opcua_semi_real"] = _count_files(root / "opcua" / "semi_real", "xml")
        counts["aas_semi_real"] = _count_files(root / "aas" / "semi_real", "json")
        print(
            f"[DATASET] stage=generate status=done counts={{'opcua_semi_real': {counts['opcua_semi_real']}, 'aas_semi_real': {counts['aas_semi_real']}}}",
            flush=True,
        )

    print(f"[DATASET] stage=complete status=done counts={counts}", flush=True)
    return counts


if __name__ == "__main__":
    final_counts = validate_or_generate()
    print("dataset-validation", final_counts)
