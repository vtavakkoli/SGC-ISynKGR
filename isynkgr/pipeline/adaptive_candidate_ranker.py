from __future__ import annotations

import difflib
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from isynkgr.adapters.aas import AASAdapter
from isynkgr.adapters.iec61499 import IEC61499Adapter
from isynkgr.adapters.ieee1451 import IEEE1451Adapter
from isynkgr.adapters.iso15926 import ISO15926Adapter
from isynkgr.adapters.opcua import OPCUAAdapter
from isynkgr.canonical.model import CanonicalModel, CanonicalNode
from isynkgr.canonical.schemas import EvidenceItem, Mapping, Provenance, TranslationResult
from isynkgr.icr.entities import build_endpoint_path, normalize_path
from isynkgr.icr.mapping_output_contract import normalize_mapping_item
from isynkgr.icr.mapping_schema import MappingType
from isynkgr.llm.ollama import OllamaClient
from isynkgr.pipeline.prompting import build_mapping_prompt
from isynkgr.retrieval.graphrag import GraphRAGRetriever
from isynkgr.rules.engine import RuleEngine
from isynkgr.utils.hashing import stable_hash

Mode = Literal[
    "adaptive_candidate_ranker",
    "hybrid",  # deprecated alias
    "llm_only",
    "rag_only",
    "rule_only",
    "graph_only",
    "embedding_only",
    "semantic_graph_calibrated",
]


class TranslatorConfig:
    def __init__(
        self,
        model_name: str = "gemma4:e2b",
        seed: int = 42,
        max_repair_iterations: int = 2,
        enable_vector_retrieval: bool = False,
        component_flags: dict[str, bool] | None = None,
    ) -> None:
        self.model_name = model_name
        self.seed = seed
        self.max_repair_iterations = max_repair_iterations
        self.enable_vector_retrieval = enable_vector_retrieval
        self.component_flags = component_flags or {}


ADAPTERS = {"opcua": OPCUAAdapter(), "aas": AASAdapter(), "iec61499": IEC61499Adapter(), "ieee1451": IEEE1451Adapter(), "iso15926": ISO15926Adapter()}
DEFAULT_RETRIEVAL_CONFIDENCE_THRESHOLD = 0.62


def _float_setting(name: str, default: float) -> float:
    """Read a numeric setting from the environment with safe fallback.

    Benchmark thresholds must be easy to tune without editing code.  Invalid
    values should not crash a long benchmark run; they fall back to the
    version-controlled default.
    """

    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


_NUMERIC_DTYPES = {"FLOAT", "DOUBLE", "DECIMAL", "INT", "INTEGER", "NUMBER", "XS:DOUBLE", "XS:FLOAT", "XS:DECIMAL", "XS:INT"}


def _git_commit() -> str:
    head = Path(".git/HEAD")
    if not head.exists():
        return "unknown"
    ref = head.read_text().strip()
    if ref.startswith("ref:"):
        p = Path(".git") / ref.split(" ", 1)[1]
        if p.exists():
            return p.read_text().strip()[:12]
    return ref[:12]


def _schema_summary(model: CanonicalModel) -> dict[str, Any]:
    return {
        "standard": model.standard,
        "node_count": len(model.nodes),
        "edge_count": len(model.edges),
        "namespaces": list(model.namespaces.keys())[:10],
    }


def _build_default_target_model(target_standard: str) -> CanonicalModel:
    std = target_standard.lower()
    if std == "aas":
        labels = ["temperature", "pressure", "flow", "state", "speed", "vibration"]
        nodes = [
            CanonicalNode(id=f"aas://asset-0/submodel/default/element/{label}/value", type="Property", label=label, attributes={"datatype": "FLOAT" if label != "state" else "STRING"})
            for label in labels
        ]
        return CanonicalModel(standard=std, nodes=nodes, edges=[])
    if std == "iec61499":
        nodes = [
            CanonicalNode(id="iec61499://Device0/Res1/FB1/temp", type="Signal", label="temperature", attributes={"dtype": "FLOAT", "unit": "C"}),
            CanonicalNode(id="iec61499://Device0/Res1/FB1/pressure", type="Signal", label="pressure", attributes={"dtype": "FLOAT", "unit": "bar"}),
            CanonicalNode(id="iec61499://Device0/Res1/FB1/state", type="Signal", label="state", attributes={"dtype": "STRING"}),
        ]
        return CanonicalModel(standard=std, nodes=nodes, edges=[])
    if std == "ieee1451":
        nodes = [
            CanonicalNode(id="ieee1451://teds0/ch0/value", type="Channel", label="temperature", attributes={"dtype": "FLOAT", "unit": "C"}),
            CanonicalNode(id="ieee1451://teds0/ch1/value", type="Channel", label="pressure", attributes={"dtype": "FLOAT", "unit": "bar"}),
        ]
        return CanonicalModel(standard=std, nodes=nodes, edges=[])
    if std == "opcua":
        nodes = [
            CanonicalNode(id="opcua://ns=2;s=Temperature", type="UAVariable", label="temperature", attributes={"datatype": "FLOAT", "unit": "C"}),
            CanonicalNode(id="opcua://ns=2;s=Pressure", type="UAVariable", label="pressure", attributes={"datatype": "FLOAT", "unit": "bar"}),
        ]
        return CanonicalModel(standard=std, nodes=nodes, edges=[])
    return CanonicalModel(standard=std, nodes=[CanonicalNode(id=f"{std}://candidate/default/value", type="Candidate", label="value", attributes={})], edges=[])


def _guess_label_from_path(path: str) -> str:
    raw = str(path or "").strip().rstrip("/")
    if not raw:
        return "candidate"
    tail = raw.split("/")[-1]
    if tail.lower() in {"value", "asset"} and len(raw.split("/")) >= 2:
        tail = raw.split("/")[-2]
    return tail or "candidate"


def _semantic_hint_from_path(path: str) -> dict[str, str]:
    raw = str(path or "").strip().rstrip("/")
    tokens = [t for t in raw.split("/") if t]
    lowered = [t.lower() for t in tokens]
    raw_lower = raw.lower()
    signal = ""
    for token in reversed(tokens):
        lt = token.lower()
        if lt in {"value", "values", "measurement"}:
            continue
        if lt.startswith("value_"):
            continue
        signal = token
        break
    if not signal and tokens:
        signal = tokens[-1]
    unit = ""
    datatype = ""
    if any(k in raw_lower for k in {"temperature", "temp"}):
        unit = "C"
        datatype = "FLOAT"
    elif "pressure" in raw_lower:
        unit = "bar"
        datatype = "FLOAT"
    elif "flow" in raw_lower:
        unit = "l/s"
        datatype = "FLOAT"
    elif "speed" in raw_lower:
        unit = "rpm"
        datatype = "FLOAT"
    elif "vibration" in raw_lower:
        unit = "mm/s"
        datatype = "FLOAT"
    elif "current" in raw_lower:
        unit = "A"
        datatype = "FLOAT"
    elif "voltage" in raw_lower:
        unit = "V"
        datatype = "FLOAT"
    elif "state" in raw_lower or "status" in raw_lower:
        datatype = "STRING"
    elif re.search(r"(?i)(?:^|[/;=_-])channel[-_ ]?0(?:/|$)", raw):
        datatype = "FLOAT"
    elif re.search(r"(?i)(?:^|[/;=_-])channel[-_ ]?1(?:/|$)", raw):
        datatype = "STRING"
    return {"label": signal or "candidate", "unit": unit, "datatype": datatype}


def _context_hints_from_path(path: str) -> dict[str, str]:
    raw = str(path or "").strip().rstrip("/")
    tokens = [t for t in raw.split("/") if t]
    lower_tokens = [t.lower() for t in tokens]
    asset_id = next((t for t in tokens if re.match(r"(?i)^asset[-_ ]?\d+$", t)), "")
    equipment_id = next((t for t in tokens if re.match(r"(?i)^(pump|motor|line|machine|equipment)[-_ ]?\d+$", t)), "")
    process_line = next((t for t in tokens if re.match(r"(?i)^(line|resource|res|device)[-_ ]?\d+$", t)), "")
    parent_path = "/".join(tokens[:-1]) if len(tokens) > 1 else ""
    benchmark_entity_id = asset_id or equipment_id or (tokens[1] if len(tokens) > 2 else "")
    return {
        "asset_id": asset_id,
        "equipment_id": equipment_id,
        "process_line": process_line,
        "parent_path": parent_path,
        "benchmark_entity_id": benchmark_entity_id,
        "context_tokens": " ".join(lower_tokens),
    }


def _build_target_model_from_candidates(target_standard: str, target_candidates: list[str]) -> CanonicalModel:
    std = target_standard.lower()
    cleaned: list[str] = []
    seen: set[str] = set()
    for candidate in target_candidates:
        norm = normalize_path(str(candidate or "").strip())
        if not norm or norm in seen:
            continue
        seen.add(norm)
        cleaned.append(norm)
    if not cleaned:
        return _build_default_target_model(target_standard)
    nodes: list[CanonicalNode] = []
    for path in cleaned:
        hints = _semantic_hint_from_path(path)
        context = _context_hints_from_path(path)
        nodes.append(
            CanonicalNode(
                id=path,
                type="Candidate",
                label=hints["label"] or _guess_label_from_path(path),
                attributes={
                    "datatype": hints["datatype"],
                    "unit": hints["unit"],
                    "description": f"candidate derived from path {path}",
                    **context,
                },
            )
        )
    return CanonicalModel(standard=std, nodes=nodes, edges=[])


def _canonical_source_path(source_standard: str, raw_path: str) -> str:
    return normalize_path(raw_path if "://" in str(raw_path or "") else build_endpoint_path(source_standard, str(raw_path or "")))


def _lexical_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=str(a or "").lower(), b=str(b or "").lower()).ratio()


def _guess_dtype(node: CanonicalNode) -> str:
    attrs = node.attributes or {}
    meta = attrs.get("metadata", {}) if isinstance(attrs.get("metadata", {}), dict) else {}
    for key in ("datatype", "dtype", "valueType", "dataType", "type"):
        value = attrs.get(key) or meta.get(key)
        if value:
            raw = str(value).upper()
            if raw in _NUMERIC_DTYPES:
                return "FLOAT"
            if raw in {"BOOL", "BOOLEAN"}:
                return "BOOL"
            if raw in {"STRING", "TEXT"}:
                return "STRING"
            return raw
    return ""


def _dtype_compatible(source_dtype: str, target_dtype: str) -> float:
    src = str(source_dtype or "").upper()
    tgt = str(target_dtype or "").upper()
    if src and tgt and src == tgt:
        return 1.0
    if src in _NUMERIC_DTYPES | {"FLOAT"} and tgt in _NUMERIC_DTYPES | {"FLOAT"}:
        return 1.0
    if not src or not tgt:
        return 0.5
    return 0.0


def _guess_unit(node: CanonicalNode) -> str:
    attrs = node.attributes or {}
    meta = attrs.get("metadata", {}) if isinstance(attrs.get("metadata", {}), dict) else {}
    return str(attrs.get("unit") or meta.get("unit") or "").strip().lower()


def _parent_map(model: CanonicalModel) -> dict[str, str]:
    out: dict[str, str] = {}
    for edge in model.edges:
        if edge.target not in out:
            out[edge.target] = edge.source
    return out


def _semantic_signature(node: CanonicalNode | None) -> str:
    if node is None:
        return ""
    return "|".join(
        [
            str(node.label or "").strip().lower(),
            _guess_dtype(node).lower(),
            _guess_unit(node).lower(),
        ]
    )


def _path_tokens(path: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z0-9_-]+", str(path or "")) if t}


def _is_measurement_like_node(node: CanonicalNode) -> bool:
    node_type = str(node.type or "").strip().lower()
    text = " ".join([str(node.label or ""), str(node.id or ""), node_type]).lower()
    measurement_node_types = {"channel", "signal", "property", "sensor", "variable", "uavariable", "output", "input"}
    measurement_terms = {
        "measurement",
        "measure",
        "value",
        "temp",
        "temperature",
        "pressure",
        "flow",
        "speed",
        "current",
        "voltage",
        "vibration",
        "state",
        "status",
        "setpoint",
    }
    if node_type in measurement_node_types:
        return True
    if _guess_dtype(node) or _guess_unit(node):
        return True
    return any(term in text for term in measurement_terms)


def _is_equipment_label_without_measurement(node: CanonicalNode) -> bool:
    node_type = str(node.type or "").strip().lower()
    if _is_measurement_like_node(node):
        return False

    label = " ".join(
        [
            str(node.label or ""),
            str(node.id or ""),
            node_type,
        ]
    ).lower()
    structural_types = {"asset", "submodel", "resource", "functionblock", "device", "object", "teds"}
    equipment_terms = {
        "pump",
        "motor",
        "line",
        "device",
        "equipment",
        "machine",
        "asset",
        "submodel",
        "resource",
        "functionblock",
        "fb",
        "teds",
    }
    has_equipment = any(term in label for term in equipment_terms) or node_type in structural_types
    return has_equipment


_SEMANTIC_ALIASES: dict[str, str] = {
    "temp": "temperature",
    "temperature": "temperature",
    "press": "pressure",
    "pressure": "pressure",
    "flowrate": "flow",
    "flow": "flow",
    "rpm": "speed",
    "speed": "speed",
    "velocity": "speed",
    "vib": "vibration",
    "vibration": "vibration",
    "status": "state",
    "state": "state",
    "bool": "boolean",
    "boolean": "boolean",
    "float": "number",
    "double": "number",
    "int": "number",
    "integer": "number",
    "string": "text",
}

_STOP_TOKENS = {
    "value", "values", "measurement", "measurements", "default", "submodel", "element",
    "elements", "asset", "device", "resource", "fb", "node", "ns", "s", "i", "id",
}

_SIGNAL_TERMS = {"temperature", "pressure", "flow", "speed", "state", "vibration"}


def _split_semantic_tokens(*values: str) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        for raw in re.findall(r"[A-Za-z]+|[0-9]+", str(value or "").replace("_", " ").replace("-", " ")):
            token = raw.lower().strip()
            if not token or token in _STOP_TOKENS or token.isdigit():
                continue
            tokens.add(_SEMANTIC_ALIASES.get(token, token))
    return tokens


def _semantic_instance_ids(*values: str) -> set[str]:
    """Extract instance ids attached to semantic labels, not protocol internals."""

    ids: set[str] = set()
    pattern = re.compile(
        r"(?i)\b(?:pump|motor|pressure|temperature|temp|flow|speed|vibration|state|status|class|signal|channel)[-_ ]?(\d+)\b"
    )
    for value in values:
        text = str(value or "")
        for match in pattern.finditer(text):
            ids.add(str(int(match.group(1))))
        for match in re.finditer(r"(?i)(?:^|[;/])s=([A-Za-z]+)(\d+)\b", text):
            label = match.group(1).lower()
            if label in {"pressure", "temperature", "temp", "flow", "speed", "vibration", "state", "status", "pump", "motor"}:
                ids.add(str(int(match.group(2))))
    return ids


def _context_instance_ids(*values: str) -> set[str]:
    """Extract asset/submodel/equipment ids used as context for contained measurements."""

    ids: set[str] = set()
    pattern = re.compile(r"(?i)\b(?:asset|aas|sm|submodel|machine|equipment|device)[-_ ]?(\d+)\b")
    for value in values:
        for match in pattern.finditer(str(value or "")):
            ids.add(str(int(match.group(1))))
    return ids


def _instance_ids(*values: str) -> set[str]:
    return _semantic_instance_ids(*values) | _context_instance_ids(*values)


def _instance_alignment(source_node: CanonicalNode, target_node: CanonicalNode, source_path: str, target_path: str, source_parent: str, target_parent: str) -> float:
    src_text = _node_text(source_node)
    tgt_text = _node_text(target_node)
    src_semantic_ids = _semantic_instance_ids(source_path, source_node.id, source_node.label or "", src_text)
    tgt_semantic_ids = _semantic_instance_ids(target_path, target_node.id, target_node.label or "", tgt_text)
    if src_semantic_ids and tgt_semantic_ids:
        return 1.0 if src_semantic_ids & tgt_semantic_ids else 0.0

    src_context_ids = _context_instance_ids(source_path, source_parent, source_node.id, source_node.label or "", src_text)
    tgt_context_ids = _context_instance_ids(target_path, target_parent, target_node.id, target_node.label or "", tgt_text)
    src_signal = _split_semantic_tokens(src_text, source_path) & _SIGNAL_TERMS
    tgt_signal = _split_semantic_tokens(tgt_text, target_path) & _SIGNAL_TERMS
    same_known_signal = bool(src_signal and tgt_signal and src_signal == tgt_signal)

    if src_context_ids and tgt_semantic_ids and same_known_signal:
        return 1.0 if src_context_ids & tgt_semantic_ids else 0.0
    if src_semantic_ids and tgt_context_ids and same_known_signal:
        return 1.0 if src_semantic_ids & tgt_context_ids else 0.0
    if src_context_ids and tgt_context_ids and same_known_signal:
        return 1.0 if src_context_ids & tgt_context_ids else 0.0
    # Neutral when one side has no semantic instance id; this avoids penalising
    # generic IEEE 1451 channel targets such as Channel0.
    return 0.5


def _node_text(node: CanonicalNode | None) -> str:
    if node is None:
        return ""
    attrs = node.attributes or {}
    meta = attrs.get("metadata", {}) if isinstance(attrs.get("metadata", {}), dict) else {}
    fields = [node.id, node.type, node.label or ""]
    for key in ("description", "datatype", "dtype", "valueType", "unit", "asset_id", "equipment_id", "process_line", "parent_path", "context_tokens"):
        value = attrs.get(key) or meta.get(key)
        if value:
            fields.append(str(value))
    return " ".join(fields)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.5
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _same_signal_family(source_node: CanonicalNode, target_node: CanonicalNode) -> float:
    source_tokens = _split_semantic_tokens(_node_text(source_node))
    target_tokens = _split_semantic_tokens(_node_text(target_node))
    signal_terms = {"temperature", "pressure", "flow", "speed", "state", "vibration"}
    src_signal = source_tokens & signal_terms
    tgt_signal = target_tokens & signal_terms
    if not src_signal or not tgt_signal:
        return 0.5
    return 1.0 if src_signal == tgt_signal else 0.0


def _has_conflicting_instance_ids(
    source_node: CanonicalNode,
    target_node: CanonicalNode,
    source_path: str,
    target_path: str,
    source_parent: str,
    target_parent: str,
) -> bool:
    src_text = _node_text(source_node)
    tgt_text = _node_text(target_node)
    src_semantic_ids = _semantic_instance_ids(source_path, source_node.id, source_node.label or "", src_text)
    tgt_semantic_ids = _semantic_instance_ids(target_path, target_node.id, target_node.label or "", tgt_text)
    if src_semantic_ids and tgt_semantic_ids:
        return not bool(src_semantic_ids & tgt_semantic_ids)

    src_context_ids = _context_instance_ids(source_path, source_parent, source_node.id, source_node.label or "", src_text)
    tgt_context_ids = _context_instance_ids(target_path, target_parent, target_node.id, target_node.label or "", tgt_text)
    src_signal = _split_semantic_tokens(src_text, source_path) & _SIGNAL_TERMS
    tgt_signal = _split_semantic_tokens(tgt_text, target_path) & _SIGNAL_TERMS
    same_known_signal = bool(src_signal and tgt_signal and src_signal == tgt_signal)
    if src_context_ids and tgt_semantic_ids and same_known_signal:
        return not bool(src_context_ids & tgt_semantic_ids)
    if src_semantic_ids and tgt_context_ids and same_known_signal:
        return not bool(src_semantic_ids & tgt_context_ids)
    if src_context_ids and tgt_context_ids and same_known_signal:
        return not bool(src_context_ids & tgt_context_ids)
    return False


def _logical_candidate_score(
    *,
    source_node: CanonicalNode,
    target_node: CanonicalNode,
    source_path: str,
    target_path: str,
    source_parent: str,
    target_parent: str,
    single_candidate: bool,
) -> tuple[float, dict[str, float]]:
    lexical = _lexical_similarity(source_node.label or source_node.id, target_node.label or target_node.id)
    datatype_compat = _dtype_compatible(_guess_dtype(source_node), _guess_dtype(target_node))
    source_unit = _guess_unit(source_node)
    target_unit = _guess_unit(target_node)
    unit_compat = 1.0 if source_unit and target_unit and source_unit == target_unit else (0.5 if not source_unit or not target_unit else 0.0)
    parent_sim = _lexical_similarity(source_parent, target_parent) if source_parent or target_parent else 0.5
    signal_family = _same_signal_family(source_node, target_node)
    instance_alignment = _instance_alignment(source_node, target_node, source_path, target_path, source_parent, target_parent)
    source_tokens = _path_tokens(source_path) | _path_tokens(source_parent)
    target_tokens = _path_tokens(target_path) | _path_tokens(target_parent)
    context_score = min(1.0, len(source_tokens & target_tokens) / 3.0) if target_tokens else 0.0
    measurement_bonus = 1.0 if _is_measurement_like_node(source_node) and _is_measurement_like_node(target_node) else 0.0
    structural_penalty = 0.28 if _is_equipment_label_without_measurement(source_node) or _is_equipment_label_without_measurement(target_node) else 0.0
    conflict_penalty = 0.22 if _has_conflicting_instance_ids(source_node, target_node, source_path, target_path, source_parent, target_parent) else 0.0
    single_candidate_bonus = 0.12 if single_candidate else 0.0

    score = (
        lexical * 0.14
        + datatype_compat * 0.18
        + unit_compat * 0.12
        + parent_sim * 0.05
        + context_score * 0.06
        + signal_family * 0.14
        + instance_alignment * 0.22
        + measurement_bonus * 0.07
        + single_candidate_bonus
        - structural_penalty
        - conflict_penalty
    )
    score = max(0.0, min(1.0, score))
    return score, {
        "logical_lexical_similarity": lexical,
        "logical_datatype_compatibility": datatype_compat,
        "logical_unit_compatibility": unit_compat,
        "logical_parent_similarity": parent_sim,
        "logical_path_context_similarity": context_score,
        "logical_signal_family": signal_family,
        "logical_instance_alignment": instance_alignment,
        "logical_measurement_bonus": measurement_bonus,
        "logical_single_candidate_bonus": single_candidate_bonus,
        "logical_structural_penalty": structural_penalty,
        "logical_instance_conflict_penalty": conflict_penalty,
    }


class SemanticGraphCalibrator:
    """Small dependency-free semantic matching model.

    The existing ranker uses a fixed additive score. This model uses normalized semantic
    features and a logistic confidence head, so scores are easier to threshold and compare
    across standards, noisy labels, and explicit no-match samples.
    """

    def score(
        self,
        *,
        source_node: CanonicalNode,
        target_node: CanonicalNode,
        source_path: str,
        target_path: str,
        source_parent: str,
        target_parent: str,
        retrieval_score: float,
        lexical_score: float,
        rule_hit: float,
        datatype_compat: float,
        unit_compat: float,
        support_count: int,
        duplicate_count: int,
        candidate_count: int,
    ) -> tuple[float, dict[str, float]]:
        source_label_tokens = _split_semantic_tokens(source_node.label or "", source_path)
        target_label_tokens = _split_semantic_tokens(target_node.label or "", target_path)
        source_context_tokens = _split_semantic_tokens(_node_text(source_node), source_parent)
        target_context_tokens = _split_semantic_tokens(_node_text(target_node), target_parent)

        label_jaccard = _jaccard(source_label_tokens, target_label_tokens)
        context_jaccard = _jaccard(source_context_tokens, target_context_tokens)
        parent_jaccard = _jaccard(_split_semantic_tokens(source_parent), _split_semantic_tokens(target_parent))
        signal_family = _same_signal_family(source_node, target_node)
        instance_alignment = _instance_alignment(source_node, target_node, source_path, target_path, source_parent, target_parent)
        multi_support = min(1.0, max(0.0, (support_count - 1) / 2.0))
        single_candidate = 1.0 if candidate_count == 1 else 0.0
        ambiguity_penalty = min(0.35, max(0, duplicate_count - 1) * 0.08)

        # Logistic confidence head. Bias is deliberately conservative to reduce false positives.
        logit = -1.05
        logit += 2.15 * label_jaccard
        logit += 1.25 * context_jaccard
        logit += 0.65 * parent_jaccard
        logit += 1.15 * datatype_compat
        logit += 0.90 * unit_compat
        logit += 1.10 * retrieval_score
        logit += 0.85 * rule_hit
        logit += 0.70 * signal_family
        logit += 1.65 * instance_alignment
        if instance_alignment <= 0.0:
            logit -= 1.20
        logit += 0.45 * lexical_score
        logit += 0.50 * multi_support
        logit += 2.00 * single_candidate
        logit -= 1.15 * ambiguity_penalty
        confidence = 1.0 / (1.0 + math.exp(-logit))
        confidence = max(0.0, min(1.0, confidence))

        return confidence, {
            "semantic_label_jaccard": label_jaccard,
            "semantic_context_jaccard": context_jaccard,
            "semantic_parent_jaccard": parent_jaccard,
            "semantic_signal_family": signal_family,
            "semantic_instance_alignment": instance_alignment,
            "semantic_multi_support": multi_support,
            "semantic_single_candidate": single_candidate,
            "semantic_ambiguity_penalty": ambiguity_penalty,
            "semantic_logistic_confidence": confidence,
        }


@dataclass
class CandidateState:
    mapping: Mapping
    total_score: float
    score_breakdown: dict[str, float]
    support: set[str]
    rejected_reasons: list[str]


class AdaptiveCandidateRankerPipeline:
    def __init__(self, llm: OllamaClient, retriever: GraphRAGRetriever, rules: RuleEngine) -> None:
        self.llm = llm
        self.retriever = retriever
        self.rules = rules

    def run(
        self,
        source_standard: str,
        target_standard: str,
        source_raw: str | bytes | dict,
        mode: Mode,
        config: TranslatorConfig,
        target_candidates: list[str] | None = None,
    ) -> TranslationResult:
        resolved_mode = "adaptive_candidate_ranker" if mode == "hybrid" else mode
        flags = {
            "rules": True,
            "retrieval": True,
            "llm": True,
            "allow_synthetic_benchmark_shortcuts": False,
            # Conservative defaults after the first 90%+ tuning run.
            # The previous 0.55/0.02 calibration made the benchmark too easy
            # and allowed low-evidence matches to pass when the candidate list
            # was small.  Keep these values configurable for threshold sweeps.
            "uncertainty_threshold": _float_setting("SGC_UNCERTAINTY_THRESHOLD", 0.82),
            "ambiguity_margin": _float_setting("SGC_AMBIGUITY_MARGIN", 0.08),
            "max_candidates_per_source": int(_float_setting("SGC_MAX_CANDIDATES_PER_SOURCE", 8)),
            "retrieval_confidence_threshold": _float_setting("SGC_RETRIEVAL_CONFIDENCE_THRESHOLD", DEFAULT_RETRIEVAL_CONFIDENCE_THRESHOLD),
            "calibrated_accept_threshold": _float_setting("SGC_CALIBRATED_ACCEPT_THRESHOLD", 0.68),
            "calibrated_margin": _float_setting("SGC_CALIBRATED_MARGIN", 0.05),
        }
        flags.update(config.component_flags or {})

        if resolved_mode == "rule_only":
            flags.update({"retrieval": False, "llm": False})
        elif resolved_mode in {"rag_only", "graph_only", "embedding_only"}:
            flags.update({"rules": False, "llm": False, "retrieval": True})
        elif resolved_mode == "llm_only":
            flags.update({"rules": False, "retrieval": False, "llm": True})
        elif resolved_mode == "semantic_graph_calibrated":
            flags.update({"rules": True, "retrieval": True, "llm": False, "adaptive_selection": True})

        src = ADAPTERS[source_standard]
        tgt = ADAPTERS[target_standard]
        source_model = src.parse(source_raw)
        source_paths = [_canonical_source_path(source_standard, node.id) for node in source_model.nodes]
        source_index = {path: node for path, node in zip(source_paths, source_model.nodes)}
        explicit_target_candidates = {normalize_path(str(candidate or "").strip()) for candidate in (target_candidates or []) if str(candidate or "").strip()}
        target_model = _build_target_model_from_candidates(target_standard, target_candidates or [])
        target_index = {node.id: node for node in target_model.nodes}
        source_parent = _parent_map(source_model)
        target_parent = _parent_map(target_model)
        allowed_external_targets = set(explicit_target_candidates)
        strict_target_existence = resolved_mode in {"adaptive_candidate_ranker", "rule_only", "hybrid"}

        evidence: list[EvidenceItem] = []
        retrieval_by_source: dict[str, list[EvidenceItem]] = {p: [] for p in source_paths}
        if flags["retrieval"]:
            evidence = self.retriever.retrieve(
                source_model,
                target_standard,
                target_model=target_model,
                top_k=int(flags.get("max_candidates_per_source", 5)),
                enable_vector=config.enable_vector_retrieval,
            )
            for ev in evidence:
                source_node = str((ev.payload or {}).get("source_node", "")).strip()
                if not source_node and len(source_paths) == 1:
                    source_node = source_paths[0]
                if source_node in retrieval_by_source:
                    retrieval_by_source[source_node].append(ev)
            for source_node in retrieval_by_source:
                retrieval_by_source[source_node] = sorted(retrieval_by_source[source_node], key=lambda x: float(x.score), reverse=True)
            target_prefix = f"{target_standard.lower()}://"
            for items in retrieval_by_source.values():
                for ev in items:
                    candidate_path = str((ev.payload or {}).get("candidate_path") or "").strip()
                    if not candidate_path or not candidate_path.startswith(target_prefix):
                        continue
                    allowed_external_targets.add(candidate_path)
                    if candidate_path in target_index:
                        continue
                    hints = _semantic_hint_from_path(candidate_path)
                    target_index[candidate_path] = CanonicalNode(
                        id=candidate_path,
                        type="Candidate",
                        label=hints["label"] or _guess_label_from_path(candidate_path),
                        attributes={"datatype": hints["datatype"], "unit": hints["unit"]},
                    )

        if target_candidates and not flags["retrieval"]:
            normalized_candidates = [normalize_path(str(candidate or "").strip()) for candidate in target_candidates if str(candidate or "").strip()]
            normalized_candidates = list(dict.fromkeys(candidate for candidate in normalized_candidates if candidate))
            single_candidate = len(normalized_candidates) == 1
            for source_path in source_paths:
                source_node = source_index[source_path]
                src_parent = source_parent.get(source_node.id, "")
                for candidate in normalized_candidates:
                    hints = _semantic_hint_from_path(candidate)
                    target_node = target_index.get(candidate)
                    if target_node is None:
                        target_node = CanonicalNode(
                            id=candidate,
                            type="Candidate",
                            label=hints["label"] or _guess_label_from_path(candidate),
                            attributes={"datatype": hints["datatype"], "unit": hints["unit"]},
                        )
                    logical_score, logical_breakdown = _logical_candidate_score(
                        source_node=source_node,
                        target_node=target_node,
                        source_path=source_path,
                        target_path=candidate,
                        source_parent=src_parent,
                        target_parent=target_parent.get(target_node.id, ""),
                        single_candidate=single_candidate,
                    )
                    item = EvidenceItem(
                        id=f"candidate:{source_path}:{candidate}",
                        kind="target_candidate",
                        text=hints["label"] or candidate,
                        score=logical_score,
                        payload={
                            "source_node": source_path,
                            "candidate_path": candidate,
                            "target_hint": candidate,
                            "label": hints["label"] or candidate.rsplit("/", 2)[-2],
                            "datatype": hints["datatype"],
                            "unit": hints["unit"],
                            "score_breakdown": {
                                **logical_breakdown,
                                "explicit_candidate": 1.0,
                                "candidate_scored_by_logical_ranker": 1.0,
                            },
                        },
                    )
                    retrieval_by_source[source_path].append(item)
                    evidence.append(item)
            for source_node in retrieval_by_source:
                retrieval_by_source[source_node] = sorted(retrieval_by_source[source_node], key=lambda x: float(x.score), reverse=True)

        rules_by_source: dict[str, list[Mapping]] = {p: [] for p in source_paths}
        if flags["rules"]:
            rule_mappings = self.rules.apply_rules(
                source_model,
                target_standard,
                target=target_model,
                allow_synthetic_shortcuts=bool(flags.get("allow_synthetic_benchmark_shortcuts", True)),
            )
            for rm in rule_mappings:
                normalized = normalize_mapping_item(rm.model_dump(), source_standard, target_standard)
                source_key = _canonical_source_path(source_standard, normalized.source_path)
                if source_key in rules_by_source:
                    rules_by_source[source_key].append(normalized)

        candidates_by_source: dict[str, dict[str, CandidateState]] = {p: {} for p in source_paths}
        for source_path in source_paths:
            for ev in retrieval_by_source.get(source_path, []):
                target_path = str((ev.payload or {}).get("candidate_path") or "").strip()
                if not target_path:
                    continue
                mapping = normalize_mapping_item(
                    {
                        "source_path": source_path,
                        "target_path": target_path,
                        "mapping_type": "equivalent",
                        "transform": None,
                        "confidence": float(ev.score),
                        "rationale": "Retrieved target candidate.",
                        "evidence": ["retrieval:ranked_candidate"],
                    },
                    source_standard,
                    target_standard,
                )
                candidates_by_source[source_path][target_path] = CandidateState(mapping=mapping, total_score=0.0, score_breakdown={}, support={"retrieval"}, rejected_reasons=[])
            for rm in rules_by_source.get(source_path, []):
                if rm.mapping_type == MappingType.NO_MATCH:
                    continue
                state = candidates_by_source[source_path].get(rm.target_path)
                if state is None:
                    candidates_by_source[source_path][rm.target_path] = CandidateState(mapping=rm, total_score=0.0, score_breakdown={}, support={"rules"}, rejected_reasons=[])
                else:
                    state.support.add("rules")
                    if rm.confidence > state.mapping.confidence:
                        state.mapping = rm

        llm_raw_output: list[dict[str, Any]] = []
        llm_by_source: dict[str, list[Mapping]] = {p: [] for p in source_paths}
        llm_invocation_log: list[dict[str, Any]] = []
        llm_explicit_no_match_sources: set[str] = set()

        prompt = ""
        if flags["llm"]:
            prompt = build_mapping_prompt(
                source_protocol=source_standard,
                target_protocol=target_standard,
                source_schema_summary=_schema_summary(source_model),
                target_schema_summary=_schema_summary(target_model),
                source_model=source_model,
                evidence=evidence,
                use_reasoning_prompt=True,
            )

        source_decisions: dict[str, dict[str, Any]] = {}
        semantic_calibrator = SemanticGraphCalibrator() if resolved_mode == "semantic_graph_calibrated" else None
        for source_path in source_paths:
            source_node = source_index[source_path]
            src_dtype = _guess_dtype(source_node)
            src_unit = _guess_unit(source_node)
            src_parent = source_parent.get(source_node.id, "")
            src_tokens = _path_tokens(source_path) | _path_tokens(src_parent)
            signature_counts: dict[str, int] = {}
            for target_path in candidates_by_source[source_path]:
                signature = _semantic_signature(target_index.get(target_path))
                if signature:
                    signature_counts[signature] = signature_counts.get(signature, 0) + 1

            scored: list[CandidateState] = []
            for target_path, state in candidates_by_source[source_path].items():
                target_node = target_index.get(target_path)
                if target_node is None:
                    state.rejected_reasons.append("target_path_missing")
                    continue
                retrieval_item = next((i for i in retrieval_by_source.get(source_path, []) if str(i.payload.get("candidate_path")) == target_path), None)
                retrieval_score = float(retrieval_item.score) if retrieval_item else 0.0
                retrieval_breakdown = (retrieval_item.payload or {}).get("score_breakdown", {}) if retrieval_item else {}
                lexical = float(retrieval_breakdown.get("lexical", _lexical_similarity(source_node.label or source_node.id, target_node.label or target_node.id)))
                embedding_similarity = float(retrieval_breakdown.get("vector_boost", 0.0))
                datatype_compat = _dtype_compatible(src_dtype, _guess_dtype(target_node))
                unit_compat = 1.0 if src_unit and _guess_unit(target_node) and src_unit == _guess_unit(target_node) else (0.5 if not src_unit or not _guess_unit(target_node) else 0.0)
                parent_sim = _lexical_similarity(src_parent, target_parent.get(target_node.id, "")) if src_parent or target_parent.get(target_node.id, "") else 0.5
                rule_hit = 1.0 if "rules" in state.support else 0.0
                target_parent_path = target_parent.get(target_node.id, "")
                target_tokens = _path_tokens(target_path) | _path_tokens(str((target_node.attributes or {}).get("parent_path", ""))) | _path_tokens(target_parent_path)
                context_overlap = len(src_tokens & target_tokens)
                context_score = min(1.0, context_overlap / 3.0) if target_tokens else 0.0
                duplicate_count = signature_counts.get(_semantic_signature(target_node), 1)
                duplicate_penalty = max(0.0, min(0.2, (duplicate_count - 1) * 0.05))
                if context_score > 0.0:
                    duplicate_penalty *= 0.5
                signal_family = _same_signal_family(source_node, target_node)
                instance_alignment = max(
                    _instance_alignment(source_node, target_node, source_path, target_path, src_parent, target_parent_path),
                    float(retrieval_breakdown.get("instance_match", 0.5)),
                )
                instance_conflict_penalty = 0.16 if _has_conflicting_instance_ids(source_node, target_node, source_path, target_path, src_parent, target_parent_path) else 0.0
                measurement_bonus = 1.0 if _is_measurement_like_node(source_node) and _is_measurement_like_node(target_node) else 0.0
                single_candidate = 1.0 if len(candidates_by_source[source_path]) == 1 else 0.0

                breakdown = {
                    "lexical_similarity": lexical,
                    "embedding_similarity": embedding_similarity,
                    "rule_hit": rule_hit,
                    "datatype_compatibility": datatype_compat,
                    "unit_compatibility": unit_compat,
                    "parent_context_similarity": parent_sim,
                    "path_context_similarity": context_score,
                    "signal_family": signal_family,
                    "instance_alignment": instance_alignment,
                    "instance_conflict_penalty": instance_conflict_penalty,
                    "measurement_node_bonus": measurement_bonus,
                    "single_candidate_bonus": single_candidate,
                    "duplicate_semantic_count": float(duplicate_count),
                    "duplicate_ambiguity_penalty": duplicate_penalty,
                    "retrieval_score": retrieval_score,
                    "explicit_candidate": 1.0 if target_path in explicit_target_candidates else 0.0,
                    "fallback_injected": float(retrieval_breakdown.get("fallback_injected", 0.0)),
                }
                if semantic_calibrator is not None:
                    total_score, semantic_breakdown = semantic_calibrator.score(
                        source_node=source_node,
                        target_node=target_node,
                        source_path=source_path,
                        target_path=target_path,
                        source_parent=src_parent,
                        target_parent=target_parent.get(target_node.id, ""),
                        retrieval_score=retrieval_score,
                        lexical_score=lexical,
                        rule_hit=rule_hit,
                        datatype_compat=datatype_compat,
                        unit_compat=unit_compat,
                        support_count=len(state.support),
                        duplicate_count=duplicate_count,
                        candidate_count=len(candidates_by_source[source_path]),
                    )
                    breakdown.update(semantic_breakdown)
                else:
                    total_score = (
                        lexical * 0.16
                        + embedding_similarity * 0.05
                        + rule_hit * 0.18
                        + datatype_compat * 0.13
                        + unit_compat * 0.11
                        + parent_sim * 0.06
                        + context_score * 0.06
                        + retrieval_score * 0.16
                        + signal_family * 0.11
                        + instance_alignment * 0.18
                        + measurement_bonus * 0.06
                        + single_candidate * 0.06
                    )
                    total_score -= duplicate_penalty
                    total_score -= instance_conflict_penalty
                    if len(state.support) > 1:
                        total_score += 0.08
                state.score_breakdown = breakdown
                state.total_score = min(1.0, total_score)
                scored.append(state)

            scored.sort(key=lambda x: x.total_score, reverse=True)
            top1 = scored[0].total_score if scored else 0.0
            top2 = scored[1].total_score if len(scored) > 1 else 0.0
            uncertainty = top1 < float(flags["uncertainty_threshold"]) or (top1 - top2) <= float(flags["ambiguity_margin"])

            rules_target = rules_by_source[source_path][0].target_path if rules_by_source.get(source_path) else ""
            retrieval_target = str(retrieval_by_source[source_path][0].payload.get("candidate_path")) if retrieval_by_source.get(source_path) else ""
            disagreement = bool(rules_target and retrieval_target and rules_target != retrieval_target)
            uncertainty = uncertainty or disagreement

            if flags["llm"] and (resolved_mode == "llm_only" or uncertainty):
                raw = self.llm.complete_json(prompt, "MappingOutputContract", config.seed)
                llm_raw_output.append(
                    {
                        "method": resolved_mode,
                        "source_protocol": source_standard,
                        "target_protocol": target_standard,
                        "prompt": prompt,
                        "raw": raw,
                        "source_path": source_path,
                    }
                )
                raw_mappings = raw.get("mappings", [])
                for item in raw_mappings:
                    try:
                        normalized = normalize_mapping_item(item, source_standard, target_standard)
                    except Exception:
                        continue
                    normalized_source = _canonical_source_path(source_standard, normalized.source_path)
                    if normalized_source != source_path:
                        if len(source_paths) == 1:
                            normalized = normalize_mapping_item(
                                {**normalized.model_dump(), "source_path": source_path},
                                source_standard,
                                target_standard,
                            )
                        else:
                            continue
                    source_candidates = [str(i.payload.get("candidate_path", "")).strip() for i in retrieval_by_source.get(source_path, []) if str(i.payload.get("candidate_path", "")).strip()]
                    if source_candidates and normalized.target_path not in source_candidates:
                        snapped_target = ""
                        if len(source_candidates) == 1:
                            snapped_target = source_candidates[0]
                        else:
                            scored = sorted(
                                ((_lexical_similarity(normalized.target_path, c), c) for c in source_candidates),
                                reverse=True,
                            )
                            if scored and scored[0][0] >= 0.72:
                                snapped_target = scored[0][1]
                        if snapped_target:
                            normalized = normalize_mapping_item(
                                {
                                    **normalized.model_dump(),
                                    "target_path": snapped_target,
                                    "rationale": f"{normalized.rationale} (snapped to retrieved candidate)",
                                    "evidence": [*normalized.evidence, "llm:candidate_snap"],
                                },
                                source_standard,
                                target_standard,
                            )
                    llm_by_source[source_path].append(normalized)
                    if normalized.mapping_type == MappingType.NO_MATCH:
                        llm_explicit_no_match_sources.add(source_path)
                    if normalized.mapping_type != MappingType.NO_MATCH:
                        state = candidates_by_source[source_path].get(normalized.target_path)
                        llm_conf = float(normalized.confidence)
                        if state is None:
                            candidates_by_source[source_path][normalized.target_path] = CandidateState(mapping=normalized, total_score=llm_conf, score_breakdown={"llm_confidence": llm_conf}, support={"llm"}, rejected_reasons=[])
                        else:
                            state.support.add("llm")
                            state.score_breakdown["llm_confidence"] = llm_conf
                            state.total_score = min(1.0, max(state.total_score + 0.06, llm_conf))
                if not llm_by_source[source_path]:
                    llm_by_source[source_path].append(
                        normalize_mapping_item(
                            {
                                "source_path": source_path,
                                "target_path": "",
                                "mapping_type": "no_match",
                                "transform": None,
                                "confidence": 0.0,
                                "rationale": "LLM produced no usable mapping for this source variable.",
                                "evidence": ["llm:no_decision"],
                            },
                            source_standard,
                            target_standard,
                        )
                    )
                llm_invocation_log.append({"source_path": source_path, "invoked": True, "reason": "uncertain_or_disagreement", "top1": top1, "top2": top2, "rules_retrieval_disagree": disagreement})
            else:
                llm_invocation_log.append({"source_path": source_path, "invoked": False, "reason": "high_confidence_non_ambiguous", "top1": top1, "top2": top2, "rules_retrieval_disagree": disagreement})

            source_decisions[source_path] = {
                "top1": top1,
                "top2": top2,
                "score_gap": max(0.0, top1 - top2),
                "rules_retrieval_disagree": disagreement,
                "candidate_count": len(scored),
            }

        final_by_source: dict[str, Mapping] = {}
        ranking_trace: dict[str, list[dict[str, Any]]] = {}
        valid_states_by_source: dict[str, list[CandidateState]] = {}

        retrieval_confidence_threshold = float(flags["retrieval_confidence_threshold"])

        for source_path in source_paths:
            states = list(candidates_by_source[source_path].values())
            states.sort(key=lambda x: x.total_score, reverse=True)
            ranking_trace[source_path] = [
                {
                    "target_path": s.mapping.target_path,
                    "total_score": s.total_score,
                    "score_breakdown": s.score_breakdown,
                    "support": sorted(s.support),
                    "rejected_reasons": s.rejected_reasons,
                }
                for s in states[:8]
            ]
            valid_states: list[CandidateState] = []
            for state in states:
                mapping = state.mapping
                if mapping.mapping_type not in {
                    MappingType.EQUIVALENT,
                    MappingType.APPROXIMATE,
                    MappingType.TRANSFORM,
                    MappingType.LABEL_MATCH,
                }:
                    continue
                retrieval_score = float(state.score_breakdown.get("retrieval_score", 0.0))
                retrieval_only = "retrieval" in state.support and "rules" not in state.support and "llm" not in state.support
                if retrieval_only and retrieval_score < retrieval_confidence_threshold:
                    if not (resolved_mode == "semantic_graph_calibrated" and state.total_score >= float(flags["calibrated_accept_threshold"])):
                        state.rejected_reasons.append(
                            f"retrieval_score_below_min_threshold:{retrieval_score:.3f}<{retrieval_confidence_threshold:.3f}"
                        )
                        continue
                if strict_target_existence and mapping.target_path not in target_index and mapping.target_path not in allowed_external_targets:
                    continue
                if mapping.target_path in target_index:
                    src_dtype = _guess_dtype(source_index[source_path])
                    tgt_dtype = _guess_dtype(target_index[mapping.target_path])
                    if src_dtype and tgt_dtype and _dtype_compatible(src_dtype, tgt_dtype) <= 0.0:
                        continue
                    src_unit = _guess_unit(source_index[source_path])
                    tgt_unit = _guess_unit(target_index[mapping.target_path])
                    if src_unit and tgt_unit and src_unit != tgt_unit:
                        continue
                valid_states.append(state)
            valid_states_by_source[source_path] = valid_states

        ambiguity_margin = float(flags["ambiguity_margin"])
        forced_no_match_sources: set[str] = set()
        for source_path, states in valid_states_by_source.items():
            if _is_equipment_label_without_measurement(source_index[source_path]):
                forced_no_match_sources.add(source_path)
                continue
            if len(states) < 2:
                continue
            top = states[0]
            runner_up = states[1]
            if (top.total_score - runner_up.total_score) > ambiguity_margin:
                continue
            top_target = target_index.get(top.mapping.target_path)
            runner_target = target_index.get(runner_up.mapping.target_path)
            if _semantic_signature(top_target) != _semantic_signature(runner_target):
                continue
            top_context = float(top.score_breakdown.get("path_context_similarity", 0.0))
            runner_context = float(runner_up.score_breakdown.get("path_context_similarity", 0.0))
            if max(top_context, runner_context) >= 0.34:
                continue
            if "rules" in top.support and "rules" in runner_up.support and top.mapping.target_path != runner_up.mapping.target_path:
                continue
            forced_no_match_sources.add(source_path)

        if resolved_mode == "semantic_graph_calibrated":
            accept_threshold = float(flags["calibrated_accept_threshold"])
            calibrated_margin = float(flags["calibrated_margin"])
            for source_path, states in valid_states_by_source.items():
                if source_path in forced_no_match_sources or not states:
                    continue
                states.sort(key=lambda x: x.total_score, reverse=True)
                top_score = states[0].total_score
                runner_score = states[1].total_score if len(states) > 1 else 0.0
                if top_score < accept_threshold or (top_score - runner_score) < calibrated_margin:
                    forced_no_match_sources.add(source_path)

        assigned_sources: set[str] = set()
        used_targets: set[str] = set()
        global_ranked: list[tuple[float, str, CandidateState]] = []
        for source_path, states in valid_states_by_source.items():
            for state in states:
                global_ranked.append((state.total_score, source_path, state))
        global_ranked.sort(key=lambda x: x[0], reverse=True)
        for _, source_path, state in global_ranked:
            if source_path in forced_no_match_sources:
                continue
            if source_path in assigned_sources:
                continue
            if state.mapping.target_path in used_targets:
                continue
            selected_confidence = max(float(state.mapping.confidence), state.total_score)
            winner = normalize_mapping_item(
                {
                    **state.mapping.model_dump(),
                    "source_path": source_path,
                    "confidence": selected_confidence,
                    "rationale": (f"Semantic graph calibrated ranker selected highest valid candidate with support={sorted(state.support)}." if resolved_mode == "semantic_graph_calibrated" else f"Adaptive candidate ranker selected highest valid candidate with support={sorted(state.support)}."),
                    "evidence": [*state.mapping.evidence, ("semantic_graph_calibrated:final_selection" if resolved_mode == "semantic_graph_calibrated" else "ranker:final_selection")],
                },
                source_standard,
                target_standard,
            )
            final_by_source[source_path] = winner
            assigned_sources.add(source_path)
            if winner.target_path:
                used_targets.add(winner.target_path)

        for source_path in source_paths:
            if source_path in final_by_source:
                continue
            if source_path in forced_no_match_sources:
                final_by_source[source_path] = normalize_mapping_item(
                    {
                        "source_path": source_path,
                        "target_path": "",
                        "mapping_type": "no_match",
                        "transform": None,
                        "confidence": 0.0,
                        "rationale": "Semantic graph calibrated model rejected the mapping because confidence or margin was insufficient." if resolved_mode == "semantic_graph_calibrated" else "Ambiguous duplicate candidates share the same semantic signature and lack asset-level context.",
                        "evidence": ["semantic_graph_calibrated:no_match"] if resolved_mode == "semantic_graph_calibrated" else ["ranker:ambiguous_duplicate_candidates"],
                    },
                    source_standard,
                    target_standard,
                )
                continue
            if source_path in llm_explicit_no_match_sources:
                retrieval_fallback = next((state for state in valid_states_by_source.get(source_path, []) if "retrieval" in state.support), None)
                if retrieval_fallback is not None:
                    retrieval_score = float(retrieval_fallback.score_breakdown.get("retrieval_score", 0.0))
                    if retrieval_score >= retrieval_confidence_threshold:
                        winner = normalize_mapping_item(
                            {
                                **retrieval_fallback.mapping.model_dump(),
                                "source_path": source_path,
                                "confidence": max(float(retrieval_fallback.mapping.confidence), retrieval_fallback.total_score),
                                "rationale": "Fallback to retrieval candidate after LLM no_match with sufficient retrieval confidence.",
                                "evidence": [*retrieval_fallback.mapping.evidence, "ranker:llm_no_match_retrieval_fallback"],
                            },
                            source_standard,
                            target_standard,
                        )
                        final_by_source[source_path] = winner
                        continue
            final_by_source[source_path] = normalize_mapping_item(
                {
                    "source_path": source_path,
                    "target_path": "",
                    "mapping_type": "no_match",
                    "transform": None,
                    "confidence": 0.0,
                    "rationale": (
                        "No valid target candidate after constraint enforcement."
                        if source_path not in llm_explicit_no_match_sources
                        else "LLM returned no_match and retrieval fallback did not meet confidence threshold."
                    ),
                    "evidence": (
                        ["ranker:no_valid_candidate"]
                        if source_path not in llm_explicit_no_match_sources
                        else ["ranker:llm_no_match_retrieval_rejected"]
                    ),
                },
                source_standard,
                target_standard,
            )

        mappings = [final_by_source[source_path] for source_path in source_paths]
        target_artifact = tgt.serialize(target_model, [m.model_dump() for m in mappings])
        validation = tgt.validate(target_artifact)

        metadata: dict[str, Any] = {
            "mode": resolved_mode,
            "legacy_mode_alias_used": mode == "hybrid",
            "deprecation_warning": "mode='hybrid' is deprecated; use mode='adaptive_candidate_ranker'." if mode == "hybrid" else "",
            "llm_raw_output": llm_raw_output,
            "component_outputs": {
                "retrieval": ({
                    source_node: [
                        {
                            "id": item.id,
                            "score": item.score,
                            "candidate_path": item.payload.get("candidate_path"),
                            "datatype": item.payload.get("datatype", ""),
                            "unit": item.payload.get("unit", ""),
                            "breakdown": item.payload.get("score_breakdown", {}),
                        }
                        for item in items
                    ]
                    for source_node, items in retrieval_by_source.items()
                } if (flags["retrieval"] or target_candidates) else {}),
                "rules": {k: [m.model_dump() for m in v] for k, v in rules_by_source.items()},
                "rule_engine": {k: [m.model_dump() for m in v] for k, v in rules_by_source.items()},
                "llm": {k: [m.model_dump() for m in v] for k, v in llm_by_source.items()},
                "ranking": ranking_trace,
                "final": [m.model_dump() for m in mappings],
                "merged": [m.model_dump() for m in mappings],
            },
            "source_decisions": source_decisions,
            "decision_log": [
                {"source_path": source_path, **details, "selected_strategy": resolved_mode}
                for source_path, details in source_decisions.items()
            ],
            "llm_invocations": llm_invocation_log,
            "execution": {
                "selected_strategy": resolved_mode,
                "rules_ran": flags["rules"],
                "retrieval_ran": flags["retrieval"],
                "llm_ran": flags["llm"],
                "one_to_one_enforced": True,
            },
        }

        prov = Provenance(
            model_name=config.model_name,
            prompt_hash=stable_hash({"mode": resolved_mode, "source": source_standard, "target": target_standard}),
            seed=config.seed,
            git_commit=_git_commit(),
            adapter_versions={"source": "1.0", "target": "1.0"},
            metadata=metadata,
        )
        return TranslationResult(target_artifact=target_artifact, mappings=mappings, evidence=evidence, provenance=prov, validation_report=validation)


# Backward compatibility class alias.
HybridPipeline = AdaptiveCandidateRankerPipeline
