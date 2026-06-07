from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from isynkgr.canonical.model import CanonicalModel, CanonicalNode
from isynkgr.canonical.schemas import EvidenceItem
from isynkgr.icr.entities import build_endpoint_path, normalize_path
from isynkgr.retrieval.vector import SqliteFTSRetriever


@dataclass
class TargetCandidate:
    target_path: str
    label: str
    datatype: str
    unit: str
    parent: str


def _norm(text: str) -> str:
    return str(text or "").strip().lower().replace("_", " ")


def _semantic_instance_ids(*values: str) -> set[str]:
    ids: set[str] = set()
    pattern = re.compile(
        r"(?i)\b(?:pump|motor|line|pressure|temperature|temp|flow|speed|vibration|state|status|class|signal|channel)[-_ ]?(\d+)\b"
    )
    for value in values:
        text = str(value or "")
        for match in pattern.finditer(text):
            ids.add(str(int(match.group(1))))
        for match in re.finditer(r"(?i)(?:^|[;/])s=([A-Za-z]+)(\d+)\b", text):
            if match.group(1).lower() in {"pressure", "temperature", "temp", "flow", "speed", "vibration", "state", "status", "pump", "motor"}:
                ids.add(str(int(match.group(2))))
    return ids


def _context_instance_ids(*values: str) -> set[str]:
    ids: set[str] = set()
    pattern = re.compile(r"(?i)\b(?:asset|aas|sm|submodel|machine|equipment|device)[-_ ]?(\d+)\b")
    for value in values:
        for match in pattern.finditer(str(value or "")):
            ids.add(str(int(match.group(1))))
    return ids


def _instance_ids(*values: str) -> set[str]:
    return _semantic_instance_ids(*values) | _context_instance_ids(*values)


def _instance_match(source_node: CanonicalNode, candidate: "TargetCandidate") -> float:
    src_text = _node_text(source_node)
    tgt_text = _candidate_text(candidate) if "_candidate_text" in globals() else " ".join([candidate.target_path, candidate.label, candidate.parent])
    src_semantic_ids = _semantic_instance_ids(source_node.id, source_node.label or "", src_text)
    tgt_semantic_ids = _semantic_instance_ids(candidate.target_path, candidate.label, candidate.parent, tgt_text)
    if src_semantic_ids and tgt_semantic_ids:
        return 1.0 if src_semantic_ids & tgt_semantic_ids else 0.0

    src_context_ids = _context_instance_ids(source_node.id, source_node.label or "", src_text)
    tgt_context_ids = _context_instance_ids(candidate.target_path, candidate.label, candidate.parent, tgt_text)
    src_signal = _split_semantic_tokens(src_text) & _SIGNAL_TERMS if "_split_semantic_tokens" in globals() else set()
    tgt_signal = _split_semantic_tokens(tgt_text) & _SIGNAL_TERMS if "_split_semantic_tokens" in globals() else set()

    if src_context_ids and tgt_semantic_ids and src_signal and tgt_signal and src_signal == tgt_signal:
        return 1.0 if src_context_ids & tgt_semantic_ids else 0.0
    if src_semantic_ids and tgt_context_ids and src_signal and tgt_signal and src_signal == tgt_signal:
        return 1.0 if src_semantic_ids & tgt_context_ids else 0.0
    if src_context_ids and tgt_context_ids and src_signal and tgt_signal and src_signal == tgt_signal:
        return 1.0 if src_context_ids & tgt_context_ids else 0.0
    return 0.5


def _generic_label_penalty(label: str) -> float:
    lowered = _norm(label)
    if lowered in {"value", "measurement", "variable", "candidate"}:
        return 0.18
    if lowered.startswith("value "):
        return 0.15
    return 0.0


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
    "teds", "channel", "candidate",
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


def _node_text(node: CanonicalNode) -> str:
    attrs = node.attributes or {}
    metadata = attrs.get("metadata", {}) if isinstance(attrs.get("metadata", {}), dict) else {}
    fields = [node.id, node.type, node.label or ""]
    for key in ("description", "datatype", "dtype", "valueType", "unit", "asset_id", "equipment_id", "process_line", "parent_path", "context_tokens"):
        value = attrs.get(key) or metadata.get(key)
        if value:
            fields.append(str(value))
    return " ".join(fields)


def _candidate_text(candidate: TargetCandidate) -> str:
    return " ".join([candidate.target_path, candidate.label, candidate.datatype, candidate.unit, candidate.parent])


def _same_signal_family(source_node: CanonicalNode, candidate: TargetCandidate) -> float:
    src_signal = _split_semantic_tokens(_node_text(source_node)) & _SIGNAL_TERMS
    tgt_signal = _split_semantic_tokens(_candidate_text(candidate)) & _SIGNAL_TERMS
    if not src_signal or not tgt_signal:
        return 0.5
    return 1.0 if src_signal == tgt_signal else 0.0


def _is_measurement_like_source(node: CanonicalNode) -> bool:
    node_type = str(node.type or "").strip().lower()
    text = _node_text(node).lower()
    if node_type in {"channel", "signal", "property", "sensor", "variable", "uavariable", "output", "input"}:
        return True
    if _guess_datatype(node) or _guess_unit(node):
        return True
    return any(term in text for term in {"measurement", "measure", "value", "temp", "temperature", "pressure", "flow", "speed", "current", "voltage", "vibration", "state", "status", "setpoint"})


def _is_measurement_like_candidate(candidate: TargetCandidate) -> bool:
    text = _candidate_text(candidate).lower()
    if candidate.datatype or candidate.unit:
        return True
    return any(term in text for term in {"measurement", "measure", "value", "temp", "temperature", "pressure", "flow", "speed", "current", "voltage", "vibration", "state", "status", "setpoint", "channel", "signal"})


def _is_structural_without_measurement_source(node: CanonicalNode) -> bool:
    if _is_measurement_like_source(node):
        return False
    text = _node_text(node).lower()
    node_type = str(node.type or "").strip().lower()
    return node_type in {"asset", "submodel", "resource", "functionblock", "device", "object", "teds"} or any(term in text for term in {"asset", "submodel", "resource", "functionblock", "device", "equipment", "machine"})


def _is_structural_without_measurement_candidate(candidate: TargetCandidate) -> bool:
    if _is_measurement_like_candidate(candidate):
        return False
    text = _candidate_text(candidate).lower()
    return any(term in text for term in {"asset", "submodel", "resource", "functionblock", "device", "equipment", "machine", "teds"})


def _guess_datatype(node: CanonicalNode) -> str:
    attrs = node.attributes or {}
    metadata = attrs.get("metadata", {}) if isinstance(attrs.get("metadata", {}), dict) else {}
    for key in ("datatype", "dtype", "valueType", "dataType", "type"):
        value = attrs.get(key) or metadata.get(key)
        if value:
            return str(value).upper()
    return ""


def _guess_unit(node: CanonicalNode) -> str:
    attrs = node.attributes or {}
    metadata = attrs.get("metadata", {}) if isinstance(attrs.get("metadata", {}), dict) else {}
    return str(attrs.get("unit") or metadata.get("unit") or "").strip()


def _parent_map(model: CanonicalModel) -> dict[str, str]:
    parent: dict[str, str] = {}
    for edge in model.edges:
        if edge.target not in parent:
            parent[edge.target] = edge.source
    return parent


def _build_target_candidates(model: CanonicalModel) -> list[TargetCandidate]:
    parent = _parent_map(model)
    candidates: list[TargetCandidate] = []
    for node in model.nodes:
        candidates.append(
            TargetCandidate(
                target_path=node.id,
                label=str(node.label or node.id).strip(),
                datatype=_guess_datatype(node),
                unit=_guess_unit(node),
                parent=parent.get(node.id, ""),
            )
        )
    return candidates


class GraphRAGRetriever:
    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        self._vector = SqliteFTSRetriever()

    def retrieve(
        self,
        source: CanonicalModel,
        target_schema_hint: str,
        target_model: CanonicalModel | None = None,
        top_k: int | None = None,
        enable_vector: bool = False,
    ) -> list[EvidenceItem]:
        target = target_model or CanonicalModel(standard=target_schema_hint, nodes=[], edges=[])
        pool = _build_target_candidates(target)
        if not pool:
            return []

        limit = max(1, top_k or self.top_k)
        scored: list[EvidenceItem] = []
        vector_rows = self._vector.retrieve(source, target_schema_hint) if enable_vector else []
        vector_boost = max((float(item.score) for item in vector_rows), default=0.0) * 0.1

        for src_node in source.nodes:
            src_path = normalize_path(src_node.id if "://" in src_node.id else build_endpoint_path(source.standard, src_node.id))
            src_label = _norm(src_node.label or src_node.id)
            src_dtype = _guess_datatype(src_node)
            src_unit = _guess_unit(src_node)

            ranked: list[tuple[float, TargetCandidate, dict[str, float]]] = []
            for candidate in pool:
                cand_label = _norm(candidate.label)
                label_similarity = difflib.SequenceMatcher(a=src_label, b=cand_label).ratio()
                token_overlap = len(set(src_label.split()) & set(cand_label.split())) / max(len(set(src_label.split()) | set(cand_label.split())), 1)
                lexical = (label_similarity * 0.65) + (token_overlap * 0.35)
                dtype_match = 1.0 if src_dtype and candidate.datatype and src_dtype == candidate.datatype else (0.4 if not src_dtype or not candidate.datatype else 0.0)
                unit_match = 1.0 if src_unit and candidate.unit and src_unit.lower() == candidate.unit.lower() else (0.5 if not src_unit or not candidate.unit else 0.0)
                context_hint = 1.0 if candidate.parent and _norm(candidate.parent).split("/")[-1] in src_label else 0.0
                penalty = _generic_label_penalty(candidate.label)
                instance_match = _instance_match(src_node, candidate)
                signal_family = _same_signal_family(src_node, candidate)
                measurement_bonus = 1.0 if _is_measurement_like_source(src_node) and _is_measurement_like_candidate(candidate) else 0.0
                single_candidate_bonus = 1.0 if len(pool) == 1 else 0.0
                structural_penalty = 0.22 if _is_structural_without_measurement_source(src_node) or _is_structural_without_measurement_candidate(candidate) else 0.0
                instance_conflict_penalty = 0.18 if instance_match == 0.0 else 0.0
                score = min(
                    1.0,
                    max(
                        0.0,
                        (lexical * 0.26)
                        + (dtype_match * 0.16)
                        + (unit_match * 0.10)
                        + (context_hint * 0.05)
                        + (instance_match * 0.20)
                        + (signal_family * 0.13)
                        + (measurement_bonus * 0.08)
                        + (single_candidate_bonus * 0.12)
                        + vector_boost
                        - penalty
                        - structural_penalty
                        - instance_conflict_penalty,
                    ),
                )
                breakdown = {
                    "lexical": lexical,
                    "label_similarity": label_similarity,
                    "token_overlap": token_overlap,
                    "datatype_match": dtype_match,
                    "unit_match": unit_match,
                    "context_hint": context_hint,
                    "instance_match": instance_match,
                    "signal_family": signal_family,
                    "measurement_bonus": measurement_bonus,
                    "single_candidate_bonus": single_candidate_bonus,
                    "structural_penalty": structural_penalty,
                    "instance_conflict_penalty": instance_conflict_penalty,
                    "vector_boost": vector_boost,
                    "generic_label_penalty": penalty,
                }
                ranked.append((score, candidate, breakdown))

            ranked.sort(key=lambda row: row[0], reverse=True)
            for rank, (score, candidate, breakdown) in enumerate(ranked[:limit], start=1):
                scored.append(
                    EvidenceItem(
                        id=f"node:{src_path}:cand:{rank}",
                        kind="target_candidate",
                        text=candidate.label,
                        score=score,
                        payload={
                            "source_node": src_path,
                            "candidate_rank": rank,
                            "candidate_path": candidate.target_path,
                            "target_hint": candidate.target_path,
                            "label": candidate.label,
                            "datatype": candidate.datatype,
                            "unit": candidate.unit,
                            "parent": candidate.parent,
                            "score_breakdown": breakdown,
                        },
                    )
                )

        scored.sort(key=lambda item: (str(item.payload.get("source_node", "")), -float(item.score)))
        return scored
