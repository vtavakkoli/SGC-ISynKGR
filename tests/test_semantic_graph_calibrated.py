from __future__ import annotations

from dataclasses import dataclass

from isynkgr.canonical.model import CanonicalModel, CanonicalNode
from isynkgr.canonical.schemas import EvidenceItem, Mapping
from isynkgr.pipeline.hybrid import HybridPipeline, TranslatorConfig


@dataclass
class _FakeAdapter:
    standard: str

    def parse(self, _raw):
        return CanonicalModel(
            standard=self.standard,
            nodes=[CanonicalNode(id="opcua://ns=2;s=Pump1.Temp", type="signal", label="temp", attributes={"dtype": "FLOAT", "unit": "C"})],
            edges=[],
        )

    def serialize(self, _model, mappings):
        return {"mappings": mappings}

    def validate(self, _artifact):
        return {"valid": True, "violations": []}


class _Retriever:
    def retrieve(self, *_args, **_kwargs):
        return [
            EvidenceItem(
                id="candidate:temp",
                kind="target_candidate",
                text="temperature",
                score=0.93,
                payload={
                    "source_node": "opcua://ns=2;s=Pump1.Temp",
                    "candidate_path": "aas://asset-1/submodel/default/element/temperature/value",
                    "label": "temperature",
                    "datatype": "FLOAT",
                    "unit": "C",
                    "score_breakdown": {"lexical": 0.72},
                },
            ),
            EvidenceItem(
                id="candidate:pressure",
                kind="target_candidate",
                text="pressure",
                score=0.61,
                payload={
                    "source_node": "opcua://ns=2;s=Pump1.Temp",
                    "candidate_path": "aas://asset-1/submodel/default/element/pressure/value",
                    "label": "pressure",
                    "datatype": "FLOAT",
                    "unit": "bar",
                    "score_breakdown": {"lexical": 0.18},
                },
            ),
        ]


class _Rules:
    def apply_rules(self, *_args, **_kwargs):
        return [
            Mapping(
                source_path="opcua://ns=2;s=Pump1.Temp",
                target_path="aas://asset-1/submodel/default/element/temperature/value",
                mapping_type="label_match",
                confidence=0.82,
                rationale="rule selected temp",
                evidence=["rule"],
            )
        ]


class _LLM:
    def complete_json(self, *_args, **_kwargs):  # pragma: no cover - semantic mode does not call LLM
        raise AssertionError("semantic_graph_calibrated should not call LLM")


def test_semantic_graph_calibrated_selects_alias_candidate(monkeypatch):
    from isynkgr.pipeline import hybrid as hybrid_mod

    monkeypatch.setattr(hybrid_mod, "ADAPTERS", {"opcua": _FakeAdapter("opcua"), "aas": _FakeAdapter("aas")})
    pipeline = HybridPipeline(llm=_LLM(), retriever=_Retriever(), rules=_Rules())

    result = pipeline.run("opcua", "aas", source_raw="x", mode="semantic_graph_calibrated", config=TranslatorConfig())

    assert result.provenance.metadata["execution"]["selected_strategy"] == "semantic_graph_calibrated"
    assert result.provenance.metadata["execution"]["llm_ran"] is False
    assert result.mappings[0].target_path == "aas://asset-1/submodel/default/element/temperature/value"
    ranking = result.provenance.metadata["component_outputs"]["ranking"]["opcua://ns=2;s=Pump1.Temp"][0]
    assert "semantic_logistic_confidence" in ranking["score_breakdown"]
