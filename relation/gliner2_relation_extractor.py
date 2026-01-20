from __future__ import annotations

from typing import Dict, Any, List, Iterable, Mapping

from model.model import RelationMention, NERMention
from ner.ner_extractor import ExtractorRunMetadata
from util import new_id, now_utc_iso


class GLiNER2RelationExtractor:
    """
    Placeholder for a GLiNER2 relation extraction adapter.

    You can wire this to whichever relation model/package you are using.
    The important part is the interface and output shape.

    Params:
      model_name: id/path (optional depending on your RE library)
      device: optional
      mode: optional
    """
    name = "gliner2_re"

    def __init__(self, params: Dict[str, Any]):
        self.model_name = params.get("model_name") or params.get("name")
        if not self.model_name:
            raise ValueError("GLiNER2RelationExtractor requires params.model_name (or params.name)")
        self.device = params.get("device")
        self.extra = params.get("extra") or {}

        try:
            from gliner2 import GLiNER2  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Could not import `gliner2`. Install the correct GLiNER2 package."
            ) from e

        self._GLiNER = GLiNER2
        self._model = GLiNER2.from_pretrained(self.model_name)

        if self.device:
            try:
                self._model.to(self.device)
            except Exception:
                pass

        backend_version = None
        try:
            import gliner2
            backend_version = getattr(gliner2, "__version__", None)
        except Exception:
            pass

        self._run_meta = ExtractorRunMetadata(
            run_id=new_id("re"),
            timestamp_utc=now_utc_iso(),
            backend=self.name,
            backend_version=backend_version,
            config={
                "model_name": self.model_name,
                "device": self.device,
                **({"extra": self.extra} if self.extra else {}),
            },
        )

    def get_run_meta(self) -> ExtractorRunMetadata:
        return self._run_meta

    def extract(
            self,
            text: str,
            entities: List[NERMention],
            relation_types: List[str],
            threshold: float,
    ) -> List[RelationMention]:
        if not relation_types or not entities:
            return []

        entity_payload = [
            {
                "id": entity.get("id"),
                "label": entity.get("label"),
                "text": entity.get("text"),
                "start": entity.get("start"),
                "end": entity.get("end"),
            }
            for entity in entities
        ]

        if not hasattr(self._model, "extract_relations"):
            raise RuntimeError(
                "GLiNER2RelationExtractor requires a model with extract_relations()."
            )

        raw_relations = self._model.extract_relations(
            text,
            entities=entity_payload,
            relation_types=relation_types,
            threshold=threshold,
        )
        normalized = self._normalize_relations(raw_relations, entities, text)
        return normalized

    def _normalize_relations(
            self,
            raw_relations: Any,
            entities: List[NERMention],
            text: str,
    ) -> List[RelationMention]:
        if isinstance(raw_relations, Mapping):
            relations = raw_relations.get("relations") or raw_relations.get("predictions") or []
        else:
            relations = raw_relations

        if not isinstance(relations, Iterable):
            return []

        results: List[RelationMention] = []
        for rel in relations:
            mention = self._relation_from_item(rel, entities, text)
            if mention:
                results.append(mention)
        return results

    def _relation_from_item(
            self,
            rel: Any,
            entities: List[NERMention],
            text: str,
    ) -> RelationMention | None:
        if isinstance(rel, (list, tuple)) and len(rel) >= 3:
            head_ref = rel[0]
            tail_ref = rel[1]
            rel_type = rel[2]
            score = rel[3] if len(rel) > 3 else None
            return self._build_relation(rel_type, score, head_ref, tail_ref, None, entities, text)

        if isinstance(rel, Mapping):
            rel_type = rel.get("relation") or rel.get("type") or rel.get("label")
            score = rel.get("confidence") or rel.get("score")
            head_ref = rel.get("head") or rel.get("subject") or rel.get("source")
            tail_ref = rel.get("tail") or rel.get("object") or rel.get("target")
            evidence = rel.get("evidence")
            return self._build_relation(rel_type, score, head_ref, tail_ref, evidence, entities, text)

        return None

    def _build_relation(
            self,
            rel_type: Any,
            score: Any,
            head_ref: Any,
            tail_ref: Any,
            evidence: Any,
            entities: List[NERMention],
            text: str,
    ) -> RelationMention | None:
        if not rel_type:
            return None

        subject_id = self._resolve_entity_ref(head_ref, entities)
        object_id = self._resolve_entity_ref(tail_ref, entities)
        if not subject_id or not object_id:
            return None

        evidence_payload = self._build_evidence(evidence, subject_id, object_id, entities, text)
        confidence = self._coerce_confidence(score)
        return RelationMention(
            id=new_id("rm"),
            type=str(rel_type),
            confidence=confidence,
            subject={"mention_id": subject_id},
            object={"mention_id": object_id},
            evidence=evidence_payload or {},
        )

    def _resolve_entity_ref(self, ref: Any, entities: List[NERMention]) -> str | None:
        if ref is None:
            return None
        if isinstance(ref, int):
            if 0 <= ref < len(entities):
                return entities[ref].get("id")
            return None
        if isinstance(ref, str):
            for ent in entities:
                if ent.get("id") == ref:
                    return ent.get("id")
            for ent in entities:
                if ent.get("text") == ref:
                    return ent.get("id")
            return None
        if isinstance(ref, Mapping):
            if ref.get("mention_id"):
                return ref.get("mention_id")
            if ref.get("id"):
                return ref.get("id")
            start = ref.get("start")
            end = ref.get("end")
            if start is not None and end is not None:
                for ent in entities:
                    if ent.get("start") == start and ent.get("end") == end:
                        return ent.get("id")
            label = ref.get("label")
            text = ref.get("text")
            for ent in entities:
                if label and text:
                    if ent.get("label") == label and ent.get("text") == text:
                        return ent.get("id")
            if text:
                for ent in entities:
                    if ent.get("text") == text:
                        return ent.get("id")
        return None

    def _build_evidence(
            self,
            evidence: Any,
            subject_id: str,
            object_id: str,
            entities: List[NERMention],
            text: str,
    ) -> Dict[str, Any] | None:
        if isinstance(evidence, Mapping):
            payload = dict(evidence)
        else:
            payload = {}

        if "span" in payload and "snippet" in payload:
            return payload

        subject = next((ent for ent in entities if ent.get("id") == subject_id), None)
        obj = next((ent for ent in entities if ent.get("id") == object_id), None)
        if not subject or not obj:
            return payload or None

        start_vals = [val for val in [subject.get("start"), obj.get("start")] if isinstance(val, int)]
        end_vals = [val for val in [subject.get("end"), obj.get("end")] if isinstance(val, int)]
        if not start_vals or not end_vals:
            return payload or None

        span_start = min(start_vals)
        span_end = max(end_vals)
        snippet = text[span_start:span_end]
        payload.setdefault("span", [span_start, span_end])
        payload.setdefault("snippet", snippet)
        return payload

    @staticmethod
    def _coerce_confidence(score: Any) -> float | None:
        if score is None:
            return None
        try:
            return float(score)
        except (TypeError, ValueError):
            return None
