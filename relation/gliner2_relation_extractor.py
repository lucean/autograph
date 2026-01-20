from __future__ import annotations

from typing import Dict, Any, List, Iterable, Mapping

from model.model import RelationMention, NERMention
from ner.ner_extractor import ExtractorRunMetadata
from relation.relation import RelationExtractor
from util import new_id, now_utc_iso


class GLiNER2RelationExtractor(RelationExtractor):
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
        if not relation_types:
            return []

        if not hasattr(self._model, "extract_relations"):
            raise RuntimeError(
                "GLiNER2RelationExtractor requires a model with extract_relations()."
            )

        raw_relations = self._model.extract_relations(
            text,
            relation_types=relation_types,
            threshold=threshold,
        )

        print(raw_relations)

        return raw_relations

