from typing import Dict, Any, List

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
        self.device = params.get("device")
        self.mode = params.get("mode", "unwired")
        self.extra = params.get("extra") or {}
        self._model = None

        self._run_meta = ExtractorRunMetadata(
            run_id=new_id("re"),
            timestamp_utc=now_utc_iso(),
            backend=self.name,
            backend_version=None,
            config={
                "model_name": self.model_name,
                "device": self.device,
                "mode": self.mode,
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
        # If not wired, return empty but keep deterministic behaviour.
        if self._model is None:
            return []

        # Example structure you should return once wired:
        # return [
        #   RelationMention(
        #     id=new_id("rm"),
        #     type="has_email_address",
        #     confidence=0.87,
        #     subject={"mention_id": entities[0]["id"]},
        #     object={"mention_id": entities[1]["id"]},
        #     evidence={"span": [..,..], "snippet": "..."},
        #     meta={"backend": self.name},
        #   )
        # ]
        return []