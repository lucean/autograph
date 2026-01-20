import uuid
from typing import Dict, Any, List

from model.model import NERMention
from ner.ner_extractor import NERExtractor, ExtractorRunMetadata
from util import new_id, now_utc_iso


class GLiNER2NERExtractor(NERExtractor):
    name = "gliner2"

    def __init__(self, params: Dict[str, Any]):
        self.model_name = params.get("model_name") or params.get("name")
        if not self.model_name:
            raise ValueError("GLiNER2NERExtractor requires params.model_name (or params.name)")
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
                # Some builds do not implement .to(); ignore.
                pass

        # Best-effort version reporting
        backend_version = None
        try:
            import gliner2
            backend_version = getattr(gliner2, "__version__", None)
        except Exception:
            pass

        self._run_meta = ExtractorRunMetadata(
            run_id=new_id("ner"),
            timestamp_utc=now_utc_iso(),
            backend=self.name,
            backend_version=backend_version,
            config={"model_name": self.model_name, "device": self.device,
                    **({"extra": self.extra} if self.extra else {})},
        )

    def get_run_meta(self) -> ExtractorRunMetadata:
        return self._run_meta

    def extract(self, text: str, labels: List[str], threshold: float, spans: bool) -> List[
        NERMention]:
        results = self._model.extract_entities(text, labels, include_confidence=True, threshold=threshold,
                                               include_spans=spans)

        out: List[NERMention] = []
        for label, entities in results['entities'].items():
            for entity in entities:
                out.append(NERMention(
                    id=str(uuid.uuid4()),
                    label=label,
                    text=entity['text'],
                    confidence=entity['confidence'],
                    start=entity.get('start') or None,
                    end=entity.get('end') or None
                ))
        return out
