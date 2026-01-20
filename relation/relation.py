from typing import runtime_checkable, Protocol, List

from model.model import RelationMention, NERMention
from ner.ner_extractor import ExtractorRunMetadata


@runtime_checkable
class RelationExtractor(Protocol):
    """
    Implementations must:
    - expose a stable name
    - accept the raw text and entity mentions (segment-relative offsets)
    - return a list of RelationMention dicts with confidence
    """
    name: str

    def get_run_meta(self) -> ExtractorRunMetadata: ...

    def extract(
            self,
            text: str,
            entities: List[NERMention],
            relation_types: List[str],
            threshold: float,
    ) -> List[RelationMention]: ...
