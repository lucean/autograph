from typing import TypedDict, Dict, Any


class NERMention(TypedDict, total=False):
    id: str
    label: str
    text: str
    start: int | None
    end: int | None
    confidence: float | None


class RelationMention(TypedDict, total=False):
    id: str
    type: str
    confidence: float
    subject: Dict[str, Any]
    object: Dict[str, Any]
    evidence: Dict[str, Any]
