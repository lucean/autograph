import dataclasses
from typing import runtime_checkable, Protocol, Dict, Any, List

from model.model import NERMention


@dataclasses.dataclass(frozen=True)
class ExtractorRunMetadata:
    run_id: str
    timestamp_utc: str
    backend: str
    backend_version: str | None = None
    config: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        out = {
            "run_id": self.run_id,
            "timestamp_utc": self.timestamp_utc,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "config": self.config,
        }
        return {k: v for k, v in out.items() if v is not None}


@runtime_checkable
class NERExtractor(Protocol):
    name: str

    def get_run_meta(self) -> ExtractorRunMetadata: ...

    def extract(self, text: str, labels: List[str], threshold: float, spans: bool) -> List[NERMention]: ...
