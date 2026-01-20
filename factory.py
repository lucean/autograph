# -----------------------------
# Factories / registries
# -----------------------------
from typing import Dict, Any

from ner.ner_extractor import NERExtractor
from relation.relation import RelationExtractor


class ExtractorFactory:
    """
    Create extractors from externalised config.

    Expected config shape (YAML/JSON):
      extraction:
        ner:
          backend: "gliner2"
          params: {...}
        re:
          backend: "none" | "gliner2_re" | ...
          params: {...}
    """

    _ner_builders: Dict[str, Any] = {}
    _re_builders: Dict[str, Any] = {}

    @classmethod
    def register_ner(cls, backend: str, builder) -> None:
        cls._ner_builders[backend] = builder

    @classmethod
    def register_re(cls, backend: str, builder) -> None:
        cls._re_builders[backend] = builder

    @classmethod
    def create_ner(cls, config: Dict[str, Any]) -> NERExtractor:
        ner_cfg = (config.get("extraction") or {}).get("ner") or {}
        backend = ner_cfg.get("backend", "gliner2")
        params = ner_cfg.get("params") or {}
        if backend not in cls._ner_builders:
            raise ValueError(f"Unknown NER backend: {backend!r}. Registered: {sorted(cls._ner_builders)}")
        return cls._ner_builders[backend](params)

    @classmethod
    def create_re(cls, config: Dict[str, Any]) -> RelationExtractor:
        re_cfg = (config.get("extraction") or {}).get("re") or {}
        backend = re_cfg.get("backend", "none")
        params = re_cfg.get("params") or {}
        if backend not in cls._re_builders:
            raise ValueError(f"Unknown RE backend: {backend!r}. Registered: {sorted(cls._re_builders)}")
        return cls._re_builders[backend](params)
