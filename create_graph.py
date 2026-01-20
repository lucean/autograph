from __future__ import annotations

import json
from typing import Any, Dict

from config import load_toml_config
from factory import ExtractorFactory
from ner.gliner2_ner_extractor import GLiNER2NERExtractor
from relation.gliner2_relation_extractor import GLiNER2RelationExtractor

# Register backends
ExtractorFactory.register_ner("gliner2", lambda params: GLiNER2NERExtractor(params))
ExtractorFactory.register_re("gliner2_re", lambda params: GLiNER2RelationExtractor(params))


# -----------------------------
# Minimal usage example (to be integrated into your pipeline)
# -----------------------------

def example_run(config: Dict[str, Any], segment_text: str) -> Dict[str, Any]:
    ner = ExtractorFactory.create_ner(config)

    extraction_cfg = config.get("extraction") or {}
    entity_labels = extraction_cfg.get("entity_labels") or []
    relation_types = extraction_cfg.get("relation_types") or []
    ner_threshold = float(extraction_cfg.get("ner_threshold", 0.55))
    re_threshold = float(extraction_cfg.get("re_threshold", 0.50))
    with_spans = extraction_cfg.get("with_spans") or False

    ents = ner.extract(segment_text, entity_labels, threshold=ner_threshold, spans=with_spans)

    # Only do RE if relation types exist AND backend is not disabled
    re_cfg = extraction_cfg.get("re") or {}
    re_backend = re_cfg.get("backend", "none")
    do_re = bool(relation_types) and re_backend not in (None, "", "none", "null", "disabled", False)

    rels = []
    re_meta = None
    if do_re:
        re_extractor = ExtractorFactory.create_re(config)
        rels = re_extractor.extract(segment_text, ents, relation_types, threshold=re_threshold)
        re_meta = re_extractor.get_run_meta().as_dict()

    return {
        "run": {
            "ner": ner.get_run_meta().as_dict(),
            **({"re": re_meta} if re_meta else {}),
        },
        "entity_mentions": ents,
        "relation_mentions": rels,
    }


config = load_toml_config("config.toml")

result = example_run(config,
                     segment_text="Apple Inc. CEO Tim Cook announced the new iPhone 15 in Cupertino, California on September 12, 2023.")

print(json.dumps(result, indent=2))
