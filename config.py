from pathlib import Path
from typing import Any, Dict

def load_toml_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    import tomllib

    data = tomllib.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError("TOML root must parse to a table/dict")

    extraction = data.get("extraction")
    if not isinstance(extraction, dict):
        raise ValueError("Config must contain [extraction] table")

    entity_labels = extraction.get("entity_labels")
    if entity_labels is not None and not (
        isinstance(entity_labels, list) and all(isinstance(x, str) for x in entity_labels)
    ):
        raise ValueError("[extraction].entity_labels must be a list of strings")

    return data
