import json

_KB_CACHE: dict | None = None


def get_kb() -> dict:
    global _KB_CACHE
    if _KB_CACHE is None:
        with open("app/data/knowledge_base.json", encoding="utf-8") as f:
            _KB_CACHE = json.load(f)
    return _KB_CACHE


def pick_docs(tag: str, kb: dict) -> list[dict]:
    return [{**doc, "tag": tag, "type": t} for t, doc in kb.get(tag, {}).items()]


def validate_sources(llm_source_ids: list[str], kb: dict) -> list[dict]:
    id_index = {
        doc["id"]: {"id": doc["id"], "source": doc["source"], "tag": tag, "type": t}
        for tag, types in kb.items()
        for t, doc in types.items()
    }
    return [id_index[sid] for sid in llm_source_ids if sid in id_index]
