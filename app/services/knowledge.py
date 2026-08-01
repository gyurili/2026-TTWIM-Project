import json

_KB_CACHE: dict | None = None


def get_kb() -> dict:
    global _KB_CACHE
    if _KB_CACHE is None:
        with open("app/data/knowledge_base.json", encoding="utf-8") as f:
            _KB_CACHE = json.load(f)
    return _KB_CACHE


def pick_docs(tag: str, kb: dict) -> list[dict]:
    return [{**doc, "tag": tag, "type": t}
            for t, docs in kb.get(tag, {}).items()
            for doc in docs]
