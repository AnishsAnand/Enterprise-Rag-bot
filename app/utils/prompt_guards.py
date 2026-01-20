def is_meta_or_system_prompt(text: str) -> bool:
    if not text:
        return False

    q = text.strip().lower()

    return (
        q.startswith("### task") or
        q.startswith("system:") or
        q.startswith("assistant:") or
        "<chat_history>" in q or
        "suggest 3-5 relevant follow-up" in q
    )
