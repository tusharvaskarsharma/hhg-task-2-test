def resolve_language(query: str, requested_lang: str = None) -> str:
    if requested_lang in ["hi", "en", "bn"]:
        return requested_lang
    return "hi"
