def normalize_language(value: str | None) -> str | None:
    """
    Maps hi, hi-IN, hindi -> hi;
    bn, bn-IN, bengali -> bn;
    en, en-IN, english -> en.
    Returns None for unknown values.
    """
    if not value:
        return None
    val = value.lower().strip()
    if val.startswith("hi") or val == "hindi":
        return "hi"
    if val.startswith("bn") or val == "bengali":
        return "bn"
    if val.startswith("en") or val == "english":
        return "en"
    return None

def detect_text_language(text: str) -> str | None:
    """
    Detects language based on Unicode blocks.
    Devanagari -> hi
    Bengali -> bn
    Latin/Default -> en
    """
    if not text:
        return None
        
    deva_count = 0
    bengali_count = 0
    latin_count = 0
    
    for char in text:
        cp = ord(char)
        if 0x0900 <= cp <= 0x097F:
            deva_count += 1
        elif 0x0980 <= cp <= 0x09FF:
            bengali_count += 1
        elif 0x0000 <= cp <= 0x007F and char.isalpha():
            latin_count += 1
            
    total = deva_count + bengali_count + latin_count
    if total == 0:
        return "en" # fallback if punctuation only
        
    if deva_count > bengali_count and deva_count > latin_count:
        return "hi"
    if bengali_count > deva_count and bengali_count > latin_count:
        return "bn"
    return "en"

def resolve_language(query: str, requested_lang: str = None) -> str:
    norm = normalize_language(requested_lang)
    if norm:
        return norm
    detected = detect_text_language(query)
    return detected or "hi"
