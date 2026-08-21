import re
import unicodedata

def preprocess_query(query: str) -> str:
    # Unicode NFC normalization (critical for Hindi/Bengali consistency)
    query = unicodedata.normalize("NFC", query)
    
    # Handle leading/trailing whitespace
    query = query.strip()
    
    # Normalize repeated whitespace
    query = re.sub(r'\s+', ' ', query)
    
    # Do not transliterate Hindi/Bengali/English. Do not aggressively lowercase.
    return query

