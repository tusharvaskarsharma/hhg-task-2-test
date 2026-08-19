import re

def preprocess_query(query: str) -> str:
    # Handle leading/trailing whitespace
    query = query.strip()
    
    # Normalize repeated whitespace
    query = re.sub(r'\s+', ' ', query)
    
    # Do not transliterate Hindi/Bengali/English. Do not aggressively lowercase.
    return query
