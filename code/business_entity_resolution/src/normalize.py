"""
Normalization and Tokenization Engine for Business Entity Resolution.
Supports US, India, France, and generic open-set multilingual business records.
"""

import re
import unicodedata
from typing import Dict, List, Set, Tuple, Optional, Any


# Common business legal suffixes across US, India, and France
LEGAL_SUFFIXES = {
    # US / UK / India
    "inc", "incorporated", "corp", "corporation", "llc", "llp", "ltd", "limited",
    "co", "company", "pvt", "private", "holdings", "holding", "group", "enterprises",
    "enterprise", "associates", "consulting", "services", "solutions",
    # France
    "sarl", "sas", "sasu", "sci", "snc", "sa", "eurl", "gie",
}

# Address street types and abbreviations
ADDRESS_REPLACEMENTS = {
    r"\bst\b": "street",
    r"\brd\b": "road",
    r"\bave\b": "avenue",
    r"\bav\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bbd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bhwy\b": "highway",
    r"\bpkwy\b": "parkway",
    r"\bste\b": "suite",
    r"\bapt\b": "apartment",
    r"\bfl\b": "floor",
    r"\bflr\b": "floor",
    r"\bopp\b": "opposite",
    r"\bnr\b": "near",
}

US_STATE_MAP = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire",
    "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee",
    "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}

IN_STATE_MAP = {
    "mh": "maharashtra", "dl": "delhi", "ka": "karnataka", "tn": "tamil nadu",
    "up": "uttar pradesh", "gj": "gujarat", "wb": "west bengal", "rj": "rajasthan",
    "ap": "andhra pradesh", "ts": "telangana", "tg": "telangana", "kl": "kerala",
    "mp": "madhya pradesh", "pb": "punjab", "hr": "haryana", "or": "odisha",
    "od": "odisha", "as": "assam", "br": "bihar", "jh": "jharkhand", "uk": "uttarakhand",
}


def strip_accents(text: str) -> str:
    """Normalize unicode characters (NFKD) and strip accents (e.g. é -> e)."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def clean_basic(text: str) -> str:
    """Basic clean: lowercase, strip accents, separate digits from letters, clean punctuation."""
    if not isinstance(text, str):
        return ""
    text = strip_accents(text.lower())
    text = text.replace("&", " and ")
    # Separate attached letters and digits: e.g. 'No127' -> 'No 127', 'Flat3A' -> 'Flat 3 A'
    text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_business_name(name: str) -> Dict[str, Any]:
    """
    Produce multi-view representations for a business name:
      - raw: original string
      - clean: lowercased, accent-stripped, clean punctuation
      - stripped_legal: clean name with legal suffixes removed
      - compact: alphanumeric stripped of whitespace and domain suffixes
      - acronym: first letter of each significant word
      - tokens: list of tokens
      - sig_tokens: tokens excluding legal suffixes
      - bigrams: word pairs for robust multi-word indexing
    """
    if not isinstance(name, str):
        name = ""
        
    raw = name.strip()
    clean = clean_basic(raw)
    
    # Strip URL domain extensions (e.g. 'maurewilliamscolombier.com', '.in', '.org')
    clean = re.sub(r"^www\s+", "", clean)
    clean = re.sub(r"\s*(com|org|net|in|fr|co|io|biz|info)$", "", clean)
    
    tokens = clean.split() if clean else []
    
    # Significant tokens (excluding legal words)
    sig_tokens = [t for t in tokens if t not in LEGAL_SUFFIXES and len(t) > 1]
    stripped_legal = " ".join(sig_tokens)
    
    # Compact alphanumeric (helps match 'telefutureindia' with 'tele future india')
    compact = re.sub(r"[^a-z0-9]", "", stripped_legal) if stripped_legal else re.sub(r"[^a-z0-9]", "", clean)
    
    # Acronym representation
    acronym = "".join(t[0] for t in sig_tokens if t[0].isalnum()) if sig_tokens else ""
    
    # Name bigrams (first two significant words, e.g. 'tele_future')
    bigram = f"{sig_tokens[0]}_{sig_tokens[1]}" if len(sig_tokens) >= 2 else ""

    return {
        "raw": raw,
        "clean": clean,
        "stripped_legal": stripped_legal,
        "compact": compact,
        "acronym": acronym,
        "bigram": bigram,
        "tokens": tokens,
        "sig_tokens": sig_tokens,
    }


def normalize_business_address(address: str) -> Dict[str, Any]:
    """
    Produce multi-view representations for a business address:
      - raw: original string
      - clean: standardized abbreviations (rd->road, st->street, etc.)
      - numbers: set of extracted numeric strings (e.g. house number, postal code)
      - postal_candidates: 5-digit or 6-digit postal/PIN candidates
      - tokens: list of clean words
    """
    if not isinstance(address, str):
        address = ""
        
    raw = address.strip()
    clean = clean_basic(raw)
    
    # Apply standard address replacements
    for pattern, replacement in ADDRESS_REPLACEMENTS.items():
        clean = re.sub(pattern, replacement, clean)
        
    tokens = clean.split() if clean else []
    
    # Extract numeric components (house numbers, floor numbers, PIN codes)
    numbers = set(re.findall(r"\b\d+\b", clean))
    
    # Identify postal code candidates (5 digits for US/FR, 6 digits for India)
    postal_candidates = set(re.findall(r"\b\d{5,6}\b", clean))
    
    # Extract significant non-numeric words (len >= 3)
    text_tokens = [t for t in tokens if not t.isdigit() and len(t) >= 3]
    
    return {
        "raw": raw,
        "clean": clean,
        "numbers": numbers,
        "postal_candidates": postal_candidates,
        "tokens": tokens,
        "text_tokens": text_tokens,
    }


if __name__ == "__main__":
    test_name = "Maure Williams Colombier Inc."
    test_addr = "85 Wanye Ave, Ticonderoga Townshiip, NY 12883"
    
    n_res = normalize_business_name(test_name)
    a_res = normalize_business_address(test_addr)
    
    print("Name normalization test:")
    print(n_res)
    print("\nAddress normalization test:")
    print(a_res)
