"""
Loading and text normalization: cleans business_name / business_address noise
(abbreviations, punctuation, case) so downstream blocking and features see
consistent text instead of raw noisy strings.
"""
import re
import pandas as pd

NAME_ABBREV = {
    "pvt": "private", "ltd": "limited", "co": "company", "corp": "corporation",
    "inc": "incorporated", "llc": "limited liability company", "llp": "limited liability partnership",
    "intl": "international", "mfg": "manufacturing", "svc": "service", "svcs": "services",
    "assoc": "associates", "bros": "brothers", "grp": "group", "dept": "department",
    "natl": "national", "tech": "technology", "ind": "industries", "indl": "industrial",
}

ADDR_ABBREV = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard", "dr": "drive",
    "ln": "lane", "ct": "court", "pl": "place", "sq": "square", "apt": "apartment",
    "bldg": "building", "fl": "floor", "flr": "floor", "no": "number", "nr": "near",
    "opp": "opposite", "stn": "station", "sec": "sector", "hwy": "highway",
    "ste": "suite", "blk": "block", "twp": "township",
}


def load_source(path: str) -> pd.DataFrame:
    """Load one source1/source2/source3 tsv file with correct dtypes and no NaNs."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
    df["business_name"] = df["business_name"].fillna("")
    df["business_address"] = df["business_address"].fillna("")
    df["country"] = df["country"].fillna("")
    return df


def basic_clean(text: str) -> str:
    text = str(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def expand_tokens(text: str, abbrev_map: dict) -> str:
    return " ".join(abbrev_map.get(t, t) for t in text.split())


def normalize_name(text: str) -> str:
    return expand_tokens(basic_clean(text), NAME_ABBREV)


def normalize_address(text: str) -> str:
    return expand_tokens(basic_clean(text), ADDR_ABBREV)


def extract_digits(text: str) -> set:
    return set(re.findall(r"\d+", str(text)))


def add_normalized_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Adds norm_name, norm_addr, name_tokens, addr_tokens, addr_digits, country_norm."""
    df = df.copy()
    df["norm_name"] = df["business_name"].apply(normalize_name)
    df["norm_addr"] = df["business_address"].apply(normalize_address)
    df["name_tokens"] = df["norm_name"].apply(lambda s: set(s.split()))
    df["addr_tokens"] = df["norm_addr"].apply(lambda s: set(s.split()))
    df["addr_digits"] = df["business_address"].apply(extract_digits)
    df["country_norm"] = df["country"].str.strip().str.lower()
    return df


def parse_ground_truth(gt_df: pd.DataFrame) -> dict:
    """train_ground_truth.tsv -> {source1_entity_id: set(matched_entity_ids)}.
    Uses itertuples (not iterrows) - itertuples is roughly 10-50x faster on large
    files, since iterrows boxes every row into a Series (expensive) while
    itertuples yields lightweight namedtuples."""
    gt_map = {}
    for row in gt_df.itertuples(index=False):
        s1id = row.source1_entity_id
        matched = row.matched_entity_ids
        gt_map[s1id] = set(x.strip() for x in matched.split(",") if x.strip()) if matched.strip() else set()
    return gt_map