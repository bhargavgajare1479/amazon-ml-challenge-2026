"""
Shared configuration for the entity-resolution pipeline.

Tune the blocking constants here first if you need to trade recall against
candidate-set size (candidate_pairs.tsv size now counts toward the final
ranking, alongside recall - see the blocking module for details).
"""

# --- Blocking / candidate generation ---
TOP_K_BLOCK = 20                 # max candidates fetched per entity per source, per signal (name/address), before pruning
MIN_SIM = 0.05                   # min cosine similarity to even consider a TF-IDF blocking candidate
ADAPTIVE_MARGIN = 0.20           # keep candidates within this cosine-sim margin of the entity's best match
ADAPTIVE_MIN_KEEP = 3            # always keep at least this many top candidates per entity (if they exist)
FALLBACK_TOP_TOKENS = 8          # token-overlap fallback candidates - only used when TF-IDF found too few
FALLBACK_TRIGGER_BELOW = 3       # fire the fallback only if an entity has fewer than this many TF-IDF candidates
MAX_CANDIDATES_PER_ENTITY = 20   # hard cap - candidate_pairs.tsv size counts toward your ranking too, keep this bounded

# --- TF-IDF vectorizer memory control ---
# On large datasets, an uncapped vectorizer (min_df=1) can build a vocabulary of
# millions of char n-grams and blow up RAM (visible as high memory + 100% disk +
# LOW cpu = your machine swapping to disk, not actually computing). These two
# settings bound memory usage regardless of dataset size.
#
# These are DEFAULTS ONLY - every value in this file can be overridden from the
# command line without editing this file, e.g.:
#   python -m src.pipeline --tfidf-max-features 60000 --min-sim 0.08
# Run `python -m src.pipeline --help` to see every available override. This
# matters because the right value depends on the machine actually running it
# (a Kaggle notebook has different RAM than a laptop) - tune per-run via CLI
# flags rather than editing this file and re-pushing.
#
# NOTE: blocking now fits TWO vectorizers (name + address), so memory roughly
# doubles vs. before at the same value. The 30000 default below was tuned on a
# ~3,000-entity SAMPLE and worked comfortably. The official test set is much
# larger (~1.7M entities per the challenge's own validator) - on the full
# dataset, start here, watch memory on the first run, and lower via
# --tfidf-max-features if needed (this trades a little recall for memory
# safety - re-check the recall printout after changing it).
TFIDF_MIN_DF = 2          # ignore n-grams that appear in fewer than this many names (mostly noise anyway)
TFIDF_MAX_FEATURES = 30000  # hard cap on vocabulary size, PER vectorizer (name and address each get their own)

# --- Nearest-neighbor search parallelism ---
# The brute-force cosine similarity search in blocking can use multiple CPU
# cores. At full dataset scale, CPU time (not memory) becomes the dominant
# cost, so on a strong machine this is the highest-leverage speed knob.
# -1 = use all available cores. Override with --n-jobs to leave some free.
NN_N_JOBS = -1

# --- Train/val split & model ---
RANDOM_STATE = 42
VAL_SIZE = 0.2

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "is_unbalance": True,
    "verbosity": -1,
    "seed": RANDOM_STATE,
}
LGB_NUM_BOOST_ROUND = 500
LGB_EARLY_STOPPING_ROUNDS = 30

# --- Threshold search ---
THRESHOLD_GRID = (0.30, 0.96, 0.02)  # (start, stop, step) for np.arange

# --- Feature columns (base pairwise features; aggregate features appended in features.py) ---
FEATURE_COLS = [
    "tfidf_sim", "name_ratio", "name_token_sort", "name_token_set", "name_jaccard",
    "addr_ratio", "addr_token_set", "addr_jaccard", "digit_jaccard", "country_match",
    "name_len_ratio",
]
AGGREGATE_FEATURE_COLS = ["rank_in_entity", "num_candidates", "gap_to_top", "top_margin"]
FEATURE_COLS_FULL = FEATURE_COLS + AGGREGATE_FEATURE_COLS