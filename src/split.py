"""
Entity-level train/validation split. Splitting by entity (not by row) matters:
every candidate pair belonging to one Source-1 entity must stay entirely in
train or entirely in validation, or you leak information and get an
overoptimistic validation score.
"""
from sklearn.model_selection import train_test_split

from . import config


def split_entities(s1_entity_ids, test_size: float = config.VAL_SIZE,
                    random_state: int = config.RANDOM_STATE):
    train_ids, val_ids = train_test_split(
        list(s1_entity_ids), test_size=test_size, random_state=random_state
    )
    return set(train_ids), set(val_ids)
