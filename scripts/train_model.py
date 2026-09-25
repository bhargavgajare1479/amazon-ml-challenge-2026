"""
Run supervised matcher training.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.train import train_matcher

if __name__ == "__main__":
    train_matcher(sample_s1_count=20000, neg_to_pos_ratio=6, seed=42)
