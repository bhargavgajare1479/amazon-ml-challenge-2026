"""
Multi-Channel Blocking and Candidate Generation Engine.
Implements optimized, single-pass indexed candidate retrieval per country.
"""

import os
import re
import gc
import json
import time
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Any
import pandas as pd

from .normalize import (
    clean_basic,
    normalize_business_name,
    normalize_business_address,
)


class CandidateBlocker:
    """
    Inverted-index blocker operating strictly within country boundaries.
    Channels:
      1. Exact stripped name (legal suffixes removed)
      2. Compact alphanumeric name (catches domains like 'telefutureindia.com' vs 'Tele Future India')
      3. Name word bigram (first two significant words, e.g. 'tele_future')
      4. Exact clean address
      5. Address number + first significant address word (e.g. '127_sriananthammalcompx')
      6. Rare name tokens
      7. Name 6-char prefix + location token
    """

    def __init__(self, max_token_freq: int = 15000, max_candidates_per_s1: int = 120):
        self.max_token_freq = max_token_freq
        self.max_candidates_per_s1 = max_candidates_per_s1
        
        # Inverted index tables
        self.name_exact_index = defaultdict(list)
        self.name_compact_index = defaultdict(list)
        self.name_bigram_index = defaultdict(list)
        self.addr_exact_index = defaultdict(list)
        self.addr_num_token_index = defaultdict(list)
        self.rare_token_index = defaultdict(list)
        self.name_prefix_index = defaultdict(list)

    def index_target_records(self, df_targets: pd.DataFrame):
        """
        Build inverted indexes over target records in a memory-efficient single pass.
        df_targets must contain: ['entity_id', 'business_name', 'business_address']
        """
        print(f"Building optimized target index for {len(df_targets):,} records...")
        t0 = time.time()

        for row in df_targets.itertuples(index=False):
            eid = row.entity_id
            name = row.business_name if isinstance(row.business_name, str) else ""
            addr = row.business_address if isinstance(row.business_address, str) else ""

            n_meta = normalize_business_name(name)
            a_meta = normalize_business_address(addr)

            # 1. Exact stripped name
            stripped_name = n_meta["stripped_legal"]
            if len(stripped_name) >= 3:
                self.name_exact_index[stripped_name].append(eid)

            # 2. Compact alphanumeric name
            compact = n_meta["compact"]
            if len(compact) >= 5:
                self.name_compact_index[compact].append(eid)

            # 3. Name bigram
            bigram = n_meta["bigram"]
            if bigram:
                self.name_bigram_index[bigram].append(eid)

            # 4. Exact clean address
            clean_addr = a_meta["clean"]
            if len(clean_addr) >= 8:
                self.addr_exact_index[clean_addr].append(eid)

            # 5. Address number + first significant address token
            if a_meta["numbers"] and a_meta["text_tokens"]:
                first_num = sorted(list(a_meta["numbers"]))[0]
                first_word = a_meta["text_tokens"][0]
                num_key = f"{first_num}_{first_word}"
                self.addr_num_token_index[num_key].append(eid)

            # 6. Significant name tokens (capped length)
            for t in n_meta["sig_tokens"]:
                if len(t) >= 5:
                    self.rare_token_index[t].append(eid)

            # 7. Name prefix (first 6 letters) + location
            if len(stripped_name) >= 6 and a_meta["text_tokens"]:
                pref_key = f"{stripped_name[:6]}_{a_meta['text_tokens'][0]}"
                self.name_prefix_index[pref_key].append(eid)

        elapsed = round(time.time() - t0, 2)
        print(f"Target index built in {elapsed}s.")
        print(f"  Exact name keys:    {len(self.name_exact_index):,}")
        print(f"  Compact name keys:  {len(self.name_compact_index):,}")
        print(f"  Name bigram keys:   {len(self.name_bigram_index):,}")
        print(f"  Exact addr keys:    {len(self.addr_exact_index):,}")
        print(f"  Addr num+word keys: {len(self.addr_num_token_index):,}")
        print(f"  Token keys:         {len(self.rare_token_index):,}")

    def generate_candidates_for_s1(
        self,
        business_name: str,
        business_address: str,
    ) -> Set[str]:
        """Query indexes to retrieve candidate target IDs for a single S1 record."""
        candidates = set()
        
        n_meta = normalize_business_name(business_name)
        a_meta = normalize_business_address(business_address)

        # Channel 1: Exact stripped name (High priority, up to 50)
        stripped_name = n_meta["stripped_legal"]
        if stripped_name in self.name_exact_index:
            candidates.update(self.name_exact_index[stripped_name][:50])

        # Channel 2: Compact alphanumeric name (Domains & compound words, up to 30)
        compact = n_meta["compact"]
        if compact in self.name_compact_index:
            candidates.update(self.name_compact_index[compact][:30])

        # Channel 3: Name bigram (Word pairs, up to 30)
        bigram = n_meta["bigram"]
        if bigram and bigram in self.name_bigram_index:
            candidates.update(self.name_bigram_index[bigram][:30])

        # Channel 4: Exact clean address (Up to 50)
        clean_addr = a_meta["clean"]
        if clean_addr in self.addr_exact_index:
            candidates.update(self.addr_exact_index[clean_addr][:50])

        # Channel 5: Address number + word (Up to 30)
        if a_meta["numbers"] and a_meta["text_tokens"]:
            first_num = sorted(list(a_meta["numbers"]))[0]
            first_word = a_meta["text_tokens"][0]
            num_key = f"{first_num}_{first_word}"
            if num_key in self.addr_num_token_index:
                candidates.update(self.addr_num_token_index[num_key][:30])

        # Channel 6: Name tokens (Filter overly common tokens)
        for t in n_meta["sig_tokens"]:
            if len(t) >= 5 and t in self.rare_token_index:
                cand_list = self.rare_token_index[t]
                if len(cand_list) <= self.max_token_freq:
                    candidates.update(cand_list[:25])

        # Channel 7: Name prefix + location
        if len(stripped_name) >= 6 and a_meta["text_tokens"]:
            pref_key = f"{stripped_name[:6]}_{a_meta['text_tokens'][0]}"
            if pref_key in self.name_prefix_index:
                candidates.update(self.name_prefix_index[pref_key][:25])

        # Cap candidates per S1 to prevent runaway size
        if len(candidates) > self.max_candidates_per_s1:
            candidates = set(list(candidates)[:self.max_candidates_per_s1])

        return candidates
