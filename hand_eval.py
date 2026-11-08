"""
hand_eval.py - self-contained Short Deck Omaha evaluator used by the strategy.

Scores match the rules in the participant guide (flush > full house,
trips > straight, A-6-7-8-9 wheel). Higher score = better hand.
No imports from the starter repo, so it can be tested on its own.
"""

from itertools import combinations, permutations

HIGH_CARD, ONE_PAIR, TWO_PAIR, STRAIGHT, TRIPS, FULL_HOUSE, FLUSH, QUADS, STRAIGHT_FLUSH = range(9)
SHIFT = 20
ACE = 8

PAIRS = tuple(combinations(range(5), 2))
TRIPLES = tuple(combinations(range(5), 3))


def _pack(cat, ranks):
    s = cat
    for r in ranks:
        s = (s << 4) | r
    for _ in range(5 - len(ranks)):
        s <<= 4
    return s


def _straight_top(uniq_desc):
    if len(uniq_desc) != 5:
        return None
    if uniq_desc[0] - uniq_desc[4] == 4:
        return uniq_desc[0]
    if uniq_desc == (8, 3, 2, 1, 0):  # A-6-7-8-9 wheel, nine high
        return 3
    return None


def score_ranks(ranks_desc, is_flush):
    counts = {}
    for r in ranks_desc:
        counts[r] = counts.get(r, 0) + 1
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    shape = tuple(c for _, c in groups)
    order = tuple(r for r, _ in groups)
    top = _straight_top(tuple(sorted(counts, reverse=True)))
    if top is not None and is_flush:
        return _pack(STRAIGHT_FLUSH, (top,))
    if shape == (4, 1):
        return _pack(QUADS, order)
    if is_flush:
        return _pack(FLUSH, ranks_desc)
    if shape == (3, 2):
        return _pack(FULL_HOUSE, order)
    if shape == (3, 1, 1):
        return _pack(TRIPS, order)
    if top is not None:
        return _pack(STRAIGHT, (top,))
    if shape == (2, 2, 1):
        return _pack(TWO_PAIR, order)
    if shape == (2, 1, 1, 1):
        return _pack(ONE_PAIR, order)
    return _pack(HIGH_CARD, ranks_desc)


def build_table():
    """{36-bit card mask: score} for all 376,992 five-card hands."""
    table = {}
    cache = {}
    for a, b, c, d, e in combinations(range(36), 5):
        ranks = (e >> 2, d >> 2, c >> 2, b >> 2, a >> 2)
        fl = (a & 3) == (b & 3) == (c & 3) == (d & 3) == (e & 3)
        key = (ranks, fl)
        s = cache.get(key)
        if s is None:
            s = score_ranks(ranks, fl)
            cache[key] = s
        table[(1 << a) | (1 << b) | (1 << c) | (1 << d) | (1 << e)] = s
    return table


def pair_masks(hole):
    return [(1 << hole[i]) | (1 << hole[j]) for i, j in PAIRS]


def triple_masks(board):
    return [(1 << board[i]) | (1 << board[j]) | (1 << board[k]) for i, j, k in TRIPLES]


def best_score(table, pms, tms):
    best = 0
    for p in pms:
        for t in tms:
            s = table[p | t]
            if s > best:
                best = s
    return best


# ---- suit-isomorphism canonical key for 5 hole cards (preflop table) ----
_SUIT_PERMS = tuple(permutations(range(4)))


def canonical(hole):
    best = None
    for perm in _SUIT_PERMS:
        key = tuple(sorted(((c >> 2) << 2) | perm[c & 3] for c in hole))
        if best is None or key < best:
            best = key
    return best


def canonical_id(hole):
    """Pack the canonical 5-card tuple into one int (6 bits per card)."""
    k = canonical(hole)
    return (k[0] << 24) | (k[1] << 18) | (k[2] << 12) | (k[3] << 6) | k[4]
