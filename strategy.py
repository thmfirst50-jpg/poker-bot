"""
strategy.py - the brain of the bot. agent.py calls into this.

Approach
  * Preflop: lookup table of every 5-card holding's equity vs a random hand
    (precomputed offline, preflop_data.bin) -> percentile-based open/limp/fold,
    and equity-vs-estimated-raising-range when facing a raise.
  * Postflop: Monte Carlo equity with a time-adaptive sample count. Opponent
    hands are sampled from a range narrowed by what they did this hand
    (preflop raise -> stronger preflop hands; postflop raises -> stronger
    made hands), scaled by how often this opponent raises in general.
  * Pot odds decide call/fold; equity thresholds decide value raises;
    semi-bluffs only against opponents who actually fold.
  * Opponent model: raise frequencies (preflop / postflop) and how often they
    fold to our bets, all kept in `memory` (reset every half).
  * Time bank: tracks its own thinking time and shrinks simulations so a half
    stays well under the 8 s bank. Any exception -> safe legal action.

Only depends on the standard library + hand_eval.py + preflop_data.bin.
Actions/streets are compared by .name so this works with game.py's enums.
"""

import os
import random
from array import array
from math import comb
from time import perf_counter

import hand_eval as he

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------
TOTAL_BUDGET_S = 6.2        # target charged time per half (bank is 8 s; the rest
                            # is margin for a slower / shared tournament CPU)
OVERHEAD_S = 0.0006         # per-request server cost (JSON, FastAPI) the judge may
                            # charge beyond its free first 1 ms of every request
HARD_STOP_S = 6.6           # estimated total beyond this -> minimal simulations
PANIC_S = 7.0               # beyond this -> no simulations at all
BANK_S = 7.3                # the judge's 8 s bank, minus a safety margin
EST_DECISIONS_PER_HAND = 2.3   # floors for the measured per-hand decision rates
EST_POST_DECISIONS_PER_HAND = 1.2
MIN_TRIALS, MAX_TRIALS = 12, 600
HANDS_PER_HALF = 1000

CALL_MARGIN = {"FLOP": 0.05, "TURN": 0.03, "RIVER": 0.0}
VALUE_RAISE = {"FLOP": 0.66, "TURN": 0.70, "RIVER": 0.74}   # bet/raise for value
POT_RAISE = {"FLOP": 0.76, "TURN": 0.80, "RIVER": 0.84}     # use the big size
PF_OPEN_RAISE = 0.62        # button: raise hands at/above this preflop percentile
PF_OPEN_MIX = 0.45          #   ...and raise this band sometimes
PF_MIX_P = 0.25
PF_LIMP = 0.15              # button: fold below this percentile
PF_BB_RAISE = 0.70          # big blind after a limp: raise at/above
PF_3BET = 0.58              # facing a raise: re-raise when equity vs range >= this
PF_CALL_MARGIN = 0.04       # facing a raise: extra equity needed over pot odds
BLUFF_FOLD = 0.40           # semi-bluff only if opponent folds to bets more than this
BLUFF_EQ = 0.35             # ...and our equity is at least this
LEARN_CALL_LIMIT = 0.025    # max mid-match tightening/loosening of postflop calls
LEARN_RAISE_LIMIT = 0.025   # max mid-match shift of postflop raise thresholds
LEARN_RATE = 0.2            # EMA weight of each new hand result
LEARN_CALL_GAIN = 0.025     # threshold shift per unit of (negative) average call result
SD_BLEND = 0.0              # weight of showdown-measured bluff rate in range narrowing
SD_SCALE = 1.5              # bluff rate -> share of weak hands kept in their betting range
SD_TIGHTEN = 0.0            # one-way showdown learning: only ever TIGHTENS the range of an
                            # opponent whose bets are shown to be honest (never loosens vs maniacs)
SD_CONF = 20.0              # showdowns needed for half weight in SD_TIGHTEN
SD_TSCALE = 2.0             # weak-showdown rate -> share of weak hands kept (random bettor: 0.5 -> 1.0)

# Constants the offline growth loop (bot_tools/grow.py) may override via brain.json
TUNABLE = ("CALL_MARGIN", "VALUE_RAISE", "POT_RAISE", "PF_OPEN_RAISE", "PF_OPEN_MIX",
           "PF_MIX_P", "PF_LIMP", "PF_BB_RAISE", "PF_3BET", "PF_CALL_MARGIN", "BLUFF_FOLD",
           "BLUFF_EQ", "LEARN_CALL_LIMIT", "LEARN_RAISE_LIMIT", "LEARN_RATE",
           "LEARN_CALL_GAIN", "SD_BLEND", "SD_SCALE", "SD_TIGHTEN", "SD_CONF", "SD_TSCALE", "TOTAL_BUDGET_S")
BRAIN_INFO = {"generation": 0}

# --------------------------------------------------------------------------
# Static data (built once at import)
# --------------------------------------------------------------------------
TABLE = None
_GET = None
PF_EQ = None      # array('H'): colex rank -> equity*1000 vs random
PF_PCT = None     # array('H'): colex rank -> weighted percentile*1000
RANGE_MEAN = []   # RANGE_MEAN[i]: mean equity of the top (i+1)*5% of hands
_BINOM = [[comb(n, k) for k in range(6)] for n in range(37)]
N_HANDS = comb(36, 5)


def load_brain(path=None):
    """Apply learned constants from brain.json (written by bot_tools/grow.py).
    Missing or malformed file -> keep the built-in defaults."""
    import json
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "brain.json")
    try:
        with open(path) as f:
            brain = json.load(f)
    except (OSError, ValueError):
        return
    if not isinstance(brain, dict):
        return
    g = globals()
    for k, v in brain.get("params", {}).items():
        if k not in TUNABLE:
            continue
        cur = g[k]
        try:
            if isinstance(cur, dict) and isinstance(v, dict):
                new = dict(cur)
                new.update({st: float(x) for st, x in v.items() if st in cur})
                g[k] = new
            elif isinstance(cur, (int, float)) and not isinstance(v, (dict, list)):
                g[k] = float(v)
        except (TypeError, ValueError):
            pass
    BRAIN_INFO.update({k: v for k, v in brain.items() if k != "params"})


BENCH_BASE = None


def _bench():
    """~20 us of fixed CPU work; its wall time tracks how loaded the machine is."""
    t0 = perf_counter()
    x = 0
    for i in range(400):
        x += i * i
    return perf_counter() - t0


def init():
    """Build lookup tables. Call once at server start (agent.py does)."""
    global TABLE, _GET, PF_EQ, PF_PCT, RANGE_MEAN, BENCH_BASE
    if BENCH_BASE is None:
        BENCH_BASE = min(_bench() for _ in range(200))
    if TABLE is not None:
        return
    load_brain()
    TABLE = he.build_table()
    _GET = TABLE.__getitem__
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preflop_data.bin")
    data = array("H")
    with open(path, "rb") as f:
        data.fromfile(f, 2 * N_HANDS)
    PF_EQ, PF_PCT = data[:N_HANDS], data[N_HANDS:]
    # mean equity of the top-f range, f = 5%, 10%, ..., 100%
    buckets = [[0, 0] for _ in range(20)]
    for eq, pct in zip(PF_EQ, PF_PCT):
        top = 1000 - pct                       # 0 = best hand
        for i in range(min(19, top * 20 // 1000), 20):
            buckets[i][0] += eq
            buckets[i][1] += 1
    RANGE_MEAN = [s / max(1, n) / 1000.0 for s, n in buckets]


def colex(cards):
    """Index of a 5-card set (any order) in 0..C(36,5)-1."""
    c = sorted(cards)
    b = _BINOM
    return b[c[0]][1] + b[c[1]][2] + b[c[2]][3] + b[c[3]][4] + b[c[4]][5]


def range_mean(f):
    i = min(19, max(0, int(f * 20 + 0.5) - 1))
    return RANGE_MEAN[i]


# --------------------------------------------------------------------------
# Memory / opponent model
# --------------------------------------------------------------------------
def new_half(memory):
    import time
    memory["half_id"] = int(time.time() * 1000)
    memory.update(
        t_used=0.0,
        decisions=0, post_decisions=0,
        hand=-1, seen=0,
        pf_act=3.0, pf_raise=1.0,          # priors: ~1/3 raise frequency
        post_act=3.0, post_raise=1.0,
        bet_faced=3.0, bet_folded=1.0,     # how often opp folds to our bets
        pending_bet=False,
        net=0, hands_done=0, showdowns=0, sd_won=0,   # progress stats (local tracking)
        slow=1.0,                         # machine slowdown vs startup (micro-benchmark)
        ov_used=0.0,                      # estimated hidden server overhead charged so far
        rng=random.Random(12345),
        us_flop=60.0, us_river=30.0,      # measured cost per MC trial (EMA)
        fixed_s=0.0004,                   # measured non-simulation cost per decision (EMA)
        mc_s=0.0,                         # simulation time of the current decision
        learning_pending={},              # hand index -> unique call/raise contexts
        learning_stats={},                # context -> (sample count, EMA reward)
        sd_n=4.0, sd_bluff=1.0,           # showdowns after opp postflop aggression / weak ones
        sd_m=4.0, sd_weak=2.0,            # same showdowns / hand below a random hand's median
        opp_post_aggr=False,              # opp bet or raised postflop this hand
    )


def _absorb_history(obs, memory):
    """Update opponent stats with any opponent actions we haven't seen yet."""
    if obs.hand_index != memory["hand"]:
        memory["hand"] = obs.hand_index
        memory["opp_post_aggr"] = False
        memory["seen"] = 0
        memory["pending_bet"] = False
    hist = obs.action_history
    for a in hist[memory["seen"]:]:
        if a.is_me:
            continue
        name = a.action.name
        raised = name.startswith("RAISE")
        if a.street.name == "PREFLOP":
            memory["pf_act"] += 1
            memory["pf_raise"] += raised
        else:
            memory["post_act"] += 1
            memory["post_raise"] += raised
            if raised:
                memory["opp_post_aggr"] = True
        if memory["pending_bet"]:
            memory["bet_faced"] += 1          # they responded without folding
            memory["pending_bet"] = False
    memory["seen"] = len(hist)


def _record_learning_decision(obs, action, memory):
    """Remember call/raise contexts until this hand's outcome arrives."""
    name = action.name
    if name == "CHECK_CALL" and obs.to_call > 0:
        kind = "call"
    elif name.startswith("RAISE"):
        kind = "raise"
    else:
        return
    key = f"{kind}_{obs.street.name}"
    pending = memory.setdefault("learning_pending", {})
    contexts = pending.setdefault(str(obs.hand_index), set())
    contexts.add(key)


def _learned_adjustment(memory, context, limit, gain=None):
    """Return a small, sample-shrunk threshold adjustment from past results."""
    count, avg_reward = memory.get("learning_stats", {}).get(context, (0, 0.0))
    if count == 0:
        return 0.0
    confidence = count / (count + 8.0)
    # Losing outcomes make the corresponding action more selective; winning
    # outcomes relax it slightly. Clamp so noisy short-term results cannot
    # overwhelm the hand-strength and pot-odds rules.
    g = limit if gain is None else gain
    return max(-limit, min(limit, -avg_reward * g * confidence))


STATS_FILE = os.environ.get("BOT_STATS_FILE")    # local progress tracking only; unset in the tournament


def _write_stats(memory, final=False):
    """Atomically dump a progress snapshot for bot_tools/dashboard.py. Never raises."""
    try:
        import json
        import time
        h = max(1, memory["hands_done"])
        snap = {
            "time": time.time(), "final": final, "half_id": memory.get("half_id"),
            "hands": memory["hands_done"], "net_chips": memory["net"],
            "bb_per_hand": memory["net"] / 2.0 / h,
            "showdowns": memory["showdowns"], "sd_won": memory["sd_won"],
            "think_s": round(memory["t_used"], 3),
            "brain_generation": BRAIN_INFO.get("generation", 0),
            "opp": {
                "pf_raise": round(memory["pf_raise"] / memory["pf_act"], 3),
                "post_raise": round(memory["post_raise"] / memory["post_act"], 3),
                "fold_to_bet": round(memory["bet_folded"] / memory["bet_faced"], 3),
                "sd_bluff": round(memory["sd_bluff"] / memory["sd_n"], 3),
                "sd_weak": round(memory.get("sd_weak", 2.0) / memory.get("sd_m", 4.0), 3),
            },
            "learned": {k: round(_learned_adjustment(memory, k, LEARN_CALL_LIMIT), 4)
                        for k in sorted(memory.get("learning_stats", {}))},
        }
        tmp = STATS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(snap, f)
        os.replace(tmp, STATS_FILE)
    except Exception:
        pass


def on_hand_end(result, memory):
    if "net" in memory:
        memory["net"] += result.my_net
        memory["hands_done"] += 1
        if result.went_to_showdown:
            memory["showdowns"] += 1
            memory["sd_won"] += result.winner == "me"
        if STATS_FILE and (memory["hands_done"] % 25 == 0 or result.hand_index >= HANDS_PER_HALF - 1):
            _write_stats(memory, final=result.hand_index >= HANDS_PER_HALF - 1)
    if memory.get("opp_post_aggr"):
        memory["opp_post_aggr"] = False
        if result.went_to_showdown and result.opp_hole and len(result.board) == 5:
            board = list(result.board)
            tms = he.triple_masks(board)
            val = _best(he.pair_masks(list(result.opp_hole)), tms)
            memory["sd_n"] += 1
            memory["sd_bluff"] += (val >> he.SHIFT) < he.TWO_PAIR
            # Weak = beaten by most random hands on this board. With 5 hole cards even
            # random hands usually make two pair, so a fixed category can't tell bluffs.
            used = set(board) | set(result.opp_hole)
            rest = [c for c in range(36) if c not in used]
            rng = memory.get("rng") or random
            beat = sum(_best(he.pair_masks(rng.sample(rest, 5)), tms) > val for _ in range(24))
            memory["sd_m"] = memory.get("sd_m", 4.0) + 1
            memory["sd_weak"] = memory.get("sd_weak", 2.0) + (beat > 12)
    if memory.get("pending_bet"):
        memory["bet_faced"] += 1
        if not result.went_to_showdown and result.winner == "me":
            memory["bet_folded"] += 1
        memory["pending_bet"] = False

    pending = memory.setdefault("learning_pending", {})
    contexts = pending.pop(str(result.hand_index), ())
    if contexts:
        reward = max(-1.0, min(1.0, result.my_net / max(1, result.pot)))
        stats = memory.setdefault("learning_stats", {})
        for context in contexts:
            count, avg_reward = stats.get(context, (0, 0.0))
            # EMA with a conservative fixed learning rate; outcome is normalized
            # by pot size so one unusually large hand has limited influence.
            stats[context] = (count + 1, (1.0 - LEARN_RATE) * avg_reward + LEARN_RATE * reward)


# --------------------------------------------------------------------------
# Equity
# --------------------------------------------------------------------------
def _best(pms, tms):
    return max(map(_GET, [p | t for p in pms for t in tms]))


def _postflop_equity(obs, memory, n_trials, opp_pf_raised, opp_post_raises):
    rng = memory["rng"]
    hole = list(obs.hole_cards)
    board = list(obs.board)
    known = set(hole) | set(board)
    rest = [c for c in range(36) if c not in known]
    need = 5 - len(board)
    my_pms = he.pair_masks(hole)
    cur_tms = he.triple_masks(board) if len(board) == 5 else None
    if len(board) == 3:
        cur_board_tms = [(1 << board[0]) | (1 << board[1]) | (1 << board[2])]
    elif len(board) == 4:
        cur_board_tms = [(1 << board[i]) | (1 << board[j]) | (1 << board[k])
                         for i, j, k in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))]
    else:
        cur_board_tms = cur_tms
    my_river = _best(my_pms, cur_tms) if cur_tms else None

    # Range narrowing, scaled by how informative this opponent's raises are.
    pf_freq = memory["pf_raise"] / memory["pf_act"]
    post_freq = memory["post_raise"] / memory["post_act"]
    pf_cut = int(1000 * (1.0 - min(1.0, pf_freq * 1.3))) if opp_pf_raised else 0
    pf_keep = min(1.0, max(0.25, pf_freq * 1.2))
    if opp_post_raises:
        cat_needed = he.TWO_PAIR if opp_post_raises == 1 else he.STRAIGHT
        post_keep = min(1.0, max(0.15, post_freq * 1.2))
        if SD_BLEND:
            sd_keep = min(1.0, SD_SCALE * memory["sd_bluff"] / memory["sd_n"])
            post_keep = (1.0 - SD_BLEND) * post_keep + SD_BLEND * max(0.05, sd_keep)
        extra = memory["sd_m"] - 4.0                   # real showdowns beyond the prior
        if SD_TIGHTEN and extra > 0:
            sd_keep = max(0.02, SD_TSCALE * memory["sd_weak"] / memory["sd_m"])
            if sd_keep < post_keep:
                w = SD_TIGHTEN * extra / (extra + SD_CONF)
                post_keep = (1.0 - w) * post_keep + w * sd_keep
    else:
        cat_needed, post_keep = -1, 1.0

    wins = 0.0
    n = 0
    attempts = 0
    max_attempts = n_trials * 4
    pct = PF_PCT
    sample = rng.sample
    rand = rng.random
    shift = he.SHIFT
    while n < n_trials and attempts < max_attempts:
        attempts += 1
        draw = sample(rest, 5 + need)
        opp = draw[:5]
        if pf_cut and pct[colex(opp)] < pf_cut and rand() > pf_keep:
            continue
        opp_pms = he.pair_masks(opp)
        if cat_needed >= 0 and (_best(opp_pms, cur_board_tms) >> shift) < cat_needed \
                and rand() > post_keep:
            continue
        if need:
            tms = he.triple_masks(board + draw[5:])
            a = _best(my_pms, tms)
            b = _best(opp_pms, tms)
        else:
            a = my_river
            b = _best(opp_pms, cur_tms)
        wins += 1.0 if a > b else (0.5 if a == b else 0.0)
        n += 1
    return wins / n if n else 0.5


def _trial_budget(obs, memory, cost_us):
    dec = memory["decisions"]
    spent = memory["t_used"] + memory["ov_used"]       # what the judge has likely charged
    if spent > PANIC_S:
        return 0
    hands_done = obs.hand_index + 1
    all_ph = max(EST_DECISIONS_PER_HAND, dec / hands_done)
    post_ph = max(EST_POST_DECISIONS_PER_HAND, memory["post_decisions"] / hands_done)
    hands_left = max(1, HANDS_PER_HALF - obs.hand_index)
    fixed = OVERHEAD_S * memory["slow"] + memory["fixed_s"]
    think_left = TOTAL_BUDGET_S - spent - fixed * hands_left * all_ph
    if spent > HARD_STOP_S or think_left <= 0:
        # Out of planned budget: tiny simulations while the real bank has room,
        # then the zero-cost estimate, so the bank never runs dry (dry = check/fold).
        bank_left = BANK_S - spent - fixed * hands_left * all_ph
        return MIN_TRIALS if bank_left > hands_left * post_ph * MIN_TRIALS * cost_us * 1e-6 else 0
    per_decision = think_left / (hands_left * post_ph)
    n = int(per_decision * 1e6 / cost_us)
    return max(MIN_TRIALS, min(MAX_TRIALS, n))


_CAT_EQ = (0.18, 0.36, 0.56, 0.66, 0.72, 0.86, 0.90, 0.97, 0.99)


def _cheap_equity(obs):
    """Near-zero-cost equity guess from our current made-hand category."""
    b = list(obs.board)
    if len(b) == 5:
        tms = he.triple_masks(b)
    else:
        n = len(b)
        tms = [(1 << b[i]) | (1 << b[j]) | (1 << b[k])
               for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)]
    cat = _best(he.pair_masks(list(obs.hole_cards)), tms) >> he.SHIFT
    eq = _CAT_EQ[cat]
    return eq - 0.06 * _opp_raises(obs) if len(b) == 5 else eq


# --------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------
def _pick(obs, *names):
    """First action among `names` that is legal, else CHECK_CALL."""
    legal = {a.name: a for a in obs.legal_actions}
    for nm in names:
        if nm in legal:
            return legal[nm]
    return legal["CHECK_CALL"]


def _opp_raises(obs, street_name=None):
    return sum(1 for a in obs.action_history
               if not a.is_me and a.action.name.startswith("RAISE")
               and (street_name is None or a.street.name == street_name))


def _preflop(obs, memory):
    cid = colex(obs.hole_cards)
    eq = PF_EQ[cid] / 1000.0
    pct = PF_PCT[cid] / 1000.0
    rng = memory["rng"]
    opp_raised = _opp_raises(obs, "PREFLOP") > 0

    if not opp_raised:
        if obs.to_call > 0:                       # button, first to act (limp/raise/fold)
            raise_adjust = _learned_adjustment(memory, "raise_PREFLOP", 0.025)
            if pct >= PF_OPEN_RAISE + raise_adjust or (pct >= PF_OPEN_MIX + raise_adjust and rng.random() < PF_MIX_P):
                return _pick(obs, "RAISE_POT")
            if pct >= PF_LIMP:
                return _pick(obs, "CHECK_CALL")
            return _pick(obs, "FOLD")
        # big blind after a limp: free flop, raise the best hands
        raise_adjust = _learned_adjustment(memory, "raise_PREFLOP", 0.025)
        if pct >= PF_BB_RAISE + raise_adjust:
            return _pick(obs, "RAISE_POT")
        return _pick(obs, "CHECK_CALL")

    # Facing a raise: equity vs the opponent's estimated raising range.
    f = min(1.0, max(0.10, memory["pf_raise"] / memory["pf_act"] * 1.3))
    eq_vs = 0.5 + eq - range_mean(f)
    need = obs.to_call / (obs.pot + obs.to_call)
    call_adjust = _learned_adjustment(memory, "call_PREFLOP", 0.02)
    if eq_vs >= PF_3BET + call_adjust and obs.raises_this_street < 4:
        return _pick(obs, "RAISE_POT", "CHECK_CALL")
    if obs.to_call == 0 or eq_vs >= need + PF_CALL_MARGIN + call_adjust:
        return _pick(obs, "CHECK_CALL")
    return _pick(obs, "FOLD")


def _postflop(obs, memory):
    st = obs.street.name
    opp_pf_raised = _opp_raises(obs, "PREFLOP") > 0
    opp_post = sum(1 for a in obs.action_history
                   if not a.is_me and a.street.name != "PREFLOP"
                   and a.action.name.startswith("RAISE"))
    key = "us_river" if st == "RIVER" else "us_flop"
    n = _trial_budget(obs, memory, memory[key])
    t0 = perf_counter()
    if n:
        eq = _postflop_equity(obs, memory, n, opp_pf_raised, opp_post)
        memory["mc_s"] = perf_counter() - t0
        memory[key] = 0.8 * memory[key] + 0.2 * memory["mc_s"] * 1e6 / n
    else:
        eq = _cheap_equity(obs)

    rng = memory["rng"]
    fold_rate = memory["bet_folded"] / memory["bet_faced"]
    can_raise = any(a.name.startswith("RAISE") for a in obs.legal_actions)
    big = "RAISE_POT"
    # vs. calling stations bet bigger for value; vs. folders size down slightly
    small = "RAISE_POT" if fold_rate < 0.2 else "RAISE_HALF"
    raise_adjust = _learned_adjustment(memory, f"raise_{st}", LEARN_RAISE_LIMIT)
    call_adjust = _learned_adjustment(memory, f"call_{st}", LEARN_CALL_LIMIT, LEARN_CALL_GAIN)

    if obs.to_call == 0:
        if can_raise and eq >= POT_RAISE[st] + raise_adjust:
            return _pick(obs, big, small)
        if can_raise and eq >= VALUE_RAISE[st] + raise_adjust:
            return _pick(obs, small, big)
        # semi-bluff / bluff only vs. opponents who fold enough
        if can_raise and st != "RIVER" and eq >= BLUFF_EQ and fold_rate > BLUFF_FOLD:
            if rng.random() < min(0.6, (fold_rate - 0.35) * 2.0):
                return _pick(obs, "RAISE_HALF", "RAISE_POT")
        if can_raise and st == "RIVER" and eq < 0.15 and fold_rate > 0.55 and obs.is_button:
            if rng.random() < (fold_rate - 0.5):
                return _pick(obs, "RAISE_HALF", "RAISE_POT")
        return _pick(obs, "CHECK_CALL")

    need = obs.to_call / (obs.pot + obs.to_call)
    reraise_bar = POT_RAISE[st] + raise_adjust + 0.04 * obs.raises_this_street
    if can_raise and eq >= reraise_bar:
        return _pick(obs, big, small)
    if eq >= need + CALL_MARGIN[st] + call_adjust:
        return _pick(obs, "CHECK_CALL")
    return _pick(obs, "FOLD")


def choose_action(obs, memory):
    t0 = perf_counter()
    try:
        if "t_used" not in memory:
            new_half(memory)
        memory["mc_s"] = 0.0
        _absorb_history(obs, memory)
        memory["decisions"] += 1
        if memory["decisions"] % 40 == 1 and BENCH_BASE:
            # The judge charges server overhead we can't time directly; it grows with
            # machine load, so scale our estimate by a measured slowdown factor.
            r = min(_bench() for _ in range(3)) / BENCH_BASE
            memory["slow"] = 0.6 * memory["slow"] + 0.4 * min(8.0, max(1.0, r))
        memory["ov_used"] += OVERHEAD_S * memory["slow"]
        if obs.street.name == "PREFLOP":
            action = _preflop(obs, memory)
        else:
            memory["post_decisions"] += 1
            action = _postflop(obs, memory)
        if action not in obs.legal_actions:
            action = _pick(obs, "CHECK_CALL")
        _record_learning_decision(obs, action, memory)
        memory["pending_bet"] = action.name.startswith("RAISE")
        return action
    finally:
        dt = perf_counter() - t0
        memory["t_used"] = memory.get("t_used", 0.0) + dt
        if "fixed_s" in memory:
            memory["fixed_s"] = 0.95 * memory["fixed_s"] + 0.05 * max(0.0, dt - memory["mc_s"])
