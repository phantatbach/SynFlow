from math import inf
import numpy as np
import pandas as pd
from typing import Iterable, List, Tuple, Dict
from .freq import freq_all_slots_by_period

# THIS IS UNDER DEVELOPMENT AND HAS NOT BEEN USED OFFICIALLY
# Shifting frequency of the slot level
# Split into smaller periods
def split_periods(periods: Iterable, starting_points: Iterable[int]) -> Tuple[List[List[int]], List[Tuple[int,int]]]:
    """
    Split the given periods into smaller blocks based on the provided starting points.

    Parameters:
        periods (Iterable): A list of period values.
        starting_points (Iterable[int]): A list of starting cutoff points.

    Returns:
        Tuple[List[List[int]], List[Tuple[int,int]]]: A tuple containing two lists.
            The first list contains the blocks of period values.
            The second list contains tuples of (low_bound, high_bound) for each block.
    """
    period_list = []
    for period in periods:
        try:
            period_list.append(int(period))
        except (TypeError, ValueError):
            continue
    period_set = sorted(set(period_list))

    blocks = []

    if starting_points:
        # Get a set of starting cutoff points
        cuts = sorted({int(starting_point) for starting_point in starting_points})    
        for i, low_bound in enumerate(cuts):
            high_bound = cuts[i+1] if i+1 < len(cuts) else inf
            block = [period for period in period_set if low_bound <= period < high_bound]
            if block:
                blocks.append(block)
    else:
        blocks = [period_set]

    return blocks

# Get the sub data frame of each period
def get_sub_freq_df(block, slot_raw_freq_df):
    sub_cols = [str(y) for y in block]
    sub_freq_df = slot_raw_freq_df.loc[:, slot_raw_freq_df.columns.intersection(sub_cols)].sort_index(axis=1).copy()
    return sub_freq_df

# shift left -> right
def shift_left_right(row_series: pd.Series, min_cov_freq: int) -> Tuple[pd.Series, Dict]:
    """
    shift the frequencies of the slot level from left to right.

    Parameters:
        row_series (pd.Series): A Series containing slot frequencies.
        min_cov_freq (int): Minimum coverage frequency to shift.

    Returns:
        Tuple[pd.Series, Dict]: A tuple containing the shifted Series and a dict of shifting notes.
    """
    series = row_series.fillna(0)

    # chronological order (try numeric)
    try:
        order = np.argsort([int(x) for x in series.index])
    except Exception:
        order = np.argsort(series.index.astype(str))

    idx = series.index[order]
    vals = series.values[order].astype(float)

    out = np.zeros_like(vals, dtype=float)
    plan: List[Dict] = []
    accum = 0.0
    start = 0
    last_period_pos = None

    # Loop through the series and shift left -> right
    for i, val in enumerate(vals):
        accum += val
        if accum >= min_cov_freq:
            out[i] = accum  # collapse this block to the rightmost period i
            plan.append({"source": list(idx[start:i+1]), "target": idx[i], "sum": float(accum)})
            last_period_pos = i
            accum = 0.0
            start = i + 1

    # Collapse the tail to the last period
    if accum > 0 and last_period_pos is not None:
        out[last_period_pos] += accum
        # cập nhật group cuối cùng
        g = plan[-1]
        g["source"] = g["source"] + list(idx[start:len(idx)])   # append tail sources only
        g["sum"]   = float(out[last_period_pos])
    
    elif last_period_pos is None and len(vals) > 0:
        # không có khối nào hoàn tất: dồn hết vào ô cuối
        out[-1] = accum  # ở đây accum = tổng toàn chuỗi
        plan.append({"source": list(idx[start:len(idx)]), "target": idx[-1], "sum": float(accum)})

    # return aligned to original index order
    shifted = pd.Series(out, index=idx)
    return shifted.reindex(series.index), {"algo": "left_right", "groups": plan}

# shift right -> left
def shift_right_left(row_series: pd.Series, min_cov_freq: int) -> Tuple[pd.Series, Dict]:
    """
    shift the frequencies of the slot level from right to left.

    Parameters:
        row_series (pd.Series): A Series containing slot frequencies.
        min_cov_freq (int): Minimum coverage frequency to shift.

    Returns:
        Tuple[pd.Series, Dict]: A tuple containing the shifted Series and a dict of shifting notes.
    """
    series = row_series.fillna(0)

    # chronological order (try numeric)
    try:
        order = np.argsort([int(x) for x in series.index])
    except Exception:
        order = np.argsort(series.index.astype(str))

    idx = series.index[order]
    vals = series.values[order].astype(float)
    out = np.zeros_like(vals, dtype=float)
    plan: List[Dict] = []
    end = len(vals) - 1
    accum = 0.0
    first_period_pos = None

    # iterate right -> left
    for i in range(len(vals) - 1, -1, -1):
        accum += vals[i]
        if accum >= min_cov_freq:
            out[i] = accum   # collapse this block to the leftmost period i
            plan.append({"source": list(idx[i:end+1]), "target": idx[i], "sum": float(accum)})
            first_period_pos = i
            accum = 0.0
            end = i - 1

    # Collapse the tail to the last period
    if accum > 0 and first_period_pos is not None:
        out[first_period_pos] += accum
        g = plan[-1]                             # leftmost completed block
        left_tail = list(idx[0:first_period_pos])  # periods strictly left of the block
        g["source"] = left_tail + g["source"]    # prepend tail
        g["sum"]    = float(out[first_period_pos])
            
    elif first_period_pos is None and len(vals) > 0:
        # không có khối nào hoàn tất: dồn hết vào ô đầu
        out[0] = accum  # accum = tổng toàn chuỗi
        plan.append({"source": list(idx[0:end+1]), "target": idx[0], "sum": float(accum)})

    shifted = pd.Series(out, index=idx)
    return shifted.reindex(series.index), {"algo": "right_left", "groups": plan}

# Some metrics
def _has_deficit(s: pd.Series, min_val: float) -> bool:
    """Any 0 < value < min_val?"""
    s = s.fillna(0).astype(float)
    return bool(((s > 0) & (s < min_val)).any())
def _nonzero_count(s: pd.Series) -> int:
    """How many cells >0?"""
    return int((s.fillna(0).astype(float) > 0).sum())
def _count_singles_gt0(s):
    """How many runs of >0?"""
    mask = (s.fillna(0).to_numpy() > 0)
    singles = runsz = 0
    for v in np.r_[mask, False]:  # Add False guard at the end to always end the run
        if v: 
            runsz += 1
        elif runsz:
            if runsz == 1: 
                singles += 1
            runsz = 0
    return singles
def _longest_streak_gt0(s):
    """Longest run of >0?"""
    mask = (s.fillna(0).to_numpy() > 0)
    streak = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        if cur > streak: 
            streak = cur
    return streak

# Choosing the better option
def choose_shifted_series(LR_SN, RL_SN, min_val: float) -> Tuple[pd.Series, Dict]:
    """
    Safe option = left_series.
    1) Any deficit → left
    2) More nonzeros → prefer
    3) Fewer singles → prefer
    4) Larger longest_streak → prefer
    5) Tie → left
    """
    # Unpack the tuple
    LS, _ = LR_SN
    RS, _ = RL_SN

    # Calculate the matrix
    L_def = _has_deficit(LS, min_val); R_def = _has_deficit(RS, min_val)
    L_nz = _nonzero_count(LS); R_nz = _nonzero_count(RS)
    L_singles = _count_singles_gt0(LS); R_singles = _count_singles_gt0(RS)
    L_streak = _longest_streak_gt0(LS); R_streak = _longest_streak_gt0(RS)

    # 1) any deficit → safe (left)
    if L_def or R_def:
        return LR_SN

    # 2) more nonzeros
    elif L_nz != R_nz:
        return (LR_SN if L_nz > R_nz else RL_SN)

    # 3) less single > 0
    elif L_singles != R_singles:
        return LR_SN if L_singles < R_singles else RL_SN

    # 4) longest_streak
    elif L_streak != R_streak:
        return LR_SN if L_streak > R_streak else RL_SN
    
    # 5) tie → safe
    else:
        return LR_SN
    
# Main shifting function
def shift_slot_freq(slot_json_path, min_cov_freq, starting_points=None):
    # Get the raw frequencies and create a reconstruction df
    recon = freq_all_slots_by_period(json_path=slot_json_path).T
    # Get the periods
    periods = recon.columns.to_list()
    # Create a notes dict
    shift_notes = {}
    # Split the periods into blocks
    period_blocks = split_periods(periods, starting_points)

    # shift the frequencies of the slots in each block
    for block in period_blocks:
        block_cols = [str(y) for y in block]

        sub_freq_df = get_sub_freq_df(block, recon)

        for slot, row in sub_freq_df.iterrows():
            row_lr = shift_left_right(row, min_cov_freq)
            row_rl = shift_right_left(row, min_cov_freq)
            final_shift, final_shift_note = choose_shifted_series(row_lr, row_rl, min_cov_freq)
            
            # Update the reconstruction DF
            recon.loc[slot, block_cols] = final_shift.values
            # Get the notes
            shift_notes[(slot, tuple(block_cols))] = final_shift_note

    return recon, shift_notes

def coverage_filter_df(df, min_cov_freq=1, coverage_per=.5):
    periods = df.columns

    coverage = (df[periods] > min_cov_freq).sum(axis=1)

    keep = coverage >= coverage_per * len(periods)
    cov_filtered_df = (
        df.loc[keep].copy()
        .assign(coverage=coverage.loc[keep])
        .sort_values('coverage', ascending=False)
    )
    return cov_filtered_df