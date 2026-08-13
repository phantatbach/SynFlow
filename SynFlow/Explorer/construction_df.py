from pathlib import Path
from typing import Optional, Union
import pandas as pd
#--------------------------------------------------------------------------
# Convert individual slot path df to construction df for distance calculation
#--------------------------------------------------------------------------
def parse_frequency(value: object) -> int:
    if pd.isna(value):
        raise ValueError("frequency cannot be NaN")
    frequency = float(value)
    if not frequency.is_integer() or frequency < 0:
        raise ValueError(f"frequency must be a non-negative integer, got {value!r}")
    return int(frequency)
    
def construction_from_row(row: pd.Series, slot_cols: list[str]) -> Optional[str]:
    slots = []
    for col in slot_cols:
        value = row[col]
        if pd.isna(value):
            continue
        slot = str(value).strip()
        if slot:
            slots.append(slot)
    if not slots:
        return None
    return " & ".join(sorted(set(slots)))

def spath_to_constructiondf(spath_df: pd.DataFrame, output_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Convert a lowercase spath combination DataFrame into sfiller_df-like construction rows."""
    required_cols = {"subfolder", "frequency", "target"}
    missing_cols = required_cols - set(spath_df.columns)
    if missing_cols:
        raise ValueError("spath_df must contain lowercase columns: subfolder, frequency, target")

    slot_cols = [col for col in spath_df.columns if col not in required_cols]
    records = []
    counters: dict[str, int] = {}

    sort_cols = ["subfolder", "target", *slot_cols]
    sorted_df = spath_df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    for _, row in sorted_df.iterrows():
        subfolder = str(row["subfolder"]).strip()
        target = str(row["target"]).strip()
        construction = construction_from_row(row, slot_cols)
        if construction is None:
            continue

        frequency = parse_frequency(row["frequency"])
        for _ in range(frequency):
            counters[subfolder] = counters.get(subfolder, 0) + 1
            records.append({
                "id": f"{subfolder}_{counters[subfolder]}",
                "subfolder": subfolder,
                "target": [(target,)],
                "construction": [(construction,)],
            })

    out = pd.DataFrame(records, columns=["id", "subfolder", "target", "construction"])
    if output_path is not None:
        out.to_csv(output_path, index=False)
    return out
