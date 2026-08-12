import json
from pathlib import Path

import pandas as pd


def spaths_json_to_slotfiller_df(
    json_path: str | Path,
    exclude: list[str] | None = None,
) -> pd.DataFrame:
    """Convert spaths.json to an sfiller_df-like DataFrame."""
    exclude_set = set(exclude or [])

    with open(json_path, encoding="utf-8") as f:
        spaths = json.load(f)

    target = Path(json_path).name.replace("_spaths.json", "")
    rows = []

    for subfolder, slot_counts in spaths.items():
        idx = 1
        for slot, freq in slot_counts.items():
            if slot in exclude_set:
                continue

            for _ in range(int(freq)):
                rows.append(
                    {
                        "id": f"{subfolder}_{idx}",
                        "subfolder": str(subfolder),
                        "target": [(target,)],
                        "slot": [(slot,)],
                    }
                )
                idx += 1

    return pd.DataFrame(rows, columns=["id", "subfolder", "target", "slot"])