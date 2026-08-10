from __future__ import annotations

import json
import pathlib
from os import PathLike
from typing import Sequence

import numpy as np
import pandas as pd

NON_SLOT_COLS = {"Subfolder", "Frequency", "Target"}


def trimming(spath_df: pd.DataFrame, trimmed_rels: Sequence[str]) -> pd.DataFrame:
    """
    Xóa các slot từ trimmed_rels trở đi trong mỗi cell của các slot.

    Args:
        spath_df (pd.DataFrame): DataFrame chứa các cột Subfolder, Frequency, Target và Slot_*.
        trimmed_rels (list): List các trimmed_rels cần trim (VD: ['chi_punct', 'chi_subj'])

    Returns:
        pd.DataFrame: DataFrame đã trim slot.
    """
    if not isinstance(spath_df, pd.DataFrame):
        raise TypeError("spath_df must be a pandas DataFrame, not a path.")

    df = spath_df.copy()

    # Lấy các cột slot (bỏ Subfolder, Frequency và Target)
    slot_cols = [c for c in df.columns if c not in NON_SLOT_COLS]

    def trim_cell(cell):
        """
        Trims the relations in a slot cell based on specified trimmed relations.

        Args:
            cell (str): A string representation of slot relations in the format '> rel1 > rel2 > ...'.

        Returns:
            str: The trimmed slot relations, retaining only those not in trimmed_rels,
                formatted as '> rel1 > rel2 > ...', or an empty string if all are trimmed.
        """
        if not isinstance(cell, str):
            return cell
        # Bỏ dấu > đầu tiên nếu có
        cell = cell.lstrip("> ").strip()
        # Tách các relation
        parts = [p.strip() for p in cell.split(">") if p.strip()]
        new_parts = []
        for part in parts:
            if part in trimmed_rels:
                # Gặp trimmed_rels, dừng ngay, không thêm nó
                break
            new_parts.append(part)
        if new_parts:
            return "> " + " > ".join(new_parts)
        else:
            return ""

    # Áp dụng cho từng slot col
    for col in slot_cols:
        df[col] = df[col].apply(trim_cell)
    
    # Fill cell rỗng với NaN
    df[slot_cols] = df[slot_cols].replace("", np.nan)

    return df


def merging(df: pd.DataFrame, output_path: str | PathLike[str] | None = None) -> pd.DataFrame:
    """
    Merge các row có cùng slot list (đã loại duplicate theo chiều ngang).

    Args:
        df (pd.DataFrame): DataFrame sau khi trim.
        output_path (str | PathLike | None): File CSV để lưu kết quả nếu cần.

    Returns:
        pd.DataFrame: DataFrame đã merge.
    """
    df = df.copy()
    slot_cols = [c for c in df.columns if c not in NON_SLOT_COLS]

    # Loại duplicate trong từng row (chiều ngang), sort để nhất quán
    df["slot_key"] = df[slot_cols].apply(
        lambda row: tuple(sorted(set(row.dropna()))),
        axis=1
    )

    # Loại dòng mà slot_key rỗng
    df = df[df["slot_key"].apply(lambda x: len(x) > 0)]

    # Merge theo slot_key và Target
    merged = (
        df.groupby(["Subfolder", "Target", "slot_key"], as_index=False)
        .agg({"Frequency": "sum"})
    )

    if merged.empty:
        merged = merged[["Subfolder", "Frequency", "Target"]]
    else:
        # Tách slot_key ra lại thành cột
        max_len = max(merged["slot_key"].apply(len))
        slot_df = pd.DataFrame(
            merged["slot_key"].apply(lambda x: list(x) + [np.nan] * (max_len - len(x))).tolist(),
            columns=[f"Slot_{i + 1}" for i in range(max_len)],
        )
        merged = pd.concat([merged[["Subfolder", "Frequency", "Target"]], slot_df], axis=1)

    merged = merged.sort_values(["Subfolder","Frequency"], ascending=[True, False]).reset_index(drop=True)

    if output_path is not None:
        merged.to_csv(output_path, sep="&", index=False)
        print(f"Saved merged file to {output_path}")

    return merged


def trim_and_merge(
    spath_df: pd.DataFrame,
    trimmed_rels: Sequence[str],
    output_path: str | PathLike[str] | None = None,
) -> pd.DataFrame:
    """Trim slot relations in a DataFrame, then merge duplicate slot combinations."""
    df = trimming(spath_df, trimmed_rels)
    merged = merging(df, output_path=output_path)
    return merged

def spe_group(spath_df: pd.DataFrame, output_folder: str, target_lemma: str):
    """
    Nhóm DataFrame có cột: Subfolder, Frequency, Target, Slot_*...
    Nhóm các row có cùng slot list (đã loại duplicate theo chiều ngang) 
    trong cùng 1 Subfolder và lưu file JSON.

    Args:
        spath_df (pd.DataFrame): DataFrame chứa các cột Subfolder, Frequency, Target và Slot_*.
        output_folder (str): Thư mục để lưu file JSON.

    Returns:
    Danh sách các node đã được nhóm trong file json
    {
      "1750": [ {id, slot_combs, frequency, specialisations:[...]}, ... ],
      "1755": [ ... ]
    }
    """
    def first_level(slot: str) -> str:
        """
        Trả về level đầu tiên của slot (tách ra bởi '>') sau khi loại bỏ
        các dấu '>' và khoảng trắng thừa.

        Args:
            slot (str): Chuỗi slot.

        Returns:
            str: Phần đầu tiên của slot.
        """
        return slot.lstrip(">").split(">")[0].strip()

    out_dir = pathlib.Path(output_folder) # Chuẩn bị thư mục output
    out_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(spath_df, pd.DataFrame):
        raise TypeError("spath_df must be a pandas DataFrame, not a path.")

    required_cols = ["Subfolder", "Frequency", "Target"]
    missing_cols = [col for col in required_cols if col not in spath_df.columns]
    if missing_cols:
        raise ValueError("DataFrame must contain columns: Subfolder, Frequency, Target")

    df = spath_df.copy()
    columns = list(df.columns)
    target_index = columns.index("Target")
    slot_cols = columns[target_index + 1:] # Các cột slots

    # bucket theo subfolder
    buckets = {}
    for _, row in df.iterrows():
        subf = str(row["Subfolder"]).strip()
        buckets.setdefault(subf, []).append(row) # Add rows into buckets of subfolders

    # xử lý từng subfolder
    spe_by_subfolder = {}

    for subf, rows in buckets.items():
        nodes = {}  # key = frozenset(first-level slots) -> node dict
        for row in rows:
            # Get all frequencies
            try:
                freq = int(str(row["Frequency"]).strip())
            except Exception:
                continue

            # Get all raw slots
            raw_slots = []
            for c in slot_cols:
                value = row[c]
                if pd.isna(value):
                    continue
                s = str(value).strip()
                if s:
                    raw_slots.append(s)
            if not raw_slots:
                continue

            # group theo tập first-level slots (logic cũ)
            flat_slots = {first_level(s) for s in raw_slots}
            key = frozenset(flat_slots) # Create dictionary keys from flat_slots

            node = nodes.setdefault(
                key,
                {
                    "id": f"{subf}_node_{len(nodes)+1}",  # reset theo subfolder
                    "slot_combs": sorted(flat_slots),
                    "frequency": 0,
                    "specialisations": []
                }
            )
            node["specialisations"].append({
                "specialisation": raw_slots,
                "frequency": freq
            })
            node["frequency"] += freq

        spe_by_subfolder[subf] = list(nodes.values()) # Add nodes to final dict, grouped by subfolders

    out_file = pathlib.Path(output_folder) / f"{target_lemma}_spath_comb_grouped.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(spe_by_subfolder, f, ensure_ascii=False, indent=2)
    print(f"Saved to {out_file}")
    return spe_by_subfolder
