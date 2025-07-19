import pandas as pd
import numpy as np
import csv, json, pathlib

def trimming(df_file, trimmed_rels):
    """
    Xóa các slot từ trimmed_rels trở đi trong mỗi cell của các slot.

    Args:
        df (pd.DataFrame): DataFrame đọc từ CSV (sep='&').
        trimmed_rels (list): List các trimmed_rels cần trim (VD: ['chi_punct', 'chi_subj'])

    Returns:
        pd.DataFrame: DataFrame đã trim slot.
    """
    df = pd.read_csv(df_file, sep='&')

    # Lấy các cột slot (bỏ Frequency và Target)
    slot_cols = df.columns[2:]

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

    # df_file = df_file.split('.')[0]
    # df.to_csv(f'{df_file}_trimmed.csv', sep='&')
    return df

def merging(df, df_file):
    """
    Merge các row có cùng slot list (đã loại duplicate theo chiều ngang) và lưu file.

    Args:
        df (pd.DataFrame): DataFrame sau khi trim.
        df_file (str): Tên file (có hoặc không có đuôi .csv).

    Returns:
        pd.DataFrame: DataFrame đã merge.
    """
    slot_cols = df.columns[2:]

    # Loại duplicate trong từng row (chiều ngang), sort để nhất quán
    df['slot_key'] = df[slot_cols].apply(
        lambda row: tuple(sorted(set(row.dropna()))),
        axis=1
    )

    # Loại dòng mà slot_key rỗng
    df = df[df['slot_key'].apply(lambda x: len(x) > 0)]

    # Merge theo slot_key và Target
    merged = (
        df.groupby(['Target', 'slot_key'], as_index=False)
        .agg({'Frequency': 'sum'})
    )

    # Tách slot_key ra lại thành cột
    max_len = max(merged['slot_key'].apply(len))
    slot_df = pd.DataFrame(merged['slot_key'].apply(lambda x: list(x) + [np.nan]*(max_len - len(x))).tolist(),
                           columns=[f"Slot_{i+1}" for i in range(max_len)])

    merged = pd.concat([merged[['Frequency', 'Target']], slot_df], axis=1)

    merged = merged.sort_values(by="Frequency", ascending=False)

    # Lưu file
    df_file = df_file.rsplit('.', 1)[0]
    merged.to_csv(f'{df_file}_trimmed.csv', sep='&', index=False)
    print(f'Saved merged file to {df_file}_trimmed.csv')

    return merged

def trim_and_merge(df_file, trimmed_rels):
    df = trimming(df_file, trimmed_rels)
    merged = merging(df, df_file)
    return merged

def spe_group(df_path: str, output_folder: str, target_lemma: str):
    """
    Nhóm các row có cùng slot list (đã loại duplicate theo chiều ngang)
    và lưu file JSON.

    Args:
        df_path (str): Đường dẫn đến file CSV (sep='&').
        output_folder (str): Thư mục để lưu file JSON.

    Returns:
        list: Danh sách các node đã được nhóm.
    """
    def first_level(slot: str) -> str:
        """
        Trả về phần đầu tiên của slot (tách ra bởi '>') sau khi loại bỏ
        các dấu '>' và khoảng trắng thừa.

        Args:
            slot (str): Chuỗi slot.

        Returns:
            str: Phần đầu tiên của slot.
        """
        return slot.lstrip('>').split('>')[0].strip()

    nodes = {}

    with open(df_path, encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter='&')

        # skip header if present
        header = next(reader, None)
        if header and header[0].strip().lower() != 'frequency':
            f.seek(0); reader = csv.reader(f, delimiter='&')

        for row in reader:
            if len(row) < 3:
                continue
            try:
                freq = int(row[0].strip())
            except ValueError:
                continue

            raw_slots  = [c.strip() for c in row[2:] if c.strip()]
            flat_slots = {first_level(s) for s in raw_slots}
            key        = frozenset(flat_slots)

            node = nodes.setdefault(
                key,
                {
                    "id": f"node_{len(nodes)+1}",
                    "slot_combs": sorted(flat_slots),
                    "frequency": 0,            # <── NEW
                    "specialisations": []
                }
            )

            node["specialisations"].append({
                "specialisation": raw_slots,
                "frequency": freq
            })
            node["frequency"] += freq           # <── NEW

    out_dir = pathlib.Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = f'{out_dir}/{target_lemma}_arg_comb_grouped.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(list(nodes.values()), f, ensure_ascii=False, indent=2)
    print(f'Saved to {out_dir}/{target_lemma}_arg_comb_grouped.json')
    return list(nodes.values())
