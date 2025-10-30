import os
import json
import csv
import ast


CSV_PATH = "/Users/jlee0/Desktop/research/therapeutic-tuning/data/TDC/MEDEX/medex_subset_1M.csv"
JSON_PATH = "/Users/jlee0/Desktop/research/therapeutic-tuning/LLaVA/playground/data/mol-instruct_llava_format.json"


def read_first_row(csv_path: str):
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    return row


def parse_molinfo(molinfo_str: str):
    return ast.literal_eval(molinfo_str)


def build_entry(entity: str, fact: str, smiles: str):
    return {
        "id": "medex_subset_1M_row0",
        "smiles": smiles,
        "conversations": [
            {
                "from": "human",
                "value": "<image>"
            },
            {
                "from": "gpt",
                "value": f"<SMILES>{smiles}</SMILES>"
            }
        ]
    }


def prepend_entry_to_json(entry: dict, json_path: str):
    tmp_path = json_path + ".tmp"
    entry_bytes = json.dumps(entry, ensure_ascii=False).encode("utf-8")
    with open(json_path, "rb") as src, open(tmp_path, "wb") as dst:
        head = src.read(1024 * 1024)
        lb = head.find(b"[")
        if lb == -1:
            raise RuntimeError("Target JSON is not a list.")
        dst.write(b"[")
        dst.write(entry_bytes)
        dst.write(b",")
        dst.write(head[lb + 1 :])
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
    os.replace(tmp_path, json_path)


def main():
    row = read_first_row(CSV_PATH)
    molinfo = parse_molinfo(row["MolInfo"])
    smiles = molinfo.get("SMILES", "").strip()
    entity = row.get("entity", "").strip()
    fact = row.get("fact", "").strip()
    entry = build_entry(entity, fact, smiles)
    prepend_entry_to_json(entry, JSON_PATH)


if __name__ == "__main__":
    main()


