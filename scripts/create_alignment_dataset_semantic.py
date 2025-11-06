import os
import json
import csv
import ast
from tqdm import tqdm
import argparse

def build_alignment_entry(smiles: str, entry_id: str):
    """
    Creates a single entry for the alignment dataset.
    The conversation teaches the model to associate the MolFormer embeddings
    (represented by <image>) with the textual SMILES representation.
    """
    return {
        "id": entry_id,
        "smiles": smiles,
        "conversations": [
            {
                "from": "human",
                "value": "The SMILES of the molecule <image> is"
            },
            {
                "from": "gpt",
                "value": f"<SMILES>{smiles}</SMILES>"
            }
        ]
    }

def create_alignment_dataset(csv_path: str, output_path: str, num_rows: int):
    """
    Reads a MEDEX CSV file, processes a specified number of rows, and creates
    a JSON dataset for the alignment training stage.
    """
    print(f"Reading from: {csv_path}")
    
    all_entries = []
    skipped_count = 0
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(tqdm(reader, total=num_rows, desc="Processing MEDEX data")):
            if i >= num_rows:
                break
            
            try:
                molinfo_str = row.get("MolInfo", "{}")
                molinfo = ast.literal_eval(molinfo_str)
                smiles = molinfo.get("SMILES", "").strip()
                
                if smiles:
                    if len(smiles) > 100:
                        skipped_count += 1
                        continue
                    entry_id = f"medex_alignment_{i}"
                    entry = build_alignment_entry(smiles, entry_id)
                    all_entries.append(entry)
            except (ValueError, SyntaxError) as e:
                print(f"Skipping row {i} due to parsing error: {e}")
                continue

    print(f"\nProcessed {len(all_entries)} valid entries.")
    print(f"Skipped {skipped_count} entries with SMILES longer than 100 characters.")
    print(f"Saving alignment dataset to: {output_path}")

    with open(output_path, "w") as f:
        json.dump(all_entries, f, indent=2)

    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Create a LLaVA-formatted alignment dataset from a MEDEX CSV file.")
    parser.add_argument(
        "--csv-path", 
        type=str, 
        default="/Users/jlee0/Desktop/research/therapeutic-tuning/data/TDC/MEDEX/medex_subset_1M.csv",
        help="Path to the input MEDEX CSV file."
    )
    parser.add_argument(
        "--output-path", 
        type=str, 
        default="/Users/jlee0/Desktop/research/therapeutic-tuning/LLaVA/playground/data/llava_medex_alignment_100k.json",
        help="Path to save the output JSON alignment file."
    )
    parser.add_argument(
        "--num-rows", 
        type=int, 
        default=100000,
        help="Number of rows to process from the CSV file."
    )
    args = parser.parse_args()
    
    create_alignment_dataset(args.csv_path, args.output_path, args.num_rows)

if __name__ == "__main__":
    main()
