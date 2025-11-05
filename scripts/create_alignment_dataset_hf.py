import os
import json
import csv
import ast
from tqdm import tqdm
import argparse

# Optional import for Hugging Face datasets
try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

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
                "value": "<image>"
            },
            {
                "from": "gpt",
                "value": f"<SMILES>{smiles}</SMILES>"
            }
        ]
    }

def create_alignment_dataset_from_rows(rows, output_path: str, num_rows: int):
    """
    Processes an iterable of rows (dictionaries) that contain a `MolInfo` field
    and writes a LLaVA-style alignment JSON file to `output_path`.
    """
    all_entries = []
    skipped_count = 0
    for i, row in enumerate(tqdm(rows, total=num_rows, desc="Processing MEDEX data")):
        if i >= num_rows:
            break

        try:
            # Prefer direct 'SMILES' column if present (Hugging Face dataset provides it)
            smiles = row
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


def create_alignment_dataset(csv_path: str, output_path: str, num_rows: int, use_hf: bool = False, hf_id: str = None):
    """
    Wrapper that either reads from a local CSV (`csv_path`) or from a Hugging Face
    dataset (`hf_id`) and then delegates to `create_alignment_dataset_from_rows`.
    """
    if use_hf:
        if load_dataset is None:
            raise RuntimeError("datasets library is not installed. Install via `pip install datasets` to use --use-hf.")
        if not hf_id:
            raise ValueError("--hf-id must be provided when --use-hf is set")

        print(f"Loading HF dataset: {hf_id}")
        ds = load_dataset(hf_id)
        # Prefer the 'train' split if present, else take the first split available
        split = 'train' if 'train' in ds.keys() else list(ds.keys())[0]
        # The HF dataset uses columns: 'entity', 'fact', 'SMILES' (all strings).
        # Map each row to a dict with a MolInfo-like structure so downstream parsing
        # works unchanged.
        def hf_row_generator():
            for r in ds[split]:
                # r may be a Dataset object row; cast to dict
                rec = dict(r)
                smiles = rec.get('SMILES', '')
                # wrap into MolInfo dict for compatibility
                yield smiles

        create_alignment_dataset_from_rows(hf_row_generator(), output_path, num_rows)
    else:
        print(f"Reading from local CSV: {csv_path}")
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            create_alignment_dataset_from_rows(reader, output_path, num_rows)

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
    parser.add_argument(
        "--use-hf",
        action="store_true",
        help="Load the dataset from the Hugging Face hub via --hf-id instead of a local CSV."
    )
    parser.add_argument(
        "--hf-id",
        type=str,
        default=None,
        help="Hugging Face dataset id (e.g. 'jiosephlee/Medex_TDC_1M_Facts')."
    )
    args = parser.parse_args()
    
    create_alignment_dataset(args.csv_path, args.output_path, args.num_rows, use_hf=args.use_hf, hf_id=args.hf_id)

if __name__ == "__main__":
    main()
