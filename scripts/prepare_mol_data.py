import os
import json
import glob
import selfies as sf
from tqdm import tqdm

def preprocess_mol_instruct_data():
    """
    Processes all JSON files in the mol-instruct directory, converts SELFIES to SMILES,
    reformats the data into LLaVA conversation format, and saves it as a single
    JSON file.
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, 'playground', 'data', 'mol-instruct')
    output_file = os.path.join(base_dir, 'playground', 'data', 'mol_instruct_1M.json')

    all_data = []
    json_files = glob.glob(os.path.join(data_dir, '*.json'))
    
    # Exclude specific files
    excluded_files = {'reagent_prediction.json', 'forward_reaction_prediction.json', "retrosynthesis.json"}
    json_files = [f for f in json_files if os.path.basename(f) not in excluded_files]

    for file_path in json_files:
        file_name = os.path.basename(file_path)
        print(f"Processing {file_name}...")
        with open(file_path, 'r') as f:
            data = json.load(f)

        for i, entry in enumerate(tqdm(data, desc=f"Converting {file_name}")):
            try:
                selfies_string = entry['input']
                smiles_string = sf.decoder(selfies_string)

                reformatted_entry = {
                    "id": f"{os.path.splitext(file_name)[0]}_{i}",
                    "smiles": smiles_string,
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"{entry['instruction']}\n<SMILES>{smiles_string}</SMILES>\n<image>"
                        },
                        {
                            "from": "gpt",
                            "value": str(entry['output'])
                        }
                    ]
                }
                all_data.append(reformatted_entry)
            except Exception as e:
                print(f"Skipping entry {i} in {file_name} due to error: {e}")

    print(f"\nProcessed {len(all_data)} total entries.")
    print(f"Saving combined data to {output_file}...")

    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2)

    print("Done.")

if __name__ == "__main__":
    preprocess_mol_instruct_data()
