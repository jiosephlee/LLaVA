import json
import csv
import ast
from tqdm import tqdm
import argparse

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit import RDLogger
    # Disable RDKit warnings globally to suppress console output
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Install with: pip install rdkit")


def compute_molecular_properties(smiles: str):
    """
    Computes molecular properties using RDKit.
    Returns a dictionary of property name to value, or None if warnings occur or molecule is invalid.
    """
    if not RDKIT_AVAILABLE:
        return None
    
    try:
        # Use sanitize=False to avoid RDKit trying to fix issues (which generates warnings)
        # If sanitize fails, we'll get None and skip this molecule
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        
        if mol is None:
            return None
        
        # Try to sanitize explicitly - if this fails, we'll skip the molecule
        # This will catch molecules with structural issues
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None
        
        # Check if molecule has any issues by trying to compute a simple property
        # If this fails, the molecule likely has issues
        try:
            _ = mol.GetNumAtoms()
        except Exception:
            return None
        
        if mol.GetNumAtoms() == 0:
            return None
        
        props = {}
        props['Molecular Weight'] = round(Descriptors.MolWt(mol), 2)
        props['LogP'] = round(Descriptors.MolLogP(mol), 2)
        props['Number of Atoms'] = mol.GetNumAtoms()
        props['Number of Heavy Atoms'] = Descriptors.HeavyAtomCount(mol)
        props['Number of Rings'] = Descriptors.RingCount(mol)
        props['Number of Aromatic Rings'] = Descriptors.NumAromaticRings(mol)
        props['Number of Rotatable Bonds'] = Descriptors.NumRotatableBonds(mol)
        props['Number of H-Bond Donors'] = Descriptors.NumHDonors(mol)
        props['Number of H-Bond Acceptors'] = Descriptors.NumHAcceptors(mol)
        props['TPSA'] = round(Descriptors.TPSA(mol), 2)
        props['Formal Charge'] = Chem.rdmolops.GetFormalCharge(mol)
        
        return props
    except Exception:
        return None

def get_property_description(property_name: str) -> str:
    """
    Returns a description of what the molecular property means.
    """
    descriptions = {
    "Molecular Weight": "Molecular weight is the sum of the atomic weights of all atoms in a molecule, expressed in atomic mass units.",
    "LogP": "LogP is the logarithm of the octanol-water partition coefficient, which measures a molecule's lipophilicity (fat-solubility) versus its hydrophilicity (water-solubility). A higher positive LogP value indicates a more lipophilic (fat-loving) molecule, while a more negative value indicates a more hydrophilic (water-loving) molecule. This is a critical property used in drug discovery to predict how a drug will be absorbed, distributed, and interact with the body.",
    "Number of Atoms": "The Number of Atoms is the total count of all atoms in a molecule.",
    "Number of Heavy Atoms": "The Number of Heavy Atoms is the total count of all non-hydrogen atoms in a molecule.",
    "Number of Rings": "The Number of Rings is the total count of ring structures in a molecule.",
    "Number of Aromatic Rings": "The Number of Aromatic Rings is the total count of ring structures in a molecule that meet the criteria for aromaticity. To be considered aromatic, a ring system must typically be cyclic, planar, fully conjugated, and contain a specific number of delocalized pi electrons that follows Hückel’s rule ($4n + 2$). A higher count usually implies increased chemical stability.",
    "Number of Rotatable Bonds": "The Number of Rotatable Bonds is the total count of single, non-terminal bonds between non-ring, non-hydrogen atoms that allow free rotation. Molecules with more rotatable bonds are generally more flexible, which can reduce binding specificity and membrane permeability.",
    "Number of H-Bond Donors": "The Number of H-Bond Donors is the count of atoms (usually N–H or O–H groups) capable of donating hydrogen bonds. More donors typically increase polarity and water solubility but can reduce membrane permeability.",
    "Number of H-Bond Acceptors": "The Number of H-Bond Acceptors is the count of atoms (commonly O or N atoms with lone pairs) capable of accepting hydrogen bonds. High counts suggest strong intermolecular interactions and solubility in polar environments, but again may limit passive diffusion across lipid membranes.",
    "TPSA": "Topological Polar Surface Area (TPSA) represents the sum of the surface areas of polar atoms (mainly oxygen and nitrogen and their attached hydrogens). It correlates strongly with drug transport properties: lower TPSA (< 140) often implies better membrane permeability and oral bioavailability.",
    "Formal Charge": "Formal Charge is the net electrical charge of the molecule, derived from the sum of formal charges on individual atoms. It indicates electron distribution; charged molecules tend to have higher solubility in water but lower permeability through lipid membranes."
}
    # descriptions = {
    #     "Molecular Weight": "Molecular weight is the sum of atomic weights of all atoms in the molecule.",
    #     "LogP": "LogP is the logarithm of the partition coefficient between octanol and water, measuring lipophilicity.",
    #     "Number of Atoms": "The number of atoms is the total count of all atoms in the molecule.",
    #     "Number of Heavy Atoms": "The number of heavy atoms is the count of non-hydrogen atoms in the molecule.",
    #     "Number of Rings": "The number of rings is the count of ring structures present in the molecule.",
    #     "Number of Aromatic Rings": "The number of aromatic rings is the count of aromatic ring systems in the molecule.",
    #     "Number of Rotatable Bonds": "The number of rotatable bonds is the count of single bonds that can rotate freely.",
    #     "Number of H-Bond Donors": "The number of H-bond donors is the count of atoms that can donate hydrogen bonds.",
    #     "Number of H-Bond Acceptors": "The number of H-bond acceptors is the count of atoms that can accept hydrogen bonds.",
    #     "TPSA": "TPSA (Topological Polar Surface Area) is the sum of polar surface areas of atoms in the molecule.",
    #     "Formal Charge": "Formal charge is the charge assigned to the entire molecule based on its bonding pattern."
    # }
    return descriptions.get(property_name, f"{property_name} is a molecular property.")

def build_rdkit_alignment_entry(smiles: str, property_name: str, property_value, entry_id: str):
    """
    Creates a single entry for the RDKit property alignment dataset.
    Each entry corresponds to one property prediction.
    The conversation teaches the model to predict molecular properties
    using both the MolFormer embeddings (represented by <image>) and SMILES.
    """
    # Determine appropriate phrasing based on property type
    # Count-based properties use "in this molecule", others use "of this molecule"
    if property_name.startswith("Number of"):
        preposition = "in"
    else:
        preposition = "of"
    
    # Get property description
    property_description = get_property_description(property_name)
    
    return {
        "id": entry_id,
        "smiles": smiles,
        "conversations": [
            {
                "from": "human",
                "value": f"{property_description} Consider the molecule <image> whose SMILES is <SMILES>{smiles}</SMILES>. The {property_name} {preposition} this molecule is"
            },
            {
                "from": "gpt",
                "value": str(property_value)
            }
        ]
    }

def create_rdkit_alignment_dataset(csv_path: str, output_path: str, num_rows: int):
    """
    Reads a MEDEX CSV file, processes a specified number of rows, computes
    RDKit properties, and creates a JSON dataset for property prediction alignment.
    """
    if not RDKIT_AVAILABLE:
        print("Error: RDKit is required but not installed.")
        print("Install with: pip install rdkit")
        return
    
    print(f"Reading from: {csv_path}")
    
    all_entries = []
    skipped_count = 0
    entry_counter = 0
    molecules_processed = 0
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
                    
                    properties = compute_molecular_properties(smiles)
                    if properties is None:
                        skipped_count += 1
                        continue
                    
                    # Create one entry per property
                    for prop_name, prop_value in properties.items():
                        entry_id = f"medex_rdkit_alignment_{entry_counter}"
                        entry = build_rdkit_alignment_entry(smiles, prop_name, prop_value, entry_id)
                        all_entries.append(entry)
                        entry_counter += 1
                    
                    molecules_processed += 1
            except (ValueError, SyntaxError) as e:
                print(f"Skipping row {i} due to parsing error: {e}")
                continue

    print(f"\nProcessed {len(all_entries)} property entries from {molecules_processed} molecules.")
    print(f"Skipped {skipped_count} entries (invalid SMILES or SMILES longer than 100 characters).")
    print(f"Saving RDKit property alignment dataset to: {output_path}")

    with open(output_path, "w") as f:
        json.dump(all_entries, f, indent=2)

    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Create a LLaVA-formatted RDKit property alignment dataset from a MEDEX CSV file.")
    parser.add_argument(
        "--csv-path", 
        type=str, 
        default="/Users/jlee0/Desktop/research/therapeutic-tuning/data/TDC/MEDEX/medex_subset_1M.csv",
        help="Path to the input MEDEX CSV file."
    )
    parser.add_argument(
        "--output-path", 
        type=str, 
        default="/Users/jlee0/Desktop/research/therapeutic-tuning/LLaVA/playground/data/llava_medex_rdkit_alignment_100k.json",
        help="Path to save the output JSON alignment file."
    )
    parser.add_argument(
        "--num-rows", 
        type=int, 
        default=10000,
        help="Number of rows to process from the CSV file."
    )
    args = parser.parse_args()
    
    create_rdkit_alignment_dataset(args.csv_path, args.output_path, args.num_rows)

if __name__ == "__main__":
    main()

