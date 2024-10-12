import pandas as pd
from openbabel import pybel as pb
import os
from datetime import datetime

def read_smiles_csv(file_path):
    """
    Read a CSV file containing SMILES strings.

    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        DataFrame: DataFrame containing SMILES strings.
    """
    try:
        data = pd.read_csv(file_path)
        print("CSV file successfully read!")
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def smiles_to_xyz(smiles_string):
    """
    Convert a SMILES string to XYZ format.

    Args:
        smiles_string (str): SMILES string.
    
    Returns:
        str: Corresponding XYZ format string.
    """
    mol = pb.readstring('smiles', smiles_string)
    mol.make3D()
    return mol.write(format='xyz')

def smi2xyz_main(file_path, output_dir='data'):
    """
    Main function: Reads SMILES from CSV and converts them to XYZ files.

    Args:
        file_path (str): Path to the CSV file.
        output_dir (str): Root path for output directory.
    """
    smiles_data = read_smiles_csv(file_path)
    
    if smiles_data is not None:
        # Get the current timestamp and format it as YYYYMMDD-HHMMSS
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        # Create a new directory based on the timestamp
        output_path = os.path.join(output_dir, timestamp)
        os.makedirs(output_path, exist_ok=True)

        # Assuming SMILES data is in a column named 'SMILES'
        # Add a new column 'XYZ', containing the converted XYZ data
        smiles_data['XYZ'] = smiles_data['SMILES'].apply(smiles_to_xyz)
        print("Conversion complete!")

        # Save each XYZ data to a separate file
        for idx, row in smiles_data.iterrows():
            file_name = os.path.join(output_path, f"{idx+1:06d}.xyz")  # Modify file path to include timestamp directory
            with open(file_name, 'w') as file:
                file.write(row['XYZ'])
        
        print(f"XYZ files have been saved in {output_path}.")

if __name__ == "__main__":
    # Interactive input for CSV file path
    file_path = input("Please enter your CSV file path: ")
    smi2xyz_main(file_path)