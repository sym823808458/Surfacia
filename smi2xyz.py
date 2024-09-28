import pandas as pd
from openbabel import pybel as pb

def read_smiles_csv(file_path):
    """
    Read a CSV file containing SMILES strings.

    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        DataFrame: DataFrame containing SMILES strings.
    """
    try:
        # Read the CSV file
        data = pd.read_csv(file_path)
        print("CSV file read successfully!")
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

if __name__ == "__main__":
    # Interactive input for CSV file path
    file_path = input("Enter the path to your CSV file: ")
    smiles_data = read_smiles_csv(file_path)
    
    if smiles_data is not None:
        # Assuming the SMILES data is in a column named 'SMILES'
        # Add a new column 'XYZ' containing the converted XYZ data
        smiles_data['XYZ'] = smiles_data['SMILES'].apply(smiles_to_xyz)
        print("Conversion completed!")

        # Save each XYZ data to a separate file
        for idx, row in smiles_data.iterrows():
            file_name = f"{idx+1:06d}.xyz"  # Naming files as '00001.xyz', '00002.xyz', etc.
            with open(file_name, 'w') as file:
                file.write(row['XYZ'])
        print("XYZ files have been saved.")