"""
SMILES到XYZ转换模块
"""

import pandas as pd
from openbabel import pybel as pb
import os
import re
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

def clean_column_name(column_name):
    """
    Clean column names by removing or replacing problematic characters.
    """
    if not isinstance(column_name, str):
        column_name = str(column_name)
    
    cleaned = column_name.lower()
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', cleaned)
    cleaned = re.sub(r'_+', '_', cleaned)
    cleaned = cleaned.strip('_')
    
    if cleaned and cleaned[0].isdigit():
        cleaned = 'exp_' + cleaned
    
    if not cleaned:
        cleaned = 'unnamed_feature'
    
    return cleaned

def validate_csv_columns(data):
    """
    Validate that the CSV contains required 'smiles' and 'target' columns.
    """
    if data is None:
        return False
    
    columns_lower = [col.lower() for col in data.columns]
    
    if 'smiles' not in columns_lower:
        print("Error: CSV file must contain a 'smiles' column (case-insensitive)")
        return False
    
    if 'target' not in columns_lower:
        print("Error: CSV file must contain a 'target' column (case-insensitive)")
        return False
    
    if len(data.columns) < 2:
        print("Error: CSV file must contain at least 2 columns")
        return False
    
    print("✓ Required columns 'smiles' and 'target' found")
    return True

def smiles_to_xyz(smiles_string, sample_name):
    """
    Convert a SMILES string to XYZ format.
    """
    try:
        mol = pb.readstring('smiles', smiles_string)
        mol.make3D()
        return mol.write(format='xyz'), True
    except Exception as e:
        print(f"Error converting SMILES {smiles_string} to XYZ: {e}")
        return None, False

def smi2xyz_main(file_path, check_extensions=['.fchk']):
    """
    Main function: Reads SMILES from CSV and converts them to XYZ files.
    """
    smiles_data = read_smiles_csv(file_path)
    
    if not validate_csv_columns(smiles_data):
        print("Please ensure your CSV file contains 'smiles' and 'target' columns (case-insensitive)")
        return
    
    conversion_count = 0
    skip_count = 0
    
    mapping_data = []
    column_mapping = {}
    column_name_changes = []
    
    for col in smiles_data.columns:
        col_lower = col.lower()
        if col_lower == 'smiles':
            column_mapping[col] = 'smiles'
        elif col_lower == 'target':
            column_mapping[col] = 'target'
        else:
            cleaned_name = clean_column_name(col)
            column_mapping[col] = cleaned_name
            if col != cleaned_name:
                column_name_changes.append(f"'{col}' -> '{cleaned_name}'")
    
    smiles_data_normalized = smiles_data.rename(columns=column_mapping)
    
    if column_name_changes:
        print(f"\n✓ Column names cleaned:")
        for change in column_name_changes:
            print(f"  {change}")
    
    duplicated_cols = smiles_data_normalized.columns.duplicated()
    if duplicated_cols.any():
        print("Warning: Found duplicated column names after cleaning. Adding suffixes...")
        cols = smiles_data_normalized.columns.tolist()
        for i in range(len(cols)):
            if duplicated_cols[i]:
                suffix = 1
                base_name = cols[i]
                while f"{base_name}_{suffix}" in cols:
                    suffix += 1
                cols[i] = f"{base_name}_{suffix}"
        smiles_data_normalized.columns = cols
    
    smiles_col = 'smiles'
    target_col = 'target'
    
    print(f"\n✓ SMILES column: '{smiles_col}'")
    print(f"✓ Target column: '{target_col}'")
    
    experimental_feature_cols = [col for col in smiles_data_normalized.columns 
                               if col not in [smiles_col, target_col]]
    
    if experimental_feature_cols:
        print(f"✓ Experimental feature columns found ({len(experimental_feature_cols)}):")
        for exp_col in experimental_feature_cols:
            print(f"  - {exp_col}")
    else:
        print("ℹ No additional experimental features found (only smiles and target columns)")

    print(f"\nData validation:")
    print(f"Total samples: {len(smiles_data_normalized)}")
    
    smiles_null_count = smiles_data_normalized[smiles_col].isnull().sum()
    if smiles_null_count > 0:
        print(f"Warning: {smiles_null_count} samples have missing SMILES values")
    
    target_null_count = smiles_data_normalized[target_col].isnull().sum()
    if target_null_count > 0:
        print(f"Warning: {target_null_count} samples have missing target values")

    if experimental_feature_cols:
        print("Experimental features null value check:")
        for exp_col in experimental_feature_cols:
            null_count = smiles_data_normalized[exp_col].isnull().sum()
            if null_count > 0:
                print(f"  - {exp_col}: {null_count} missing values")

    for idx, row in smiles_data_normalized.iterrows():
        sample_name = f"{idx+1:06d}"
        smiles_string = row[smiles_col]
        target_value = row[target_col]
        
        if pd.isnull(smiles_string) or smiles_string == '':
            print(f"Skipping sample {sample_name}: empty SMILES")
            continue
        
        mapping_record = {
            'Sample Name': sample_name,
            'smiles': smiles_string,
            'target': target_value
        }
        
        for exp_col in experimental_feature_cols:
            mapping_record[exp_col] = row[exp_col]
        
        mapping_data.append(mapping_record)
        
        file_exists = False
        for ext in check_extensions:
            if os.path.exists(sample_name + ext):
                print(f"File {sample_name + ext} already exists, skipping conversion.")
                file_exists = True
                skip_count += 1
                break
        
        if not file_exists:
            xyz_data, conversion_success = smiles_to_xyz(smiles_string, sample_name)
            
            if conversion_success:
                file_name = f"{sample_name}.xyz"
                with open(file_name, 'w') as file:
                    file.write(xyz_data)
                conversion_count += 1
                print(f"Converted {sample_name}: {smiles_string}")
            else:
                print(f"Failed to convert {sample_name}: {smiles_string}")

    if mapping_data:
        mapping_df = pd.DataFrame(mapping_data)
        mapping_df.to_csv('sample_mapping.csv', index=False)
        print(f"\n✓ Sample mapping file 'sample_mapping.csv' created successfully!")
        
        print(f"\nConversion Summary:")
        print(f"✓ Converted {conversion_count} new files")
        print(f"ℹ Skipped {skip_count} existing files")
        print(f"✓ Total samples in mapping: {len(mapping_data)}")
        
        print(f"✓ Mapping file columns: {list(mapping_df.columns)}")
        
        if experimental_feature_cols:
            print(f"✓ Experimental features included: {len(experimental_feature_cols)}")
            print("Final experimental feature names:")
            for exp_col in experimental_feature_cols:
                print(f"  - {exp_col}")
        else:
            print("ℹ No experimental features (only smiles and target)")
            
        if column_name_changes:
            mapping_record_file = 'column_name_mapping.log'
            with open(mapping_record_file, 'w') as f:
                f.write("Original Column Name -> Cleaned Column Name\n")
                f.write("=" * 50 + "\n")
                for change in column_name_changes:
                    f.write(change + "\n")
            print(f"✓ Column name mapping saved to '{mapping_record_file}'")
    else:
        print("Error: No valid samples found to process")
