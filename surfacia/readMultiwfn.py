import csv
import os
import glob
import subprocess
import logging
import time
import pandas as pd
import numpy as np
import re
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

def extract_after(text, keyword):
    """
    Extracts text after the specified keyword.
    """
    partitioned_text = text.partition(keyword)
    if partitioned_text[1] == '':
        return None
    else:
        return partitioned_text[2].strip()

def extract_before(text, keyword):
    """
    Extracts text before the specified keyword.
    """
    partitioned_text = text.partition(keyword)
    if partitioned_text[1] == '':
        return text.strip()
    else:
        return partitioned_text[0].strip()

def extract_between(text, start_delimiter, end_delimiter):
    """
    Extracts text between two specified delimiters.
    """
    start_index = text.find(start_delimiter)
    if start_index == -1:
        return None
    start_index += len(start_delimiter)
    end_index = text.find(end_delimiter, start_index)
    if end_index == -1:
        return None
    return text[start_index:end_index].strip()

def read_first_matches_csv(csv_path):
    """
    Reads the first_matches CSV and returns a dictionary mapping xyz_file to list of indices.
    """
    first_matches = {}
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            xyz_file = row['xyz_file']
            first_match = row['first_match']
            indices = [int(i.strip()) for i in first_match.strip().split()]
            first_matches[xyz_file] = indices
    return first_matches

def create_descriptors_content(fragment_indices_list):
    # Adjust indices by adding one for 1-based indexing, if necessary
    adjusted_indices = [i + 1 for i in fragment_indices_list]
    indices_str = ",".join(map(str, adjusted_indices))
    print(f"Descriptors content indices: {indices_str}")
    content = f"""0
100
21
size
0
MPP
a
n
q
0
300
5
0
8
8
1
h-1
h
l
l+1
0
-10
12
2
-4
1
1
0.002
0
11
n
12
{indices_str}
n
-1
2
1
1
1
0.002
0
11
n
12
{indices_str}
n
-1
2
2
1
1
1
0.002
0
11
n
12
{indices_str}
n
-1

"""
    return content

def run_multiwfn_on_fchk_files(input_path='.', first_matches={}):
    original_dir = os.getcwd()
    os.chdir(input_path)
    fchk_files = glob.glob('*.fchk')
    processed_files = []

    for fchk_file in fchk_files:
        sample_name = os.path.splitext(fchk_file)[0]  # e.g., '003' if fchk_file is '003.fchk'
        xyz_file = sample_name + '.xyz'

        if xyz_file in first_matches:
            fragment_indices = first_matches[xyz_file]
        else:
            fragment_indices = []
            logging.warning(f"No fragment indices found for {xyz_file}, using empty fragment list.")

        descriptors_content = create_descriptors_content(fragment_indices)
        print(f"Descriptors content for {sample_name}:\n{descriptors_content}")
        output_file = f"{sample_name}.txt"

        # Construct the command argument list, only adding the -silent option
        command = ["Multiwfn", fchk_file, "-silent"]

        try:
            with open(output_file, 'w') as outfile:
                subprocess.run(
                    command,
                    input=descriptors_content,
                    text=True,
                    stdout=outfile,
                    stderr=subprocess.PIPE,
                    check=True
                )
            logging.info(f"Multiwfn output saved to {output_file}")
            processed_files.append(output_file)
        except subprocess.CalledProcessError as e:
            logging.warning(f"Warning: An error occurred while running Multiwfn on {fchk_file}: {e}")
            logging.warning(f"stderr: {e.stderr}")

        if os.path.exists(output_file):
            if output_file not in processed_files:
                processed_files.append(output_file)
                logging.info(f"Output file {output_file} was created. Good!")
        else:
            logging.error(f"Error: Output file {output_file} was not created for {fchk_file}.")

    os.chdir(original_dir)

def process_txt_files(input_directory, output_directory, smiles_target_csv_path, first_matches_csv_path, descriptor_option=1):
    """
    Process all .txt files in the specified directory, extract features, and generate a feature matrix.

    Args:
        input_directory (str): Directory containing .txt files.
        output_directory (str): Directory to save output files.
        smiles_target_csv_path (str): Path to the CSV file containing SMILES and target.
        first_matches_csv_path (str): Path to the CSV file containing fragment indices.
        descriptor_option (int): Select descriptor type. 1=Only molecular properties, 2=Molecular + specific atom properties, 3=Molecular + specific atom + fragment properties.

    Returns:
        None
    """
    Version = '2.0'
    c_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    DIR = os.path.join(output_directory, f'Surfacia_{Version}_{c_time}')
    os.makedirs(DIR, exist_ok=True)

    current_directory = input_directory
    file_list = [f for f in os.listdir(current_directory) if f.endswith('.txt')]

    print("Found text files:", file_list)

    df = pd.DataFrame()

    # Iterate through each .txt file to extract features
    for filename in file_list:
        data = {}
        with open(os.path.join(input_directory, filename), 'r') as file:
            lines_iter = iter(file.readlines())

        sample_name = filename[:-4]
        print('Processing sample:', sample_name)
        data['Sample Name'] = sample_name
        matrix = {}
        odi_values = []
        summary_count = 0  # Counts the number of Summary sections encountered
        in_fragment_section = False

        while True:
            try:
                line = next(lines_iter)
                # Basic Information Extraction
                if 'Atoms:' in line:
                    atom_num = int(extract_between(line, "Atoms: ", ","))
                    data['Atom Number'] = atom_num
                if 'Molecule weight:' in line:
                    weight = float(extract_between(line, "Molecule weight:", "Da"))
                    data['Molecule Weight'] = weight
                if 'Orbitals from 1 to' in line:
                    occupied_orbitals = float(extract_between(line, "Orbitals from 1 to", "are occupied"))
                    data['Occupied Orbitals'] = occupied_orbitals
                if 'Atom list:' in line:
                    atoms = []
                    xyz = []
                    for _ in range(atom_num):
                        line = next(lines_iter).strip()
                        if line:
                            atom_info = line.split()
                            atom = atom_info[0].split('(')[1].split(')')[0]
                            x, y, z = map(float, atom_info[-3:])
                            atoms.append(atom)
                            xyz.append([x, y, z])
                        else:
                            break
                # HOMO/LUMO Energy Levels
                if 'is HOMO, energy:' in line:
                    homo_energy = extract_between(line, "is HOMO, energy:", "a.u.")
                    data['HOMO'] = float(homo_energy)
                if 'LUMO, energy:' in line:
                    lumo_energy = extract_between(line, "is LUMO, energy:", "a.u.")
                    data['LUMO'] = float(lumo_energy)
                if 'HOMO-LUMO gap:' in line:
                    gap_energy = extract_between(line, "gap:", "a.u.")
                    data['HOMO-LUMO Gap'] = float(gap_energy)
                # Molecular Shape
                if 'Farthest distance:' in line:
                    farthest_distance = float(extract_between(line, "):", "Angstrom"))
                    data['Farthest Distance'] = farthest_distance
                if 'Radius of the system: ' in line:
                    mol_radius = float(extract_between(line, ":", "Angstrom"))
                    data['Molecular Radius'] = mol_radius
                if 'Length of the three sides:' in line:
                    mol_size = list(map(float, extract_between(line, ":", "Angstrom").split()))
                    mol_size.sort()
                    data['Molecular Size Short'] = mol_size[0]
                    data['Molecular Size Medium'] = mol_size[1]
                    data['Molecular Size Long'] = mol_size[2]
                    data['Long/Sum Size Ratio'] = mol_size[2] / sum(mol_size)
                    data['Length/Diameter'] = mol_size[2] / (2 * mol_radius)
                if 'Molecular planarity parameter (MPP) is' in line:
                    mpp = float(extract_before(extract_after(line, "is"), "Angstrom"))
                    data['MPP'] = mpp
                if 'Span of deviation from plane' in line:
                    sdp = float(extract_before(extract_after(line, "is"), "Angstrom"))
                    data['SDP'] = sdp
                # Dipole Moment
                if 'Magnitude of dipole moment:' in line:
                    dipole_moment = float(extract_between(line, 'Magnitude of dipole moment:', "a.u."))
                    data['Dipole Moment (a.u.)'] = dipole_moment
                if 'Magnitude: |Q_2|=' in line:
                    quadrupole_moment = float(extract_after(line, "Magnitude: |Q_2|="))
                    data['Quadrupole Moment'] = quadrupole_moment
                if 'Magnitude: |Q_3|=' in line:
                    octopole_moment = float(extract_after(line, "|Q_3|= "))
                    data['Octopole Moment'] = octopole_moment
                # ODI Index
                if 'Orbital delocalization index:' in line:
                    odi_value = float(extract_after(line, "index:"))
                    odi_values.append(odi_value)
                    data['ODI LUMO+1'] = odi_values[0] if len(odi_values) > 0 else None
                    data['ODI LUMO'] = odi_values[1] if len(odi_values) > 1 else None
                    data['ODI HOMO'] = odi_values[2] if len(odi_values) > 2 else None
                    data['ODI HOMO-1'] = odi_values[3] if len(odi_values) > 3 else None
                    data['ODI Mean'] = np.mean(odi_values) if odi_values else None
                    data['ODI Std'] = np.std(odi_values) if odi_values else None
                if 'Isosurface area:' in line:
                    isosurface_area = float(extract_between(line, 'Bohr^2  (', "Angstrom^2)"))
                    data['Isosurface area'] = isosurface_area
                if 'Sphericity:' in line:
                    sphericity = float(extract_after(line, "Sphericity:"))
                    data['Sphericity'] = sphericity
                # LEAE, ESP, ALIE Sections
                try:
                    if '================= Summary of surface analysis =================' in line:
                        summary_count += 1
                        print('Summary of surface analysis', summary_count)
                        if summary_count == 1:
                            # LEAE Section
                            while True:
                                line = next(lines_iter)
                                if 'Minimal value:' in line and not in_fragment_section:
                                    # Extract molecule's minimal and maximal values
                                    data['LEAE Minimal Value'] = float(extract_between(line, "Minimal value:", 'eV,   Maximal value:'))
                                    data['LEAE Maximal Value'] = float(extract_between(line, "eV,   Maximal value:", 'eV'))
                                elif 'Overall average value:' in line and not in_fragment_section:
                                    data['LEAE Average Value'] = float(extract_between(line, "a.u. (", 'eV'))
                                elif 'Variance:' in line and not in_fragment_section:
                                    data['LEAE Variance'] = float(extract_between(line, "a.u.^2  (", 'eV'))
                                if 'Note: Below minimal and maximal values are in eV' in line:
                                    next(lines_iter)  # Skip note line
                                    matrix_data = []
                                    for _ in range(atom_num):
                                        matrix_line = next(lines_iter).strip()
                                        if matrix_line:
                                            matrix_data.append(matrix_line)
                                        else:
                                            break
                                    next(lines_iter)  # Skip empty line
                                    line = next(lines_iter)
                                    if 'Note: Average and variance below are in eV and eV^2 respectively' in line:
                                        next(lines_iter)  # Skip note line
                                    matrix_data2 = []
                                    for _ in range(atom_num):
                                        matrix_line = next(lines_iter).strip()
                                        if matrix_line:
                                            matrix_data2.append(matrix_line)
                                        else:
                                            break
                                    num_rows = min(len(matrix_data), len(matrix_data2))
                                    data['Surface exposed atom num'] = num_rows
                                    for i in range(num_rows):
                                        matrix_data[i] += ' ' + ' '.join(matrix_data2[i].split()[-6:])
                                    matrix['Matrix Data'] = matrix_data
                                # Check for fragment properties section
                                if 'Properties on the surface of this fragment:' in line:
                                    in_fragment_section = True
                                    # Begin parsing fragment properties
                                    while True:
                                        line = next(lines_iter).strip()
                                        if 'Minimal value:' in line:
                                            frag_min_val_str = extract_between(line, "Minimal value:", 'eV,   Maximal value:')
                                            frag_max_val_str = extract_between(line, "eV,   Maximal value:", 'eV')
                                            # Handle possible '**************' in fragment min and max values
                                            if frag_min_val_str.strip() == '**************':
                                                data['Frag LEAE Minimal Value'] = np.nan
                                            else:
                                                data['Frag LEAE Minimal Value'] = float(frag_min_val_str)
                                            if frag_max_val_str.strip() == '**************':
                                                data['Frag LEAE Maximal Value'] = np.nan
                                            else:
                                                data['Frag LEAE Maximal Value'] = float(frag_max_val_str)
                                        # Extract fragment's overall average value
                                        elif 'Overall average value:' in line:
                                            data['Frag LEAE Average Value'] = float(extract_between(line, "a.u. (", 'eV'))
                                            break
                                    in_fragment_section = False
                                    break
                        elif summary_count == 2:
                            # ESP Section
                            while True:
                                line = next(lines_iter)
                                if 'Volume:' in line:
                                    data['Volume (Angstrom^3)'] = float(extract_between(line, "Bohr^3  (", 'Angstrom^3)'))
                                elif 'Estimated density according to mass and volume (M/V):' in line:
                                    data['Density (g/cm^3)'] = float(extract_between(line, "M/V):", 'g/cm^3'))
                                elif 'Minimal value:' in line:
                                    data['ESP Minimal Value'] = float(extract_between(line, "Minimal value:", 'kcal/mol   Maximal value:'))
                                    data['ESP Maximal Value'] = float(extract_between(line, "kcal/mol   Maximal value:", 'kcal/mol'))
                                elif 'Overall average value:' in line:
                                    data['ESP Overall Average Value (kcal/mol)'] = float(extract_between(line, "a.u. (", 'kcal/mol)'))
                                elif 'Overall variance (sigma^2_tot):' in line:
                                    data['ESP Overall Variance ((kcal/mol)^2)'] = float(extract_between(line, "a.u.^2 (", '(kcal/mol)^2)'))
                                elif 'Balance of charges (nu):' in line:
                                    data['Balance of Charges (nu)'] = float(extract_after(line, "Balance of charges (nu):"))
                                elif 'Product of sigma^2_tot and nu:' in line:
                                    data['Product of sigma^2_tot and nu ((kcal/mol)^2)'] = float(extract_between(line, "a.u.^2 (", '(kcal/mol)^2)'))
                                elif 'Internal charge separation (Pi):' in line:
                                    data['Internal Charge Separation (Pi) (kcal/mol)'] = float(extract_between(line, "a.u. (", 'kcal/mol)'))
                                elif 'Molecular polarity index (MPI):' in line:
                                    data['Molecular Polarity Index (MPI) (kcal/mol)'] = float(extract_between(line, "eV (", 'kcal/mol)'))
                                elif 'Polar surface area (|ESP| > 10 kcal/mol):' in line:
                                    data['Polar Surface Area (Angstrom^2)'] = float(extract_between(line, "Polar surface area (|ESP| > 10 kcal/mol):", 'Angstrom^2'))
                                    data['Polar Surface Area (%)'] = float(extract_between(line, "Angstrom^2  (", '%)'))
                                if 'Note: Minimal and maximal value below are in kcal/mol' in line:
                                    next(lines_iter)  # Skip note line
                                    matrix_data3 = []
                                    for _ in range(atom_num):
                                        matrix_line = next(lines_iter).strip()
                                        if matrix_line:
                                            matrix_data3.append(matrix_line)
                                        else:
                                            break
                                    next(lines_iter)  # Skip empty line
                                    line = next(lines_iter)
                                    if 'Note: Average and variance below are in' in line:
                                        next(lines_iter)  # Skip note line
                                    matrix_data4 = []
                                    for _ in range(atom_num):
                                        matrix_line = next(lines_iter).strip()
                                        if matrix_line:
                                            matrix_data4.append(matrix_line)
                                        else:
                                            break
                                    next(lines_iter)  # Skip another line
                                    line = next(lines_iter)
                                    if 'Note: Internal charge separation' in line:
                                        next(lines_iter)  # Skip note line
                                    matrix_data5 = []
                                    for _ in range(atom_num):
                                        matrix_line = next(lines_iter).strip()
                                        if matrix_line:
                                            matrix_data5.append(matrix_line)
                                        else:
                                            break
                                    for i in range(num_rows):
                                        matrix_data[i] += ' ' + ' '.join(matrix_data3[i].split()[-5:]) + ' ' + ' '.join(matrix_data4[i].split()[-6:]) + ' ' + ' '.join(matrix_data5[i].split()[-3:])
                                    matrix['Matrix Data'] = matrix_data
                                # Check for fragment properties section
                                if 'Properties on the surface of this fragment:' in line:
                                    in_fragment_section = True
                                    # Begin parsing fragment properties
                                    while True:
                                        line = next(lines_iter).strip()
                                        if 'Minimal value:' in line:
                                            frag_min_val_str = extract_between(line, "Minimal value:", 'kcal/mol   Maximal value:')
                                            frag_max_val_str = extract_between(line, "kcal/mol   Maximal value:", 'kcal/mol')
                                            # Handle possible '**************' in fragment min and max values
                                            if frag_min_val_str.strip() == '**************':
                                                data['Frag ESP Minimal Value'] = np.nan
                                            else:
                                                data['Frag ESP Minimal Value'] = float(frag_min_val_str)
                                            if frag_max_val_str.strip() == '**************':
                                                data['Frag ESP Maximal Value'] = np.nan
                                            else:
                                                data['Frag ESP Maximal Value'] = float(frag_max_val_str)
                                        # Extract fragment's overall surface area
                                        elif 'Overall surface area:' in line:
                                            data['Frag ESP Overall Surface Area (Angstrom^2)'] = float(extract_between(line, '(', 'Angstrom^2)'))
                                        # Extract fragment's overall average value
                                        elif 'Overall average value:' in line:
                                            data['Frag ESP Average Value'] = float(extract_between(line, "a.u. (", 'kcal/mol)'))
                                        elif 'Overall variance (sigma^2_tot):' in line:
                                            data['Frag ESP variance Value'] = float(extract_between(line, "a.u.^2 (", '(kcal/mol)'))
                                        elif 'Internal charge separation (Pi):' in line:
                                            data['Frag ESP Pi Value'] = float(extract_between(line, "a.u. (", 'kcal/mol)'))
                                            break
                                    in_fragment_section = False
                                    break
                        elif summary_count == 3:
                            # ALIE Section
                            while True:
                                line = next(lines_iter)
                                if 'Minimal value:' in line:
                                    data['ALIE Minimal Value'] = float(extract_between(line, "Minimal value:", 'eV,   Maximal value:'))
                                    data['ALIE Maximal Value'] = float(extract_between(line, "eV,   Maximal value:", 'eV'))
                                elif 'Average value:' in line:
                                    data['ALIE Average Value'] = float(extract_between(line, "a.u. (", 'eV'))
                                elif 'Variance:' in line:
                                    data['ALIE Variance'] = float(extract_between(line, "a.u.^2  (", 'eV'))
                                if 'Minimal, maximal and average value are in eV, variance is in eV^2' in line:
                                    next(lines_iter)  # Skip note line
                                    matrix_data6 = []
                                    for _ in range(atom_num):
                                        matrix_line = next(lines_iter).strip()
                                        if matrix_line:
                                            matrix_data6.append(matrix_line)
                                        else:
                                            break
                                    titles = [
                                        "Atom#", "LEAE All area", "LEAE Positive area", "LEAE Negative area",
                                        "LEAE Minimal value", "LEAE Maximal value", "LEAE All average", "LEAE Positive average",
                                        "LEAE Negative average", "LEAE All variance", "LEAE Positive variance", "LEAE Negative variance",
                                        "ESP All area (Ang^2)", "ESP Positive area (Ang^2)", "ESP Negative area (Ang^2)",
                                        "ESP Minimal value (kcal/mol)", "ESP Maximal value (kcal/mol)",
                                        "ESP All average (kcal/mol)", "ESP Positive average (kcal/mol)", "ESP Negative average (kcal/mol)",
                                        "ESP All variance (kcal/mol)^2", "ESP Positive variance (kcal/mol)^2", "ESP Negative variance (kcal/mol)^2",
                                        "ESP Pi (kcal/mol)", "ESP nu", "ESP nu*sigma^2",
                                        "ALIE Area(Ang^2)", "ALIE Min value", "ALIE Max value", "ALIE Average", "ALIE Variance"
                                    ]
                                    for i in range(num_rows):
                                        matrix_data[i] += ' ' + ' '.join(matrix_data6[i].split()[-5:])
                                    matrix['Matrix Data'] = matrix_data
                                # Check for fragment properties section
                                if 'Properties on the surface of this fragment:' in line:
                                    in_fragment_section = True
                                    # Begin parsing fragment properties
                                    while True:
                                        line = next(lines_iter).strip()
                                        if 'Minimal value:' in line:
                                            frag_min_val_str = extract_between(line, "Minimal value:", 'eV   Maximal value:')
                                            frag_max_val_str = extract_between(line, "eV   Maximal value:", 'eV')
                                            # Handle possible '**************' in fragment min and max values
                                            if frag_min_val_str.strip() == '**************':
                                                data['Frag ALIE Minimal Value'] = np.nan
                                            else:
                                                data['Frag ALIE Minimal Value'] = float(frag_min_val_str)
                                            if frag_max_val_str.strip() == '**************':
                                                data['Frag ALIE Maximal Value'] = np.nan
                                            else:
                                                data['Frag ALIE Maximal Value'] = float(frag_max_val_str)
                                        # Extract fragment's overall average value
                                        elif 'Average value:' in line:
                                            data['Frag ALIE Average Value'] = float(extract_between(line, "a.u. (", 'eV'))
                                        elif 'Variance:' in line:
                                            data['Frag ALIE Variance Value'] = float(extract_between(line, "a.u.^2  (", 'eV'))
                                            break
                                    in_fragment_section = False
                                    break
                except ValueError as e:
                    continue
            except StopIteration:
                break
        temp_df = pd.DataFrame([data])
        df = pd.concat([df, temp_df], ignore_index=True)

        if 'Matrix Data' in matrix:
            matrix_data = matrix['Matrix Data']
            new_filename = 'AtomProp_' + sample_name + '.csv'  # Add '.csv' extension

            max_index = max(int(row.split()[0]) for row in matrix_data)
            merged_data = [['NaN'] * (len(matrix_data[0].split()) + len(xyz[0])) for _ in range(max_index)]
            num_xyz_columns = len(xyz[0])
            num_matrix_columns = len(matrix_data[0].split())

            for row in matrix_data:
                parts = row.split()
                index = int(parts[0]) - 1  # Convert 1-based index to 0-based index
                if index < len(xyz):
                    merged_data[index] = xyz[index] + parts[:]  # Exclude index part from matrix_data

            while len(merged_data) < len(xyz):
                merged_data.append(['NaN'] * (num_xyz_columns + num_matrix_columns))

            for i in range(len(xyz)):
                if len(merged_data[i]) > num_xyz_columns:
                    if merged_data[i][num_xyz_columns] == 'NaN':
                        merged_data[i] = xyz[i] + ['NaN'] * num_matrix_columns
                else:
                    additional_nans = ['NaN'] * (num_xyz_columns + 1 - len(merged_data[i]))
                    merged_data[i] = merged_data[i] + additional_nans
                    if merged_data[i][num_xyz_columns] == 'NaN':
                        merged_data[i] = xyz[i] + ['NaN'] * num_matrix_columns

            for i in range(len(merged_data)):
                if i < len(atoms):
                    merged_data[i].insert(0, atoms[i])
                else:
                    merged_data[i].insert(0, 'NaN')

            title_parts = ['Element', 'X', 'Y', 'Z'] + titles
            merged_data.insert(0, title_parts)

            output_filename = Path(DIR, new_filename)
            with open(output_filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(merged_data)
            print(f"Data successfully written to {output_filename}")
        else:
            # If there is no matrix_data, still generate an empty AtomProp file so that subsequent processing won't fail
            new_filename = 'AtomProp_' + sample_name + '.csv'
            output_filename = Path(DIR, new_filename)
            with open(output_filename, 'w', newline='') as file:
                pass  # Create an empty file

    RECORD_NAME1 = 'RawFull_' + str(df.shape[0]) + '_' + str(df.shape[1]) + '.csv'
    RECORD_NAME = Path(DIR, RECORD_NAME1)
    df.to_csv(RECORD_NAME, index=False)

    df_sorted = df.sort_values(by='Sample Name')
    df_cleaned = df_sorted.dropna(axis=1)
    RECORD_NAME_CLEAN = 'Full0_' + str(df_cleaned.shape[0]) + '_' + str(df_cleaned.shape[1]) + '.csv'
    df_cleaned.to_csv(Path(DIR, RECORD_NAME_CLEAN), index=False)

    # Choose processing method based on descriptor_option
    if descriptor_option == 1:
        # Consider only molecular properties
        final_df = df_cleaned.copy()
    else:
        featnums = [9, 10, 11, 17, 20, 21, 22, 28, 32, 33, 34, 35]  # Adjust as needed

        # Get all AtomProp_*.csv files in the current directory
        filelist = [
            f
            for f in os.listdir(DIR)
            if f.startswith("AtomProp_") and f.endswith(".csv")
        ]

        if not filelist:
            print("No AtomProp_*.csv files found.")
            exit()

        # Get feature names from the header of the first file
        first_file = Path(DIR, filelist[0])
        try:
            titles_df = pd.read_csv(first_file, nrows=0)
            titles = titles_df.columns.tolist()
        except pd.errors.EmptyDataError:
            titles = []

        # Create mapping from featnum to column name
        featnum_to_colname = {}
        for featnum in featnums:
            if featnum - 1 < len(titles):
                colname = titles[featnum - 1]
                featnum_to_colname[featnum] = colname
            else:
                logging.warning(f"Feature number {featnum} exceeds number of columns in atom properties")
                featnum_to_colname[featnum] = f'Feature{featnum}'

        # Initialize a dataframe to store feature data
        feature_df = pd.DataFrame()
        # Read fragment indices
        first_matches = read_first_matches_csv(first_matches_csv_path)
        # Process each file in the file list
        for i, file in enumerate(filelist):
            # Read data from the file
            df_atom = pd.read_csv(Path(DIR, file))
            sample_name = file.replace('AtomProp_', '').replace('.csv', '')
            temp_features = {'Sample Name': sample_name}

            # Print debug information
            print(f"Processing file: {file}")
            print(f"File contents (first few rows):\n{df_atom.head()}")
            print(f"DataFrame shape: {df_atom.shape}")


            # Choose atomlist based on descriptor_option
            if descriptor_option == 2:
                # Consider only one specific atom, assuming the first atom here
                atomlist = [1]
            elif descriptor_option == 3:
                # Read atomlist for current sample from first_matches
                xyz_file = f"{sample_name}.xyz"
                if xyz_file in first_matches:
                    raw_atomlist = first_matches[xyz_file]
                    # Filter out invalid indices, e.g., negative or exceeding atom count
                    atomlist = [idx for idx in raw_atomlist if 0 <= idx < len(df_atom)]
                    if not atomlist:
                        print(f"Warning: After filtering, atomlist for {xyz_file} is empty.")
                else:
                    atomlist = []
                    print(f"Warning: No fragment indices found for {xyz_file}, using empty atomlist.")

            print(f"Atom list for {sample_name}: {atomlist}")

            # Iterate through each combination of atomlist and featnums
            for pos, atom_idx in enumerate(atomlist, start=1):
                for featnum in featnums:
                    colname = featnum_to_colname.get(featnum, f'Feature{featnum}')
                    key = f'Atom{pos}_{colname}'

                    if atom_idx is not None and atom_idx < len(df_atom):
                        if colname in df_atom.columns:
                            temp_features[key] = df_atom.loc[atom_idx, colname]
                        else:
                            temp_features[key] = np.nan
                            print(f"Warning: Column {colname} not found in atom properties for sample {sample_name}")
                    else:
                        temp_features[key] = np.nan
                        print(f"Warning: Atom index {atom_idx} exceeds number of atoms in sample {sample_name}")

            temp_df = pd.DataFrame([temp_features], columns=temp_features.keys())
            feature_df = pd.concat([feature_df, temp_df], ignore_index=True)

        # Merge feature_df with molecular properties df_cleaned
        final_df = pd.merge(df_cleaned, feature_df, on='Sample Name', how='left')

        if descriptor_option == 3:
            # Reorder columns: Sample Name -> Molecular properties -> Frag columns -> Atom features -> smiles, target
            # 1. Molecular property columns (excluding 'Sample Name' and 'Frag' related columns)
            molecule_cols = [col for col in df_cleaned.columns if col != 'Sample Name' and not col.startswith('Frag')]
            # 2. Frag related columns
            frag_columns = [col for col in df_cleaned.columns if col.startswith('Frag')]
            frag_cols = frag_columns
            # 3. Atom feature columns
            atom_cols = [col for col in final_df.columns if col.startswith('Atom')]
            # 4. 'Sample Name' column
            sample_col = ['Sample Name']

            # Define new column order
            new_order = sample_col + molecule_cols + frag_cols + atom_cols 

            # Ensure all columns exist in final_df
            available_columns = [col for col in new_order if col in final_df.columns]

            # Reorder final_df
            final_df = final_df[available_columns]

    # Merge SMILES and target columns and save
    try:
        df_smiles_target = pd.read_csv(smiles_target_csv_path)
        df_smiles_target.columns = map(str.lower, df_smiles_target.columns)
        df_smiles_target = df_smiles_target[['smiles', 'target']]
    except ValueError:
        print("No 'smiles' or 'target' column found in the CSV.")
        exit()

    merged_df = pd.concat([final_df, df_smiles_target], axis=1)

    # Save final data
    RECORD_NAME_MERGED = f'FinalDescriptorOption{descriptor_option}_' + str(merged_df.shape[0]) + '_' + str(merged_df.shape[1]) + '.csv'
    merged_output_filename = Path(DIR, RECORD_NAME_MERGED)
    merged_df.to_csv(merged_output_filename, index=False, float_format='%.6f')

    print(f'Data written to {merged_output_filename}')
    print("Processing completed.")

    merged_df = pd.read_csv(merged_output_filename)
    S_N = merged_df.shape[0]
    F_N = merged_df.shape[1] - 3  # Assuming 'Sample Name', 'smiles', 'target' are three columns

    # Create a MachineLearning folder inside the main output directory
    ML_DIR = Path(DIR, "MachineLearning")
    ML_DIR.mkdir(exist_ok=True)

    # Save SMILES
    INPUT_SMILES = Path(ML_DIR, f'Smiles_{S_N}.csv')
    merged_df[['smiles']].to_csv(INPUT_SMILES, index=False, header=False)

    # Save labels (values)
    INPUT_Y = Path(ML_DIR, f'Values_True_{S_N}.csv')
    merged_df[['target']].to_csv(INPUT_Y, index=False, header=False)

    # Save feature matrix
    INPUT_X = Path(ML_DIR, f'Features_{S_N}_{F_N}.csv')
    merged_df.drop(['Sample Name', 'smiles', 'target'], axis=1).to_csv(INPUT_X, index=False, float_format='%.6f', header=False)

    # Save feature names with spaces replaced by underscores
    INPUT_TITLE = Path(ML_DIR, f'Title_{F_N}.csv')
    with open(INPUT_TITLE, 'w') as f:
        for col in merged_df.columns[1:-2]:
            # Replace spaces with underscores
            formatted_col = col.replace(' ', '_')
            formatted_col = formatted_col.replace('/', '_')
            f.write(formatted_col + '\n')

    print(f'Machine Learning data split and saved in {ML_DIR}:')
    print(f'  - SMILES: {INPUT_SMILES.name}')
    print(f'  - Values: {INPUT_Y.name}')
    print(f'  - Features: {INPUT_X.name}')
    print(f'  - Feature Titles: {INPUT_TITLE.name}')
    print("All processing completed.")
    
if __name__ == "__main__":

    # Interactive input for directories
    input_directory = input("Please enter the input directory path: ")
    output_directory = input("Please enter the output directory path: ")
    
    # Interactive input for CSV file paths
    smiles_target_csv_path = input("Please enter the path for the SMILES target CSV file: ")
    first_matches_csv_path = input("Please enter the path for the first matches CSV file: ")
        # Interactive input for descriptor option

    descriptor_option = input("Please choose a descriptor option (1, 2, or 3): ")
    process_txt_files(input_directory, output_directory, smiles_target_csv_path, first_matches_csv_path, descriptor_option=1)