import os
import glob
import numpy as np
import argparse
import logging

def setup_logging(log_file_path):
    # Clear any existing logging handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Configure logging to write to a file and stdout
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler()
        ]
    )

def convert_fchk_to_xyz(input_path='.'):
    multiwfn_commands = "100\n2\n2\n\n"
    multiwfn_executable = "Multiwfn"

    # Store the current working directory to restore later
    original_cwd = os.getcwd()
    
    # Change to the input directory
    os.chdir(input_path)
    
    # Create the temporary input file in the current directory
    input_file = 'temp_input.txt'
    with open(input_file, 'w') as f:
        f.write(multiwfn_commands)

    fchk_files = glob.glob('*.fchk')
    generated_xyz_files = []

    for fchk_file in fchk_files:
        logging.info(f"Processing {fchk_file}...")
        output_xyz = os.path.splitext(fchk_file)[0] + '.xyz'

        if os.path.exists(output_xyz):
            logging.info(f"{output_xyz} already exists. Skipping.")
            generated_xyz_files.append(os.path.join(input_path, output_xyz))
            continue

        command = f'{multiwfn_executable} "{fchk_file}" < {input_file} > /dev/null 2>&1'
        os.system(command)

        # Check if the output .xyz file was created with the correct name
        if os.path.exists(output_xyz):
            print(f"Generated {output_xyz}")
        else:
            print(f"Warning: {output_xyz} was not created for {fchk_file}. Skipping.")

    # Clean up the temporary input file
    if os.path.exists(input_file):
        os.remove(input_file)
    
    # Restore the original working directory
    os.chdir(original_cwd)
    
    return generated_xyz_files  # Return the list of generated .xyz files

def read_xyz(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    atoms = []
    coords = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            atoms.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
    return atoms, np.array(coords)

def euclidean_distance(coord1, coord2):
    return np.linalg.norm(coord1 - coord2)

def find_substructure(target_atoms, target_coords, substructure_atoms, substructure_coords, threshold=1.01):
    substructure_center = np.mean(substructure_coords, axis=0)
    center_distances = np.array([euclidean_distance(coord, substructure_center) for coord in substructure_coords])
    center_atom_index = np.argmin(center_distances)
    center_atom = substructure_atoms[center_atom_index]
    center_atom_coord = substructure_coords[center_atom_index]
    substructure_pairs = sorted(
        [(substructure_atoms[i], euclidean_distance(center_atom_coord, coord)) for i, coord in enumerate(substructure_coords) if i != center_atom_index],
        key=lambda x: x[1]
    )
    matches = []
    for i, target_atom in enumerate(target_atoms):
        if target_atom == center_atom:
            all_matches_found = True
            matched_indices = [i]
            for sub_atom, sub_dist in substructure_pairs:
                found_match = False
                for j, (t_atom, t_coord) in enumerate(zip(target_atoms, target_coords)):
                    if j not in matched_indices and t_atom == sub_atom:
                        if abs(euclidean_distance(target_coords[i], t_coord) - sub_dist) <= threshold:
                            found_match = True
                            matched_indices.append(j)
                            break
                if not found_match:
                    all_matches_found = False
                    break
            if all_matches_found:
                matches.append(matched_indices)
    return matches

def find_substructure_center_atom(substructure_coords):
    geometric_center = np.mean(substructure_coords, axis=0)
    center_atom_index = np.argmin([euclidean_distance(coord, geometric_center) for coord in substructure_coords])
    return center_atom_index

def sort_substructure_atoms(substructure_atoms, substructure_coords):
    center_atom_index = find_substructure_center_atom(substructure_coords)
    center_atom_coord = substructure_coords[center_atom_index]
    distances = [euclidean_distance(center_atom_coord, coord) for coord in substructure_coords]
    index_with_distances = list(enumerate(distances))
    index_with_distances = [index_distance for index_distance in index_with_distances if index_distance[0] != center_atom_index]
    sorted_indices_with_distances = sorted(index_with_distances, key=lambda x: x[1])
    sorted_indices, _ = zip(*sorted_indices_with_distances)
    sorted_indices = (center_atom_index,) + sorted_indices
    return list(sorted_indices)

def fchk2matches_main(input_path='.', xyz1_path='sub.xyz1', threshold=1.01):
    # Set up logging
    log_file = os.path.join(input_path, 'process.log')
    setup_logging(log_file)

    # Log the parameters
    logging.info(f"Input path: {input_path}")
    logging.info(f"Substructure file path: {xyz1_path}")
    logging.info(f"Threshold: {threshold}\n")

    # Step 1: Convert .fchk files to .xyz files
    xyz_files = convert_fchk_to_xyz(input_path)
    logging.info("\nConversion of .fchk to .xyz completed.\n")

    # Step 2: Read the substructure from the provided xyz1 file
    if not os.path.exists(xyz1_path):
        error_msg = f"Substructure file '{xyz1_path}' not found."
        logging.error(error_msg)
        print(error_msg)
        return

    substructure_atoms, substructure_coords = read_xyz(xyz1_path)
    # Sort substructure atoms based on center atom
    sorted_indices = sort_substructure_atoms(substructure_atoms, substructure_coords)

    # Files to write results
    matches_output_file = os.path.join(input_path, 'matches_output.csv')  # Original detailed matches
    match_counts_file = os.path.join(input_path, 'match_counts.csv')     # Counts of matches
    first_matches_file = os.path.join(input_path, 'first_matches.csv')   # First match per xyz file

    # Open files for writing
    with open(matches_output_file, 'w') as matches_file, \
         open(match_counts_file, 'w') as counts_file, \
         open(first_matches_file, 'w') as first_match_file:

        # Write headers
        matches_file.write('xyz_file,matches\n')
        counts_file.write('xyz_file,number_of_matches\n')
        first_match_file.write('xyz_file,first_match\n')

        # Iterate over each xyz file
        for xyz_file in xyz_files:
            # Read target structure
            target_atoms, target_coords = read_xyz(xyz_file)

            # Find matching substructures with the given threshold
            matches = find_substructure(target_atoms, target_coords, substructure_atoms, substructure_coords, threshold=threshold)

            # Sort matches and write to files
            for match in matches:
                sorted_match = [match[i] for i in sorted_indices]
                matches_file.write(f"{os.path.basename(xyz_file)},{' '.join(map(str, sorted_match))}\n")

            num_matches = len(matches)
            if num_matches > 0:
                # Write to match counts file
                counts_file.write(f"{os.path.basename(xyz_file)},{num_matches}\n")
                # Write the first match to the first matches file
                first_match_indices = [matches[0][i] for i in sorted_indices]
                first_match_file.write(f"{os.path.basename(xyz_file)},{' '.join(map(str, first_match_indices))}\n")
                # Log the matches
                logging.info(f"Found matches in {os.path.basename(xyz_file)}: {matches}")
            else:
                # Write zero matches to counts file
                counts_file.write(f"{os.path.basename(xyz_file)},0\n")
                # Log no matches found
                logging.info(f"No matches found in {os.path.basename(xyz_file)}")

    logging.info(f"\nMatching results saved to '{matches_output_file}'.")
    logging.info(f"Match counts saved to '{match_counts_file}'.")
    logging.info(f"First matches saved to '{first_matches_file}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process .fchk files, convert them to .xyz, and perform substructure matching.')

    parser.add_argument('--input_path', type=str, default='.',
                        help='Path to the directory containing .fchk files. Defaults to current directory.')
    parser.add_argument('--xyz1_path', type=str, required=True,
                        help='Path to the substructure .xyz1 file.')
    parser.add_argument('--threshold', type=float, default=1.01,
                        help='Threshold for substructure matching. Default is 1.01.')

    args = parser.parse_args()

    fchk2matches_main(input_path=args.input_path, xyz1_path=args.xyz1_path, threshold=args.threshold)