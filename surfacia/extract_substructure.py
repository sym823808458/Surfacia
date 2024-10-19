# extract_substructure.py

import os
import numpy as np

def read_xyz(filename):
    """Read an XYZ file and return a list of atoms and an array of coordinates."""
    with open(filename, 'r') as file:
        lines = file.readlines()
    atoms = []
    coords = []
    for line in lines[2:]:  # Skip the first two lines
        parts = line.split()
        if len(parts) >= 4:
            atoms.append(parts[0])  # Atom type
            coords.append([float(x) for x in parts[1:4]])  # Coordinates
    return atoms, np.array(coords)

def euclidean_distance(coord1, coord2):
    """Calculate the Euclidean distance between two coordinates."""
    return np.linalg.norm(coord1 - coord2)

def find_substructure_center_atom(substructure_coords):
    """Find the index of the atom closest to the geometric center of the substructure."""
    # Calculate the geometric center
    geometric_center = np.mean(substructure_coords, axis=0)
    # Find the atom closest to the geometric center
    center_atom_index = np.argmin([euclidean_distance(coord, geometric_center) for coord in substructure_coords])
    return center_atom_index

def sort_substructure_atoms(substructure_atoms, substructure_coords):
    """Sort substructure atoms by their distance from the center atom moving outward."""
    center_atom_index = find_substructure_center_atom(substructure_coords)
    center_atom_coord = substructure_coords[center_atom_index]

    # Calculate distances from the center atom to other atoms
    distances = [euclidean_distance(center_atom_coord, coord) for coord in substructure_coords]

    # Create a list of tuples containing index and distance
    index_with_distances = list(enumerate(distances))

    # Filter out the center atom
    index_with_distances = [index_with_distance for index_with_distance in index_with_distances if index_with_distance[0] != center_atom_index]

    # Sort the indices based on distance
    sorted_indices_with_distances = sorted(index_with_distances, key=lambda x: x[1])

    # Unpack the sorted indices
    sorted_indices, _ = zip(*sorted_indices_with_distances) if sorted_indices_with_distances else ([], [])

    # Add the center atom index to the beginning of the sorted list
    sorted_indices = (center_atom_index,) + sorted_indices

    return list(sorted_indices)

def find_substructure(target_atoms, target_coords, substructure_atoms, substructure_coords, threshold=1.1):
    """Find matching substructures in the target molecule and return a list of matched atom indices."""
    # Calculate the geometric center of the substructure
    substructure_center = np.mean(substructure_coords, axis=0)

    # Find distances from substructure center to each atom
    center_distances = np.array([euclidean_distance(coord, substructure_center) for coord in substructure_coords])
    center_atom_index = np.argmin(center_distances)
    center_atom = substructure_atoms[center_atom_index]
    center_atom_coord = substructure_coords[center_atom_index]

    # Record distances and pairs, including atom types
    substructure_pairs = sorted(
        [(substructure_atoms[i], euclidean_distance(center_atom_coord, coord)) for i, coord in enumerate(substructure_coords) if i != center_atom_index],
        key=lambda x: x[1]
    )

    # Search for matches in the target structure
    matches = []  # This list will contain complete matched substructures
    for i, target_atom in enumerate(target_atoms):
        if target_atom == center_atom:  # Check if the center atom matches
            all_matches_found = True
            matched_indices = [i]  # Store indices of the matched substructure

            for sub_atom, sub_dist in substructure_pairs:
                found_match = False
                for j, (t_atom, t_coord) in enumerate(zip(target_atoms, target_coords)):
                    if j not in matched_indices and t_atom == sub_atom:
                        if abs(euclidean_distance(target_coords[i], t_coord) - sub_dist) <= threshold:
                            found_match = True
                            matched_indices.append(j)  # Add the matched atom index to the list
                            break
                if not found_match:
                    all_matches_found = False
                    break

            if all_matches_found:
                matches.append(matched_indices)  # Add the complete matched substructure index list

    return matches

def reorder_xyz(filename, order):
    """Rearrange atoms in an XYZ file according to the specified order."""
    atoms, coords = read_xyz(filename)
    new_atoms = [atoms[i] for i in order] + [atom for i, atom in enumerate(atoms) if i not in order]
    new_coords = np.array([coords[i] for i in order] + [coord for i, coord in enumerate(coords) if i not in order])
    return new_atoms, new_coords

def write_xyz(filename, atoms, coords):
    """Write atoms and coordinates to a new XYZ file."""
    with open(filename, 'w') as file:
        file.write(f"{len(atoms)}\n\n")  # Number of atoms and a blank line
        for atom, coord in zip(atoms, coords):
            file.write(f"{atom} {' '.join(map(str, coord))}\n")

def extract_substructure_main(substructure_file='sub.xyz1', input_dir='.', output_dir='reordered_xyz', threshold=1.1):
    """
    Extract atoms matching the specified substructure and reorder them to the beginning of the XYZ file.
    """
    # Read the substructure
    substructure_atoms, substructure_coords = read_xyz(substructure_file)
    # Sort substructure atoms and coordinates from the center outward
    sorted_indices = sort_substructure_atoms(substructure_atoms, substructure_coords)

    # Get all .xyz files in the input directory
    xyz_files = [f for f in os.listdir(input_dir) if f.endswith('.xyz')]
    if not xyz_files:
        print(f"No .xyz files found in directory {input_dir}.")
        return

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open matches_output.csv file for writing
    matches_output_file = os.path.join(output_dir, 'matches_output.csv')
    with open(matches_output_file, 'w') as file:
        # Write the header row
        file.write('xyz_file,matches\n')

        # Iterate over each .xyz file
        for xyz_file in xyz_files:
            xyz_filepath = os.path.join(input_dir, xyz_file)
            # Read the target structure
            target_atoms, target_coords = read_xyz(xyz_filepath)

            # Find matched substructure atom indices
            matches = find_substructure(target_atoms, target_coords, substructure_atoms, substructure_coords, threshold=threshold)

            if matches:
                # Sort each match using the sort_substructure_atoms function
                sorted_matches = []
                for match in matches:
                    # Rearrange the match based on the sorted indices
                    sorted_match = [match[i] for i in sorted_indices]
                    sorted_matches.append(sorted_match)
                    # Write the sorted match to the file
                    file.write(f"{xyz_file},{' '.join(map(str, sorted_match))}\n")

                print(f"{xyz_file} matched indices: {sorted_matches}")

                # Rearrange atoms in the XYZ file based on the first match
                new_atoms, new_coords = reorder_xyz(xyz_filepath, sorted_matches[0])  # Use only the first match
                # Save to the output directory
                output_xyz_filepath = os.path.join(output_dir, xyz_file)
                write_xyz(output_xyz_filepath, new_atoms, new_coords)
            else:
                print(f"No matching substructure found in {xyz_file}.")

if __name__ == "__main__":
    extract_substructure_main(substructure_file='sub.xyz1', input_dir='.', output_dir='reordered_xyz', threshold=1.1)