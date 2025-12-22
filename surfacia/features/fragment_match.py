"""
分子片段匹配功能
"""
import numpy as np
import logging

def setup_logging(log_file_path):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler()
        ]
    )

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