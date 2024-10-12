import os
import numpy as np

def read_xyz(filename):
    """
    Read an XYZ file.

    Args:
        filename (str): Path to the XYZ file.
    
    Returns:
        tuple: Tuple containing the number of atoms, comments, list of atoms, and coordinates.
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
    num_atoms = int(lines[0].strip())
    comment = lines[1].strip()
    atoms = []
    coords = []
    for line in lines[2:]:  # Skip the first two lines
        parts = line.strip().split()
        if parts:  # Check if the line is not empty
            atoms.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
    return num_atoms, comment, atoms, np.array(coords)

def extract_and_count_element(element, atoms, coords):
    """
    Extract and count the specified element.

    Args:
        element (str): Element symbol to extract.
        atoms (list): List of atoms.
        coords (ndarray): Coordinates array.
    
    Returns:
        tuple: Tuple containing indices, coordinates, and count of the element.
    """
    element_indices = [i for i, atom in enumerate(atoms) if atom == element]
    element_coords = coords[element_indices]
    num_elements = len(element_indices)
    return element_indices, element_coords, num_elements

def reorder_atoms(element='P', input_dir='.', output_dir='reorder', count_file='element_count.csv'):
    """
    Reorder atoms to move specified element to the front.

    Args:
        element (str): Element symbol, default is 'P'.
        input_dir (str): Directory containing input XYZ files.
        output_dir (str): Output directory.
        count_file (str): Filename for element count results.
    """
    xyz_files = [f for f in os.listdir(input_dir) if f.endswith('.xyz')]
    os.makedirs(output_dir, exist_ok=True)
    with open(count_file, 'w') as file:
        file.write('xyz_file,number_of_atoms\n')
        
        for xyz_file in xyz_files:
            print(f"Processing file: {xyz_file}")
            num_atoms, comment, atoms, coords = read_xyz(os.path.join(input_dir, xyz_file))
            elem_indices, elem_coords, num_elements = extract_and_count_element(element, atoms, coords)
            new_order_indices = elem_indices + [i for i in range(num_atoms) if i not in elem_indices]
            new_order_atoms = [atoms[i] for i in new_order_indices]
            new_order_coords = np.array([coords[i] for i in new_order_indices])
            file.write(f"{xyz_file},{num_elements}\n")
            with open(os.path.join(output_dir, xyz_file), 'w') as reorder_file:
                reorder_file.write(f"{num_atoms}\n")
                reorder_file.write(f"{comment}\n")
                for atom, coord in zip(new_order_atoms, new_order_coords):
                    reorder_file.write(f"{atom} {' '.join(map(str, coord))}\n")
            print(f"Processed {xyz_file}: {num_elements} {element} atoms.")

if __name__ == "__main__":
    element = input("Please enter the element symbol you want to extract (e.g., 'P'): ").strip()
    reorder_atoms(element=element)