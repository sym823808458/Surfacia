import os
import numpy as np

def read_xyz(filename):
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

def extract_and_count_phosphorus(element,atoms, coords):
    p_indices = [i for i, atom in enumerate(atoms) if atom == element]
    p_coords = coords[p_indices]
    num_p_atoms = len(p_indices)
    return p_indices, p_coords, num_p_atoms


element='P'
xyz_files = [f for f in os.listdir('.') if f.endswith('.xyz')]
os.makedirs('reorder', exist_ok=True)
with open('element_count.csv', 'w') as file:

    file.write('xyz_file,number_of_atoms\n')
    
    for xyz_file in xyz_files:

        print(xyz_file)
        num_atoms, comment, atoms, coords = read_xyz(xyz_file)
        print(num_atoms, comment, atoms, coords)

        p_indices, p_coords, num_p_atoms = extract_and_count_phosphorus(element,atoms, coords)
        
        print(p_indices, p_coords, num_p_atoms)
        new_order_indices = p_indices + [i for i in range(num_atoms) if i not in p_indices]
        print(new_order_indices)
        new_order_atoms = [atoms[i] for i in new_order_indices]
        new_order_coords = np.array([coords[i] for i in new_order_indices])
        

        file.write(f"{xyz_file},{num_p_atoms}\n")
        

        with open(f'reorder/{xyz_file}', 'w') as reorder_file:
            reorder_file.write(f"{num_atoms}\n")
            reorder_file.write(f"{comment}\n")
            for atom, coord in zip(new_order_atoms, new_order_coords):
                reorder_file.write(f"{atom} {' '.join(map(str, coord))}\n")

        print(f"Processed {xyz_file}: {num_p_atoms} {element} atoms.")