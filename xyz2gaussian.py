import pandas as pd
from openbabel import pybel as pb
import os
import datetime
import shutil

def read_smiles_csv(file_path):
    """Reads a CSV file containing SMILES strings."""
    try:
        data = pd.read_csv(file_path)
        print("CSV file read successfully!")
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def smiles_to_xyz(smiles_string):
    """Converts a SMILES string to XYZ format."""
    mol = pb.readstring('smiles', smiles_string)
    mol.make3D()
    return mol.write(format='xyz')

def xyz_to_com(xyz_file, template_file):
    """Converts an XYZ file to a Gaussian .com file using a template."""
    try:
        with open(xyz_file, 'r') as f_xyz, open(template_file, 'r') as f_template:
            xyz_lines = f_xyz.readlines()
            template_lines = f_template.readlines()

        molecule_name = os.path.splitext(os.path.basename(xyz_file))[0]

        chk_index = next((i for i, line in enumerate(template_lines) if '%chk=' in line), None)
        if chk_index is not None:
            template_lines[chk_index] = f'%chk={molecule_name}.chk\n'

        name_index = template_lines.index('\n') + 1
        template_lines.insert(name_index, f'{molecule_name}\n\n')

        charge_mult_index = name_index + 1
        template_lines.insert(charge_mult_index, '  0  1\n')

        com_content = template_lines[:charge_mult_index + 1] + xyz_lines[2:] + ['\n']

        com_file = f'{molecule_name}.com'
        with open(com_file, 'w') as f_com:
            f_com.writelines(com_content)

        print(f"Successfully created {com_file}")

    except Exception as e:
        print(f"Error converting {xyz_file} to .com: {e}")

def generate_bash_script(com_files, script_path):
    """Generates a bash script for running Gaussian calculations."""
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n\n")
        
        # SLURM parameters
        f.write("#SBATCH -o job.%j.out\n")
        f.write("#SBATCH -p small_s\n")
        f.write("#SBATCH -J g16\n")
        f.write("#SBATCH --nodes=1\n")
        f.write("#SBATCH --cpus-per-task=64\n\n")
        f.write("module load gauss/16\n")
        
        for com_file in com_files:
            file_base = os.path.splitext(com_file)[0]
            f.write(f"INPUT_FILE={com_file}\n")
            f.write(f"CHK_FILE={file_base}.chk\n")
            f.write("#----> Job begins <----\n")
            f.write(f"g16  ${{INPUT_FILE%.com}}.com\n\n")
            f.write(f"echo  ${{INPUT_FILE%.com}} job ends at:'  ' `date`\n\n")

def create_and_copy_files(com_files, script_file, output_folder):
    """Creates a timestamped subfolder and copies files."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    subfolder = os.path.join(output_folder, f"gaussian_jobs_{timestamp}")
    os.makedirs(subfolder, exist_ok=True)

    for com_file in com_files:
        shutil.copy(com_file, os.path.join(subfolder, com_file))

    shutil.copy(script_file, os.path.join(subfolder, script_file))
    print(f"Copied .com files and '{script_file}' to '{subfolder}'")

def process_xyz_folder(xyz_folder, template_file, script_file, output_folder):
    """Processes all XYZ files in a folder."""
    for filename in os.listdir(xyz_folder):
        if filename.endswith(".xyz"):
            xyz_file = os.path.join(xyz_folder, filename)
            xyz_to_com(xyz_file, template_file)

    com_files = [f for f in os.listdir(xyz_folder) if f.endswith(".com")]
    generate_bash_script(com_files, script_file)
    create_and_copy_files(com_files, script_file, output_folder)

if __name__ == "__main__":
    xyz_folder = input("Enter the path to your XYZ files folder: ")
    template_file = input("Enter the path to your template.com file: ")
    script_file = "run_gaussian.sh"
    output_folder = "."  # Current directory

    process_xyz_folder(xyz_folder, template_file, script_file, output_folder)