# surfacia/xyz2gaussian.py

import os

def xyz_to_com(xyz_file, template_file, output_folder):
    """
    Convert an XYZ file to a Gaussian .com file using a template and store it in the specified folder.

    Args:
        xyz_file (str): Path to the XYZ file.
        template_file (str): Path to the template file.
        output_folder (str): Path to the output directory where .com files will be saved.
    """
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

        com_file = os.path.join(output_folder, f'{molecule_name}.com')
        with open(com_file, 'w') as f_com:
            f_com.writelines(com_content)

        print(f"Successfully created {com_file}")

    except Exception as e:
        print(f"Error converting {xyz_file} to .com: {e}")

def process_xyz_folder(xyz_folder, template_file, output_folder):
    """
    Processes all XYZ files in a folder and generates the corresponding .com files.

    Args:
        xyz_folder (str): Path to the XYZ folder.
        template_file (str): Path to the template file.
        output_folder (str): Output directory where .com files will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(xyz_folder):
        if filename.endswith(".xyz"):
            xyz_file = os.path.join(xyz_folder, filename)
            xyz_to_com(xyz_file, template_file, output_folder)

def xyz2gaussian_main(xyz_folder, template_file, output_folder='.'):
    """
    Main function: Converts XYZ files to Gaussian input files.

    Args:
        xyz_folder (str): Path to the XYZ folder.
        template_file (str): Path to the template file.
        output_folder (str): Output directory where .com files will be saved.
    """
    process_xyz_folder(xyz_folder, template_file, output_folder)

if __name__ == "__main__":
    xyz_folder = input("Enter the path to the XYZ folder: ").strip()
    template_file = input("Enter the path to the template .com file: ").strip()
    output_folder = input("Enter the path to the output folder: ").strip()

    xyz2gaussian_main(xyz_folder, template_file, output_folder)