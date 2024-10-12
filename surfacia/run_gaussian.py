# surfacia/run_gaussian.py

import subprocess
import os
import glob
import shutil

def run_gaussian(com_dir, esp_descriptor_dir):
    """
    Runs Gaussian calculations for all .com files in the specified directory,
    converts .chk files to .fchk files, and then runs Multiwfn on the .fchk files.

    Args:
        com_dir (str): The directory containing the .com files.
        esp_descriptor_dir (str): The path to the ESP_descriptor.txt1 file.
    """
    # Ensure the directory exists
    if not os.path.isdir(com_dir):
        print(f"Directory {com_dir} does not exist.")
        return

    # Ensure the ESP_descriptor.txt1 file exists
    if not os.path.isfile(esp_descriptor_dir):
        print(f"ESP_descriptor.txt1 file not found at {esp_descriptor_dir}")
        return

    # Change the working directory to the .com files directory
    os.chdir(com_dir)

    # Find all .com files in the directory
    com_files = [f for f in os.listdir(com_dir) if f.endswith('.com')]

    if not com_files:
        print("No .com files found for Gaussian calculations.")
        return

    # Run Gaussian calculations for each .com file
    for com_file in com_files:
        print(f"Running Gaussian calculation for {com_file}...")
        try:
            subprocess.run(['g16', com_file], check=True)
            print(f"{com_file} has been processed.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while processing {com_file}: {e}")
            continue

    # Convert .chk files to .fchk files using formchk
    chk_files = glob.glob('*.chk')

    if not chk_files:
        print("No .chk files found for conversion.")
        return

    for chk_file in chk_files:
        print(f"Converting {chk_file} to formatted checkpoint file...")
        try:
            subprocess.run(['formchk', chk_file], check=True)
            print(f"Successfully converted {chk_file} to .fchk")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while converting {chk_file}: {e}")

    # Run Multiwfn on .fchk files
    print("Current working directory:", os.getcwd())
    print("Multiwfn path:", subprocess.run(['which', 'Multiwfn'], capture_output=True, text=True).stdout.strip())
 
    # Copy ESP_descriptor.txt1 to the current directory
    shutil.copy(esp_descriptor_dir, '.')

    fchk_files = glob.glob('*.fchk')
    for fchk_file in fchk_files:
        print(f"Running Multiwfn on {fchk_file}...")
        try:
            output_file = f"{os.path.splitext(fchk_file)[0]}.txt"
            subprocess.run(f"Multiwfn {fchk_file} < ESP_descriptor.txt1 > {output_file}", shell=True, check=True)
            print(f"Multiwfn output saved to {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running Multiwfn on {fchk_file}: {e}")

    print("job finished")

if __name__ == '__main__':
    com_directory = input("Enter the path to the directory containing .com files: ").strip()
    esp_descriptor_path = input("Enter the path to the ESP_descriptor.txt1 file: ").strip()
    run_gaussian(com_directory, esp_descriptor_path)