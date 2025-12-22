"""
Gaussian计算相关模块
"""
import os
from pathlib import Path
import subprocess
import glob

# 用户可以直接修改这些默认设置
GAUSSIAN_KEYWORD_LINE = "# PBE1PBE/6-311g* scrf(SMD,solvent=Water) em=GD3"
DEFAULT_CHARGE = 0
DEFAULT_MULTIPLICITY = 1
DEFAULT_NPROC = 32
DEFAULT_MEMORY = "30GB"

def xyz_to_com(xyz_file):
    """
    Convert an XYZ file to a Gaussian .com file using built-in settings.

    Args:
        xyz_file (str): Path to the XYZ file.
    """
    try:
        # 读取XYZ文件
        with open(xyz_file, 'r') as f_xyz:
            xyz_lines = f_xyz.readlines()

        molecule_name = Path(xyz_file).stem
        
        # 构建.com文件内容
        com_content = f"""%nprocshared={DEFAULT_NPROC}
%mem={DEFAULT_MEMORY}
%chk={molecule_name}.chk
{GAUSSIAN_KEYWORD_LINE}

{molecule_name}

{DEFAULT_CHARGE:3d} {DEFAULT_MULTIPLICITY:2d}
"""
        
        # 添加坐标部分 (跳过XYZ文件的前两行)
        com_content += ''.join(xyz_lines[2:]) + '\n'

        # 保存.com文件
        com_file = Path(f'{molecule_name}.com')
        with open(com_file, 'w') as f_com:
            f_com.write(com_content)

        print(f"✓ Successfully created {com_file}")

    except Exception as e:
        print(f"❌ Error converting {xyz_file} to .com: {e}")

def process_xyz_files():
    """
    Processes all XYZ files in current directory and generates the corresponding .com files.
    """
    current_dir = Path('.')
    
    # 处理当前目录下的所有.xyz文件
    xyz_files = sorted(list(current_dir.glob("*.xyz")))
    
    if not xyz_files:
        print("❌ No .xyz files found in current directory")
        return

    print(f"Found {len(xyz_files)} XYZ files to process")
    print(f"Using settings:")
    print(f"  Gaussian keywords: {GAUSSIAN_KEYWORD_LINE}")
    print(f"  Charge/Multiplicity: {DEFAULT_CHARGE} {DEFAULT_MULTIPLICITY}")
    print(f"  Resources: {DEFAULT_NPROC} cores, {DEFAULT_MEMORY}")
    print("-" * 50)
    
    for xyz_file in xyz_files:
        xyz_to_com(xyz_file)

def xyz2gaussian_main():
    """
    Main function: Converts XYZ files to Gaussian input files in current directory.
    """
    print("XYZ to Gaussian .com converter")
    print("=" * 50)
    
    process_xyz_files()
    print(f"\n✓ All conversions completed!")

def run_gaussian():
    """
    Runs Gaussian calculations for all .com files in the current directory,
    converts .chk files to .fchk files, and then runs Multiwfn on the .fchk files.
    """
    current_dir = Path('.')

    # Find all .com files in the current directory and sort them by name
    com_files = sorted(list(current_dir.glob('*.com')))

    if not com_files:
        print("No .com files found for Gaussian calculations.")
        return

    # Run Gaussian calculations for each .com file in sorted order
    for com_file in com_files:
        print(f"Running Gaussian calculation for {com_file.name}...")
        try:
            subprocess.run(['g16', str(com_file)], check=True)
            print(f"{com_file.name} has been processed.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while processing {com_file.name}: {e}")
            continue

    # Convert .chk files to .fchk files using formchk, also in sorted order
    chk_files = sorted(list(current_dir.glob('*.chk')))

    if not chk_files:
        print("No .chk files found for conversion.")
        return

    for chk_file in chk_files:
        print(f"Converting {chk_file.name} to formatted checkpoint file...")
        try:
            subprocess.run(['formchk', str(chk_file)], check=True)
            print(f"Successfully converted {chk_file.name} to .fchk")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while converting {chk_file.name}: {e}")

    print("Job finished")

def rerun_failed_calculations():
    """
    Identifies and reruns calculations in two cases:
    1. Empty .fchk files with existing .com files
    2. Existing .xyz files without corresponding .fchk files
    """
    current_dir = Path('.')
    failed_jobs = []
    
    # Case 1: Check for empty .fchk files
    for fchk_file in current_dir.glob('*.fchk'):
        if fchk_file.stat().st_size == 0:  # Check if file is empty
            com_file = current_dir / f"{fchk_file.stem}.com"
            if com_file.exists():
                failed_jobs.append(com_file)
                # Remove empty .fchk file and corresponding .chk file
                fchk_file.unlink()  # Delete empty .fchk file
                chk_file = current_dir / f"{fchk_file.stem}.chk"
                if chk_file.exists():
                    chk_file.unlink()  # Delete corresponding .chk file

    # Case 2: Check for xyz files without fchk files
    for xyz_file in current_dir.glob('*.xyz'):
        fchk_file = current_dir / f"{xyz_file.stem}.fchk"
        com_file = current_dir / f"{xyz_file.stem}.com"
        if not fchk_file.exists() and com_file.exists():
            if com_file not in failed_jobs:  # Avoid duplicates
                failed_jobs.append(com_file)

    if not failed_jobs:
        print("No failed calculations or missing fchk files found.")
        return

    print(f"Found {len(failed_jobs)} jobs to run. Starting calculations...")
    
    # Run calculations for all identified jobs
    for com_file in sorted(failed_jobs):
        print(f"Running Gaussian calculation for {com_file.name}...")
        try:
            subprocess.run(['g16', str(com_file)], check=True)
            print(f"{com_file.name} has been processed.")
            
            # Convert new .chk file to .fchk
            chk_file = current_dir / f"{com_file.stem}.chk"
            if chk_file.exists():
                print(f"Converting {chk_file.name} to formatted checkpoint file...")
                subprocess.run(['formchk', str(chk_file)], check=True)
                print(f"Successfully converted {chk_file.name} to .fchk")
            
        except subprocess.CalledProcessError as e:
            print(f"Error processing {com_file.name}: {e}")
            continue

    print("All jobs finished")