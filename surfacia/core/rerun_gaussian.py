#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rerun failed Gaussian calculations module
"""

import subprocess
import os
import glob
from pathlib import Path

def rerun_failed_gaussian_calculations():
    """
    Identifies and reruns calculations in two cases:
    1. Empty .fchk files with existing .com files
    2. Existing .xyz files without corresponding .fchk files
    """
    current_dir = Path('.')
    failed_jobs = []
    
    print("🔍 Scanning for failed Gaussian calculations...")
    
    # Case 1: Check for empty .fchk files
    empty_fchk_count = 0
    for fchk_file in current_dir.glob('*.fchk'):
        if fchk_file.stat().st_size == 0:  # Check if file is empty
            com_file = current_dir / f"{fchk_file.stem}.com"
            if com_file.exists():
                failed_jobs.append(com_file)
                empty_fchk_count += 1
                # Remove empty .fchk file and corresponding .chk file
                fchk_file.unlink()  # Delete empty .fchk file
                chk_file = current_dir / f"{fchk_file.stem}.chk"
                if chk_file.exists():
                    chk_file.unlink()  # Delete corresponding .chk file
                print(f"   Found empty .fchk: {fchk_file.name} (removed)")

    # Case 2: Check for xyz files without fchk files
    missing_fchk_count = 0
    for xyz_file in current_dir.glob('*.xyz'):
        fchk_file = current_dir / f"{xyz_file.stem}.fchk"
        com_file = current_dir / f"{xyz_file.stem}.com"
        if not fchk_file.exists() and com_file.exists():
            if com_file not in failed_jobs:  # Avoid duplicates
                failed_jobs.append(com_file)
                missing_fchk_count += 1
                print(f"   Missing .fchk for: {xyz_file.name}")

    if not failed_jobs:
        print("✅ No failed calculations or missing fchk files found.")
        return True

    print(f"\n📊 Summary:")
    print(f"   Empty .fchk files: {empty_fchk_count}")
    print(f"   Missing .fchk files: {missing_fchk_count}")
    print(f"   Total jobs to rerun: {len(failed_jobs)}")
    
    print(f"\n🚀 Starting calculations for {len(failed_jobs)} jobs...")
    
    # Run calculations for all identified jobs
    success_count = 0
    for i, com_file in enumerate(sorted(failed_jobs), 1):
        print(f"\n[{i}/{len(failed_jobs)}] Running Gaussian calculation for {com_file.name}...")
        try:
            subprocess.run(['g16', str(com_file)], check=True)
            print(f"   ✅ {com_file.name} completed successfully")
            
            # Convert new .chk file to .fchk
            chk_file = current_dir / f"{com_file.stem}.chk"
            if chk_file.exists():
                print(f"   🔄 Converting {chk_file.name} to formatted checkpoint file...")
                subprocess.run(['formchk', str(chk_file)], check=True)
                print(f"   ✅ Successfully converted to .fchk")
                success_count += 1
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error processing {com_file.name}: {e}")
            continue
        except FileNotFoundError:
            print(f"   ❌ Error: 'g16' command not found. Please ensure Gaussian is installed and in PATH.")
            return False

    print(f"\n🎉 Rerun completed: {success_count}/{len(failed_jobs)} jobs successful")
    return success_count == len(failed_jobs)

def main():
    """Main function for standalone execution"""
    choice = input("Enter '1' for normal run, '2' for rerunning failed calculations: ").strip()
    
    if choice == '1':
        print('Skipping rerun check.')
    elif choice == '2':
        rerun_failed_gaussian_calculations()
    else:
        print("Invalid choice. Please enter either '1' or '2'.")

if __name__ == '__main__':
    main()