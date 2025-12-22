"""
XTB Geometry Optimization Module
"""
import os
import subprocess
from pathlib import Path

def run_xtb_opt(param_file: str = None):
    """
    Run xtb optimization on all .xyz files in the current directory.
    Outputs will overwrite the original files in the same directory.
    
    Parameters:
    - param_file: Path to a text file containing xtb parameters. If not provided, default parameters are used.
    """
    print("Starting XTB geometry optimization...")
    
    # Use current directory
    current_dir = Path('.')

    # Default xtb parameters if no param_file is provided
    default_xtb_options = "--opt tight --gfn 2 --molden --alpb water"
    
    # Read xtb parameters from the param_file if it exists, otherwise use default options
    if param_file and os.path.exists(param_file):
        with open(param_file, 'r') as f:
            xtb_options = f.read().strip()
        print(f"Using parameters from file: {param_file}")
    else:
        if param_file:
            print(f"Parameter file {param_file} not found. Using default parameters: {default_xtb_options}")
        else:
            print(f"No parameters file provided. Using default parameters: {default_xtb_options}")
        xtb_options = default_xtb_options

    # Get all .xyz files in the current directory
    xyz_files = list(current_dir.glob("*.xyz"))

    if not xyz_files:
        print("No .xyz files found in current directory!")
        return

    print(f"Found {len(xyz_files)} .xyz files")
    
    success_count = 0
    failed_count = 0

    for xyz_file in xyz_files:
        base_name = xyz_file.stem  # Filename without extension
        output_xyz = current_dir / f"{base_name}.xyz"
        output_molden = current_dir / f"{base_name}.molden.input"
        output_log = current_dir / f"{base_name}.out"
        
        print(f"Processing: {xyz_file}")
        
        # Run xtb command
        xtb_command = f"xtb {xyz_file} {xtb_options}"
        print(f"Running: {xtb_command}")

        try:
            # Redirect the command output to a log file
            with open(output_log, 'w') as log_file:
                result = subprocess.run(
                    xtb_command, 
                    shell=True, 
                    check=True, 
                    stdout=log_file, 
                    stderr=subprocess.STDOUT,
                    timeout=300  # 5 minutes timeout
                )

            # Move and overwrite optimized files
            molden_input = Path("molden.input")
            if molden_input.exists():
                molden_input.replace(output_molden)  # replace() will overwrite if file exists

            # Move and overwrite optimized xyz file
            xtbopt_xyz = Path("xtbopt.xyz")
            if xtbopt_xyz.exists():
                xtbopt_xyz.replace(output_xyz)  # replace() will overwrite if file exists
                print(f"✓ Optimization completed: {output_xyz}")
                success_count += 1
            else:
                print(f"✗ Optimization failed: {xyz_file} (output file not generated)")
                failed_count += 1

            # Clean up any temporary files that xtb might have created
            temp_files = ['charges', 'wbo', 'xtbrestart', 'xtbtopo.mol']
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass

        except subprocess.TimeoutExpired:
            print(f"✗ Optimization timeout: {xyz_file}")
            failed_count += 1
        except subprocess.CalledProcessError as e:
            print(f"✗ Error processing {xyz_file}: {e}")
            failed_count += 1
        except FileNotFoundError:
            print("Error: xtb program not found. Please ensure XTB is properly installed.")
            return
        except Exception as e:
            print(f"✗ Optimization error: {xyz_file}, Error: {e}")
            failed_count += 1

    print(f"\nXTB optimization completed!")
    print(f"Success: {success_count} files")
    print(f"Failed: {failed_count} files")
    print("All tasks completed!")

# Usage example
if __name__ == "__main__":
    # Run without parameter file (uses default parameters)
    run_xtb_opt()