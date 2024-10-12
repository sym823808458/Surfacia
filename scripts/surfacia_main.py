# scripts/surfacia_main.py

import argparse
import configparser
import os
import sys
import subprocess

# Import the main functions from the surfacia package
from surfacia import (
    smi2xyz_main,
    reorder_atoms,
    xyz2gaussian_main,
    run_gaussian,
    process_txt_files,
    xgb_stepwise_regression,
)


def main():
    parser = argparse.ArgumentParser(description='SURF Atomic Chemical Interaction Analyzer - Surfacia')

    parser.add_argument('task', choices=[
        'smi2xyz',
        'reorder',
        'xyz2gaussian',
        'run_gaussian',
        'readmultiwfn',
        'split',
        'machinelearning'
    ], help='Specify the task to execute')

    parser.add_argument('--config', type=str, default='config/setting.ini', help='Path to the configuration file')

    # Common arguments
    parser.add_argument('--smiles_csv', type=str, help='Path to the CSV file containing SMILES strings')
    parser.add_argument('--input_dir', type=str, default='.', help='Input directory path')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory path')
    parser.add_argument('--element', type=str, default='P', help='Element symbol to extract (default is P)')

    # xyz2gaussian arguments
    parser.add_argument('--xyz_folder', type=str, default='.', help='Directory containing XYZ files')
    parser.add_argument('--template_file', type=str, help='Path to the Gaussian template file')

    # run_gaussian arguments
    parser.add_argument('--com_dir', type=str, help='Directory containing .com files')  
    parser.add_argument("--esp_descriptor_dir", type=str, default="/home/yumingsu/Python/Project_Surfacia/Surfacia/config/ESP_descriptor.txt1",help="Path to the ESP_descriptor.txt1 file")
    # process_txt_files arguments
    parser.add_argument('--smiles_target_csv', type=str, help='Path to the CSV file containing SMILES and target values')

    # xgb_stepwise_regression arguments
    parser.add_argument('--input_x', type=str, help='Path to the feature matrix file')
    parser.add_argument('--input_y', type=str, help='Path to the label (target values) file')
    parser.add_argument('--input_title', type=str, help='Path to the feature names file')
    parser.add_argument('--ml_input_dir', type=str, help='Directory containing input files for machine learning')
#    parser.add_argument('--version', type=str, default='V2.2sym', help='Version number')
    parser.add_argument('--epoch', type=int, default=32, help='Number of epochs')
    parser.add_argument('--core_num', type=int, default=32, help='Number of CPU cores to use')
    parser.add_argument('--train_test_split_ratio', type=float, default=0.85, help='Train-test split ratio')
    parser.add_argument('--step_feat_num', type=int, default=2, help='Number of features to select')
    parser.add_argument('--know_ini_feat', action='store_true', help='Whether the initial features are known')
    parser.add_argument('--ini_feat', type=int, nargs='+', help='List of initial feature indices')
    parser.add_argument('--test_indices', type=int, nargs='+', help='List of test set indices')

    args = parser.parse_args()

    # Read the configuration file
    config = configparser.ConfigParser()
    if os.path.exists(args.config):
        config.read(args.config)
    else:
        print(f"Configuration file {args.config} not found.")
        sys.exit(1)

    # Execute the specified task
    if args.task == 'smi2xyz':
        smiles_csv = args.smiles_csv or config.get('DEFAULT', 'smiles_prop_path', fallback=None)
        if not smiles_csv:
            print("SMILES CSV file not specified.")
            sys.exit(1)
        smi2xyz_main(smiles_csv)

    elif args.task == 'reorder':
        element = args.element or config.get('DEFAULT', 'element', fallback='P')
        input_dir = args.input_dir or config.get('DEFAULT', 'input_dir', fallback='.')
        output_dir = args.output_dir or config.get('DEFAULT', 'output_dir', fallback='reorder')
        reorder_atoms(element=element, input_dir=input_dir, output_dir=output_dir)

    elif args.task == 'xyz2gaussian':
        xyz_folder = args.xyz_folder or config.get('DEFAULT', 'xyz_folder', fallback='.')
        template_file = args.template_file or config.get('DEFAULT', 'gaussian_template', fallback=None)
        output_dir = args.output_dir or config.get('DEFAULT', 'output_dir', fallback='.')
        if not template_file:
            print("Gaussian template file not specified.")
            sys.exit(1)
        xyz2gaussian_main(xyz_folder=xyz_folder, template_file=template_file, output_folder=output_dir)

    elif args.task == 'run_gaussian':
        com_dir = args.com_dir or config.get('DEFAULT', 'com_dir', fallback=None)
        esp_descriptor_dir=args.esp_descriptor_dir or config.get('DEFAULT', 'esp_descriptor_dir', fallback=None)
        if not com_dir:
            print("Directory containing .com files not specified.")
            sys.exit(1)
        run_gaussian(com_dir, esp_descriptor_dir)

    elif args.task == 'readmultiwfn':
        input_dir = args.input_dir or config.get('DEFAULT', 'input_dir', fallback='.')
        output_dir = args.output_dir or config.get('DEFAULT', 'output_dir', fallback='output')
        smiles_target_csv = args.smiles_target_csv or config.get('DEFAULT', 'smiles_target_csv', fallback=None)
        if not smiles_target_csv:
            print("SMILES and target CSV file not specified.")
            sys.exit(1)
        process_txt_files(input_directory=input_dir, output_directory=output_dir, smiles_target_csv_path=smiles_target_csv)


    elif args.task == 'machinelearning':
        # First, attempt to get the individual input file paths
        input_x = args.input_x or config.get('DEFAULT', 'input_x', fallback=None)
        input_y = args.input_y or config.get('DEFAULT', 'input_y', fallback=None)
        input_title = args.input_title or config.get('DEFAULT', 'input_title', fallback=None)
        
        # If no individual input files are specified, attempt to get them from the specified directory
        ml_input_dir = args.ml_input_dir or config.get('DEFAULT', 'ml_input_dir', fallback=None)
        if ml_input_dir and (not input_x or not input_y or not input_title):
            
            # Function to match files by keyword
            def find_file_by_keyword(directory, keyword):
                for filename in os.listdir(directory):
                    if keyword.lower() in filename.lower():  # Case-insensitive matching
                        return os.path.join(directory, filename)
                return None
            
            # Use the function to find files based on keywords
            input_x = input_x or find_file_by_keyword(ml_input_dir, 'Features')
            input_y = input_y or find_file_by_keyword(ml_input_dir, 'Values')
            input_title = input_title or find_file_by_keyword(ml_input_dir, 'Title')
        
        # Check if input files exist
        if not input_x or not input_y or not input_title:
            print("Input data files (input_x, input_y, input_title) not specified.")
            sys.exit(1)
        else:
            # Verify the existence of files
            if not os.path.isfile(input_x):
                print(f"Input X file not found: {input_x}")
                sys.exit(1)
            if not os.path.isfile(input_y):
                print(f"Input Y file not found: {input_y}")
                sys.exit(1)
            if not os.path.isfile(input_title):
                print(f"Input Title file not found: {input_title}")
                sys.exit(1)
        
        # Retrieve other parameters (version-related code removed)
        epoch = args.epoch
        core_num = args.core_num
        train_test_split_ratio = args.train_test_split_ratio
        step_feat_num = args.step_feat_num
        know_ini_feat = args.know_ini_feat
        ini_feat = args.ini_feat if args.ini_feat else []
        test_indices = args.test_indices if args.test_indices else []
        
        # Call the xgb_stepwise_regression function
        xgb_stepwise_regression(
            input_x=input_x,
            input_y=input_y,
            input_title=input_title,
            epoch=epoch,
            core_num=core_num,
            train_test_split_ratio=train_test_split_ratio,
            step_feat_num=step_feat_num,
            know_ini_feat=know_ini_feat,
            ini_feat=ini_feat,
            test_indices=test_indices,
            output_dir=ml_input_dir
        )
    else:
        print("Unknown task.")
        sys.exit(1)

if __name__ == '__main__':
    main()