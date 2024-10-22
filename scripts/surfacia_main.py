# scripts/surfacia_main.py

import argparse
import configparser
import os
import sys
import subprocess

# Import main functions from the Surfacia package
from surfacia import (
    smi2xyz_main,
    reorder_atoms,
    extract_substructure_main,
    xyz2gaussian_main,
    run_gaussian,
    process_txt_files,
    xgb_stepwise_regression,
    fchk2matches_main,
    run_multiwfn_on_fchk_files,  # Add this line
    read_first_matches_csv,  # Add this line as well
    generate_feature_matrix,
)
def main():
    parser = argparse.ArgumentParser(description='SURF Atomic Chemical Interaction Analyzer - Surfacia')
    parser.add_argument('--config', type=str, default='config/setting.ini', help='Path to the configuration file')

    # Create subparsers
    subparsers = parser.add_subparsers(dest='task', help='Specify the task to execute')

    # smi2xyz Task
    parser_smi2xyz = subparsers.add_parser('smi2xyz', help='Convert SMILES to XYZ')
    parser_smi2xyz.add_argument('--smiles_csv', type=str, required=True, help='Path to the CSV file containing SMILES strings')

    # reorder Task
    parser_reorder = subparsers.add_parser('reorder', help='Reorder atoms')
    parser_reorder.add_argument('--element', type=str, default='P', help='Element symbol to extract (default is P)')
    parser_reorder.add_argument('--input_dir', type=str, default='.', help='Input directory path')
    parser_reorder.add_argument('--output_dir', type=str, default='reorder', help='Output directory path')

    # extract_substructure Task
    parser_extract = subparsers.add_parser('extract_substructure', help='Extract substructure')
    parser_extract.add_argument('--substructure_file', type=str, default='sub.xyz1', help='Path to the substructure file')
    parser_extract.add_argument('--input_dir', type=str, default='.', help='Input directory path')
    parser_extract.add_argument('--output_dir', type=str, default='reordered_xyz', help='Output directory path')
    parser_extract.add_argument('--threshold', type=float, default=1.1, help='Threshold for substructure matching (default is 1.1)')

    # xyz2gaussian Task
    parser_xyz2gaussian = subparsers.add_parser('xyz2gaussian', help='Convert XYZ to Gaussian input')
    parser_xyz2gaussian.add_argument('--xyz_folder', type=str, default='.', help='Directory containing XYZ files')
    parser_xyz2gaussian.add_argument('--template_file', type=str, required=True, help='Path to the Gaussian template file')
    parser_xyz2gaussian.add_argument('--output_dir', type=str, default='.', help='Output directory path')

    # run_gaussian Task
    parser_run_gaussian = subparsers.add_parser('run_gaussian', help='Run Gaussian calculations')
    parser_run_gaussian.add_argument('--com_dir', type=str, required=True, help='Directory containing .com files')

    # readmultiwfn Task
    parser_readmultiwfn = subparsers.add_parser('readmultiwfn', help='Read Multiwfn outputs')
    parser_readmultiwfn.add_argument('--input_dir', type=str, default='.', help='Input directory path')
    parser_readmultiwfn.add_argument('--output_dir', type=str, default='output', help='Output directory path')
    parser_readmultiwfn.add_argument('--smiles_target_csv', type=str, required=True, help='Path to the CSV file containing SMILES and target values')
    parser_readmultiwfn.add_argument('--first_matches_csv', type=str, required=True, help='Path to the CSV file containing first matches')
    parser_readmultiwfn.add_argument('--descriptor_option', type=int, default=1, help='Descriptor option (1, 2, or 3)')

    # machinelearning Task
    parser_ml = subparsers.add_parser('machinelearning', help='Machine Learning tasks')
    parser_ml.add_argument('--full_csv', type=str, required=True, help='Path to the full CSV file containing title, smiles, features, and target')
    parser_ml.add_argument('--output_dir', type=str, default='MachineLearning', help='Output directory path (default: MachineLearning)')
    parser_ml.add_argument('--nan_handling', type=str, default='drop_rows', choices=['drop_rows', 'drop_columns'], help='How to handle NaN values')
    parser_ml.add_argument('--epoch', type=int, default=32, help='Number of epochs')
    parser_ml.add_argument('--core_num', type=int, default=32, help='Number of CPU cores to use')
    parser_ml.add_argument('--train_test_split_ratio', type=float, default=0.85, help='Train-test split ratio')
    parser_ml.add_argument('--step_feat_num', type=int, default=2, help='Number of features to select')
    parser_ml.add_argument('--know_ini_feat', action='store_true', help='Whether the initial features are known')
    parser_ml.add_argument('--ini_feat', type=int, nargs='+', help='List of initial feature indices')
    parser_ml.add_argument('--test_indices', type=int, nargs='+', help='List of test set indices')

    # fchk2matches Task
    parser_fchk2matches = subparsers.add_parser('fchk2matches', help='FCHK to Matches')
    parser_fchk2matches.add_argument('--input_path', type=str, default='.', help='Path to the directory containing .fchk files. Defaults to current directory.')
    parser_fchk2matches.add_argument('--xyz1_path', type=str, required=True, help='Path to the substructure .xyz1 file.')
    parser_fchk2matches.add_argument('--threshold', type=float, default=1.01, help='Threshold for substructure matching (default is 1.01).')

    args = parser.parse_args()

    # Read configuration file
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

    elif args.task == 'extract_substructure':
        substructure_file = args.substructure_file or config.get('DEFAULT', 'substructure_file', fallback='sub.xyz1')
        input_dir = args.input_dir or config.get('DEFAULT', 'input_dir', fallback='.')
        output_dir = args.output_dir or config.get('DEFAULT', 'output_dir', fallback='reordered_xyz')
        threshold = args.threshold or config.getfloat('DEFAULT', 'threshold', fallback=1.1)
        extract_substructure_main(substructure_file=substructure_file, input_dir=input_dir, output_dir=output_dir, threshold=threshold)

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
        if not com_dir:
            print("Directory containing .com files not specified.")
            sys.exit(1)
        run_gaussian(com_dir)

    elif args.task == 'readmultiwfn':
        # Retrieve arguments or use defaults from the config
        input_dir = args.input_dir or config.get('DEFAULT', 'input_dir', fallback='.')
        output_dir = args.output_dir or config.get('DEFAULT', 'output_dir', fallback='output')
        smiles_target_csv = args.smiles_target_csv or config.get('DEFAULT', 'smiles_target_csv', fallback=None)
        first_matches_csv = args.first_matches_csv or config.get('DEFAULT', 'first_matches_csv', fallback=None)
    
        # Convert descriptor_option to integer
        try:
            descriptor_option = int(args.descriptor_option) if args.descriptor_option else config.getint('DEFAULT', 'descriptor_option', fallback=1)
        except ValueError:
            print("Descriptor option must be an integer (1, 2, or 3).")
            sys.exit(1)
    
        # Validate required CSV paths
        if not smiles_target_csv:
            print("SMILES and target CSV file not specified.")
            sys.exit(1)
        if not first_matches_csv:
            print("First matches CSV file not specified.")
            sys.exit(1)
    
        # Read fragment indices from the first_matches CSV
        first_matches = read_first_matches_csv(first_matches_csv)
    
        # Run Multiwfn on FCHK files
        run_multiwfn_on_fchk_files(input_path=input_dir, first_matches=first_matches)
    
        # Process the extracted TXT files to generate descriptors
        process_txt_files(input_directory=input_dir,
                          output_directory=output_dir,
                          smiles_target_csv_path=smiles_target_csv,
                          first_matches_csv_path=first_matches_csv,
                          descriptor_option=descriptor_option)

    elif args.task == 'generate_feature_matrix':
        merged_csv = args.merged_csv or config.get('DEFAULT', 'merged_csv', fallback=None)
        output_dir = args.output_dir or config.get('DEFAULT', 'output_dir', fallback=None)
        nan_handling = args.nan_handling or config.get('DEFAULT', 'nan_handling', fallback='drop_rows')

        if not merged_csv or not output_dir:
            print("Merged CSV file or output directory not specified.")
            sys.exit(1)

        result = generate_feature_matrix(merged_csv, output_dir, nan_handling)
        print("Generated files:")
        for key, value in result.items():
            print(f"  {key}: {value}")
            
    elif args.task == 'machinelearning':
        full_csv = args.full_csv or config.get('DEFAULT', 'full_csv', fallback=None)
        output_dir = args.output_dir or config.get('DEFAULT', 'output_dir', fallback='MachineLearning')
        nan_handling = args.nan_handling or config.get('DEFAULT', 'nan_handling', fallback='drop_rows')
        epoch = args.epoch
        core_num = args.core_num
        train_test_split_ratio = args.train_test_split_ratio
        step_feat_num = args.step_feat_num
        know_ini_feat = args.know_ini_feat
        ini_feat = args.ini_feat if args.ini_feat else []
        test_indices = args.test_indices if args.test_indices else []

        if not full_csv:
            print("Full CSV file (--full_csv) not specified.")
            sys.exit(1)

        if not os.path.isfile(full_csv):
            print(f"Full CSV file not found: {full_csv}")
            sys.exit(1)


        print("generate_feature_matrix ing")
        generated_files = generate_feature_matrix(
            merged_output_filename=full_csv,
            output_dir=output_dir,
            nan_handling=nan_handling
        )


        input_x = generated_files['features']
        input_y = generated_files['values']
        input_title = generated_files['titles']
        ml_input_dir = generated_files['ml_dir']


        for key in ['features', 'values', 'titles']:
            file_path = generated_files[key]
            if not os.path.isfile(file_path):
                print(f"not found{file_path}")
                sys.exit(1)


        print("begin xgb_stepwise_regression")
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

    elif args.task == 'fchk2matches':
        input_path = args.input_path or config.get('DEFAULT', 'input_path', fallback='.')
        xyz1_path = args.xyz1_path or config.get('DEFAULT', 'xyz1_path', fallback=None)
        threshold = args.threshold or config.getfloat('DEFAULT', 'threshold', fallback=1.01)

        if not xyz1_path:
            print("Substructure file path (xyz1_path) not specified.")
            sys.exit(1)

        fchk2matches_main(input_path=input_path, xyz1_path=xyz1_path, threshold=threshold)

    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
