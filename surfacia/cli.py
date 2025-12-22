#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Surfacia command line interface - Complete version with all tools
"""

import argparse
import sys
import os
import glob
from pathlib import Path

from .core.smi2xyz import smi2xyz_main
from .core.xtb_opt import run_xtb_opt
from .core.gaussian import xyz2gaussian_main, run_gaussian
from .core.multiwfn import run_multiwfn_on_fchk_files, process_txt_files
from .core.rerun_gaussian import rerun_failed_gaussian_calculations
from .features.atom_properties import run_atom_prop_extraction
from .visualization.interactive_shap_viz import InteractiveSHAPAnalyzer, MultiwfnPDBGenerator, interactive_shap_viz_main
from .visualization.mol_drawer import draw_molecules_from_csv, draw_single_molecule
from .utils.mol_properties import analyze_molecules_from_csv, analyze_single_molecule
from .core.workflow import workflow_main
from .ml.chem_ml_analyzer_v2 import ChemMLWorkflow

# 标准入口函数现在总是可用
INTERACTIVE_SHAP_MAIN_AVAILABLE = True

def find_latest_surfacia_folder():
    """Find the latest Surfacia output folder"""
    folders = glob.glob("Surfacia_3.0_*")
    if not folders:
        return None
    # Sort by modification time and return the latest
    return max(folders, key=os.path.getmtime)

def check_calculation_status():
    """
    检查当前目录中计算的完成状态
    返回: (start_step, step_name, status_info)
    """
    current_dir = Path('.')
    
    # 检查各种文件的存在情况
    xyz_files = list(current_dir.glob('*.xyz'))
    com_files = list(current_dir.glob('*.com'))
    fchk_files = list(current_dir.glob('*.fchk'))
    
    status_info = {
        'xyz_count': len(xyz_files),
        'com_count': len(com_files),
        'fchk_count': len(fchk_files),
        'empty_fchk': 0,
        'missing_fchk': 0,
        'complete_fchk': 0
    }
    
    # 检查 .fchk 文件的完整性
    empty_fchk_files = []
    missing_fchk_files = []
    
    for fchk_file in fchk_files:
        if fchk_file.stat().st_size == 0:
            status_info['empty_fchk'] += 1
            empty_fchk_files.append(fchk_file.name)
        else:
            status_info['complete_fchk'] += 1
    
    # 检查缺失的 .fchk 文件
    for xyz_file in xyz_files:
        fchk_file = current_dir / f"{xyz_file.stem}.fchk"
        if not fchk_file.exists():
            status_info['missing_fchk'] += 1
            missing_fchk_files.append(xyz_file.name)
    
    # 检查 Surfacia 输出文件夹和各阶段文件
    surfacia_folder = find_latest_surfacia_folder()
    fulloption_files = []
    finalfull_files = []
    training_files = []
    
    if surfacia_folder:
        # 检查 FullOption 文件 (Step 5 输出)
        fulloption_pattern = os.path.join(surfacia_folder, "FullOption*_*.csv")
        fulloption_files = glob.glob(fulloption_pattern)
        
        # 检查 FinalFull 文件 (Step 6 输出)
        finalfull_pattern = os.path.join(surfacia_folder, "FinalFull*.csv")
        finalfull_files = glob.glob(finalfull_pattern)
        
        # 检查训练文件 (Step 7 输出)
        training_pattern = os.path.join(surfacia_folder, "**/Training_Set_Detailed*.csv")
        training_files = glob.glob(training_pattern, recursive=True)
    
    # 更新状态信息
    status_info.update({
        'fulloption_files': len(fulloption_files),
        'finalfull_files': len(finalfull_files),
        'training_files': len(training_files)
    })
    
    # 决定从哪一步开始 (按优先级从高到低)
    if training_files:
        start_step = 8  # 直接从 SHAP 可视化开始
        step_name = "Interactive SHAP Visualization"
    elif finalfull_files:
        start_step = 7  # 从机器学习分析开始
        step_name = "Machine Learning Analysis"
    elif fulloption_files:
        start_step = 6  # 从特征提取开始
        step_name = "Feature Extraction"
    elif status_info['complete_fchk'] > 0 and status_info['empty_fchk'] == 0 and status_info['missing_fchk'] == 0:
        start_step = 5  # 从 Multiwfn 分析开始
        step_name = "Multiwfn Analysis"
    elif status_info['xyz_count'] > 0 and (status_info['empty_fchk'] > 0 or status_info['missing_fchk'] > 0):
        start_step = 4  # 从 Gaussian 计算开始（续算模式）
        step_name = "Gaussian Calculations (Resume Mode)"
    elif status_info['com_count'] > 0:
        start_step = 4  # 从 Gaussian 计算开始
        step_name = "Gaussian Calculations"
    elif status_info['xyz_count'] > 0:
        start_step = 3  # 从生成 Gaussian 输入开始
        step_name = "Gaussian Input Generation"
    else:
        start_step = 1  # 从头开始
        step_name = "SMILES to XYZ Conversion"
    
    return start_step, step_name, status_info

def main():
    parser = argparse.ArgumentParser(
        description='Surfacia - Surface Atomic Chemical Interaction Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🧪 SURFACIA COMMAND OVERVIEW:

MAIN WORKFLOW:
  workflow          Complete end-to-end analysis pipeline
  
INDIVIDUAL STEPS:
  smi2xyz           Convert SMILES to XYZ coordinates
  xtb-opt           XTB geometry optimization
  xyz2gaussian      Generate Gaussian input files
  run-gaussian      Execute Gaussian calculations
  multiwfn          Run Multiwfn analysis on .fchk files
  extract-features  Extract atomic properties and features
  ml-analysis       Machine learning analysis with feature selection
  shap-viz          Interactive SHAP visualization with AI assistant

UTILITY TOOLS (Independent):
  mol-draw          Generate 2D molecular structure images
  mol-info          Calculate and display molecular properties
  rerun-gaussian    Rerun failed Gaussian calculations

📚 For detailed help on any command: surfacia <command> --help
🌐 Documentation: https://github.com/your-repo/surfacia
        """)
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # === MAIN WORKFLOW COMMANDS ===
    
    # smi2xyz command
    parser_smi2xyz = subparsers.add_parser('smi2xyz',
                                          help='Convert SMILES to XYZ coordinates',
                                          formatter_class=argparse.RawDescriptionHelpFormatter,
                                          epilog="""
Examples:
  # Convert SMILES from CSV file
  surfacia smi2xyz -i molecules.csv
                                          """)
    parser_smi2xyz.add_argument('-i', '--input', required=True, help='Input CSV file containing SMILES')

    # xtb-opt command
    parser_xtb = subparsers.add_parser('xtb-opt', 
                                      help='XTB geometry optimization',
                                      formatter_class=argparse.RawDescriptionHelpFormatter,
                                      epilog="""
Examples:
  # Basic XTB optimization with default settings
  surfacia xtb-opt
  
  # Use GFN2-xTB method with tight optimization
  surfacia xtb-opt --method gfn2 --opt-level tight
  
  # Optimize in water solvent
  surfacia xtb-opt --solvent water
  
  # Use custom parameter file
  surfacia xtb-opt --params my_xtb_params.txt
                                      """)
    parser_xtb.add_argument('--method', choices=['gfn1', 'gfn2'], default='gfn2', help='XTB method (default: gfn2)')
    parser_xtb.add_argument('--opt-level', choices=['crude', 'sloppy', 'loose', 'normal', 'tight', 'verytight'], 
                           default='normal', help='Optimization convergence level (default: normal)')
    parser_xtb.add_argument('--solvent', default='none', help='Solvent for ALPB model (default: none)')
    parser_xtb.add_argument('--params', help='Custom XTB parameter file')

    # xyz2gaussian command
    parser_xyz2gauss = subparsers.add_parser('xyz2gaussian', 
                                            help='Generate Gaussian input files',
                                            formatter_class=argparse.RawDescriptionHelpFormatter,
                                            epilog="""
Examples:
  # Generate Gaussian inputs with default DFT settings
  surfacia xyz2gaussian
  
  # Use custom method and basis set
  surfacia xyz2gaussian --keywords "# B3LYP/6-31G* opt freq"
  
  # Set molecular charge and multiplicity
  surfacia xyz2gaussian --charge -1 --multiplicity 2
  
  # Use more processors and memory
  surfacia xyz2gaussian --nproc 64 --memory 50GB
                                            """)
    parser_xyz2gauss.add_argument('--keywords', default="# PBE1PBE/6-311g* scrf(SMD,solvent=Water) em=GD3", 
                                 help='Gaussian calculation keywords')
    parser_xyz2gauss.add_argument('--charge', type=int, default=0, help='Molecular charge (default: 0)')
    parser_xyz2gauss.add_argument('--multiplicity', type=int, default=1, help='Spin multiplicity (default: 1)')
    parser_xyz2gauss.add_argument('--nproc', type=int, default=32, help='Number of processors (default: 32)')
    parser_xyz2gauss.add_argument('--memory', default="30GB", help='Memory allocation (default: 30GB)')

    # run-gaussian command
    parser_run_gauss = subparsers.add_parser('run-gaussian', 
                                            help='Execute Gaussian calculations',
                                            formatter_class=argparse.RawDescriptionHelpFormatter,
                                            epilog="""
Examples:
  # Run all .com files in current directory
  surfacia run-gaussian
  
  # This command will:
  # - Find all .com files
  # - Submit them to Gaussian
  # - Convert .chk files to .fchk format
  # - Monitor progress and report results
                                            """)

    # multiwfn command
    parser_multiwfn = subparsers.add_parser('multiwfn', 
                                           help='Run Multiwfn analysis on .fchk files',
                                           formatter_class=argparse.RawDescriptionHelpFormatter,
                                           epilog="""
Examples:
  # Analyze all .fchk files in current directory
  surfacia multiwfn
  
  # Analyze files in specific directory
  surfacia multiwfn --input-dir ./calculations --output-dir ./results
  
  # This will generate:
  # - Surface analysis files
  # - Electronic property calculations
  # - Molecular descriptors
                                           """)
    parser_multiwfn.add_argument('--input-dir', default='.', help='Directory containing .fchk files (default: current)')
    parser_multiwfn.add_argument('--output-dir', default='.', help='Output directory (default: current)')

    # extract-features command
    parser_extract = subparsers.add_parser('extract-features', 
                                          help='Extract atomic properties and features',
                                          formatter_class=argparse.RawDescriptionHelpFormatter,
                                          epilog="""
Examples:
  # Extract features in Mode 1 (element-specific)
  surfacia extract-features -i ./Surfacia*/FullOption2.csv --mode 1 --element S
  
  # Extract features in Mode 2 (fragment-specific)
  surfacia extract-features -i ./Surfacia*/FullOption2.csv --mode 2 --xyz1 fragment.xyz
  
  # Extract features in Mode 3 (LOFFI comprehensive)
  surfacia extract-features -i ./Surfacia*/FullOption2.csv --mode 3
  
  # Use custom threshold for surface analysis
  surfacia extract-features -i ./Surfacia*/FullOption2.csv --mode 1 --element N --threshold 0.002
                                          """)
    parser_extract.add_argument('-i', '--input', required=True, help='Input CSV file')
    parser_extract.add_argument('--mode', type=int, choices=[1, 2, 3], required=True, 
                               help='Feature extraction mode: 1=element, 2=fragment, 3=LOFFI')
    parser_extract.add_argument('--element', help='Target element for mode 1 (e.g., S, N, O)')
    parser_extract.add_argument('--xyz1', help='Fragment XYZ file for mode 2')
    parser_extract.add_argument('--threshold', type=float, default=0.001, 
                               help='Surface analysis threshold (default: 0.001)')

    # ml-analysis command
    parser_ml = subparsers.add_parser('ml-analysis', 
                                     help='Machine learning analysis with feature selection',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="""
Examples:
  # Automatic workflow mode (default) - uses stepwise regression for feature selection
  surfacia ml-analysis -i ./Surfacia*/FinalFull.csv --test-samples "79,22,82,36,70,80" --max-features 5 --stepreg-runs 3
  
  # Manual feature selection mode
  surfacia ml-analysis -i ./Surfacia*/FinalFull.csv --manual --manual-features "S_ALIE_min,C_LEAE_min,Fun_ESP_delta"
  
  # Use all features in manual mode
  surfacia ml-analysis -i ./Surfacia*/FinalFull.csv --manual --manual-features "Full"
  
  # Custom training parameters
  surfacia ml-analysis -i ./Surfacia*/FinalFull.csv --epoch 128 --cores 64 --train-test-split 0.8

Modes:
  - Default (workflow): Automatic feature selection using stepwise regression
  - Manual (--manual): Use specific features you provide
                                     """)
    parser_ml.add_argument('-i', '--input', required=True, 
                          help='Input CSV file (FinalFull.csv from feature extraction)')
    parser_ml.add_argument('--test-samples', 
                          help='Comma-separated test sample names or numbers (e.g., "79,22,82,36,70,80")')
    parser_ml.add_argument('--nan-handling', choices=['drop_rows', 'drop_columns'], default='drop_columns',
                          help='How to handle NaN values (default: drop_columns)')
    
    # Mode selection
    parser_ml.add_argument('--manual', action='store_true', 
                          help='Use manual feature selection mode (default: automatic workflow mode)')
    
    # Manual mode parameters
    manual_group = parser_ml.add_argument_group('Manual Mode Options', 
                                               'Used when --manual is specified')
    manual_group.add_argument('--manual-features', 
                             help='Comma-separated feature names for manual mode, or "Full" for all features')
    manual_group.add_argument('--no-generate-fitting', dest='generate_fitting', action='store_false',
                             help='Disable generation of fitting plots for manual mode')
    
    # Workflow mode parameters (default)
    workflow_group = parser_ml.add_argument_group('Workflow Mode Options (Default)', 
                                                 'Used when --manual is NOT specified')
    workflow_group.add_argument('--max-features', type=int, default=5, 
                               help='Maximum features for automatic selection (default: 5)')
    workflow_group.add_argument('--stepreg-runs', type=int, default=3, 
                               help='Number of stepwise regression runs (default: 3)')
    workflow_group.add_argument('--initial-features', 
                               help='Comma-separated initial features for stepwise regression')
    workflow_group.add_argument('--shap-fit-threshold', type=float, default=0.3, 
                               help='SHAP R² threshold for feature recommendation (default: 0.3)')
    
    # Common parameters
    common_group = parser_ml.add_argument_group('Common Options', 
                                               'Used in both manual and workflow modes')
    common_group.add_argument('--train-test-split', type=float, default=0.85, 
                             help='Train/test split ratio (default: 0.85)')
    common_group.add_argument('--epoch', type=int, default=64, 
                             help='Number of training epochs (default: 64)')
    common_group.add_argument('--cores', type=int, default=32, 
                             help='Number of CPU cores to use (default: 32)')

    # shap-viz command
    parser_shap = subparsers.add_parser('shap-viz',
                                        help='Interactive SHAP analysis with 3D molecular visualization',
                                        formatter_class=argparse.RawDescriptionHelpFormatter,
                                        epilog="""
Examples:
  # Basic SHAP visualization
  surfacia shap-viz -i Training_Set_Detailed.csv -x ./xyz_folder
  
  # Include test set visualization
  surfacia shap-viz -i Training_Set_Detailed.csv -x ./xyz_folder --test-csv Test_Set_Detailed.csv
  
  # Use AI assistant with API key
  surfacia shap-viz -i Training_Set_Detailed.csv -x ./xyz_folder --api-key your_api_key
  
  # Custom server settings
  surfacia shap-viz -i Training_Set_Detailed.csv -x ./xyz_folder --port 8080 --host localhost

Features:
  - Interactive SHAP scatter plots
  - 3D molecular visualization with isosurfaces
  - AI-powered analysis assistant
  - Real-time molecular property exploration
                                        """)
    parser_shap.add_argument('-i', '--input', required=True,
                            help='Training_Set_Detailed CSV file from ML analysis')
    parser_shap.add_argument('-x', '--xyz-folder', required=True,
                            help='Folder containing XYZ files and molecular structure files')
    parser_shap.add_argument('--test-csv',
                            help='Test_Set_Detailed CSV file for test set visualization (optional)')
    parser_shap.add_argument('--api-key',
                            help='ZhipuAI API key for AI assistant features (optional)')
    parser_shap.add_argument('--skip-surface-gen', action='store_true',
                            help='Skip surface PDB generation if files already exist')
    parser_shap.add_argument('--port', type=int, default=8052,
                            help='Port for the web server (default: 8052)')
    parser_shap.add_argument('--host', default='0.0.0.0',
                            help='Host for the web server (default: 0.0.0.0 for all interfaces)')

    # workflow command
    parser_workflow = subparsers.add_parser('workflow',
                                           help='Complete end-to-end analysis pipeline',
                                           formatter_class=argparse.RawDescriptionHelpFormatter,
                                           epilog="""
Examples:
  # Complete workflow with default settings
  surfacia workflow -i molecules.csv
  
  # Resume workflow (smart continuation from existing calculations)
  surfacia workflow -i molecules.csv --resume
  
  # Skip XTB optimization step
  surfacia workflow -i molecules.csv --skip-xtb
  
  # Custom Gaussian settings
  surfacia workflow -i molecules.csv --keywords "# B3LYP/6-31G* opt freq" --nproc 64
  
  # Include test samples for ML analysis
  surfacia workflow -i molecules.csv --test-samples "1,5,10,15,20"
  
  # Custom ML parameters
  surfacia workflow -i molecules.csv --max-features 8 --stepreg-runs 5 --epoch 128

Pipeline Steps:
  1. SMILES → XYZ conversion
  2. XTB geometry optimization (optional)
  3. Gaussian input generation
  4. Gaussian calculations (most time-consuming)
  5. Multiwfn analysis
  6. Feature extraction
  7. Machine learning analysis
  8. Interactive SHAP visualization

Resume Mode:
  - Automatically detects completed .fchk files
  - Reruns failed calculations only
  - Skips to feature extraction if all calculations are complete
  - Saves significant computation time
                                           """)
    parser_workflow.add_argument('-i', '--input', required=True, help='Initial CSV file with SMILES')
    parser_workflow.add_argument('--resume', action='store_true', help='Resume workflow from existing calculations (smart continuation)')
    parser_workflow.add_argument('--skip-xtb', action='store_true', help='Skip XTB optimization')
    parser_workflow.add_argument('--test-samples', help='Test samples for ML analysis')
    parser_workflow.add_argument('--keywords', default="# PBE1PBE/6-311g* scrf(SMD,solvent=Water) em=GD3", help='Gaussian keywords')
    parser_workflow.add_argument('--charge', type=int, default=0, help='Molecular charge')
    parser_workflow.add_argument('--multiplicity', type=int, default=1, help='Spin multiplicity')
    parser_workflow.add_argument('--nproc', type=int, default=32, help='Number of processors')
    parser_workflow.add_argument('--memory', default="30GB", help='Memory allocation')
    
    # Feature extraction parameters
    parser_workflow.add_argument('--extract-mode', type=int, choices=[1, 2, 3], default=3,
                                help='Feature extraction mode: 1=element-specific, 2=fragment-specific, 3=LOFFI comprehensive (default: 3)')
    parser_workflow.add_argument('--extract-element', help='Target element for mode 1 (e.g., S, N, O)')
    parser_workflow.add_argument('--extract-xyz1', help='Fragment XYZ file for mode 2')
    parser_workflow.add_argument('--extract-threshold', type=float, default=0.001,
                                help='Surface analysis threshold for mode 2 (default: 0.001)')
    
    # ML parameters
    parser_workflow.add_argument('--max-features', type=int, default=5, help='Maximum features for ML')
    parser_workflow.add_argument('--stepreg-runs', type=int, default=3, help='Number of stepwise regression runs')
    parser_workflow.add_argument('--initial-features', help='Comma-separated initial features for stepwise regression')
    parser_workflow.add_argument('--train-test-split', type=float, default=0.85, help='Train/test split ratio')
    parser_workflow.add_argument('--shap-fit-threshold', type=float, default=0.3, help='SHAP R2 threshold for feature recommendation')
    parser_workflow.add_argument('--no-generate-fitting', dest='generate_fitting', action='store_false', help="Disable generation of fitting plots for manual mode")
    parser_workflow.add_argument('--epoch', type=int, default=64, help='Number of epochs')
    parser_workflow.add_argument('--cores', type=int, default=32, help='Number of CPU cores')
    parser_workflow.add_argument('--api-key', help='ZhipuAI API key for SHAP visualization')
    parser_workflow.add_argument('--port', type=int, default=8052, help='Port for SHAP visualization server')
    parser_workflow.add_argument('--host', default='0.0.0.0', help='Host for SHAP visualization server')
    
    # === UTILITY TOOLS (Independent) ===
    
    # mol-draw command - 2D分子绘图工具
    parser_draw = subparsers.add_parser('mol-draw', 
                                       help='Generate 2D molecular structure images from SMILES',
                                       formatter_class=argparse.RawDescriptionHelpFormatter,
                                       epilog="""
Examples:
  # Draw single molecule
  surfacia mol-draw --smiles "CCO" -o ethanol.png
  
  # Batch draw from CSV file
  surfacia mol-draw -i molecules.csv -o molecule_images
  
  # Custom image size
  surfacia mol-draw -i molecules.csv --size 1200 1200 -o high_res_images
  
  # Custom filename prefix
  surfacia mol-draw -i molecules.csv --prefix compound -o structures

Output:
  - High-quality PNG images (300 DPI)
  - Customizable size and styling
  - Batch processing support
                                       """)
    parser_draw.add_argument('-i', '--input', help='Input CSV file containing SMILES column')
    parser_draw.add_argument('--smiles', help='Single SMILES string to draw')
    parser_draw.add_argument('-o', '--output', help='Output directory (for CSV) or file path (for single SMILES)')
    parser_draw.add_argument('--size', nargs=2, type=int, default=[800, 800], 
                            help='Image size as width height (default: 800 800)')
    parser_draw.add_argument('--prefix', default='mol', 
                            help='Filename prefix for batch processing (default: mol)')
    
    # mol-info command - 分子信息查看工具
    parser_info = subparsers.add_parser('mol-info',
                                       help='Calculate and display molecular properties',
                                       formatter_class=argparse.RawDescriptionHelpFormatter,
                                       epilog="""
Examples:
  # Analyze single molecule
  surfacia mol-info --smiles "CCO"
  
  # Analyze molecules from CSV file
  surfacia mol-info -i molecules.csv
  
  # Save results to file
  surfacia mol-info -i molecules.csv -o molecular_properties.csv
  
  # Calculate specific properties only
  surfacia mol-info --smiles "CCO" --properties mw logp hbd hba

Properties Available:
  - mw: Molecular weight
  - logp: Partition coefficient
  - hbd: Hydrogen bond donors
  - hba: Hydrogen bond acceptors
  - tpsa: Topological polar surface area
  - rotatable_bonds: Number of rotatable bonds
  - aromatic_rings: Number of aromatic rings
  - heavy_atoms: Number of heavy atoms
                                       """)
    parser_info.add_argument('-i', '--input', help='Input CSV file containing SMILES column')
    parser_info.add_argument('--smiles', help='Single SMILES string to analyze')
    parser_info.add_argument('-o', '--output', help='Output CSV file for molecular properties')
    parser_info.add_argument('--properties', nargs='+', 
                            default=['mw', 'logp', 'hbd', 'hba', 'tpsa', 'rotatable_bonds'],
                            help='Properties to calculate: mw, logp, hbd, hba, tpsa, rotatable_bonds, etc.')
    
    # rerun-gaussian command - 重新运行失败的计算
    parser_rerun = subparsers.add_parser('rerun-gaussian',
                                        help='Rerun failed Gaussian calculations',
                                        formatter_class=argparse.RawDescriptionHelpFormatter,
                                        epilog="""
Examples:
  # Check and rerun failed calculations
  surfacia rerun-gaussian
  
  # This command will:
  # - Scan for empty .fchk files
  # - Find .xyz files without corresponding .fchk files
  # - Rerun the failed Gaussian calculations
  # - Convert .chk files to .fchk format
  
Use Cases:
  - After system crashes or interruptions
  - When some calculations failed due to resource limits
  - To complete partially finished calculation batches
  
Safety Features:
  - Automatically removes corrupted files
  - Preserves original .com files
  - Provides detailed progress reporting
                                        """)
    
    args = parser.parse_args()
    
    # Command execution logic
    if args.command == 'smi2xyz':
        # 检查输入文件
        if not os.path.exists(args.input):
            print(f"❌ Error: Input file '{args.input}' not found!")
            sys.exit(1)
        
        print(f"🚀 Converting SMILES to XYZ coordinates...")
        print(f"   Input file: {args.input}")
        
        try:
            smi2xyz_main(args.input)
            print("✅ SMILES to XYZ conversion completed successfully!")
        except Exception as e:
            print(f"❌ Error during SMILES conversion: {e}")
            sys.exit(1)
    
    elif args.command == 'xtb-opt':
        # 检查是否有 XYZ 文件
        xyz_files = glob.glob("*.xyz")
        if not xyz_files:
            print("❌ Error: No XYZ files found in current directory!")
            print("   Please run 'surfacia smi2xyz' first or ensure XYZ files are present.")
            sys.exit(1)
        
        print(f"🚀 Running XTB geometry optimization...")
        print(f"   Found {len(xyz_files)} XYZ files")
        print(f"   Method: {args.method}")
        print(f"   Optimization level: {args.opt_level}")
        print(f"   Solvent: {args.solvent}")
        
        try:
            # 构建 XTB 参数
            if hasattr(args, 'params') and args.params:
                if not os.path.exists(args.params):
                    print(f"❌ Error: Parameter file '{args.params}' not found!")
                    sys.exit(1)
                run_xtb_opt(args.params)
            else:
                # 根据命令行参数构建 XTB 选项
                xtb_options = f"--opt {args.opt_level} --gfn {args.method[-1]} --molden --alpb {args.solvent}"
                
                # 创建临时参数文件
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write(xtb_options)
                    temp_params = f.name
                
                try:
                    run_xtb_opt(temp_params)
                finally:
                    # 清理临时文件
                    if os.path.exists(temp_params):
                        os.unlink(temp_params)
            
            print("✅ XTB optimization completed successfully!")
        except Exception as e:
            print(f"❌ Error during XTB optimization: {e}")
            sys.exit(1)
        
    elif args.command == 'xyz2gaussian':
        # 检查是否有 XYZ 文件
        xyz_files = glob.glob("*.xyz")
        if not xyz_files:
            print("❌ Error: No XYZ files found in current directory!")
            print("   Please run 'surfacia smi2xyz' or 'surfacia xtb-opt' first.")
            sys.exit(1)
        
        print(f"🚀 Generating Gaussian input files...")
        print(f"   Found {len(xyz_files)} XYZ files")
        print(f"   Keywords: {args.keywords}")
        print(f"   Charge: {args.charge}, Multiplicity: {args.multiplicity}")
        print(f"   Processors: {args.nproc}, Memory: {args.memory}")
        
        try:
            # Set global variables
            from .core import gaussian
            gaussian.GAUSSIAN_KEYWORD_LINE = args.keywords
            gaussian.DEFAULT_CHARGE = args.charge
            gaussian.DEFAULT_MULTIPLICITY = args.multiplicity
            gaussian.DEFAULT_NPROC = args.nproc
            gaussian.DEFAULT_MEMORY = args.memory
            xyz2gaussian_main()
            print("✅ Gaussian input files generated successfully!")
        except Exception as e:
            print(f"❌ Error during Gaussian input generation: {e}")
            sys.exit(1)
        
    elif args.command == 'run-gaussian':
        # 检查是否有 .com 文件
        com_files = glob.glob("*.com")
        if not com_files:
            print("❌ Error: No Gaussian .com files found in current directory!")
            print("   Please run 'surfacia xyz2gaussian' first.")
            sys.exit(1)
        
        print(f"🚀 Running Gaussian calculations...")
        print(f"   Found {len(com_files)} .com files")
        print("   This may take a long time depending on your system and molecule size.")
        
        try:
            run_gaussian()
            print("✅ Gaussian calculations completed successfully!")
        except Exception as e:
            print(f"❌ Error during Gaussian calculations: {e}")
            sys.exit(1)
        
    elif args.command == 'multiwfn':
        print("🚀 Running Multiwfn analysis...")
        print(f"   Input directory: {args.input_dir}")
        print(f"   Output directory: {args.output_dir}")
        
        try:
            print("Step 1: Running Multiwfn calculations...")
            run_multiwfn_on_fchk_files(args.input_dir)
            print("\nStep 2: Processing Multiwfn output files...")
            process_txt_files(args.input_dir, args.output_dir)
            print("✅ Multiwfn analysis completed successfully!")
        except Exception as e:
            print(f"❌ Error during Multiwfn analysis: {e}")
            sys.exit(1)
        
    elif args.command == 'extract-features':
        print(f"🚀 Extracting features using Mode {args.mode}...")
        print(f"   Input file: {args.input}")
        
        if args.mode == 1 and not args.element:
            print("❌ Error: --element is required for Mode 1")
            sys.exit(1)
        elif args.mode == 2 and not args.xyz1:
            print("❌ Error: --xyz1 is required for Mode 2")
            sys.exit(1)
        
        try:
            run_atom_prop_extraction(
                args.input,
                mode=args.mode,
                target_element=args.element,
                xyz1_path=args.xyz1,
                threshold=args.threshold
            )
            print("✅ Feature extraction completed successfully!")
        except Exception as e:
            print(f"❌ Error during feature extraction: {e}")
            sys.exit(1)
        
    elif args.command == 'ml-analysis':
        # 导入新的ChemMLWorkflow
        from .ml.chem_ml_analyzer_v2 import ChemMLWorkflow
        
        print("🚀 Running machine learning analysis...")
        print(f"   Input file: {args.input}")
        
        # 检查输入文件
        if not os.path.exists(args.input):
            print(f"❌ Error: Input file '{args.input}' not found!")
            sys.exit(1)
        
        # 检查输入文件是否存在，如果不存在则尝试通配符匹配
        input_file = args.input
        if not os.path.exists(input_file):
            # 检查是否在当前目录或 Surfacia 文件夹中
            current_dir_pattern = f"FullOption*_*.csv"
            current_dir_files = glob.glob(current_dir_pattern)
            
            if current_dir_files:
                input_file = max(current_dir_files, key=os.path.getmtime)
                print(f"   Input file not found, using: {input_file}")
            else:
                # 尝试在最新的 Surfacia 文件夹中查找
                latest_folder = find_latest_surfacia_folder()
                if latest_folder:
                    folder_pattern = os.path.join(latest_folder, "FullOption*_*.csv")
                    folder_files = glob.glob(folder_pattern)
                    if folder_files:
                        input_file = max(folder_files, key=os.path.getmtime)
                        print(f"   Input file not found, using: {input_file}")
                    else:
                        print(f"❌ Error: Input file '{args.input}' not found and no FullOption files found!")
                        sys.exit(1)
                else:
                    print(f"❌ Error: Input file '{args.input}' not found!")
                    sys.exit(1)
        
        # 处理测试样本
        test_samples = None
        if args.test_samples:
            # 尝试将测试样本转换为数字，如果失败则保持字符串
            test_samples = []
            for s in args.test_samples.split(','):
                s = s.strip()
                try:
                    test_samples.append(int(s))
                except ValueError:
                    test_samples.append(s)
        
        # 决定运行模式
        if args.manual:
            # 手动模式
            mode = 'manual'
            features = None
            if args.manual_features:
                if args.manual_features.strip().lower() == 'full':
                    features = 'Full'
                else:
                    features = [f.strip() for f in args.manual_features.split(',')]
            
            print(f"   Mode: Manual feature selection")
            print(f"   Features: {features}")
            
            try:
                results = ChemMLWorkflow.run_analysis(
                    mode=mode,
                    data_file=input_file,
                    test_sample_names=test_samples,
                    nan_handling=args.nan_handling,
                    features=features,
                    epoch=args.epoch,
                    core_num=args.cores,
                    train_test_split=args.train_test_split,
                    generate_fitting=args.generate_fitting
                )
                
                print(f"\n{'='*60}")
                print("ML Analysis Completed!")
                print(f"Mode: Manual Feature Selection")
                print(f"MSE: {results['mse']:.4f}")
                print(f"R²: {results['r2']:.4f}")
                if 'selected_features' in results:
                    print(f"Features used: {len(results['selected_features'])}")
                    
            except Exception as e:
                print(f"❌ Error during ML analysis: {e}")
                sys.exit(1)
        else:
            # 工作流模式
            mode = 'workflow'
            print(f"   Mode: Automatic workflow (stepwise regression)")
            print(f"   Max features: {args.max_features}")
            print(f"   Stepwise regression runs: {args.stepreg_runs}")
            
            try:
                results = ChemMLWorkflow.run_analysis(
                    mode=mode,
                    data_file=input_file,
                    test_sample_names=test_samples,
                    nan_handling=args.nan_handling,
                    max_features=args.max_features,
                    n_runs=args.stepreg_runs,
                    epoch=args.epoch,
                    core_num=args.cores,
                    train_test_split=args.train_test_split
                )
                
                print(f"\n{'='*60}")
                print("ML Analysis Completed!")
                print(f"Mode: Workflow (Automatic Feature Selection)")
                print(f"Baseline MSE: {results['baseline']['mse']:.4f}")
                print(f"Final MSE: {results['final']['mse']:.4f}")
                print(f"Recommended features: {results['final']['selected_features']}")
                
            except Exception as e:
                print(f"❌ Error during ML analysis: {e}")
                sys.exit(1)
                
        print("✅ Machine learning analysis completed successfully!")
        
    elif args.command == 'shap-viz':
        print("🚀 Starting interactive SHAP visualization...")
        print(f"   Training data: {args.input}")
        print(f"   XYZ folder: {args.xyz_folder}")
        
        # 检查输入文件
        if not os.path.exists(args.input):
            print(f"❌ Error: Training data file '{args.input}' not found!")
            sys.exit(1)
        
        if not os.path.exists(args.xyz_folder):
            print(f"❌ Error: XYZ folder '{args.xyz_folder}' not found!")
            sys.exit(1)
        
        try:
            # 使用标准入口函数
            success = interactive_shap_viz_main(
                csv_path=args.input,
                xyz_path=args.xyz_folder,
                test_csv_path=args.test_csv if hasattr(args, 'test_csv') else None,
                api_key=args.api_key if hasattr(args, 'api_key') else None,
                skip_surface_gen=args.skip_surface_gen,
                port=args.port,
                host=args.host
            )
            if not success:
                print("❌ Interactive SHAP visualization failed!")
                sys.exit(1)
            
            print("✅ SHAP visualization completed successfully!")
            
        except Exception as e:
            print(f"❌ Error during SHAP visualization: {e}")
            sys.exit(1)

    elif args.command == 'workflow':
        import time
        start_time = time.time()
        
        print("🚀 Starting Surfacia workflow...")
        print(f"   Input file: {args.input}")
        
        # 检查输入文件并读取样本数量
        try:
            import pandas as pd
            input_df = pd.read_csv(args.input)
            sample_count = len(input_df)
            print(f"   Dataset contains: {sample_count} samples")
            
            if sample_count < 20:
                print("⚠️  WARNING: Small dataset detected!")
                print("   - Datasets with fewer than 20 samples may cause ML analysis errors")
                print("   - You may encounter 'too many indices for array' errors during baseline analysis")
                print("   - Consider using a larger dataset for more reliable results")
                print("   - Continuing anyway...")
        except Exception as e:
            print(f"   Warning: Could not read input file for sample count check: {e}")
        
        # 智能检查计算状态
        if args.resume:
            print("\n🔍 Checking existing calculations...")
            start_step, step_name, status_info = check_calculation_status()
            
            print(f"📊 Current Status:")
            print(f"   XYZ files: {status_info['xyz_count']}")
            print(f"   COM files: {status_info['com_count']}")
            print(f"   Complete .fchk files: {status_info['complete_fchk']}")
            print(f"   Empty .fchk files: {status_info['empty_fchk']}")
            print(f"   Missing .fchk files: {status_info['missing_fchk']}")
            
            print(f"\n🎯 Resume Strategy: Starting from Step {start_step} - {step_name}")
            
            if start_step > 4:
                print("   ⚡ Skipping time-consuming Gaussian calculations (already complete)")
        else:
            start_step = 1
            step_name = "Complete workflow from beginning"
            print(f"\n🎯 Running: {step_name}")
        
        # Step 1: SMILES to XYZ
        if start_step <= 1:
            print("\n=== Step 1: Converting SMILES to XYZ ===")
            try:
                smi2xyz_main(args.input)
                print("✅ SMILES to XYZ conversion completed!")
            except Exception as e:
                print(f"❌ Error in Step 1: {e}")
                sys.exit(1)
        else:
            print("\n⏭️  Step 1: SMILES to XYZ (Skipped - files exist)")
        
        # Step 2: XTB optimization (optional)
        if start_step <= 2 and not args.skip_xtb:
            print("\n=== Step 2: XTB Optimization ===")
            try:
                run_xtb_opt()
                print("✅ XTB optimization completed!")
            except Exception as e:
                print(f"❌ Error in Step 2: {e}")
                sys.exit(1)
        elif args.skip_xtb:
            print("\n⏭️  Step 2: XTB Optimization (Skipped by user)")
        else:
            print("\n⏭️  Step 2: XTB Optimization (Skipped - resuming from later step)")
        
        # Step 3: Generate Gaussian input
        if start_step <= 3:
            print("\n=== Step 3: Generating Gaussian Input ===")
            try:
                from .core import gaussian
                gaussian.GAUSSIAN_KEYWORD_LINE = args.keywords
                gaussian.DEFAULT_CHARGE = args.charge
                gaussian.DEFAULT_MULTIPLICITY = args.multiplicity
                gaussian.DEFAULT_NPROC = args.nproc
                gaussian.DEFAULT_MEMORY = args.memory
                xyz2gaussian_main()
                print("✅ Gaussian input generation completed!")
            except Exception as e:
                print(f"❌ Error in Step 3: {e}")
                sys.exit(1)
        else:
            print("\n⏭️  Step 3: Gaussian Input Generation (Skipped - files exist)")
        
        # Step 4: Run Gaussian (智能续算)
        if start_step <= 4:
            if args.resume and start_step == 4:
                print("\n=== Step 4: Gaussian Calculations (Resume Mode) ===")
                print("   🔄 Checking for failed calculations and resuming...")
                try:
                    # 使用 rerun_gaussian 功能进行智能续算
                    success = rerun_failed_gaussian_calculations()
                    if success:
                        print("✅ Gaussian calculations completed (resumed successfully)!")
                    else:
                        print("⚠️  Some calculations may have failed, but continuing...")
                except Exception as e:
                    print(f"❌ Error in Step 4 (Resume): {e}")
                    sys.exit(1)
            else:
                print("\n=== Step 4: Running Gaussian ===")
                try:
                    run_gaussian()
                    print("✅ Gaussian calculations completed!")
                except Exception as e:
                    print(f"❌ Error in Step 4: {e}")
                    sys.exit(1)
        else:
            print("\n⏭️  Step 4: Gaussian Calculations (Skipped - all .fchk files complete)")
        
        # Step 5: Multiwfn analysis
        if start_step <= 5:
            print("\n=== Step 5: Multiwfn Analysis ===")
            try:
                run_multiwfn_on_fchk_files('.')
                process_txt_files('.', '.')
                print("✅ Multiwfn analysis completed!")
            except Exception as e:
                print(f"❌ Error in Step 5: {e}")
                sys.exit(1)
        else:
            print("\n⏭️  Step 5: Multiwfn Analysis (Skipped - resuming from later step)")
        
        # Step 6: Feature extraction
        if start_step <= 6:
            print("\n=== Step 6: Feature Extraction ===")
            try:
                # 查找 Multiwfn 处理后的特征文件
                surfacia_folder = find_latest_surfacia_folder()
                if not surfacia_folder:
                    print("❌ Error: Could not find Surfacia output folder")
                    sys.exit(1)
                
                # 查找 FullOption 文件
                fulloption_pattern = os.path.join(surfacia_folder, "FullOption*_*.csv")
                fulloption_files = glob.glob(fulloption_pattern)
                
                if not fulloption_files:
                    print("❌ Error: No FullOption CSV files found in Surfacia folder")
                    sys.exit(1)
                
                # 使用最新的 FullOption 文件
                fulloption_file = max(fulloption_files, key=os.path.getmtime)
                print(f"   Using input file: {os.path.basename(fulloption_file)}")
                print(f"   Extraction mode: {args.extract_mode}")
                
                # 验证模式参数
                if args.extract_mode == 1 and not args.extract_element:
                    print("❌ Error: --extract-element is required for mode 1")
                    sys.exit(1)
                elif args.extract_mode == 2 and not args.extract_xyz1:
                    print("❌ Error: --extract-xyz1 is required for mode 2")
                    sys.exit(1)
                
                # 运行特征提取
                run_atom_prop_extraction(
                    fulloption_file,
                    mode=args.extract_mode,
                    target_element=args.extract_element,
                    xyz1_path=args.extract_xyz1,
                    threshold=args.extract_threshold
                )
                print("✅ Feature extraction completed!")
            except Exception as e:
                print(f"❌ Error in Step 6: {e}")
                sys.exit(1)
        else:
            print("\n⏭️  Step 6: Feature Extraction (Skipped - feature files exist)")
        
        # Step 7: Machine Learning Analysis
        if start_step <= 7:
            print("\n=== Step 7: Machine Learning Analysis ===")
            try:
                # 查找 FinalFull 文件
                surfacia_folder = find_latest_surfacia_folder()
                if not surfacia_folder:
                    print("❌ Error: Could not find Surfacia output folder")
                    sys.exit(1)
                
                finalfull_pattern = os.path.join(surfacia_folder, "FinalFull*.csv")
                finalfull_files = glob.glob(finalfull_pattern)
                
                if not finalfull_files:
                    print("❌ Error: No FinalFull CSV files found. Step 6 may have failed.")
                    sys.exit(1)
                
                # 使用最新的 FinalFull 文件
                finalfull_file = max(finalfull_files, key=os.path.getmtime)
                print(f"   Using feature file: {os.path.basename(finalfull_file)}")
                
                # 准备测试样本
                test_samples = None
                if hasattr(args, 'test_samples') and args.test_samples:
                    test_samples = []
                    for s in args.test_samples.split(','):
                        s = s.strip()
                        try:
                            test_samples.append(int(s))
                        except ValueError:
                            test_samples.append(s)
                    print(f"   Test samples: {test_samples}")
                
                # 运行机器学习分析
                from .ml.chem_ml_analyzer_v2 import ChemMLWorkflow
                results = ChemMLWorkflow.run_analysis(
                    mode='workflow',
                    data_file=finalfull_file,
                    test_sample_names=test_samples,
                    nan_handling='drop_columns',
                    max_features=getattr(args, 'max_features', 10),
                    n_runs=getattr(args, 'stepreg_runs', 5),
                    epoch=getattr(args, 'epoch', 100),
                    core_num=getattr(args, 'cores', 4),
                    train_test_split=getattr(args, 'train_test_split', 0.2)
                )
                
                print(f"   Baseline MSE: {results['baseline']['mse']:.4f}")
                print(f"   Final MSE: {results['final']['mse']:.4f}")
                print(f"   Selected features: {len(results['final']['selected_features'])}")
                print("✅ Machine learning analysis completed!")
                
            except Exception as e:
                print(f"❌ Error in Step 7: {e}")
                sys.exit(1)
        else:
            print("\n⏭️  Step 7: Machine Learning Analysis (Skipped - analysis files exist)")
        
        # Step 8: Interactive SHAP Visualization
        if start_step <= 8:
            print("\n=== Step 8: Interactive SHAP Visualization ===")
            try:
                # 查找训练和测试文件
                surfacia_folder = find_latest_surfacia_folder()
                if not surfacia_folder:
                    print("❌ Error: Could not find Surfacia output folder")
                    sys.exit(1)
                
                # 查找 Auto 文件夹中的训练和测试文件
                training_pattern = os.path.join(surfacia_folder, "**/Training_Set_Detailed*.csv")
                test_pattern = os.path.join(surfacia_folder, "**/Test_Set_Detailed*.csv")
                
                training_files = glob.glob(training_pattern, recursive=True)
                test_files = glob.glob(test_pattern, recursive=True)
                
                if not training_files:
                    print("❌ Error: No Training_Set_Detailed files found. Step 7 may have failed.")
                    sys.exit(1)
                
                # 使用最新的文件
                latest_training = max(training_files, key=os.path.getmtime)
                latest_test = max(test_files, key=os.path.getmtime) if test_files else None
                
                print(f"   Training file: {os.path.basename(latest_training)}")
                if latest_test:
                    print(f"   Test file: {os.path.basename(latest_test)}")
                
                # 启动 SHAP 可视化
                success = interactive_shap_viz_main(
                    csv_path=latest_training,
                    xyz_path='.',
                    test_csv_path=latest_test,
                    api_key=getattr(args, 'api_key', None),
                    skip_surface_gen=False,
                    port=getattr(args, 'port', 8052),
                    host=getattr(args, 'host', '0.0.0.0')
                )
                if success:
                    print("✅ Interactive SHAP visualization launched successfully!")
                else:
                    print("⚠️ SHAP visualization completed with warnings")
                
            except Exception as e:
                print(f"❌ Error in Step 8: {e}")
                print("⚠️  Continuing without SHAP visualization...")
        else:
            print("\n⏭️  Step 8: Interactive SHAP Visualization (Skipped)")
        
        # 工作流完成总结
        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print("🎉 Complete Surfacia Workflow Finished Successfully!")
        print(f"   Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        
        if args.resume:
            print(f"   Resume mode: Started from Step {start_step}")
            if start_step > 4:
                print("   ⚡ Time saved by skipping completed Gaussian calculations!")
        
        # 显示生成的文件
        surfacia_folder = find_latest_surfacia_folder()
        if surfacia_folder:
            print(f"\n📁 Output folder: {surfacia_folder}")
            
            # 检查各种输出文件
            finalfull_files = glob.glob(os.path.join(surfacia_folder, "FinalFull*.csv"))
            training_files = glob.glob(os.path.join(surfacia_folder, "**/Training_Set_Detailed*.csv"), recursive=True)
            
            if finalfull_files:
                print(f"   📊 Feature file: {os.path.basename(finalfull_files[0])}")
            if training_files:
                print(f"   🤖 ML analysis: {os.path.basename(training_files[0])}")
            
            print("\n💡 Next steps:")
            print("   • Check the output folder for detailed results")
            print("   • Use individual commands for specific analysis")
            print("   • SHAP visualization should be running on http://localhost:8052")
        
        
    # === UTILITY TOOLS COMMANDS ===
    
    elif args.command == 'mol-draw':
        print("🎨 Running molecular drawing tool...")
        
        if args.smiles and args.input:
            print("❌ Error: Please specify either --smiles OR --input, not both.")
            sys.exit(1)
        
        if not args.smiles and not args.input:
            print("❌ Error: Please specify either --smiles or --input CSV file.")
            sys.exit(1)
        
        try:
            if args.smiles:
                # 绘制单个分子
                output_path = args.output or f"molecule_{args.smiles[:10].replace('/', '_')}.png"
                print(f"   Drawing molecule: {args.smiles}")
                print(f"   Output: {output_path}")
                
                img = draw_single_molecule(
                    smiles=args.smiles,
                    output_path=output_path,
                    size=tuple(args.size)
                )
                
                if img:
                    print("✅ Molecular structure image generated successfully!")
                else:
                    print("❌ Failed to generate molecular structure image.")
                    sys.exit(1)
            
            else:
                # 从CSV文件批量绘制
                if not os.path.exists(args.input):
                    print(f"❌ Error: Input file '{args.input}' not found!")
                    sys.exit(1)
                
                output_dir = args.output or 'molecule_images'
                print(f"   Input CSV: {args.input}")
                print(f"   Output directory: {output_dir}")
                print(f"   Image size: {args.size[0]}x{args.size[1]}")
                
                draw_molecules_from_csv(
                    csv_file=args.input,
                    output_dir=output_dir
                )
                
                print("✅ All molecular structure images generated successfully!")
                
        except Exception as e:
            print(f"❌ Error during molecular drawing: {e}")
            sys.exit(1)
    
    elif args.command == 'mol-info':
        print("📊 Running molecular information tool...")
        
        if args.smiles and args.input:
            print("❌ Error: Please specify either --smiles OR --input, not both.")
            sys.exit(1)
        
        if not args.smiles and not args.input:
            print("❌ Error: Please specify either --smiles or --input CSV file.")
            sys.exit(1)
        
        try:
            if args.smiles:
                # 分析单个分子
                print(f"   Analyzing molecule: {args.smiles}")
                success = analyze_single_molecule(args.smiles)
                if not success:
                    print("❌ Failed to analyze molecule.")
                    sys.exit(1)
            else:
                # 从CSV文件分析
                if not os.path.exists(args.input):
                    print(f"❌ Error: Input file '{args.input}' not found!")
                    sys.exit(1)
                
                print(f"   Input CSV: {args.input}")
                if args.output:
                    print(f"   Output file: {args.output}")
                
                success = analyze_molecules_from_csv(
                    csv_file=args.input,
                    output_file=args.output,
                    properties=args.properties
                )
                
                if not success:
                    print("❌ Failed to analyze molecules.")
                    sys.exit(1)
            
            print("✅ Molecular analysis completed successfully!")
            
        except Exception as e:
            print(f"❌ Error during molecular analysis: {e}")
            sys.exit(1)
    
    elif args.command == 'rerun-gaussian':
        print("🔄 Running Gaussian rerun utility...")
        
        try:
            success = rerun_failed_gaussian_calculations()
            if success:
                print("✅ Gaussian rerun completed successfully!")
            else:
                print("❌ Some calculations failed during rerun.")
                sys.exit(1)
                
        except Exception as e:
            print(f"❌ Error during Gaussian rerun: {e}")
            sys.exit(1)
    
    else:
        print("❌ Error: No command specified or unknown command.")
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    import multiprocessing as mp
    # Set start method for compatibility, 'fork' is good for Linux/macOS
    # 'spawn' might be needed for Windows in some cases.
    try:
        mp.set_start_method('fork')
    except RuntimeError:
        print("Start method 'fork' already set or not available.")
        
    main()