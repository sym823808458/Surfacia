#!/usr/bin/env python
# coding: utf-8

# SURF Atomic Chemical Interaction Analyzer - Surfacia
# 
# Surfacia (Surface Atomic Chemical Interaction Analyzer) is a comprehensive toolset designed to automate the workflow for analyzing surface atomic chemical interactions. It integrates molecular structure generation, quantum chemical computations, feature extraction, and machine learning analysis, providing a streamlined solution for structure-activity relationships in chemistry research , with an emphasis on generating conclusions that are interpretable by chemists.
# 
# 从SMILES到SHAP分析图  
# 250105新添功能：
# Step_reg那一步将原始数据集4个csv也存入Step_reg文件夹下，而不是machinelearning了  
# 250123新添功能：V2.3.0
# beam_search策略，选出2^n个特征集，并用集中函数对其拟合，保留拟合效果好（潜在解释性好的）特征组，并在每个rank里面已经运行过多轮交叉验证   
# 250219新添功能：V2.4.0  
# evaluate_record可以选有几个特征了，rank_performan.json中的后面的performance是Full_train_val的表现(并且花在了图里和title里)，改了一下标注   
# *关键*引入分层聚合的思想：接收一个包含分子SMILES结构的CSV文件作为输入，通过RDKit进行分子解析，计算每个原子的物理化学特性（包括LEAE、ESP、ALIE和面积），同时识别分子中的芳香环系统和各类官能团。程序会为每个特征计算最小值、最大值、平均值和变化范围，最终生成一个包含32个描述符的新CSV文件，文件名会根据结果的行数和列数自动命名。
# 250604，在第一步生成一个sample_mapping.csv，后续可以直接对应。    
# 250627，V2.5.0，第一步生成sample_mapping.csv时兼容额外的实验特征加入了;xyz2gaussian功能不用外部读取模版文件了；  
# Atom&Fun_Comprehensive_descriptor部分进行大修改，兼容3种模式的特征提取，mode1给定某元素，mode2给定官能团三维子结构（sub.xyz1给出），mode3经由loffi给出分层的特征，并大改loffi_content内容，现在更加完备且互斥。  
# 优化multiwfn描述符部分，删除四级矩八极矩ODI性质，新增形状描述  
# 250711,V3.0.0,大幅修改ML和新增交互式画图程序；  
# ML部分思路修改为用多次逐步回归获取特征空间，舍弃beam_search策略，用fit_shap_plot综合特征重要性、逐步回归重要性、模型可解释性来综合给出推荐的特征集，并保留由人工一步合成筛选特征集的功能；  
# 交互式画图程序：基于Dash的交互式分子SHAP分析可视化工具，支持点击散点图查看分子的二维结构、三维结构，以及Surfacia模式下根据特征名智能选择LEAE/ESP/ALIE等值面与分子骨架的组合三维显示，并提供VMD风格的实时参数控制。  
# 250808，V3.0.1，修改特征化mode2第一步以元素做划分，应该是原子；修改ML部分，输出模型，新增输出Test_set_detailcsv中的SHAP值；修改交互式画图工具，新增同一样本其他特征的表格在右框二维结构中，优化LLM部分，新增测试集的可视化

# smi2xyz

# In[6]:


import pandas as pd
from openbabel import pybel as pb
import os
import re
from datetime import datetime

def read_smiles_csv(file_path):
    """
    Read a CSV file containing SMILES strings.

    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        DataFrame: DataFrame containing SMILES strings.
    """
    try:
        data = pd.read_csv(file_path)
        print("CSV file successfully read!")
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def clean_column_name(column_name):
    """
    Clean column names by removing or replacing problematic characters.
    
    Args:
        column_name (str): Original column name
        
    Returns:
        str: Cleaned column name
    """
    if not isinstance(column_name, str):
        column_name = str(column_name)
    
    # Convert to lowercase
    cleaned = column_name.lower()
    
    # Replace spaces, slashes, and other problematic characters with underscores
    # Keep only alphanumeric characters and underscores
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', cleaned)
    
    # Replace multiple consecutive underscores with single underscore
    cleaned = re.sub(r'_+', '_', cleaned)
    
    # Remove leading and trailing underscores
    cleaned = cleaned.strip('_')
    
    # Ensure the name doesn't start with a number (add prefix if needed)
    if cleaned and cleaned[0].isdigit():
        cleaned = 'exp_' + cleaned
    
    # Ensure the name is not empty
    if not cleaned:
        cleaned = 'unnamed_feature'
    
    return cleaned

def validate_csv_columns(data):
    """
    Validate that the CSV contains required 'smiles' and 'target' columns.
    
    Args:
        data (DataFrame): Input DataFrame
        
    Returns:
        bool: True if valid, False otherwise
    """
    if data is None:
        return False
    
    # Convert column names to lowercase for comparison
    columns_lower = [col.lower() for col in data.columns]
    
    # Check for required columns
    if 'smiles' not in columns_lower:
        print("Error: CSV file must contain a 'smiles' column (case-insensitive)")
        return False
    
    if 'target' not in columns_lower:
        print("Error: CSV file must contain a 'target' column (case-insensitive)")
        return False
    
    if len(data.columns) < 2:
        print("Error: CSV file must contain at least 2 columns")
        return False
    
    print("✓ Required columns 'smiles' and 'target' found")
    return True

def smiles_to_xyz(smiles_string, sample_name):
    """
    Convert a SMILES string to XYZ format.

    Args:
        smiles_string (str): SMILES string.
        sample_name (str): Sample name for file naming.
    
    Returns:
        tuple: (str, bool) - XYZ format string and whether conversion was successful
    """
    try:
        mol = pb.readstring('smiles', smiles_string)
        mol.make3D()
        return mol.write(format='xyz'), True
    except Exception as e:
        print(f"Error converting SMILES {smiles_string} to XYZ: {e}")
        return None, False

def smi2xyz_main(file_path, check_extensions=['.fchk']):
    """
    Main function: Reads SMILES from CSV and converts them to XYZ files.
    Requires CSV to have 'smiles' and 'target' columns.
    Also creates a mapping file with sample names, SMILES, target and all experimental features.

    Args:
        file_path (str): Path to the CSV file.
        check_extensions (list): File extensions to check for existing files.
    """
    # Read and validate CSV file
    smiles_data = read_smiles_csv(file_path)
    
    if not validate_csv_columns(smiles_data):
        print("Please ensure your CSV file contains 'smiles' and 'target' columns (case-insensitive)")
        return
    
    conversion_count = 0
    skip_count = 0
    
    # 创建映射DataFrame，包含所有列信息
    mapping_data = []

    # 创建列名映射字典：原始名称 -> 清理后名称
    column_mapping = {}
    column_name_changes = []
    
    for col in smiles_data.columns:
        col_lower = col.lower()
        if col_lower == 'smiles':
            column_mapping[col] = 'smiles'
        elif col_lower == 'target':
            column_mapping[col] = 'target'
        else:
            # 清理实验特征列名
            cleaned_name = clean_column_name(col)
            column_mapping[col] = cleaned_name
            if col != cleaned_name:
                column_name_changes.append(f"'{col}' -> '{cleaned_name}'")
    
    # 重命名列
    smiles_data_normalized = smiles_data.rename(columns=column_mapping)
    
    # 显示列名变更信息
    if column_name_changes:
        print(f"\n✓ Column names cleaned:")
        for change in column_name_changes:
            print(f"  {change}")
    
    # 处理重复的列名
    duplicated_cols = smiles_data_normalized.columns.duplicated()
    if duplicated_cols.any():
        print("Warning: Found duplicated column names after cleaning. Adding suffixes...")
        # 为重复的列名添加后缀
        cols = smiles_data_normalized.columns.tolist()
        for i in range(len(cols)):
            if duplicated_cols[i]:
                suffix = 1
                base_name = cols[i]
                while f"{base_name}_{suffix}" in cols:
                    suffix += 1
                cols[i] = f"{base_name}_{suffix}"
        smiles_data_normalized.columns = cols
    
    # 现在可以确定找到这两列
    smiles_col = 'smiles'
    target_col = 'target'
    
    print(f"\n✓ SMILES column: '{smiles_col}'")
    print(f"✓ Target column: '{target_col}'")
    
    # 识别其他实验特征列（除了SMILES和target之外的所有列）
    experimental_feature_cols = [col for col in smiles_data_normalized.columns 
                               if col not in [smiles_col, target_col]]
    
    if experimental_feature_cols:
        print(f"✓ Experimental feature columns found ({len(experimental_feature_cols)}):")
        for exp_col in experimental_feature_cols:
            print(f"  - {exp_col}")
    else:
        print("ℹ No additional experimental features found (only smiles and target columns)")

    # 验证数据完整性
    print(f"\nData validation:")
    print(f"Total samples: {len(smiles_data_normalized)}")
    
    # 检查SMILES列是否有空值
    smiles_null_count = smiles_data_normalized[smiles_col].isnull().sum()
    if smiles_null_count > 0:
        print(f"Warning: {smiles_null_count} samples have missing SMILES values")
    
    # 检查target列是否有空值
    target_null_count = smiles_data_normalized[target_col].isnull().sum()
    if target_null_count > 0:
        print(f"Warning: {target_null_count} samples have missing target values")

    # 检查实验特征列的空值情况
    if experimental_feature_cols:
        print("Experimental features null value check:")
        for exp_col in experimental_feature_cols:
            null_count = smiles_data_normalized[exp_col].isnull().sum()
            if null_count > 0:
                print(f"  - {exp_col}: {null_count} missing values")

    for idx, row in smiles_data_normalized.iterrows():
        sample_name = f"{idx+1:06d}"
        smiles_string = row[smiles_col]
        target_value = row[target_col]
        
        # 跳过SMILES为空的行
        if pd.isnull(smiles_string) or smiles_string == '':
            print(f"Skipping sample {sample_name}: empty SMILES")
            continue
        
        # 创建映射记录，包含所有信息
        mapping_record = {
            'Sample Name': sample_name,
            'smiles': smiles_string,
            'target': target_value
        }
        
        # 添加所有实验特征
        for exp_col in experimental_feature_cols:
            mapping_record[exp_col] = row[exp_col]
        
        mapping_data.append(mapping_record)
        
        # 检查是否已存在相关文件
        file_exists = False
        for ext in check_extensions:
            if os.path.exists(sample_name + ext):
                print(f"File {sample_name + ext} already exists, skipping conversion.")
                file_exists = True
                skip_count += 1
                break
        
        # 如果没有找到已存在的文件，进行转换
        if not file_exists:
            xyz_data, conversion_success = smiles_to_xyz(smiles_string, sample_name)
            
            if conversion_success:
                # Save XYZ file in current directory
                file_name = f"{sample_name}.xyz"
                with open(file_name, 'w') as file:
                    file.write(xyz_data)
                conversion_count += 1
                print(f"Converted {sample_name}: {smiles_string}")
            else:
                print(f"Failed to convert {sample_name}: {smiles_string}")

    # 保存映射文件
    if mapping_data:
        mapping_df = pd.DataFrame(mapping_data)
        mapping_df.to_csv('sample_mapping.csv', index=False)
        print(f"\n✓ Sample mapping file 'sample_mapping.csv' created successfully!")
        
        print(f"\nConversion Summary:")
        print(f"✓ Converted {conversion_count} new files")
        print(f"ℹ Skipped {skip_count} existing files")
        print(f"✓ Total samples in mapping: {len(mapping_data)}")
        
        # 显示映射文件的列信息
        print(f"✓ Mapping file columns: {list(mapping_df.columns)}")
        
        # 显示实验特征统计
        if experimental_feature_cols:
            print(f"✓ Experimental features included: {len(experimental_feature_cols)}")
            print("Final experimental feature names:")
            for exp_col in experimental_feature_cols:
                print(f"  - {exp_col}")
        else:
            print("ℹ No experimental features (only smiles and target)")
            
        # 保存列名映射记录
        if column_name_changes:
            mapping_record_file = 'column_name_mapping.log'
            with open(mapping_record_file, 'w') as f:
                f.write("Original Column Name -> Cleaned Column Name\n")
                f.write("=" * 50 + "\n")
                for change in column_name_changes:
                    f.write(change + "\n")
            print(f"✓ Column name mapping saved to '{mapping_record_file}'")
    else:
        print("Error: No valid samples found to process")


# 使用示例
file_path = 'Shenyan80.csv'
check_extensions = ['.fchk', '.xyz']  # 可以根据需要修改要检查的文件扩展名
smi2xyz_main(file_path, check_extensions)


# run_xtb_opt, optional

# In[3]:


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
    # Use current directory
    current_dir = Path('.')

    # Default xtb parameters if no param_file is provided
    default_xtb_options = "--opt tight --gfn 2 --molden --alpb water"
    
    # Read xtb parameters from the param_file if it exists, otherwise use default options
    if param_file and os.path.exists(param_file):
        with open(param_file, 'r') as f:
            xtb_options = f.read().strip()
    else:
        print(f"No parameters file provided or file does not exist. Using default parameters: {default_xtb_options}")
        xtb_options = default_xtb_options

    # Get all .xyz files in the current directory
    xyz_files = list(current_dir.glob("*.xyz"))

    if not xyz_files:
        print("No .xyz files found in current directory")
        return

    for xyz_file in xyz_files:
        base_name = xyz_file.stem  # Filename without extension
        output_xyz = current_dir / f"{base_name}.xyz"
        output_molden = current_dir / f"{base_name}.molden.input"
        output_log = current_dir / f"{base_name}.out"
        
        # Run xtb command
        xtb_command = f"xtb {xyz_file} {xtb_options}"
        print(f"Running: {xtb_command}")

        try:
            # Redirect the command output to a log file
            with open(output_log, 'w') as log_file:
                subprocess.run(xtb_command, shell=True, check=True, stdout=log_file, stderr=subprocess.STDOUT)

            # Move and overwrite optimized files
            molden_input = Path("molden.input")
            if molden_input.exists():
                molden_input.replace(output_molden)  # replace() will overwrite if file exists

            # Move and overwrite optimized xyz file
            xtbopt_xyz = Path("xtbopt.xyz")
            if xtbopt_xyz.exists():
                xtbopt_xyz.replace(output_xyz)  # replace() will overwrite if file exists
            else:
                print(f"Warning: xtbopt.xyz file not found for {xyz_file}")

            # Clean up any temporary files that xtb might have created
            temp_files = ['charges', 'wbo', 'xtbrestart', 'xtbtopo.mol']
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        except subprocess.CalledProcessError as e:
            print(f"Error processing {xyz_file}: {e}")
            continue

    print("All tasks completed!")

# 使用示例
# 不带参数文件运行
run_xtb_opt()

# 带参数文件运行
# run_xtb_opt("params.txt")


# see_xyz in this folder now(optional)

# In[5]:


import os
import glob
import py3Dmol
from IPython.display import display, clear_output
import ipywidgets as widgets
from ipywidgets import HBox, VBox

def read_xyz_file(filename):
    """读取xyz文件的内容"""
    with open(filename, 'r') as f:
        content = f.read()
    return content

def view_molecule(xyz_file, view_container):
    """使用py3Dmol库可视化分子结构，支持动态旋转"""
    # 读取XYZ文件内容
    xyz_content = read_xyz_file(xyz_file)
    
    # 创建可视化窗口，设置大小
    view = py3Dmol.view(width=800, height=500)
    
    # 添加分子结构
    view.addModel(xyz_content, 'xyz')
    
    # 设置样式 - 球棒模型
    view.setStyle({'stick': {'radius': 0.2, 'color': 'grey'},
                   'sphere': {'scale': 0.3}})
    
    # 添加标签显示文件名
    file_name = os.path.basename(xyz_file)
    view.addLabel(file_name, {'position': {'x': 0, 'y': 0, 'z': 0}, 
                             'backgroundColor': 'white', 
                             'fontColor': 'black',
                             'backgroundOpacity': 0.5,
                             'fontSize': 18,
                             'alignment': 'bottomRight'})
    
    # 自动缩放视图以适应分子大小
    view.zoomTo()
    
    # 清除视图容器并显示新的视图
    with view_container:
        clear_output(wait=True)
        display(view)
        print(f"正在显示: {file_name} ({current_index+1}/{len(xyz_files)})")
        print("提示: 可以使用鼠标拖动旋转分子，滚轮缩放")
        print("      按「下一个」按钮或按Enter键显示下一个分子")

# 获取当前目录下所有的xyz文件
current_dir = os.getcwd()
xyz_files = sorted(glob.glob(os.path.join(current_dir, "*.xyz")))

if not xyz_files:
    print("当前目录下未找到.xyz文件")
else:
    # 初始化当前索引
    current_index = 0
    
    # 创建下拉菜单以选择分子
    file_options = [(f"{i+1}. {os.path.basename(f)}", i) for i, f in enumerate(xyz_files)]
    file_dropdown = widgets.Dropdown(
        options=file_options,
        value=0,
        description='选择分子:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )
    
    # 创建按钮
    prev_button = widgets.Button(
        description='上一个',
        button_style='info',
        icon='arrow-left',
        layout=widgets.Layout(width='100px')
    )
    
    next_button = widgets.Button(
        description='下一个',
        button_style='info',
        icon='arrow-right',
        layout=widgets.Layout(width='100px')
    )
    
    # 创建状态显示
    status_label = widgets.Label(
        value=f'文件总数: {len(xyz_files)}'
    )
    
    # 创建输出区域
    view_container = widgets.Output()
    
    # 定义按钮回调函数
    def show_prev(b):
        global current_index
        if current_index > 0:
            current_index -= 1
            file_dropdown.value = current_index
            view_molecule(xyz_files[current_index], view_container)
    
    def show_next(b):
        global current_index
        if current_index < len(xyz_files) - 1:
            current_index += 1
            file_dropdown.value = current_index
            view_molecule(xyz_files[current_index], view_container)
    
    # 定义下拉菜单回调函数
    def on_dropdown_change(change):
        global current_index
        if change['type'] == 'change' and change['name'] == 'value':
            current_index = change['new']
            view_molecule(xyz_files[current_index], view_container)
    
    # 定义Enter键回调函数
    def on_enter(sender):
        global current_index
        show_next(None)
        sender.value = ''  # 清空输入框
    
    # 注册回调
    prev_button.on_click(show_prev)
    next_button.on_click(show_next)
    file_dropdown.observe(on_dropdown_change, names='value')

    
    # 创建控制面板
    controls = HBox([prev_button, next_button, file_dropdown, status_label])
    
    # 显示界面
    display(VBox([controls, view_container]))
    
    # 显示第一个分子
    view_molecule(xyz_files[current_index], view_container)


# xyz2gaussian

# In[7]:


import os
from pathlib import Path

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

# 使用示例
if __name__ == "__main__":
    # 用户只需要修改上面的这几个变量即可：
    # GAUSSIAN_KEYWORD_LINE = "# B3LYP/6-31g* opt"  # 例如改成这样
    # DEFAULT_CHARGE = 1
    # DEFAULT_MULTIPLICITY = 2
    
    xyz2gaussian_main()


# run_gaussian

# In[ ]:


import subprocess
import os
import glob
from pathlib import Path

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


run_gaussian()


# rerun_gaussian,if some chk fails

# In[1]:


import subprocess
import os
import glob
from pathlib import Path

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

if __name__ == '__main__':
    choice = input("Enter '1' for normal run, '2' for rerunning failed calculations: ").strip()
    
    if choice == '1':
        print('skip')
    elif choice == '2':
        rerun_failed_calculations()
    else:
        print("Invalid choice. Please enter either '1' or '2'.")


# readmMultiwfn

# In[10]:


import csv
import os
import glob
import subprocess
import logging
import time
import pandas as pd
import numpy as np
import re
from pathlib import Path
import warnings
from rdkit import Chem
import math

warnings.filterwarnings("ignore")

def extract_after(text, keyword):
    """
    Extracts text after the specified keyword.
    """
    partitioned_text = text.partition(keyword)
    if partitioned_text[1] == '':
        return None
    else:
        return partitioned_text[2].strip()

def extract_before(text, keyword):
    """
    Extracts text before the specified keyword.
    """
    partitioned_text = text.partition(keyword)
    if partitioned_text[1] == '':
        return text.strip()
    else:
        return partitioned_text[0].strip()

def extract_between(text, start_delimiter, end_delimiter):
    """
    Extracts text between two specified delimiters.
    """
    start_index = text.find(start_delimiter)
    if start_index == -1:
        return None
    start_index += len(start_delimiter)
    end_index = text.find(end_delimiter, start_index)
    if end_index == -1:
        return None
    return text[start_index:end_index].strip()

def get_atomic_mass(element):
    """
    获取元素的原子质量 (amu)    
    参数:
        element (str/int): 元素符号或原子序数                          
    返回:
        float: IUPAC 2016年推荐的标准原子量 (amu)
               未知元素返回碳的质量 (12.011 amu)
    """
    
    # IUPAC 2016年推荐标准原子量 (amu) - 完整前86个元素
    atomic_masses = {
        # 主族和过渡金属元素 (1-36)
        'H': 1.008, 'He': 4.003, 'Li': 6.94, 'Be': 9.012, 'B': 10.81, 'C': 12.011, 'N': 14.007, 'O': 15.999,
        'F': 18.998, 'Ne': 20.180, 'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.085, 'P': 30.974, 'S': 32.06,
        'Cl': 35.45, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078, 'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996,
        'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38, 'Ga': 69.723, 'Ge': 72.630,
        'As': 74.922, 'Se': 78.971, 'Br': 79.904, 'Kr': 83.798,
        
        # 第五周期元素 (37-54)
        'Rb': 85.468, 'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224, 'Nb': 92.906, 'Mo': 95.95, 'Tc': 98.0, 'Ru': 101.07,
        'Rh': 102.906, 'Pd': 106.42, 'Ag': 107.868, 'Cd': 112.414, 'In': 114.818, 'Sn': 118.710, 'Sb': 121.760,
        'Te': 127.60, 'I': 126.904, 'Xe': 131.293,
        
        # 第六周期元素和镧系 (55-86)
        'Cs': 132.905, 'Ba': 137.327, 'La': 138.905, 'Ce': 140.116, 'Pr': 140.908, 'Nd': 144.242, 'Pm': 145.0,
        'Sm': 150.36, 'Eu': 151.964, 'Gd': 157.25, 'Tb': 158.925, 'Dy': 162.500, 'Ho': 164.930, 'Er': 167.259,
        'Tm': 168.934, 'Yb': 173.045, 'Lu': 174.967, 'Hf': 178.49, 'Ta': 180.948, 'W': 183.84, 'Re': 186.207,
        'Os': 190.23, 'Ir': 192.217, 'Pt': 195.084, 'Au': 196.967, 'Hg': 200.592, 'Tl': 204.38, 'Pb': 207.2,
        'Bi': 208.980, 'Po': 209.0, 'At': 210.0, 'Rn': 222.0
    }
    return atomic_masses.get(element, 12.011)  # 默认返回碳的质量

def calculate_principal_moments_of_inertia(coords, masses):
    """
    计算分子的主惯性矩
    
    参数:
        coords: 原子坐标列表 [[x1,y1,z1], [x2,y2,z2], ...]
        masses: 原子质量列表 [m1, m2, ...]
    
    返回:
        eigenvalues: 排序后的主惯性矩 [I1, I2, I3] (I1 ≤ I2 ≤ I3)
        eigenvectors: 对应的特征向量矩阵
    
    物理意义:
        - 主惯性矩是分子转动惯量的主要分量
        - I1, I2, I3 分别对应最小、中等、最大主惯性矩
        - 用于描述分子的转动特性和形状各向异性
    """
    coords = np.array(coords)
    masses = np.array(masses)
    
    # 计算质心坐标
    center_of_mass = np.average(coords, weights=masses, axis=0)
    
    # 将坐标平移到质心参考系
    relative_coords = coords - center_of_mass
    
    # 计算惯性张量 3x3 矩阵
    I_tensor = np.zeros((3, 3))
    
    for i, mass in enumerate(masses):
        r = relative_coords[i]
        r_squared = np.dot(r, r)
        
        # 惯性张量对角元素 (Ixx, Iyy, Izz)
        I_tensor[0, 0] += mass * (r[1]**2 + r[2]**2)  # Ixx
        I_tensor[1, 1] += mass * (r[0]**2 + r[2]**2)  # Iyy
        I_tensor[2, 2] += mass * (r[0]**2 + r[1]**2)  # Izz
        
        # 惯性张量非对角元素 (Ixy, Ixz, Iyz)
        I_tensor[0, 1] -= mass * r[0] * r[1]  # Ixy
        I_tensor[0, 2] -= mass * r[0] * r[2]  # Ixz
        I_tensor[1, 2] -= mass * r[1] * r[2]  # Iyz
    
    # 对称化惯性张量
    I_tensor[1, 0] = I_tensor[0, 1]
    I_tensor[2, 0] = I_tensor[0, 2]
    I_tensor[2, 1] = I_tensor[1, 2]
    
    # 求解特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(I_tensor)
    
    # 按特征值大小排序 (从小到大)
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    return eigenvalues, eigenvectors

def calculate_asphericity(I1, I2, I3):
    """
    计算非球性指数 (Asphericity)
    
    参数:
        I1, I2, I3: 主惯性矩 (amu·Å²)
    
    返回:
        asphericity: 非球性指数 [0, 0.5]
    
    物理意义:
        - 测量分子形状偏离完美球形的程度
        - 0: 完美球形 (I1 = I2 = I3)
        - 0.5: 完美线性 (I1 = I2 = 0, I3 > 0)
        - 0.25: 完美盘状 (I1 = 0, I2 = I3)
    
    算法:
        Asphericity = 0.5 × [(I1-I2)² + (I1-I3)² + (I2-I3)²] / (I1² + I2² + I3²)
    
    出处:
        - Rudolph, J. et al. Chem. Phys. Lett. 1999, 309, 589
        - 广泛用于分子形状分析和分子动力学研究
    """
    if I1 < 0 or I2 < 0 or I3 < 0:
        return np.nan
    
    numerator = (I1 - I2)**2 + (I1 - I3)**2 + (I2 - I3)**2
    denominator = I1**2 + I2**2 + I3**2
    
    if denominator > 0:
        asphericity = 0.5 * numerator / denominator
    else:
        asphericity = 0.0
    
    return asphericity

def calculate_gyradius(coords, masses):
    """
    计算回转半径 (Radius of Gyration)
    
    参数:
        coords: 原子坐标列表 [[x1,y1,z1], [x2,y2,z2], ...]
        masses: 原子质量列表 [m1, m2, ...]
    
    返回:
        gyradius: 回转半径 (Å)
    
    物理意义:
        - 描述分子质量分布的紧凑程度
        - 等价于分子中所有原子相对于质心的均方根距离
        - 值越大表示分子越"分散"，值越小表示分子越"紧凑"
        - 常用于蛋白质折叠研究和聚合物链分析
    
    算法:
        Rg = √[Σ(mi × ri²) / Σ(mi)]
        其中 ri 是第i个原子到质心的距离
    
    出处:
        - Flory, P.J. Statistical Mechanics of Chain Molecules (1969)
        - 分子动力学模拟的标准几何参数
    """
    coords = np.array(coords)
    masses = np.array(masses)
    
    if len(coords) == 0 or len(masses) == 0:
        return np.nan
    
    # 计算质心
    total_mass = np.sum(masses)
    if total_mass <= 0:
        return np.nan
    
    center_of_mass = np.average(coords, weights=masses, axis=0)
    
    # 计算每个原子到质心的距离平方
    distances_squared = np.sum((coords - center_of_mass)**2, axis=1)
    
    # 计算质量加权的均方根距离
    weighted_sum = np.sum(masses * distances_squared)
    gyradius = np.sqrt(weighted_sum / total_mass)
    
    return gyradius

def calculate_relative_gyradius(gyradius, length_short, length_medium, length_long):
    """
    计算相对回转半径 (Relative Gyradius Ratio, Relative_Gyradius)
    
    参数:
        gyradius: 实际回转半径 (Å)
        length_short, length_medium, length_long: 分子三个方向的尺寸 (Å)
    
    返回:
        rgr: 相对回转半径 [通常 0.3-1.5]
    
    物理意义:
        - 实际回转半径与等体积球形分子理论回转半径的比值
        - < 1: 比理论球形更紧凑 (如线性分子)
        - = 1: 接近理论球形
        - > 1: 比理论球形更分散 (如伸展的分子)
        - 提供了尺寸无关的紧凑度指标
    
    算法:
        1. V_box = length_short × length_medium × length_long
        2. R_equiv = ∛(3×V_box / 4π)  # 等体积球半径
        3. Rg_sphere = √(3/5) × R_equiv  # 理论球形回转半径
        4. Relative_Gyradius = Rg_actual / Rg_sphere
    
    出处:
        - 基于聚合物物理学中的标准方法
        - 常用于蛋白质构象分析
    """
    if (gyradius <= 0 or length_short <= 0 or 
        length_medium <= 0 or length_long <= 0):
        return np.nan
    
    # 计算包围盒体积
    volume_box = length_short * length_medium * length_long
    
    # 计算等体积球的半径
    radius_equiv = (3 * volume_box / (4 * np.pi))**(1/3)
    
    # 计算理论球形分子的回转半径
    # 对于均匀密度球体: Rg = √(3/5) × R
    gyradius_sphere = np.sqrt(3/5) * radius_equiv
    
    if gyradius_sphere > 0:
        Relative_Gyradius = gyradius / gyradius_sphere
    else:
        Relative_Gyradius = np.nan
    
    return Relative_Gyradius

def calculate_waist_variance(coords, masses):
    """
    计算腰围变化方差 (Waist Variance)
    
    参数:
        coords: 原子坐标列表 [[x1,y1,z1], [x2,y2,z2], ...]
        masses: 原子质量列表 [m1, m2, ...]
    
    返回:
        waist_variance: 腰围变化方差 (Å²)
    
    物理意义:
        - 描述分子沿主轴方向的截面尺寸变化程度
        - 高值: 分子形状变化大，如哑铃型、梭形
        - 低值: 分子形状均匀，如圆柱型、球形
        - 0: 完美均匀的柱状或球状分子
        - 特别适用于检测具有"颈缩"特征的分子结构
    
    算法:
        1. 计算分子主轴方向（最大惯性矩对应轴）
        2. 沿主轴方向将分子分为n个切片
        3. 计算每个切片垂直于主轴的最大跨度
        4. 计算这些跨度的方差
    
    形状特征:
        - 哑铃型分子: 高方差 (>1.0)
        - 均匀棒状: 低方差 (~0)
        - 球形分子: 低方差 (~0)
        - 梭形分子: 中等方差 (0.1-1.0)
    
    出处:
        - 基于分子几何学和计算化学的形状分析方法
        - 灵感来源于蛋白质结构分析中的"瓶颈"检测
    """
    coords = np.array(coords)
    masses = np.array(masses)
    
    if len(coords) < 2:
        return 0.0
    
    try:
        # 计算主惯性矩和主轴
        eigenvalues, eigenvectors = calculate_principal_moments_of_inertia(coords, masses)
        
        # 选择最大惯性矩对应的主轴（通常是最长的分子轴）
        principal_axis = eigenvectors[:, -1]  # 最后一列对应最大特征值
        
        # 将分子坐标投影到主轴上
        projections = np.dot(coords, principal_axis)
        min_proj, max_proj = np.min(projections), np.max(projections)
        
        if max_proj == min_proj:
            return 0.0
        
        # 将分子沿主轴分成多个切片
        n_slices = 20
        slice_positions = np.linspace(min_proj, max_proj, n_slices + 1)
        slice_widths = []
        
        for i in range(n_slices):
            # 找到位于当前切片中的原子
            in_slice = ((projections >= slice_positions[i]) & 
                       (projections < slice_positions[i + 1]))
            
            if not np.any(in_slice):
                slice_widths.append(0.0)
                continue
            
            slice_coords = coords[in_slice]
            
            # 计算切片中原子坐标垂直于主轴的投影
            # 即移除主轴方向的分量
            perp_coords = slice_coords - np.outer(
                np.dot(slice_coords, principal_axis), principal_axis)
            
            # 计算切片的最大跨度（直径）
            if len(perp_coords) > 0:
                distances = np.linalg.norm(perp_coords, axis=1)
                max_distance = np.max(distances) if len(distances) > 0 else 0.0
                slice_widths.append(max_distance * 2)  # 直径 = 2 × 半径
            else:
                slice_widths.append(0.0)
        
        slice_widths = np.array(slice_widths)
        
        # 计算方差
        if len(slice_widths) > 0 and np.max(slice_widths) > 0:
            waist_variance = np.var(slice_widths)
        else:
            waist_variance = 0.0
            
        return waist_variance
        
    except Exception as e:
        print(f"Error calculating waist variance: {e}")
        return np.nan

def calculate_geometric_asphericity(length_short, length_medium, length_long):
    """
    计算几何非球性指数 (Geometric Asphericity)
    
    参数:
        length_short, length_medium, length_long: 分子包围盒的三个维度 (Å)
    
    返回:
        geometric_asphericity: 几何非球性指数 [0, 0.5]
    
    物理意义:
        - 基于几何尺寸的非球性度量，类比惯性矩的Asphericity
        - 描述分子包围盒偏离立方体的程度
        - 0: 完美立方体 (所有边长相等)
        - 0.5: 完美线性 (两个维度极小，一个维度很大)
        - 0.25: 完美盘状 (一个维度极小，两个维度相等)
    
    算法:
        GA = 0.5 × [(Ls-Lm)² + (Ls-Ll)² + (Lm-Ll)²] / (Ls² + Lm² + Ll²)
        其中 Ls, Lm, Ll 分别为短、中、长轴尺寸
    
    形状特征:
        - 球形/立方体: GA ≈ 0
        - 线性分子: GA ≈ 0.5
        - 盘状分子: GA ≈ 0.25
        - 椭球分子: GA = 0.1-0.4
    
    出处:
        - 类比分子惯性矩非球性的几何版本
        - 提供了独立于质量分布的形状度量
    """
    if (length_short <= 0 or length_medium <= 0 or length_long <= 0):
        return np.nan
    
    Ls, Lm, Ll = length_short, length_medium, length_long
    
    # 确保按大小排序
    lengths = sorted([Ls, Lm, Ll])
    Ls, Lm, Ll = lengths[0], lengths[1], lengths[2]
    
    numerator = (Ls - Lm)**2 + (Ls - Ll)**2 + (Lm - Ll)**2
    denominator = Ls**2 + Lm**2 + Ll**2
    
    if denominator > 0:
        geometric_asphericity = 0.5 * numerator / denominator
    else:
        geometric_asphericity = 0.0
    
    return geometric_asphericity

def create_descriptors_content():
    """
    生成固定的Descriptors输入内容，不再需要传入fragment_indices。
    去除了与"12 {indices_str}"相关的行。
    """
    content = f"""0
100
21

size
0
MPP
a
n
q
0
300
5
0
8
8
1
h-1
h
l
l+1
0
-10
12
2
-4
1
1
0.01
0
11
n
-1
2
1
1
1
0.01
3
0.2
0
11
n
-1
2
2
1
1
0.01
0
11
n
-1

"""
    return content

def run_multiwfn_on_fchk_files(input_path='.',):
    """
    不再需要从 first_matches 读取分子碎片索引，统一使用固定描述符输入。
    """
    original_dir = os.getcwd()
    os.chdir(input_path)
    fchk_files = sorted(glob.glob('*.fchk'))  # Sort the list of fchk files
    processed_files = []

    for fchk_file in fchk_files:
        sample_name = os.path.splitext(fchk_file)[0]  # e.g., '003' if fchk_file is '003.fchk'
        xyz_file = sample_name + '.xyz'
        output_file = f"{sample_name}.txt"

        # 如果输出文件已存在，则跳过
        if os.path.exists(output_file):
            logging.info(f"Output file {output_file} already exists. Skipping calculation.")
            processed_files.append(output_file)
            continue

        # 统一使用固定的 descriptors_content
        descriptors_content = create_descriptors_content()
        print(f"Descriptors content for {sample_name}:\n{descriptors_content}")

        # 构造命令
        command = ["Multiwfn_noGUI", fchk_file, "-silent"]

        try:
            with open(output_file, 'w') as outfile:
                subprocess.run(
                    command,
                    input=descriptors_content,
                    text=True,
                    stdout=outfile,
                    stderr=subprocess.PIPE,
                    check=True
                )
            logging.info(f"Multiwfn output saved to {output_file}")
            processed_files.append(output_file)
        except subprocess.CalledProcessError as e:
            logging.warning(f"Warning: An error occurred while running Multiwfn on {fchk_file}: {e}")
            logging.warning(f"stderr: {e.stderr}")

        if os.path.exists(output_file):
            if output_file not in processed_files:
                processed_files.append(output_file)
                logging.info(f"Output file {output_file} was created. Good!")
        else:
            logging.error(f"Error: Output file {output_file} was not created for {fchk_file}.")

    os.chdir(original_dir)
    return processed_files

def align_with_mapping_simple(DIR, mapping_file='sample_mapping.csv'):
    """
    直接使用映射文件和RawFull文件进行对齐，自动包含所有实验特征
    """
    try:
        # 读取映射文件
        mapping_df = pd.read_csv(mapping_file)
        print(f"Successfully loaded mapping file with {len(mapping_df)} samples")
        print(f"Mapping file columns: {list(mapping_df.columns)}")
        
        # 读取RawFull文件
        rawfull_files = [f for f in os.listdir(DIR) if f.startswith('RawFull_') and f.endswith('.csv')]
        if not rawfull_files:
            print("No RawFull_*.csv file found.")
            return None
            
        rawfull_file = rawfull_files[0]  # 取第一个RawFull文件
        rawfull_path = Path(DIR, rawfull_file)
        rawfull_df = pd.read_csv(rawfull_path)
        print(f"Successfully loaded RawFull file with {len(rawfull_df)} samples")
        
        # 确保Sample Name列的数据类型一致
        mapping_df['Sample Name'] = mapping_df['Sample Name'].astype(str)
        rawfull_df['Sample Name'] = rawfull_df['Sample Name'].astype(str)
        
        # 以mapping_df为基准进行左连接
        merged_df = pd.merge(mapping_df, rawfull_df, on='Sample Name', how='left')
        
        # 动态识别列类型
        calc_cols = [col for col in rawfull_df.columns if col != 'Sample Name']
        basic_cols = ['Sample Name', 'smiles', 'target']
        exp_feature_cols = [col for col in mapping_df.columns 
                           if col not in basic_cols]
        
        # 重新排列列顺序：Sample Name, 所有计算属性, SMILES, target, 所有实验特征
        final_cols = ['Sample Name'] + calc_cols + ['smiles', 'target'] + exp_feature_cols
        
        # 确保所有列都存在于merged_df中
        available_cols = [col for col in final_cols if col in merged_df.columns]
        merged_df = merged_df[available_cols]
        
        print(f"Final merged dataset shape: {merged_df.shape}")
        print(f"Columns: {merged_df.columns.tolist()}")
        
        # 显示数据类型统计
        print(f"Computational features: {len(calc_cols)}")
        print(f"Experimental features: {len(exp_feature_cols)}")
        if exp_feature_cols:
            print(f"Experimental feature names: {exp_feature_cols}")
        
        return merged_df
        
    except Exception as e:
        print(f"Error in simple alignment: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_txt_files(input_directory, output_directory):
    """
    处理Multiwfn生成的txt文件，提取分子描述符
    新增：形状描述符计算 (Asphericity, Gyradius, Relative_Gyradius, Waist_Variance, Geometric_Asphericity)
    """
    Version = '3.0'
    c_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    DIR = os.path.join(output_directory, f'Surfacia_{Version}_{c_time}')
    os.makedirs(DIR, exist_ok=True)

    current_directory = input_directory
    file_list = [f for f in os.listdir(current_directory) if f.endswith('.txt')]

    print("Found text files:", file_list)
    df = pd.DataFrame()

    for filename in file_list:
        data = {}
        titles = []
        with open(os.path.join(input_directory, filename), 'r') as file:
            lines_iter = iter(file.readlines())

        sample_name = filename[:-4]
        print('Processing sample:', sample_name)
        data['Sample Name'] = sample_name
        matrix = {}
        odi_values = []
        summary_count = 0  # Counts the number of Summary sections encountered
        in_fragment_section = False
        
        # 定义用于形状描述符计算的变量
        atom_num = None
        xyz = []
        atoms = []
        principal_moments = None  # 存储主惯性矩
        molecular_weight = None
        mol_size = None  # 存储分子尺寸

        while True:
            try:
                line = next(lines_iter)
                
                # Basic Information Extraction
                if 'Atoms:' in line:
                    atom_num = int(extract_between(line, "Atoms: ", ","))
                    data['Atom Number'] = atom_num
                    
                if 'Molecule weight:' in line:
                    weight = float(extract_between(line, "Molecule weight:", "Da"))
                    data['Molecule Weight'] = weight
                    molecular_weight = weight
                    
                if 'Orbitals from 1 to' in line:
                    occupied_orbitals = float(extract_between(line, "Orbitals from 1 to", "are occupied"))
                    data['Occupied Orbitals'] = occupied_orbitals
                    
                # ========== 新增：提取原子坐标和类型用于形状描述符计算 ==========

                if 'Atom list:' in line:
                    atoms = []
                    xyz = []
                    for _ in range(atom_num):
                        line = next(lines_iter).strip()
                        if line:
                            atom_info = line.split()
                            # 使用旧代码的提取方式
                            atom = atom_info[0].split('(')[1].split(')')[0]
                            x, y, z = map(float, atom_info[-3:])
                            atoms.append(atom)
                            xyz.append([x, y, z])
                        else:
                            break
                
                # ========== 新增：提取主惯性矩信息 ==========
                if 'The moments of inertia relative to principal axes' in line:
                    try:
                        # 读取下一行获取主惯性矩数值
                        pmi_line = next(lines_iter)
                        pmi_match = re.search(r'([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.E+-]+)', pmi_line)
                        if pmi_match:
                            moments = [float(x) for x in pmi_match.groups()]
                            moments.sort()  # 确保 I1 ≤ I2 ≤ I3
                            principal_moments = moments
                            data['Principal_Moment_I1'] = moments[0]
                            data['Principal_Moment_I2'] = moments[1]
                            data['Principal_Moment_I3'] = moments[2]
                            print(f"  提取主惯性矩: I1={moments[0]:.3e}, I2={moments[1]:.3e}, I3={moments[2]:.3e}")
                    except (StopIteration, ValueError, AttributeError) as e:
                        print(f"  警告：无法提取主惯性矩: {e}")
                        
                # HOMO/LUMO Energy Levels
#                 if 'is HOMO, energy:' in line:
                if 'is alpha-HOMO, energy:' in line or 'is HOMO, energy:' in line:
                    if 'is alpha-HOMO, energy:' in line:
                        homo_energy = extract_between(line, "is alpha-HOMO, energy:", "a.u.")
                    else:
                        homo_energy = extract_between(line, "is HOMO, energy:", "a.u.")
                    if homo_energy is not None:
                        data['HOMO'] = float(homo_energy)

                if 'is alpha-LUMO, energy:' in line or 'is LUMO, energy:' in line:
                    if 'is alpha-LUMO, energy:' in line:
                        lumo_energy = extract_between(line, "is alpha-LUMO, energy:", "a.u.")
                    else:
                        lumo_energy = extract_between(line, "is LUMO, energy:", "a.u.")
                    if lumo_energy is not None:
                        data['LUMO'] = float(lumo_energy)

                if 'HOMO-LUMO gap of alpha orbitals:' in line or 'HOMO-LUMO gap:' in line:
                    if 'HOMO-LUMO gap of alpha orbitals:' in line:
                        gap_energy = extract_between(line, "HOMO-LUMO gap of alpha orbitals:", "a.u.")
                    else:
                        gap_energy = extract_between(line, "gap:", "a.u.")
                    if gap_energy is not None:
                        data['HOMO-LUMO Gap'] = float(gap_energy)
                    
                # Molecular Shape
                if 'Farthest distance:' in line:
                    farthest_distance = float(extract_between(line, "):", "Angstrom"))
                    data['Farthest Distance'] = farthest_distance
                if 'Radius of the system: ' in line:
                    mol_radius = float(extract_between(line, ":", "Angstrom"))
                    data['Molecular Radius'] = mol_radius
                    
                # ========== 修改：提取分子尺寸用于几何描述符计算 ==========
                if 'Length of the three sides:' in line:
                    mol_size = list(map(float, extract_between(line, ":", "Angstrom").split()))
                    mol_size.sort()  # 排序为 [短, 中, 长]
                    data['Molecular Size Short'] = mol_size[0]
                    data['Molecular Size Medium'] = mol_size[1]
                    data['Molecular Size Long'] = mol_size[2]
                    data['Long/Sum Size Ratio'] = mol_size[2] / sum(mol_size)
                    data['Length/Diameter'] = mol_size[2] / (2 * mol_radius)
                    print(f"  提取分子尺寸: {mol_size[0]:.3f} × {mol_size[1]:.3f} × {mol_size[2]:.3f} Å")
                    
                if 'Molecular planarity parameter (MPP) is' in line:
                    mpp = float(extract_before(extract_after(line, "is"), "Angstrom"))
                    data['MPP'] = mpp
                if 'Span of deviation from plane' in line:
                    sdp = float(extract_before(extract_after(line, "is"), "Angstrom"))
                    data['SDP'] = sdp
                    
                # Dipole Moment
                if 'Magnitude of dipole moment:' in line:
                    dipole_moment = float(extract_between(line, 'Magnitude of dipole moment:', "a.u."))
                    data['Dipole Moment (a.u.)'] = dipole_moment
#                 if 'Magnitude: |Q_2|=' in line:
#                     quadrupole_moment = float(extract_after(line, "Magnitude: |Q_2|="))
#                     data['Quadrupole Moment'] = quadrupole_moment
#                 if 'Magnitude: |Q_3|=' in line:
#                     octopole_moment = float(extract_after(line, "|Q_3|= "))
#                     data['Octopole Moment'] = octopole_moment
                    
                # ODI Index
                if 'Orbital delocalization index:' in line:
                    odi_value = float(extract_after(line, "index:"))
                    odi_values.append(odi_value)
                    data['ODI LUMO+1'] = odi_values[0] if len(odi_values) > 0 else None
                    data['ODI LUMO'] = odi_values[1] if len(odi_values) > 1 else None
                    data['ODI HOMO'] = odi_values[2] if len(odi_values) > 2 else None
                    data['ODI HOMO-1'] = odi_values[3] if len(odi_values) > 3 else None
                    data['ODI Mean'] = np.mean(odi_values) if odi_values else None
                    data['ODI Std'] = np.std(odi_values) if odi_values else None
                if 'Isosurface area:' in line:
                    isosurface_area = float(extract_between(line, 'Bohr^2  (', "Angstrom^2)"))
                    data['Isosurface area'] = isosurface_area
                if 'Sphericity:' in line:
                    sphericity = float(extract_after(line, "Sphericity:"))
                    data['Sphericity'] = sphericity
                    
                # LEAE, ESP, ALIE Sections [保持原有代码不变]
                try:
                    if '================= Summary of surface analysis =================' in line:
                        summary_count += 1
                        print('Summary of surface analysis', summary_count)

                        if summary_count == 1:
                            # LEAE Section
                            while True:
                                line = next(lines_iter)

                                # 仅当 LEAE 行内有 "Minimal value:" 并且带 'eV' 才解析
                                if 'Minimal value:' in line:
                                    # 如果这一行里有 'kcal/mol'，那说明已经切到 ESP 了，跳出
                                    if 'kcal/mol' in line:
                                        print("Detected 'kcal/mol' in LEAE block, skip to next summary_count.")
                                        break

                                    # 如果这一行有 'eV'
                                    if 'eV' in line:
                                        # Extract molecule's minimal and maximal values
                                        data['LEAE Minimal Value'] = float(extract_between(
                                            line, "Minimal value:", 'eV,   Maximal value:'
                                        ))
                                        data['LEAE Maximal Value'] = float(extract_between(
                                            line, "eV,   Maximal value:", 'eV'
                                        ))

                                elif 'Overall average value:' in line:
                                    data['LEAE Average Value'] = float(
                                        extract_between(line, "a.u. (", 'eV')
                                    )

                                elif 'Variance:' in line:
                                    data['LEAE Variance'] = float(
                                        extract_between(line, "a.u.^2  (", 'eV')
                                    )

                                if 'Note: Below minimal and maximal values are in eV' in line:
                                    next(lines_iter)  # Skip note line
                                    matrix_data = []
                                    for _ in range(atom_num):
                                        matrix_line = next(lines_iter).strip()
                                        if matrix_line:
                                            matrix_data.append(matrix_line)
                                        else:
                                            break
                                    next(lines_iter)  # Skip empty line
                                    line = next(lines_iter)
                                    if 'Note: Average and variance below are in eV and eV^2 respectively' in line:
                                        next(lines_iter)  # Skip note line
                                    matrix_data2 = []
                                    for _ in range(atom_num):
                                        matrix_line = next(lines_iter).strip()
                                        if matrix_line:
                                            matrix_data2.append(matrix_line)
                                        else:
                                            break
                                    num_rows = min(len(matrix_data), len(matrix_data2))
                                    data['Surface exposed atom num'] = num_rows
                                    for i in range(num_rows):
                                        matrix_data[i] += ' ' + ' '.join(matrix_data2[i].split()[-6:])
                                    matrix['Matrix Data'] = matrix_data
                                if '==========' in line:
                                    # 说明进入下一个 summary
                                    break
                        
                        elif summary_count == 2:
                            # ESP Section
                            while True:
                                line = next(lines_iter)

                                # 如果这一行是 Minimal value:
                                if 'Minimal value:' in line:
                                    # 判断是否含 "kcal/mol"
                                    if 'kcal/mol' in line:
                                        # 用原有的 kcal/mol 解析
                                        min_str = extract_between(
                                            line, "Minimal value:", "kcal/mol   Maximal value:"
                                        )
                                        max_str = extract_between(
                                            line, "kcal/mol   Maximal value:", "kcal/mol"
                                        )
                                        # 若解析成功
                                        if min_str and max_str:
                                            data['ESP Minimal Value'] = float(min_str)
                                            data['ESP Maximal Value'] = float(max_str)
                                        else:
                                            # 如果没提取到，说明格式不对 -> 给 NaN 并可视情况 break / continue
                                            data['ESP Minimal Value'] = np.nan
                                            data['ESP Maximal Value'] = np.nan
                                    elif 'eV' in line:
                                        # 说明这一行并不是 ESP 的最小最大值，而是可能又回到了 LEAE / ALIE
                                        # 这里可直接 break 或 continue
                                        break
                                elif 'Overall average value:' in line:
                                    data['ESP Overall Average Value (kcal/mol)'] = float(extract_between(line, "a.u. (", 'kcal/mol)'))
                                elif 'Overall variance (sigma^2_tot):' in line:
                                    data['ESP Overall Variance ((kcal/mol)^2)'] = float(extract_between(line, "a.u.^2 (", '(kcal/mol)^2)'))
                                elif 'Balance of charges (nu):' in line:
                                    data['Balance of Charges (nu)'] = float(extract_after(line, "Balance of charges (nu):"))
                                elif 'Product of sigma^2_tot and nu:' in line:
                                    data['Product of sigma^2_tot and nu ((kcal/mol)^2)'] = float(extract_between(line, "a.u.^2 (", '(kcal/mol)^2)'))
                                elif 'Internal charge separation (Pi):' in line:
                                    data['Internal Charge Separation (Pi) (kcal/mol)'] = float(extract_between(line, "a.u. (", 'kcal/mol)'))
                                elif 'Molecular polarity index (MPI):' in line:
                                    data['Molecular Polarity Index (MPI) (kcal/mol)'] = float(extract_between(line, "eV (", 'kcal/mol)'))
                                elif 'Polar surface area (|ESP| > 10 kcal/mol):' in line:
                                    data['Polar Surface Area (Angstrom^2)'] = float(extract_between(line, "Polar surface area (|ESP| > 10 kcal/mol):", 'Angstrom^2'))
                                    data['Polar Surface Area (%)'] = float(extract_between(line, "Angstrom^2  (", '%)'))
                                if 'Note: Minimal and maximal value below are in kcal/mol' in line:
                                    next(lines_iter)  # Skip note line
                                    matrix_data3 = []
                                    #print(num_rows)
                                    for _ in range(atom_num):
                                        matrix_line = next(lines_iter).strip()
                                        if matrix_line:
                                            matrix_data3.append(matrix_line)
                                        else:
                                            break
                                    next(lines_iter)  # Skip empty line
                                    line = next(lines_iter)
                                    if 'Note: Average and variance below are in' in line:
                                        next(lines_iter)  # Skip note line
                                    matrix_data4 = []
                                    for _ in range(atom_num):
                                        matrix_line = next(lines_iter).strip()
                                        if matrix_line:
                                            matrix_data4.append(matrix_line)
                                        else:
                                            break
                                    next(lines_iter)  # Skip another line
                                    line = next(lines_iter)
                                    if 'Note: Internal charge separation' in line:
                                        next(lines_iter)  # Skip note line
                                    matrix_data5 = []
                                    for _ in range(atom_num):
                                        matrix_line = next(lines_iter).strip()
                                        if matrix_line:
                                            matrix_data5.append(matrix_line)
                                        else:
                                            break
                                    for i in range(num_rows):
                                        #print(' ' + ' '.join(matrix_data3[i].split()[-5:]) + ' ' + ' '.join(matrix_data4[i].split()[-6:]) + ' ' + ' '.join(matrix_data5[i].split()[-3:]))
                                        matrix_data[i] += ' ' + ' '.join(matrix_data3[i].split()[-5:]) + ' ' + ' '.join(matrix_data4[i].split()[-6:]) + ' ' + ' '.join(matrix_data5[i].split()[-3:])
                                    matrix['Matrix Data'] = matrix_data
                                if '==========' in line:
                                    break
                        elif summary_count == 3:
                            # ALIE Section
                            while True:
                                line = next(lines_iter)
                                if 'Minimal value:' in line:
                                    data['ALIE Minimal Value'] = float(extract_between(line, "Minimal value:", 'eV,   Maximal value:'))
                                    data['ALIE Maximal Value'] = float(extract_between(line, "eV,   Maximal value:", 'eV'))
                                elif 'Average value:' in line:
                                    data['ALIE Average Value'] = float(extract_between(line, "a.u. (", 'eV'))
                                elif 'Variance:' in line:
                                    data['ALIE Variance'] = float(extract_between(line, "a.u.^2  (", 'eV'))
                                if 'Minimal, maximal and average value are in eV, variance is in eV^2' in line:
                                    next(lines_iter)  # Skip note line
                                    matrix_data6 = []
                                    for _ in range(atom_num):
                                        matrix_line = next(lines_iter).strip()
                                        if matrix_line:
                                            matrix_data6.append(matrix_line)
                                        else:
                                            break
                                    titles = [
                                        "Atom#", "LEAE All area", "LEAE Positive area", "LEAE Negative area",
                                        "LEAE Minimal value", "LEAE Maximal value", "LEAE All average", "LEAE Positive average",
                                        "LEAE Negative average", "LEAE All variance", "LEAE Positive variance", "LEAE Negative variance",
                                        "ESP All area (Ang^2)", "ESP Positive area (Ang^2)", "ESP Negative area (Ang^2)",
                                        "ESP Minimal value (kcal/mol)", "ESP Maximal value (kcal/mol)",
                                        "ESP All average (kcal/mol)", "ESP Positive average (kcal/mol)", "ESP Negative average (kcal/mol)",
                                        "ESP All variance (kcal/mol)^2", "ESP Positive variance (kcal/mol)^2", "ESP Negative variance (kcal/mol)^2",
                                        "ESP Pi (kcal/mol)", "ESP nu", "ESP nu*sigma^2",
                                        "ALIE Area(Ang^2)", "ALIE Min value", "ALIE Max value", "ALIE Average", "ALIE Variance"
                                    ]
                                    for i in range(num_rows):
                                        matrix_data[i] += ' ' + ' '.join(matrix_data6[i].split()[-5:])
                                    matrix['Matrix Data'] = matrix_data

                except ValueError as e:
                    continue
                    
            except StopIteration:
                break
        
        # ========== 新增：计算形状描述符 ==========
        print(f"  开始计算形状描述符...")
        
        # 检查必要数据是否可用
        if len(xyz) > 0 and len(atoms) > 0:
            try:
                # 获取原子质量
                masses = [get_atomic_mass(atom) for atom in atoms]
                
                # 1. 计算回转半径 (Gyradius)
                gyradius = calculate_gyradius(xyz, masses)
                data['Shape_Gyradius'] = gyradius
                print(f"    回转半径: {gyradius:.4f} Å")
                
                # 2. 计算非球性指数 (需要主惯性矩)
                if principal_moments is not None:
                    I1, I2, I3 = principal_moments
                    asphericity = calculate_asphericity(I1, I2, I3)
                    data['Shape_Asphericity'] = asphericity
                    print(f"    非球性指数: {asphericity:.4f}")
                else:
                    # 如果没有从Multiwfn提取到，则自己计算
                    try:
                        moments, _ = calculate_principal_moments_of_inertia(xyz, masses)
                        I1, I2, I3 = moments
                        asphericity = calculate_asphericity(I1, I2, I3)
                        data['Shape_Asphericity'] = asphericity
                        data['Principal_Moment_I1'] = I1
                        data['Principal_Moment_I2'] = I2 
                        data['Principal_Moment_I3'] = I3
                        print(f"    计算得到主惯性矩和非球性指数: {asphericity:.4f}")
                    except Exception as e:
                        print(f"    警告：无法计算主惯性矩: {e}")
                        data['Shape_Asphericity'] = np.nan
                
                # 3. 计算相对回转半径 (需要分子尺寸)
                if mol_size is not None and len(mol_size) >= 3:
                    Relative_Gyradius = calculate_relative_gyradius(gyradius, mol_size[0], mol_size[1], mol_size[2])
                    data['Shape_Relative_Gyradius'] = Relative_Gyradius
                    print(f"    相对回转半径: {Relative_Gyradius:.4f}")
                else:
                    data['Shape_Relative_Gyradius'] = np.nan
                    print(f"    警告：缺少分子尺寸信息，无法计算RGR")
                
                # 4. 计算腰围变化方差
                waist_variance = calculate_waist_variance(xyz, masses)
                data['Shape_Waist_Variance'] = waist_variance
                print(f"    腰围变化方差: {waist_variance:.4f} Å²")
                
                # 5. 计算几何非球性指数 (需要分子尺寸)
                if mol_size is not None and len(mol_size) >= 3:
                    geometric_asphericity = calculate_geometric_asphericity(
                        mol_size[0], mol_size[1], mol_size[2])
                    data['Shape_Geometric_Asphericity'] = geometric_asphericity
                    print(f"    几何非球性指数: {geometric_asphericity:.4f}")
                else:
                    data['Shape_Geometric_Asphericity'] = np.nan
                    print(f"    警告：缺少分子尺寸信息，无法计算几何非球性")
                    
            except Exception as e:
                print(f"    错误：计算形状描述符时出现异常: {e}")
                # 设置默认的NaN值
                data['Shape_Gyradius'] = np.nan
                data['Shape_Asphericity'] = np.nan
                data['Shape_Relative_Gyradius'] = np.nan
                data['Shape_Waist_Variance'] = np.nan
                data['Shape_Geometric_Asphericity'] = np.nan
        else:
            print(f"    警告：缺少原子坐标信息，无法计算形状描述符")
            # 设置默认的NaN值
            data['Shape_Gyradius'] = np.nan
            data['Shape_Asphericity'] = np.nan
            data['Shape_Relative_Gyradius'] = np.nan
            data['Shape_Waist_Variance'] = np.nan
            data['Shape_Geometric_Asphericity'] = np.nan
        
        # 添加数据到DataFrame
        temp_df = pd.DataFrame([data])
        df = pd.concat([df, temp_df], ignore_index=True)

        # 如果有 matrix_data，则将其输出为 AtomProp_*.csv
        if 'Matrix Data' in matrix:
            matrix_data = matrix['Matrix Data']
            new_filename = 'AtomProp_' + sample_name + '.csv'
            output_filename = Path(DIR, new_filename)

            max_index = max(int(row.split()[0]) for row in matrix_data)
            merged_data = [['NaN'] * (len(matrix_data[0].split()) + len(xyz[0])) for _ in range(max_index)]
            num_xyz_columns = len(xyz[0])
            num_matrix_columns = len(matrix_data[0].split())

            for row in matrix_data:
                parts = row.split()
                index = int(parts[0]) - 1  # Convert 1-based index to 0-based index
                if index < len(xyz):
                    merged_data[index] = xyz[index] + parts[:]

            while len(merged_data) < len(xyz):
                merged_data.append(['NaN'] * (num_xyz_columns + num_matrix_columns))

            for i in range(len(xyz)):
                if len(merged_data[i]) > num_xyz_columns:
                    if merged_data[i][num_xyz_columns] == 'NaN':
                        merged_data[i] = xyz[i] + ['NaN'] * num_matrix_columns
                else:
                    additional_nans = ['NaN'] * (num_xyz_columns + 1 - len(merged_data[i]))
                    merged_data[i] = merged_data[i] + additional_nans
                    if merged_data[i][num_xyz_columns] == 'NaN':
                        merged_data[i] = xyz[i] + ['NaN'] * num_matrix_columns

            for i in range(len(merged_data)):
                if i < len(atoms):
                    merged_data[i].insert(0, atoms[i])
                else:
                    merged_data[i].insert(0, 'NaN')

            title_parts = ['Element', 'X', 'Y', 'Z'] + titles
            merged_data.insert(0, title_parts)

            with open(output_filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(merged_data)
            print(f"Data successfully written to {output_filename}")
        else:
            # 如果没有 matrix_data，写一个空的占位 CSV 文件
            new_filename = 'AtomProp_' + sample_name + '.csv'
            output_filename = Path(DIR, new_filename)
            with open(output_filename, 'w', newline='') as file:
                pass

    # 保存RawFull文件
    RECORD_NAME1 = 'RawFull_' + str(df.shape[0]) + '_' + str(df.shape[1]) + '.csv'
    RECORD_NAME = Path(DIR, RECORD_NAME1)
    df.to_csv(RECORD_NAME, index=False)
    print(f"RawFull file saved: {RECORD_NAME}")
    

    # ============= 对齐方法 =============
    final_merged_df = align_with_mapping_simple(DIR, 'sample_mapping.csv')
    
    if final_merged_df is not None:
        # 获取计算属性列（排除Sample Name, SMILES, target和实验特征）
        basic_cols = ['Sample Name', 'smiles', 'target']

        # 读取mapping文件来识别实验特征列
        try:
            mapping_df_temp = pd.read_csv('sample_mapping.csv')
            exp_feature_cols = [col for col in mapping_df_temp.columns 
                               if col not in basic_cols]
        except:
            exp_feature_cols = []

        calc_cols = [col for col in final_merged_df.columns 
                    if col not in basic_cols + exp_feature_cols]

        print(f"Before removing NaN rows: {final_merged_df.shape[0]} samples")
        print(f"Computational features: {len(calc_cols)}")
        print(f"Experimental features: {len(exp_feature_cols)}")

        # 只对计算属性列进行NaN检查，保留实验特征
        final_merged_df_clean = final_merged_df

        RECORD_NAME_MERGED = f'FullOption2_' + str(final_merged_df_clean.shape[0]) + '_' + str(final_merged_df_clean.shape[1]) + '.csv'
        merged_output_filename = Path(DIR, RECORD_NAME_MERGED)
        final_merged_df_clean.to_csv(merged_output_filename, index=False, float_format='%.6f')

        print(f'Final data written to {merged_output_filename}')
        print("Processing completed. Please check out the feature matrix.")
                    
    else:
        print("Error: Failed to create final merged dataframe")

if __name__ == "__main__":
    # 使用示例
    input_directory = '.'
    output_directory = '.'
    
    # 1. 运行 Multiwfn
    print("\n步骤 1: 运行 Multiwfn 计算...")
    run_multiwfn_on_fchk_files(input_path=input_directory)

    # 2. 处理产生的 *.txt 文件与 AtomProp_*.csv
    process_txt_files(input_directory, output_directory)
    
    print("\n" + "=" * 60)
    print("描述符计算完成！")
    print("=" * 60)


# Atom&Fun_Comprehensive_descriptor

# In[11]:


import os
import glob
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import itertools as it
import argparse
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

###############################################################################
# 1. 新的LOFFI算法 - 优先级系统
###############################################################################

# 新的LOFFI定义 - 优先级系统
LOFFI_CONTENT = {
    # 第一优先级：复杂多原子官能团
    "priority_1": {
        "Carboxylic_acid": "[CX3;$([R0][#6]),$([H1R0])](=[OX1])[$([OX2H]),$([OX1-])]",
        "Carboxylic_ester": "[CX3;$([R0][#6]),$([H1R0])](=[OX1])[OX2;$([OX2][#6;!$(C=[O,N,S])])]",
        "Amide": "[CX3;$([R0][#6]),$([H1R0])](=[OX1])[#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
        "Acylhalide": "[CX3;$([R0][#6]),$([H1R0])](=[OX1])[FX1,ClX1,BrX1,IX1]",
        "Lactone": "[#6X3;$([H0R][#6])](=[OX1])[#8RX2;$([#8RX2][#6;!$(C=[O,N,S])])]",
        "Lactam": "[#6X3R](=[OX1])[#7X3;$([H1][#6;!$([#6]=[O,N,S])]),$([H0]([#6;!$([#6]=[O,N,S])])[#6;!$([#6]=[O,N,S])])]",
        "Anhydrides": "[#6X3](=[OX1])[#8X2][#6X3]=[OX1]",
        "Carboxylic_imide": "[#6X3](=[OX1])[#7X3][#6X3]=[OX1]",
        "Trifluoromethyl": "[CX4](F)(F)F",
        "Sulfur_pentafluoride": "[SX6](F)(F)(F)(F)F",
        "Urea": "[#7X3;!$([#7][!#6])][#6X3](=[OX1])[#7X3;!$([#7][!#6])]",
        "Thiourea": "[#7X3;!$([#7][!#6])][#6X3](=[SX1])[#7X3;!$([#7][!#6])]",
        "Guanidine": "[N;v3X3,v4X4+][CX3](=[N;v3X2,v4X3+])[N;v3X3,v4X4+]",
        "Isocyanate": "[NX2]=[CX2]=[OX1]",
        "Cyanate": "[OX2][CX2]#[NX1]",
        "Isothiocyanate": "[NX2]=[CX2]=[SX1]",
        "Thiocyanate": "[SX2][CX2]#[NX1]",
        "Carbodiimide": "[NX2]=[CX2]=[NX2]",
        "Carbon_dioxide": "[OX1]=[CX2]=[OX1]"
    },
    
    # 第二优先级：磷、硼、硅、金属化合物
    "priority_2": {
        "Phosphoric_acid_derivative": "[#15X4D4](=[!#6])(=[!#6])([!#6])[!#6]",
        "Phosphonic_acid_derivative": "[#15X4;$([H1]),$([H0][#6])](=[!#6])([!#6])[!#6]",
        "Phosphinic_acid_derivative": "[#15X4;$([H2]),$([H1][#6]),$([H0]([#6])[#6])](=[!#6])[!#6]",
        "Phosphonous_derivatives": "[#15X3;$([D2]),$([D3][#6])]([!#6])[!#6]",
        "Phosphinous_derivatives": "[#15X3;$([H2]),$([H1][#6]),$([H0]([#6])[#6])][!#6]",
        "Phosphine_oxide": "[#15X4;$([H3]),$([H2][#6]),$([H1]([#6])[#6]),$([H0]([#6])([#6])[#6])]=[OX1]",
        "Phosphonium": "[#15+;!$([#15]~[!#6]);!$([#15]*~[#7,#8,#15,#16])]",
        "Phosphine": "[#15X3;$([H3]),$([H2][#6]),$([H1]([#6])[#6]),$([H0]([#6])([#6])[#6])]",
        "Boronic_acid_quat_derivative": "[BX4]([!#6])([!#6])([!#6])[!#6]",
        "Boronic_acid_tri_derivative": "[BX3]([!#6])([!#6])[!#6]",
        "Boron_boron_bond": "[BX3,BX4][BX3,BX4]",
        "Boron_cage": "[BX3,BX4]1[BX3,BX4][BX3,BX4][BX3,BX4][BX3,BX4][BX3,BX4]1",
        "Boron_chain": "[BX3,BX4][BX3,BX4][BX3,BX4]",
        "Quaternary_boronane": "[BX4;$([BX4]([#6])([#6])([#6])[#6])]",
        "Trialkylborane": "[BX3;$([BX3]([#6])([#6])[#6])]",
        "Quaternary_boron": "[BX4;!$([BX4]([#6])([#6])([#6])[#6])]",
        "Tri_boron": "[BX3;!$([BX3]([#6])([#6])[#6])]",
        "Fluoroboric": "[F;$([F]B)]",
        "Quart_silane": "[SiX4;$([SiX4]([#6])([#6])([#6])[#6])]",
        "Non_quart_silane": "[SiX4;!$([SiX4]([#6])([#6])([#6])[#6])]",
        "Fluorosilica": "[F;$([F][Si])]",
        "Metal_atoms": "[Li,Na,K,Rb,Cs,Be,Mg,Ca,Sr,Ba,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Al,Ga,In,Sn,Pb,Bi]",
        "Organometallic_compounds": "[!#1;!#5;!#6;!#7;!#8;!#9;!#14;!#15;!#16;!#17;!#33;!#34;!#35;!#52;!#53;!#85]~[#6;!-]"
    },
    
    # 第三优先级：硫化合物
    "priority_3": {
        "Sulfuric_derivative": "[SX4D4](=[!#6])(=[!#6])([!#6])[!#6]",
        "Sulfonic_derivative": "[SX4;$([H1]),$([H0][#6])](=[!#6])(=[!#6])[!#6]",
        "Sulfinic_derivative": "[SX3;$([H1]),$([H0][#6])](=[!#6])[!#6]",
        "Sulfon": "[SX4;$([SX4]([#6])[#6]),$([SX42+]([#6])[#6])](~[OX1])~[OX1]",
        "Sulfoxide": "[SX3;$([SX3]([#6])[#6]),$([SX3+]([#6])[#6])]~[OX1]",
        "Disulfide": "[SX2D2][SX2D2]",
        "Thionitrite": "[SX2][NX2]=[OX1]"
    },
    
    # 第四优先级：氮氧化合物
    "priority_4": {
        "Nitrate": "[NX3+](=O)(-[OX])-[OX]",
        "Nitro": "[NX3+](=O)-[OX-]",
        "Nitrite": "[NX2](=[OX1])[O;$([X2]),$([X1-])]",
        "N_Oxide": "[$([#7+][OX1-]),$([#7v5]=[OX1]);!$([#7](~[O])~[O]);!$([#7]=[#7])][OX1]",
        "Diazo": "[#6,#6-]~[NX2+]~[NX1,NX1-]",
        "Azide": "[NX1]~[NX2]~[NX2,NX1]",
        "Azo": "[NX2]=[NX2]",
        "Hydrazine": "[NX3;$([H2]),$([H1][#6]),$([H0]([#6])[#6]);!$(NC=[O,N,S])][NX3;$([H2]),$([H1][#6]),$([H0]([#6])[#6]);!$(NC=[O,N,S])]",
        "Hydrazone": "[NX3;$([H2]),$([H1][#6]),$([H0]([#6])[#6]);!$(NC=[O,N,S])][NX2]=[#6]"
    },
    
    # 第五优先级：双键/三键系统
    "priority_5": {
        "Aldehyde": "[$([CX3H][#6]),$([CX3H2])]=[OX1]",
        "Ketone": "[CX3;$([CX3]([#6])[#6])](=[OX1])",
        "Thioaldehyde": "[$([CX3H][#6]),$([CX3H2])]=[SX1]",
        "Thioketone": "[#6X3;$([#6X3]([#6])[#6])](=[SX1])",
        "Nitrile": "[NX1]#[CX2]",
        "Isonitrile": "[CX1-]#[NX2+]",
        "Ketene": "[CX3]=[CX2]=[OX1]",
        "Allene": "[CX3]=[CX2]=[CX3]",
        "Alkyne": "[CX2]#[CX2]",
        "Imine": "[NX2,NX3+;$([N][#6]),$([NH]);!$([N][CX3]=[#7,#8,#15,#16])]=[CX3;$([CH2]),$([CH][#6]),$([C]([#6])[#6])]",
        "Oxime": "[NX2](=[CX3;$([CH2]),$([CH][#6]),$([C]([#6])[#6])])[OX2,OX1-]",
        "Enamine": "[NX3;$([NH2][CX3]),$([NH1]([CX3])[#6]),$([N]([CX3])([#6])[#6]);!$([N]*=[#7,#8,#15,#16])][CX3;$([CH]),$([C][#6])]=[CX3]",
        "Amidine": "[NX3;!$(NC=[O,S])][CX3;$([CH]),$([C][#6])]=[NX2;!$(NC=[O,S])]",
        "Imidoylhalide": "[CX3R0;$([H0][#6]),$([H1])](=[NX2;$([H1]),$([H0][#6;!$(C=[O,N,S])])])[FX1,ClX1,BrX1,IX1]"
    },
    
    # 第六优先级：缩醛/缩酮类
    "priority_6": {
        "Acetal": "[OX2][CX4;!$(C(O)(O)[!#6])][OX2]",
        "Hemiacetal": "[OX2H][CX4;!$(C(O)(O)[!#6])][OX2]",
        "Thioacetal": "[SX2][CX4;!$(C(S)(S)[!#6])][SX2]",
        "Thiohemiacetal": "[SX2][CX4;!$(C(S)(S)[!#6])][OX2H]",
        "Aminal": "[NX3v3;!$(NC=[#7,#8,#15,#16])][CX4;!$(C(N)(N)[!#6])][NX3v3;!$(NC=[#7,#8,#15,#16])]",
        "Hemiaminal": "[NX3v3;!$(NC=[#7,#8,#15,#16])][CX4;!$(C(N)(N)[!#6])][OX2H]"
    },
    
    # 第七优先级：过氧化物
    "priority_7": {
        "Hydroperoxide": "[OX2H][OX2]",
        "Peroxo": "[OX2D2][OX2D2]"
    },
    
    # 第八优先级：氧基团
    "priority_8": {
        "Alcohol": "[OX2H;$([OX2H][CX4;!$(C([OX2H])[O,S,#7,#15,F,Cl,Br,I])])]",
        "Phenol": "[OX2;$([H1][c]),$([H0]([#6X4])[c]),$([H0]([!#6])[c])]",
        "Enol": "[OX2,OX1-][CX3;$([H1]),$(C[#6])]=[CX3]",
        "Epoxide": "[OX2r3]1[#6r3][#6r3]1",
        "Dialkylether": "[OX2;$([OX2]([C;!$([C]([OX2])[O,S,#7,#15,F,Cl,Br,I])])[C;!$([C]([OX2])[O,S,#7,#15])])]",
        "Oxonium": "[O+;!$([O]~[!#6]);!$([S]*~[#7,#8,#15,#16])]"
    },
    
    # 第九优先级：硫基团
    "priority_9": {
        "Alkylthiol": "[SX2H;$([SX2H][CX4;!$(C([SX2H])~[O,S,#7,#15])])]",
        "Arylthiol": "[SX2;$([H1][c]),$([H0]([#6X4])[c]),$([H0]([!#6])[c])]",
        "Dialkylthioether": "[SX2;$([H0]([#6])[c]),$([H0]([!#6])[c])]"
    },
    
    # 第十优先级：胺类
    "priority_10": {
        "Primary_aliph_amine": "[NX3H2+0,NX4H3+;!$([N][!C]);!$(N[c])]",
        "Secondary_aliph_amine": "[NX3H1+0,NX4H2+;!$([N][!C]);!$(N[c])]", 
        "Tertiary_aliph_amine": "[NX3H0+0,NX4H1+;!$([N][!C]);!$(N[c])]",
        "Quaternary_aliph_ammonium": "[NX4H0+;!$([N][!C]);!$(N[c])]",
        "Primary_arom_amine": "[NX3H2+0,NX4H3+;$([N][c])]",
        "Secondary_arom_amine": "[NX3H1+0,NX4H2+;$([N][c])]",
        "Tertiary_arom_amine": "[NX3H0+0,NX4H1+;$([N][c])]",
        "Quaternary_arom_ammonium": "[NX4H0+;$([N][c])]"
    },
    
    # 第十一优先级：卤代物
    "priority_11": {
        "Alkylchloride": "[ClX1;$([ClX1][CX4;!$(C[O,N,S,P])])]",
        "Alkylfluoride": "[FX1;$([FX1][CX4;!$(C[O,N,S,P])])]",
        "Alkylbromide": "[BrX1;$([BrX1][CX4;!$(C[O,N,S,P])])]",
        "Alkyliodide": "[IX1;$([IX1][CX4;!$(C[O,N,S,P])])]",
        "Arylchloride": "[Cl;$([Cl][c])]",
        "Arylfluoride": "[F;$([F][c])]",
        "Arylbromide": "[Br;$([Br][c])]",
        "Aryliodide": "[I;$([I][c])]",
        "Chloroalkene": "[ClX1][CX3]=[CX3]",
        "Fluoroalkene": "[FX1][CX3]=[CX3]",
        "Bromoalkene": "[BrX1][CX3]=[CX3]",
        "Iodoalkene": "[IX1][CX3]=[CX3]"
    },
    
    # 第十二优先级：芳环特殊基团
    "priority_12": {
        "Iminoarene": "[NX2;$([NX2]=[c])]",
        "Oxoarene": "[OX1;$([OX1]=[c])]",
        "Thioarene": "[SX1;$([SX1]=[c])]"
    },
    
    # 第十三优先级：双键系统
    "priority_13": {
        "Alkene": "[CX3;$([H2]),$([H1][#6]),$([C]([#6])[#6]);!$(C=[O,S,N]);!$(C[O,N,S,P,F,Cl,Br,I]);!$([c])]=[CX3;$([H2]),$([H1][#6]),$([C]([#6])[#6]);!$(C=[O,S,N]);!$(C[O,N,S,P,F,Cl,Br,I]);!$([c])]"
    },
    
    # 第十四优先级：连接基团
    "priority_14": {
        "Aryl_connector_quaternary": "[CX4H0;$(C([c])[c]);$(C([c])([c])[c]);$(C([c])([c])([c])[c]);!r]",
        "Aryl_connector_tertiary": "[CX4H1;$(C([c])[c]);$(C([c])([c])[c]);!r]",
        "Aryl_methyl": "[CH3;$(C[c]);!$(C[NX3,NX4])]",
        "N_methyl": "[CH3;$(C[NX3,NX4])]",
        "N_methylene": "[CH2;$(C[NX3,NX4])]",
    },
    
    # 第十五优先级：基础结构单元
    "priority_15": {
        "Methane": "[CX4H4]",
        "Methyl_group": "[CH3;!$(C[O,N,S,P,F,Cl,Br,I,Si,B]);!$(C=*)]",
        "Methylene_group": "[CH2;!$(C[O,N,S,P,F,Cl,Br,I,Si,B]);!$(C=*)]",
        "Methine_group": "[CH1;!$(C[O,N,S,P,F,Cl,Br,I,Si,B]);!$(C=*);!$(C#*)]",
        "Quaternary_carbon": "[CX4H0;!$(C[O,N,S,P,F,Cl,Br,I,Si,B]);!$(C=*);!$(C([c])[c])]"
    },
    
    # 通用兜底模式
    "priority_final_universal": {
        "Carbon_sp3_4H": "[CH4]",
        "Carbon_sp3_3H": "[CH3]",
        "Carbon_sp3_2H": "[CH2]",
        "Carbon_sp3_1H": "[CH1]",
        "Carbon_sp3_0H": "[CX4H0]",
        "Carbon_sp2": "[CX3]",
        "Carbon_sp": "[CX2]",
        "Boron_any": "[B]",
        "Silicon_any": "[Si]",
        "Nitrogen_primary": "[NH2]",
        "Nitrogen_secondary": "[NH1]",
        "Nitrogen_tertiary": "[NX3H0]",
        "Nitrogen_quaternary": "[NX4]",
        "Nitrogen_sp2": "[NX2]",
        "Nitrogen_sp": "[NX1]",
        "Oxygen_alcohol": "[OH]",
        "Oxygen_ether": "[OX2H0]",
        "Oxygen_carbonyl": "[OX1]",
        "Sulfur_any": "[S]",
        "Phosphorus_any": "[P]",
        "Halogen_F": "[F]",
        "Halogen_Cl": "[Cl]",
        "Halogen_Br": "[Br]",
        "Halogen_I": "[I]",
    }
}

###############################################################################
# 2. 新的LOFFI算法核心函数
###############################################################################

def find_fused_neighbour(aro_atom_rings, ring_id):
    """找到指定芳香环的近邻稠环"""
    neighbours = []
    for idx, ring in enumerate(aro_atom_rings):
        if idx == ring_id:
            continue
        if len(set(ring) & set(aro_atom_rings[ring_id])) == 2:
            neighbours.append(idx)
    return neighbours

def process_aromatic_systems(mol):
    """处理芳香环系统，返回稠环组"""
    if mol is None:
        return []
    
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    aro_atom_rings = []
    
    # 找到所有芳香环
    for ring in atom_rings:
        if len(ring) > 0 and mol.GetAtomWithIdx(ring[0]).GetIsAromatic():
            aro_atom_rings.append(ring)
    
    if len(aro_atom_rings) == 0:
        return []
    
    # 找稠环系统
    labels = [0] * len(aro_atom_rings)
    group_id = 1
    
    for i in range(len(aro_atom_rings)):
        if labels[i] == 0:
            queue = find_fused_neighbour(aro_atom_rings, i)
            labels[i] = group_id
            for nb in queue:
                labels[nb] = group_id
            while queue:
                current = queue.pop(0)
                nbs2 = find_fused_neighbour(aro_atom_rings, current)
                for nbx in nbs2:
                    if labels[nbx] == 0:
                        labels[nbx] = group_id
                        queue.append(nbx)
            group_id += 1
    
    # 合并稠环
    fused_ring_list = []
    for grp in range(1, max(labels) + 1):
        group_atoms = []
        for i, lab in enumerate(labels):
            if lab == grp:
                group_atoms.extend(aro_atom_rings[i])
        group_atoms = list(set(group_atoms))
        fused_ring_list.append(group_atoms)
    
    return fused_ring_list

def apply_loffi_algorithm(mol, smiles_val):
    """应用LOFFI算法进行功能团识别"""
    if mol is None:
        return [], []  # ring_groups, fun_groups
    
    matched_atoms = set()
    functional_groups = {}
    aromatic_groups = {}
    
    # 第一步：处理芳香环系统
    fused_ring_list = process_aromatic_systems(mol)
    ring_groups = []
    for i, ring_atoms in enumerate(fused_ring_list, start=1):
        group_name = f"Aromatic_System_{i}"
        matched_atoms.update(set(ring_atoms))
        ring_groups.append((group_name, set(ring_atoms), ""))
    
    # 第二步：按严格优先级顺序处理功能团
    priority_order = [f"priority_{i}" for i in range(1, 16)]
    
    fun_groups = []
    for priority in priority_order:
        if priority not in LOFFI_CONTENT:
            continue
            
        patterns = LOFFI_CONTENT[priority]
        for fg_name, smarts_pattern in patterns.items():
            try:
                pattern = Chem.MolFromSmarts(smarts_pattern)
                if pattern is None:
                    continue
                    
                matches = mol.GetSubstructMatches(pattern)
                
                for k, match in enumerate(matches, start=1):
                    match_set = set(match)
                    
                    # 检查是否与已匹配原子冲突
                    conflict_atoms = match_set & matched_atoms
                    if not conflict_atoms:  # 只有无冲突才匹配
                        matched_atoms.update(match_set)
                        group_name = f"{fg_name}_{k}" if len(matches) > 1 else fg_name
                        fun_groups.append((group_name, match_set, ""))
                        
            except Exception as e:
                print(f"Error with pattern {fg_name}: {e}")
    
    # 第三步：处理剩余未匹配的原子（通用兜底）
    total_atoms = mol.GetNumAtoms()
    hydrogen_indices = set([i for i in range(total_atoms) if mol.GetAtomWithIdx(i).GetSymbol() == 'H'])
    unmatched_heavy_atoms = set(range(total_atoms)) - matched_atoms - hydrogen_indices
    
    # 用通用模式匹配剩余原子
    universal_patterns = LOFFI_CONTENT.get("priority_final_universal", {})
    for fg_name, smarts_pattern in universal_patterns.items():
        try:
            pattern = Chem.MolFromSmarts(smarts_pattern)
            if pattern is None:
                continue
                
            matches = mol.GetSubstructMatches(pattern)
            
            for k, match in enumerate(matches, start=1):
                match_set = set(match)
                
                # 只匹配未被匹配的原子
                if match_set & unmatched_heavy_atoms:
                    remaining_atoms = match_set & unmatched_heavy_atoms
                    matched_atoms.update(remaining_atoms)
                    unmatched_heavy_atoms -= remaining_atoms
                    
                    group_name = f"{fg_name}_{k}" if len([m for m in matches if set(m) & unmatched_heavy_atoms]) > 1 else fg_name
                    fun_groups.append((group_name, remaining_atoms, ""))
                    
        except Exception as e:
            print(f"Error with universal pattern {fg_name}: {e}")
    
    return ring_groups, fun_groups

###############################################################################
# 3. 工具函数（保持原有逻辑）
###############################################################################

def setup_logging(log_file_path):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler()
        ]
    )

def read_xyz(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    atoms = []
    coords = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            atoms.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
    return atoms, np.array(coords)

def euclidean_distance(coord1, coord2):
    return np.linalg.norm(coord1 - coord2)

def find_substructure(target_atoms, target_coords, substructure_atoms, substructure_coords, threshold=1.01):
    substructure_center = np.mean(substructure_coords, axis=0)
    center_distances = np.array([euclidean_distance(coord, substructure_center) for coord in substructure_coords])
    center_atom_index = np.argmin(center_distances)
    center_atom = substructure_atoms[center_atom_index]
    center_atom_coord = substructure_coords[center_atom_index]
    substructure_pairs = sorted(
        [(substructure_atoms[i], euclidean_distance(center_atom_coord, coord)) for i, coord in enumerate(substructure_coords) if i != center_atom_index],
        key=lambda x: x[1]
    )
    matches = []
    for i, target_atom in enumerate(target_atoms):
        if target_atom == center_atom:
            all_matches_found = True
            matched_indices = [i]
            for sub_atom, sub_dist in substructure_pairs:
                found_match = False
                for j, (t_atom, t_coord) in enumerate(zip(target_atoms, target_coords)):
                    if j not in matched_indices and t_atom == sub_atom:
                        if abs(euclidean_distance(target_coords[i], t_coord) - sub_dist) <= threshold:
                            found_match = True
                            matched_indices.append(j)
                            break
                if not found_match:
                    all_matches_found = False
                    break
            if all_matches_found:
                matches.append(matched_indices)
    return matches

def find_substructure_center_atom(substructure_coords):
    geometric_center = np.mean(substructure_coords, axis=0)
    center_atom_index = np.argmin([euclidean_distance(coord, geometric_center) for coord in substructure_coords])
    return center_atom_index

def sort_substructure_atoms(substructure_atoms, substructure_coords):
    center_atom_index = find_substructure_center_atom(substructure_coords)
    center_atom_coord = substructure_coords[center_atom_index]
    distances = [euclidean_distance(center_atom_coord, coord) for coord in substructure_coords]
    index_with_distances = list(enumerate(distances))
    index_with_distances = [index_distance for index_distance in index_with_distances if index_distance[0] != center_atom_index]
    sorted_indices_with_distances = sorted(index_with_distances, key=lambda x: x[1])
    sorted_indices, _ = zip(*sorted_indices_with_distances)
    sorted_indices = (center_atom_index,) + sorted_indices
    return list(sorted_indices)

def read_atomprop_csv(atomprop_file):
    try:
        df_atom = pd.read_csv(atomprop_file)
    except Exception as e:
        logging.warning(f"[WARN] Failed to read {atomprop_file}: {e}")
        return None
    
    if "Atom#" not in df_atom.columns:
        logging.warning(f"[WARN] {atomprop_file} missing 'Atom#' column.")
        return None
    
    df_atom["Atom#"] = pd.to_numeric(df_atom["Atom#"], errors="coerce").fillna(-1).astype(int)
    return df_atom

def pool_1d(values, mode="mean"):
    if len(values) == 0:
        return np.nan
    if mode == "min":
        return np.min(values)
    elif mode == "max":
        return np.max(values)
    elif mode == "mean":
        return np.mean(values)
    elif mode == "delta":
        return np.max(values) - np.min(values)
    else:
        return np.nan

def pool_4modes(col_data):
    return [
        pool_1d(col_data, "min"),
        pool_1d(col_data, "max"),
        pool_1d(col_data, "mean"),
        pool_1d(col_data, "delta"),
    ]

def calc_area_weighted_4props(arr_atom, atom_indices):
    if not atom_indices:
        return [np.nan, np.nan, np.nan, 0.0]

    sum_area = 0.0
    weighted_LEAE = 0.0
    weighted_ESP  = 0.0
    weighted_ALIE = 0.0
    
    valid_area_for_LEAE = 0.0
    valid_area_for_ESP  = 0.0
    valid_area_for_ALIE = 0.0

    for idx in atom_indices:
        this_area = arr_atom[idx, 3]
        if np.isnan(this_area):
            continue
        sum_area += this_area

        val_LEAE = arr_atom[idx, 0]
        if not np.isnan(val_LEAE):
            weighted_LEAE += val_LEAE * this_area
            valid_area_for_LEAE += this_area

        val_ESP = arr_atom[idx, 1]
        if not np.isnan(val_ESP):
            weighted_ESP += val_ESP * this_area
            valid_area_for_ESP += this_area

        val_ALIE = arr_atom[idx, 2]
        if not np.isnan(val_ALIE):
            weighted_ALIE += val_ALIE * this_area
            valid_area_for_ALIE += this_area

    if sum_area < 1e-12:
        return [np.nan, np.nan, np.nan, 0.0]

    avg_LEAE = weighted_LEAE / valid_area_for_LEAE if valid_area_for_LEAE > 1e-12 else np.nan
    avg_ESP  = weighted_ESP  / valid_area_for_ESP  if valid_area_for_ESP  > 1e-12 else np.nan
    avg_ALIE = weighted_ALIE / valid_area_for_ALIE if valid_area_for_ALIE> 1e-12 else np.nan

    return [avg_LEAE, avg_ESP, avg_ALIE, sum_area]

###############################################################################
# 4. 主函数 - 三种情形处理
###############################################################################

def run_atom_prop_extraction(original_csv_path, mode=3, target_element=None, xyz1_path=None, xyz_folder=None, threshold=1.01):
    """
    三种情形的原子性质提取:
    mode=1: 只提取指定原子(如S)的性质 + 分子性质
    mode=2: 提取指定官能团(由xyz1_path定义)的性质 + 分子性质（简化版）
    mode=3: 完全根据loffi代码处理(默认)
    """
    parent_dir = os.path.dirname(original_csv_path)
    
    # 设置xyz文件夹路径
    if xyz_folder is None:
        xyz_folder = 'xyz_path'
    xyz_dir = os.path.join(parent_dir, xyz_folder)
    
    print(f"运行模式: {mode}")
    print(f"主目录: {parent_dir}")
    print(f"XYZ文件夹: {xyz_dir}")
    
    if mode == 1:
        print(f"目标元素: {target_element}")
    elif mode == 2:
        print(f"官能团结构文件: {xyz1_path}")
        print(f"匹配阈值: {threshold}")
    elif mode == 3:
        print("使用完整的loffi功能团分析")

    try:
        # 根据模式调用相应的处理函数
        if mode == 1:
            temp_output = "temp_output.csv"
            run_specific_atom_extraction(
                parent_dir=parent_dir,
                full_option_csv=os.path.basename(original_csv_path),
                target_element=target_element,
                final_output_csv=temp_output
            )
        elif mode == 2:
            temp_output = "temp_output.csv"
            run_functional_group_extraction_simplified(
                parent_dir=parent_dir,
                full_option_csv=os.path.basename(original_csv_path),
                xyz_folder=xyz_folder,
                xyz1_path=xyz1_path,
                threshold=threshold,
                final_output_csv=temp_output
            )
        else:  # mode == 3
            temp_output = "temp_output.csv"
            run_atom_and_frag_pooling_internal(
                parent_dir=parent_dir,
                full_option_csv=os.path.basename(original_csv_path),
                out_log_dir="AtomLevelLogs",
                final_output_csv=temp_output
            )
        
        # 读取生成的文件获取实际的行数和列数
        temp_output_path = os.path.join(parent_dir, temp_output)
        if os.path.exists(temp_output_path):
            df_result = pd.read_csv(temp_output_path)
            rows, cols = df_result.shape
            
            # 生成最终的文件名
            mode_suffix = f"Mode{mode}"
            if mode == 1:
                mode_suffix += f"_{target_element}"
            elif mode == 2:
                mode_suffix += "_Fragment"
            
            final_csv = f"FinalFull_{mode_suffix}_{rows}_{cols}.csv"
            final_path = os.path.join(parent_dir, final_csv)
            
            # 重命名文件
            os.rename(temp_output_path, final_path)
            print(f"✓ Final output saved as: {final_csv}")
            print(f"✓ Result contains {rows} rows and {cols} columns")
            
    finally:
        # 清理临时文件
        temp_output_path = os.path.join(parent_dir, "temp_output.csv")
        if os.path.exists(temp_output_path):
            try:
                os.remove(temp_output_path)
            except:
                pass

def run_specific_atom_extraction(parent_dir, full_option_csv, target_element, final_output_csv):
    """
    模式1: 提取指定元素的13种原子性质
    对于每个分子，如果有多个目标原子，取所有目标原子各性质的平均值
    """
    print(f"模式1: 提取{target_element}原子的13种性质")
    
    # 定义需要提取的原始特征列（从AtomProp中读取）
    base_feature_columns = [
        'LEAE All area',              # 原子表面积
        'LEAE Minimal value',         # LEAE最小值
        'LEAE Maximal value',         # LEAE最大值  
        'LEAE All average',           # LEAE平均值
        'ESP Minimal value (kcal/mol)',    # ESP最小值
        'ESP Maximal value (kcal/mol)',    # ESP最大值
        'ESP All average (kcal/mol)',      # ESP平均值
        'ALIE Min value',             # ALIE最小值
        'ALIE Max value',             # ALIE最大值
        'ALIE Average',               # ALIE平均值
    ]
    
    # 读取原始FullOption文件
    csv_full_path = os.path.join(parent_dir, full_option_csv)
    if not os.path.exists(csv_full_path):
        raise FileNotFoundError(f"[ERROR] {csv_full_path} not found in {parent_dir}")

    df_option = pd.read_csv(csv_full_path, dtype={"Sample Name": str})
    df_option["Sample Name"] = df_option["Sample Name"].str.strip()

    # 找到AtomProp_*.csv列表
    csv_list = sorted(glob.glob(os.path.join(parent_dir, "AtomProp_*.csv")))
    if len(csv_list) == 0:
        print(f"No AtomProp_*.csv found in {parent_dir}")
        return

    results = []

    for csv_file in csv_list:
        base_name = os.path.basename(csv_file)
        sample_name_raw = base_name.replace("AtomProp_", "").replace(".csv", "").strip()
        
        try:
            sample_name = str(int(sample_name_raw))
        except ValueError:
            sample_name = sample_name_raw
            
        print(f" Processing sample {sample_name} (from file {base_name})")

        # 查找FullOption中对应行
        subset = df_option[df_option["Sample Name"] == sample_name]
        if len(subset) == 0:
            logging.warning(f"[WARN] {sample_name} not found in {full_option_csv}, skip.")
            continue

        # 读取原子CSV
        df_atom = read_atomprop_csv(csv_file)
        if df_atom is None:
            continue

        # 检查必要的列是否存在
        missing_cols = [col for col in base_feature_columns if col not in df_atom.columns]
        if missing_cols:
            logging.warning(f"[WARN] {csv_file} missing columns: {missing_cols}, skip.")
            continue

        # 查找目标元素的原子
        target_atoms = df_atom[df_atom['Element'] == target_element]
        
        row_res = {"Sample Name": sample_name}

        if len(target_atoms) == 0:
            print(f"  No {target_element} atoms found in {sample_name}")
            # 设置所有特征为NaN
            row_res[f"{target_element}_area"] = np.nan
            row_res[f"{target_element}_LEAE_min"] = np.nan
            row_res[f"{target_element}_LEAE_max"] = np.nan
            row_res[f"{target_element}_LEAE_average"] = np.nan
            row_res[f"{target_element}_LEAE_delta"] = np.nan
            row_res[f"{target_element}_ESP_min"] = np.nan
            row_res[f"{target_element}_ESP_max"] = np.nan
            row_res[f"{target_element}_ESP_average"] = np.nan
            row_res[f"{target_element}_ESP_delta"] = np.nan
            row_res[f"{target_element}_ALIE_min"] = np.nan
            row_res[f"{target_element}_ALIE_max"] = np.nan
            row_res[f"{target_element}_ALIE_average"] = np.nan
            row_res[f"{target_element}_ALIE_delta"] = np.nan
        else:
            print(f"  Found {len(target_atoms)} {target_element} atoms")
            
            # 对所有目标原子的各性质取平均
            # 1. 原子表面积
            area_values = target_atoms['LEAE All area'].dropna()
            row_res[f"{target_element}_area"] = area_values.mean() if len(area_values) > 0 else np.nan
            
            # 2-5. LEAE性质
            leae_min_values = target_atoms['LEAE Minimal value'].dropna()
            leae_max_values = target_atoms['LEAE Maximal value'].dropna()
            leae_avg_values = target_atoms['LEAE All average'].dropna()
            
            row_res[f"{target_element}_LEAE_min"] = leae_min_values.mean() if len(leae_min_values) > 0 else np.nan
            row_res[f"{target_element}_LEAE_max"] = leae_max_values.mean() if len(leae_max_values) > 0 else np.nan
            row_res[f"{target_element}_LEAE_average"] = leae_avg_values.mean() if len(leae_avg_values) > 0 else np.nan
            
            # 计算LEAE delta（每个原子的max-min，然后取平均）
            leae_delta_values = []
            for idx, atom_row in target_atoms.iterrows():
                if pd.notna(atom_row['LEAE Maximal value']) and pd.notna(atom_row['LEAE Minimal value']):
                    delta = atom_row['LEAE Maximal value'] - atom_row['LEAE Minimal value']
                    leae_delta_values.append(delta)
            row_res[f"{target_element}_LEAE_delta"] = np.mean(leae_delta_values) if len(leae_delta_values) > 0 else np.nan
            
            # 6-9. ESP性质
            esp_min_values = target_atoms['ESP Minimal value (kcal/mol)'].dropna()
            esp_max_values = target_atoms['ESP Maximal value (kcal/mol)'].dropna()
            esp_avg_values = target_atoms['ESP All average (kcal/mol)'].dropna()
            
            row_res[f"{target_element}_ESP_min"] = esp_min_values.mean() if len(esp_min_values) > 0 else np.nan
            row_res[f"{target_element}_ESP_max"] = esp_max_values.mean() if len(esp_max_values) > 0 else np.nan
            row_res[f"{target_element}_ESP_average"] = esp_avg_values.mean() if len(esp_avg_values) > 0 else np.nan
            
            # 计算ESP delta
            esp_delta_values = []
            for idx, atom_row in target_atoms.iterrows():
                if pd.notna(atom_row['ESP Maximal value (kcal/mol)']) and pd.notna(atom_row['ESP Minimal value (kcal/mol)']):
                    delta = atom_row['ESP Maximal value (kcal/mol)'] - atom_row['ESP Minimal value (kcal/mol)']
                    esp_delta_values.append(delta)
            row_res[f"{target_element}_ESP_delta"] = np.mean(esp_delta_values) if len(esp_delta_values) > 0 else np.nan
            
            # 10-13. ALIE性质
            alie_min_values = target_atoms['ALIE Min value'].dropna()
            alie_max_values = target_atoms['ALIE Max value'].dropna()
            alie_avg_values = target_atoms['ALIE Average'].dropna()
            
            row_res[f"{target_element}_ALIE_min"] = alie_min_values.mean() if len(alie_min_values) > 0 else np.nan
            row_res[f"{target_element}_ALIE_max"] = alie_max_values.mean() if len(alie_max_values) > 0 else np.nan
            row_res[f"{target_element}_ALIE_average"] = alie_avg_values.mean() if len(alie_avg_values) > 0 else np.nan
            
            # 计算ALIE delta
            alie_delta_values = []
            for idx, atom_row in target_atoms.iterrows():
                if pd.notna(atom_row['ALIE Max value']) and pd.notna(atom_row['ALIE Min value']):
                    delta = atom_row['ALIE Max value'] - atom_row['ALIE Min value']
                    alie_delta_values.append(delta)
            row_res[f"{target_element}_ALIE_delta"] = np.mean(alie_delta_values) if len(alie_delta_values) > 0 else np.nan

        results.append(row_res)

    # 创建结果DataFrame
    df_out = pd.DataFrame(results)
    print(f"Generated 13 features for {target_element} atoms")

    # 与原始数据合并
    df_merged = pd.merge(df_option, df_out, on="Sample Name", how="left")

    # 重新排列列顺序：Sample Name + 原始特征 + 新特征 + smiles + target
    all_columns = df_merged.columns.tolist()
    special_cols = ['Sample Name', 'smiles', 'target']
    
    # 分离原始特征和新特征
    new_feature_cols = [col for col in df_out.columns if col != 'Sample Name']
    original_cols = [col for col in all_columns if col not in special_cols + new_feature_cols]
    
    new_column_order = ['Sample Name'] + original_cols + new_feature_cols + ['smiles', 'target']
    df_merged = df_merged[new_column_order]

    # 保存结果
    out_path = os.path.join(parent_dir, final_output_csv)
    df_merged.to_csv(out_path, index=False)
    print(f"[Done] Wrote merged CSV => {final_output_csv} in {parent_dir}")
    
    return df_merged

def run_functional_group_extraction_simplified(parent_dir, full_option_csv, xyz_folder, xyz1_path, threshold, final_output_csv):
    """
    模式2: 简化版官能团特征提取（修正版）
    min/max取原子原始值，mean用面积加权，delta=max-min
    """
    print("模式2: 简化版官能团特征提取（修正版）")
    
    # 设置xyz文件夹路径
    xyz_dir = os.path.join(parent_dir, xyz_folder) if xyz_folder else parent_dir
    
    # 读取子结构
    if not os.path.exists(xyz1_path):
        raise FileNotFoundError(f"官能团结构文件不存在: {xyz1_path}")
    
    substructure_atoms, substructure_coords = read_xyz(xyz1_path)
    sorted_indices = sort_substructure_atoms(substructure_atoms, substructure_coords)
    
    print(f"子结构信息:")
    print(f"  原子数量: {len(substructure_atoms)}")
    print(f"  元素类型: {list(set(substructure_atoms))}")
    print(f"  原子详情: {', '.join([f'{atom}{i+1}' for i, atom in enumerate(substructure_atoms)])}")
    
    # 读取原始FullOption文件
    csv_full_path = os.path.join(parent_dir, full_option_csv)
    df_option = pd.read_csv(csv_full_path, dtype={"Sample Name": str})
    df_option["Sample Name"] = df_option["Sample Name"].str.strip()
    
    # 查找xyz文件并进行匹配
    xyz_files = sorted(glob.glob(os.path.join(xyz_dir, "*.xyz")))
    print(f"在 {xyz_dir} 中找到 {len(xyz_files)} 个xyz文件")
    
    results = []
    successful_matches = 0
    
    for xyz_file in xyz_files:
        base_name = os.path.basename(xyz_file)
        sample_name_raw = base_name.replace(".xyz", "").strip()
        
        # 尝试不同的样本名格式
        possible_names = [
            sample_name_raw,
            str(int(sample_name_raw)) if sample_name_raw.isdigit() else sample_name_raw,
            sample_name_raw.lstrip('0') if sample_name_raw.isdigit() else sample_name_raw
        ]
        
        # 在FullOption中查找对应行
        matching_row = None
        final_sample_name = None
        
        for name_candidate in possible_names:
            subset = df_option[df_option["Sample Name"] == name_candidate]
            if len(subset) > 0:
                matching_row = subset.iloc[0]
                final_sample_name = name_candidate
                break
        
        if matching_row is None:
            continue
            
        print(f"✓ 处理样本 {final_sample_name} (来自文件 {base_name})")
        
        # 进行子结构匹配
        try:
            target_atoms, target_coords = read_xyz(xyz_file)
            matches = find_substructure(target_atoms, target_coords, substructure_atoms, substructure_coords, threshold=threshold)
            
            if matches:
                print(f"  🎯 找到 {len(matches)} 个匹配的官能团")
                
                # 收集所有匹配结果
                all_match_indices = []
                for match in matches:
                    sorted_match = [match[i] for i in sorted_indices]
                    all_match_indices.extend(sorted_match)
                
                # 检查索引是否在有效范围内
                fragment_indices_0based = []
                for idx in all_match_indices:
                    if 0 <= idx < len(target_atoms):
                        fragment_indices_0based.append(idx)
                
                print(f"  ✓ 有效原子索引: {fragment_indices_0based}")
                if len(fragment_indices_0based) > 0:
                    successful_matches += 1
                
            else:
                print(f"  ❌ 未找到匹配的官能团")
                fragment_indices_0based = []
                
        except Exception as e:
            print(f"  ❌ 处理xyz文件时出错: {e}")
            fragment_indices_0based = []
        
        # 查找AtomProp文件
        possible_atomprop_files = [
            os.path.join(parent_dir, f"AtomProp_{final_sample_name.zfill(6)}.csv"),
            os.path.join(parent_dir, f"AtomProp_{sample_name_raw.zfill(6)}.csv"),
            os.path.join(parent_dir, f"AtomProp_{sample_name_raw}.csv")
        ]
        
        atomprop_file = None
        for candidate_file in possible_atomprop_files:
            if os.path.exists(candidate_file):
                atomprop_file = candidate_file
                break
        
        if atomprop_file is None:
            row_res = create_empty_simplified_result(final_sample_name)
        else:
            df_atom = read_atomprop_csv(atomprop_file)
            if df_atom is None:
                row_res = create_empty_simplified_result(final_sample_name)
            else:
                row_res = extract_fragment_features_corrected(
                    final_sample_name, df_atom, fragment_indices_0based, substructure_atoms
                )
        
        results.append(row_res)
    
    # 创建结果DataFrame
    df_out = pd.DataFrame(results)
    
    # 统计信息
    fragment_cols = [col for col in df_out.columns if col.startswith('Fragment_') and col != 'Sample Name']
    total_features = len(fragment_cols)
    
    print(f"\n📊 特征提取统计:")
    print(f"  生成特征数: {total_features}")
    print(f"  成功匹配: {successful_matches}/{len(results)}")
    
    # 检查列名冲突并处理
    df_option_cols = set(df_option.columns)
    df_out_cols = set(df_out.columns) - {'Sample Name'}
    
    conflicting_cols = df_option_cols & df_out_cols
    if conflicting_cols:
        print(f"⚠️ 发现列名冲突: {conflicting_cols}")
        for col in conflicting_cols:
            if col in df_out.columns:
                df_out = df_out.rename(columns={col: f"{col}_new"})
    
    # 与原始数据合并
    df_merged = pd.merge(df_option, df_out, on="Sample Name", how="left")
    
    # 重新排列列顺序
    all_columns = df_merged.columns.tolist()
    special_cols = ['Sample Name', 'smiles', 'target']
    fragment_cols_final = [col for col in all_columns if col.startswith('Fragment_')]
    other_cols = [col for col in all_columns if col not in special_cols + fragment_cols_final]
    new_column_order = ['Sample Name'] + other_cols + fragment_cols_final + ['smiles', 'target']
    df_merged = df_merged[new_column_order]
    
    # 保存结果
    out_path = os.path.join(parent_dir, final_output_csv)
    df_merged.to_csv(out_path, index=False)
    print(f"\n[Done] 写入合并的CSV => {final_output_csv}")
    
    # 最终统计
    non_null_samples = df_merged[fragment_cols_final].dropna(how='all').shape[0]
    print(f"✓ {non_null_samples}/{len(df_merged)} 个样本有有效的官能团特征")

def extract_fragment_features_corrected(sample_name, df_atom, fragment_indices_0based, substructure_atoms):
    """
    提取官能团特征 - 修正版
    min/max: 取所有原子的原始值中的全局最小/最大值
    mean: 面积加权平均
    delta: max - min
    """
    row_res = {"Sample Name": sample_name}
    
    # 检查必要的列 - 需要原始值列用于min/max计算
    col_LEAE_avg = "LEAE All average"
    col_LEAE_min = "LEAE Minimal value"
    col_LEAE_max = "LEAE Maximal value"
    
    col_ESP_avg = "ESP All average (kcal/mol)"
    col_ESP_min = "ESP Minimal value (kcal/mol)"
    col_ESP_max = "ESP Maximal value (kcal/mol)"
    
    col_ALIE_avg = "ALIE Average"
    col_ALIE_min = "ALIE Min value"
    col_ALIE_max = "ALIE Max value"
    
    col_AREA = "LEAE All area"
    
    needed_cols = [
        col_LEAE_avg, col_LEAE_min, col_LEAE_max,
        col_ESP_avg, col_ESP_min, col_ESP_max,
        col_ALIE_avg, col_ALIE_min, col_ALIE_max,
        col_AREA
    ]
    miss_cols = [col for col in needed_cols if col not in df_atom.columns]
    
    if miss_cols:
        print(f"    ⚠️  缺少列: {miss_cols}")
        return create_empty_simplified_result(sample_name)
    
    n_atoms = df_atom.shape[0]
    
    if len(fragment_indices_0based) == 0:
        print(f"    ❌ 没有有效的官能团原子索引")
        return create_empty_simplified_result(sample_name)
    
    # 检查索引是否在AtomProp文件的有效范围内
    valid_indices = [idx for idx in fragment_indices_0based if 0 <= idx < n_atoms]
    
    if len(valid_indices) == 0:
        print(f"    ❌ 没有有效的原子索引")
        return create_empty_simplified_result(sample_name)
    
    print(f"    ✓ 处理 {len(valid_indices)} 个有效的官能团原子")
    
    # 收集所有官能团原子的性质
    substructure_size = len(substructure_atoms)
    all_atom_props = []  # 存储所有官能团原子的性质
    
    # 按子结构大小分组处理匹配的原子
    for i in range(0, len(valid_indices), substructure_size):
        group_indices = valid_indices[i:i+substructure_size]
        
        # 至少需要80%的原子才认为是一个有效的官能团
        if len(group_indices) >= substructure_size * 0.8:
            print(f"      处理第 {i//substructure_size + 1} 个官能团组: {group_indices}")
            
            # 收集这个官能团中每个原子的性质
            for atom_idx in group_indices:
                atom_row = df_atom.iloc[atom_idx]
                
                # 获取平均值用于面积加权
                leae_avg = atom_row[col_LEAE_avg]
                esp_avg = atom_row[col_ESP_avg]
                alie_avg = atom_row[col_ALIE_avg]
                area_val = atom_row[col_AREA]
                
                # 获取原始min/max值
                leae_min_val = atom_row[col_LEAE_min]
                leae_max_val = atom_row[col_LEAE_max]
                esp_min_val = atom_row[col_ESP_min]
                esp_max_val = atom_row[col_ESP_max]
                alie_min_val = atom_row[col_ALIE_min]
                alie_max_val = atom_row[col_ALIE_max]
                
                # 只收集有效的数值
                if all(pd.notna([leae_avg, esp_avg, alie_avg, area_val])):
                    all_atom_props.append({
                        'LEAE_avg': leae_avg,
                        'LEAE_min': leae_min_val,
                        'LEAE_max': leae_max_val,
                        'ESP_avg': esp_avg,
                        'ESP_min': esp_min_val,
                        'ESP_max': esp_max_val,
                        'ALIE_avg': alie_avg,
                        'ALIE_min': alie_min_val,
                        'ALIE_max': alie_max_val,
                        'area': area_val
                    })
                    print(f"        原子{atom_idx}: LEAE_avg={leae_avg:.3f}, ESP_avg={esp_avg:.3f}, ALIE_avg={alie_avg:.3f}, area={area_val:.3f}")
    
    if len(all_atom_props) == 0:
        print(f"    ❌ 没有找到有效的原子性质")
        return create_empty_simplified_result(sample_name)
    
    print(f"    ✓ 收集到 {len(all_atom_props)} 个有效原子的性质")
    
    # 计算统计量
    all_atom_props = pd.DataFrame(all_atom_props)
    areas = all_atom_props['area'].values
    total_area = np.sum(areas)
    
    # LEAE统计
    leae_min_vals = all_atom_props['LEAE_min'].dropna().values
    leae_max_vals = all_atom_props['LEAE_max'].dropna().values
    leae_avg_vals = all_atom_props['LEAE_avg'].values
    
    row_res["Fragment_LEAE_min"] = np.min(leae_min_vals) if len(leae_min_vals) > 0 else np.nan
    row_res["Fragment_LEAE_max"] = np.max(leae_max_vals) if len(leae_max_vals) > 0 else np.nan
    row_res["Fragment_LEAE_mean"] = np.sum(leae_avg_vals * areas) / total_area if total_area > 0 else np.mean(leae_avg_vals)
    row_res["Fragment_LEAE_delta"] = row_res["Fragment_LEAE_max"] - row_res["Fragment_LEAE_min"] if pd.notna(row_res["Fragment_LEAE_max"]) and pd.notna(row_res["Fragment_LEAE_min"]) else np.nan
    
    # ESP统计
    esp_min_vals = all_atom_props['ESP_min'].dropna().values
    esp_max_vals = all_atom_props['ESP_max'].dropna().values
    esp_avg_vals = all_atom_props['ESP_avg'].values
    
    row_res["Fragment_ESP_min"] = np.min(esp_min_vals) if len(esp_min_vals) > 0 else np.nan
    row_res["Fragment_ESP_max"] = np.max(esp_max_vals) if len(esp_max_vals) > 0 else np.nan
    row_res["Fragment_ESP_mean"] = np.sum(esp_avg_vals * areas) / total_area if total_area > 0 else np.mean(esp_avg_vals)
    row_res["Fragment_ESP_delta"] = row_res["Fragment_ESP_max"] - row_res["Fragment_ESP_min"] if pd.notna(row_res["Fragment_ESP_max"]) and pd.notna(row_res["Fragment_ESP_min"]) else np.nan
    
    # ALIE统计
    alie_min_vals = all_atom_props['ALIE_min'].dropna().values
    alie_max_vals = all_atom_props['ALIE_max'].dropna().values
    alie_avg_vals = all_atom_props['ALIE_avg'].values
    
    row_res["Fragment_ALIE_min"] = np.min(alie_min_vals) if len(alie_min_vals) > 0 else np.nan
    row_res["Fragment_ALIE_max"] = np.max(alie_max_vals) if len(alie_max_vals) > 0 else np.nan
    row_res["Fragment_ALIE_mean"] = np.sum(alie_avg_vals * areas) / total_area if total_area > 0 else np.mean(alie_avg_vals)
    row_res["Fragment_ALIE_delta"] = row_res["Fragment_ALIE_max"] - row_res["Fragment_ALIE_min"] if pd.notna(row_res["Fragment_ALIE_max"]) and pd.notna(row_res["Fragment_ALIE_min"]) else np.nan
    
    # Area统计
    row_res["Fragment_area_min"] = np.min(areas)
    row_res["Fragment_area_max"] = np.max(areas)
    row_res["Fragment_area_mean"] = np.mean(areas)
    row_res["Fragment_area_delta"] = np.max(areas) - np.min(areas)
    
    # 官能团数量和总面积
    row_res["Fragment_Count"] = len(valid_indices) // substructure_size
    row_res["Fragment_Total_Area"] = total_area
    
    print(f"    LEAE: min={row_res['Fragment_LEAE_min']:.3f}, max={row_res['Fragment_LEAE_max']:.3f}, mean={row_res['Fragment_LEAE_mean']:.3f}, delta={row_res['Fragment_LEAE_delta']:.3f}")
    print(f"    ESP: min={row_res['Fragment_ESP_min']:.3f}, max={row_res['Fragment_ESP_max']:.3f}, mean={row_res['Fragment_ESP_mean']:.3f}, delta={row_res['Fragment_ESP_delta']:.3f}")
    print(f"    ALIE: min={row_res['Fragment_ALIE_min']:.3f}, max={row_res['Fragment_ALIE_max']:.3f}, mean={row_res['Fragment_ALIE_mean']:.3f}, delta={row_res['Fragment_ALIE_delta']:.3f}")
    print(f"    ✓ 官能团数量: {row_res['Fragment_Count']}, 总面积: {row_res['Fragment_Total_Area']:.3f}")
    
    return row_res

def create_empty_simplified_result(sample_name):
    """创建空的简化官能团结果"""
    row_res = {"Sample Name": sample_name}
    
    # 只保留整体特征：4个属性 × 4种统计 + 计数 + 总面积
    for pkey in ["LEAE", "ESP", "ALIE", "area"]:
        row_res[f"Fragment_{pkey}_min"] = np.nan
        row_res[f"Fragment_{pkey}_max"] = np.nan
        row_res[f"Fragment_{pkey}_mean"] = np.nan
        row_res[f"Fragment_{pkey}_delta"] = np.nan
    
    row_res["Fragment_Count"] = 0
    row_res["Fragment_Total_Area"] = np.nan
    
    return row_res

def run_atom_and_frag_pooling_internal(
    parent_dir,
    full_option_csv="FullOption2_3_47.csv",
    out_log_dir="AtomLevelLogs",
    final_output_csv="FullOption2_3_47_withAtomLevelFeatures.csv"
):
    """
    模式3: 使用新LOFFI算法进行完整功能团分析
    """
    print("模式3: 完整功能团分析")
    
    # 1) 先读原始FullOption
    csv_full_path = os.path.join(parent_dir, full_option_csv)
    if not os.path.exists(csv_full_path):
        raise FileNotFoundError(f"[ERROR] {csv_full_path} not found in {parent_dir}")

    df_option = pd.read_csv(csv_full_path, dtype={"Sample Name": str})
    df_option["Sample Name"] = df_option["Sample Name"].str.strip()

    # 2) 找到 AtomProp_*.csv 列表
    csv_list = sorted(glob.glob(os.path.join(parent_dir, "AtomProp_*.csv")))
    if len(csv_list) == 0:
        print(f"No AtomProp_*.csv found in {parent_dir}")
        return

    # 3) 确保日志目录存在
    log_dir_path = os.path.join(parent_dir, out_log_dir)
    os.makedirs(log_dir_path, exist_ok=True)

    results = []

    # 4) 遍历 prop CSV
    for csv_file in csv_list:
        base_name = os.path.basename(csv_file)
        sample_name_raw = base_name.replace("AtomProp_", "").replace(".csv", "").strip()
        
        try:
            sample_name = str(int(sample_name_raw))
        except ValueError:
            sample_name = sample_name_raw
            
        print(f" Processing sample {sample_name} (from file {base_name})")

        # 查找 FullOption 中对应行
        subset = df_option[df_option["Sample Name"] == sample_name]
        if len(subset) == 0:
            logging.warning(f"[WARN] {sample_name} not found in {full_option_csv}, skip.")
            continue
        row_info = subset.iloc[0]
        smiles_val = row_info["smiles"]
        target_val = row_info["target"]

        # 读取原子 CSV
        df_atom = read_atomprop_csv(csv_file)
        if df_atom is None:
            continue

        # Check columns
        col_LEAE = "LEAE All average"
        col_ESP  = "ESP All average (kcal/mol)"
        col_ALIE = "ALIE Average"
        col_AREA = "LEAE All area"
        needed_cols = [col_LEAE, col_ESP, col_ALIE, col_AREA]
        miss_cols = [col for col in needed_cols if col not in df_atom.columns]
        if miss_cols:
            logging.warning(f"[WARN] {csv_file} missing columns: {miss_cols}, skip.")
            continue

        # 构造 arr_atom
        n_atoms = df_atom.shape[0]
        arr_atom = np.full((n_atoms, 4), np.nan, dtype=float)
        for irow in range(n_atoms):
            arr_atom[irow, 0] = df_atom.loc[irow, col_LEAE]
            arr_atom[irow, 1] = df_atom.loc[irow, col_ESP]
            arr_atom[irow, 2] = df_atom.loc[irow, col_ALIE]
            arr_atom[irow, 3] = df_atom.loc[irow, col_AREA]

        # RDKit 解析
        mol = Chem.MolFromSmiles(smiles_val)

        # 打开日志
        log_path = os.path.join(log_dir_path, f"{sample_name}.txt")
        with open(log_path, 'w', encoding='utf-8') as f_log:
            f_log.write(f"Sample Name: {sample_name}\n")
            f_log.write(f"SMILES: {smiles_val}\n")
            f_log.write(f"target: {target_val}\n\n")

            # 原子级别
            f_log.write("Atom Part (LEAE, ESP, ALIE, area):\n")
            f_log.write(str(arr_atom))
            f_log.write("\n\n" + "=" * 80 + "\n")

            # 使用新的LOFFI算法
            f_log.write("Using LOFFI Algorithm with Priority System:\n")
            ring_groups, fun_groups = apply_loffi_algorithm(mol, smiles_val)
            
            f_log.write("Aromatic Systems:\n")
            if ring_groups:
                for gname, atm_set, grp_smiles in ring_groups:
                    f_log.write(f"  {gname} => atoms={sorted(atm_set)}\n")
            else:
                f_log.write("  No aromatic systems found.\n")
            f_log.write("\n" + "=" * 80 + "\n")

            f_log.write("Functional Groups (with Priority):\n")
            if fun_groups:
                for gname, atm_set, grp_smiles in fun_groups:
                    f_log.write(f"  {gname} => atoms={sorted(atm_set)}\n")
            else:
                f_log.write("  No functional groups found.\n")
            f_log.write("\n" + "=" * 80 + "\n")

            # area-weighted 4值计算
            f_log.write("Area-weighted Properties (one row per group):\n")
            all_groups = ring_groups + fun_groups

            summary_rows = []
            if not all_groups:
                f_log.write("No group matched.\n")
            else:
                for (grp_name, atm_set, grp_smiles) in all_groups:
                    vals_4 = calc_area_weighted_4props(arr_atom, atm_set)
                    f_log.write(f"  {grp_name} => atoms={sorted(atm_set)}\n")
                    f_log.write("    => [LEAE={:.4f}, ESP={:.4f}, ALIE={:.4f}, area={:.4f}]\n".format(
                        vals_4[0], vals_4[1], vals_4[2], vals_4[3]
                    ))
                    summary_rows.append(vals_4)

                mat = np.array(summary_rows)  # (M,4)
                f_log.write("\nSummary matrix:\n")
                f_log.write(str(mat))
                f_log.write("\n\n" + "="*80 + "\n")

        # 组装 result 行
        row_res = {
            "Sample Name": sample_name,
            "smiles": smiles_val,
            "target": target_val
        }

        # ========== 原子级 16 项 ==========
        for icol, pkey in enumerate(["LEAE", "ESP", "ALIE", "area"]):
            col_data = arr_atom[:, icol]
            valid_vals = col_data[~np.isnan(col_data)]
            pmin, pmax, pmean, pdelta = pool_4modes(valid_vals) if len(valid_vals) > 0 else (np.nan,)*4
            row_res[f"Atom_{pkey}_min"]   = pmin
            row_res[f"Atom_{pkey}_max"]   = pmax
            row_res[f"Atom_{pkey}_mean"]  = pmean
            row_res[f"Atom_{pkey}_delta"] = pdelta

        # ========== 功能团 16 项 ==========
        if len(summary_rows) > 0:
            mat_fun = np.array(summary_rows)  # shape=(M,4)
            for icol, pkey in enumerate(["LEAE", "ESP", "ALIE", "area"]):
                col_data = mat_fun[:, icol]
                valid_vals = col_data[~np.isnan(col_data)]
                pmin, pmax, pmean, pdelta = pool_4modes(valid_vals) if len(valid_vals)>0 else (np.nan,)*4
                row_res[f"Fun_{pkey}_min"]   = pmin
                row_res[f"Fun_{pkey}_max"]   = pmax
                row_res[f"Fun_{pkey}_mean"]  = pmean
                row_res[f"Fun_{pkey}_delta"] = pdelta
        else:
            for pkey in ["LEAE", "ESP", "ALIE", "area"]:
                row_res[f"Fun_{pkey}_min"]   = np.nan
                row_res[f"Fun_{pkey}_max"]   = np.nan
                row_res[f"Fun_{pkey}_mean"]  = np.nan
                row_res[f"Fun_{pkey}_delta"] = np.nan

        results.append(row_res)

    # 结果汇总
    df_out = pd.DataFrame(results)

    # 合并
    df_merged = pd.merge(
        df_option,
        df_out.drop(["smiles","target"], axis=1),
        on="Sample Name",
        how="left"
    )

    # 重新排列列顺序
    all_columns = df_merged.columns.tolist()
    special_cols = ['Sample Name', 'smiles', 'target']
    other_cols = [col for col in all_columns if col not in special_cols]
    new_column_order = ['Sample Name'] + other_cols + ['smiles', 'target']
    df_merged = df_merged[new_column_order]

    # 保存文件
    out_path = os.path.join(parent_dir, final_output_csv)
    df_merged.to_csv(out_path, index=False)

    print(f"[Done] Wrote merged CSV => {final_output_csv} in {parent_dir}")
    print(f"Logs => {log_dir_path}/<SampleName>.txt")

###############################################################################
# 5. 使用示例
###############################################################################

if __name__ == "__main__":
    # 示例1: 模式1 - 只提取S原子的性质
#     original_csv_path = "/home/yumingsu/Python/Project_Surfacia/250627_ShenyanS/Surfacia_3.0_20250711_184324/FullOption2_82_55.csv"  
#     run_atom_prop_extraction(original_csv_path, mode=1, target_element='S')
    
    # 示例2: 模式2 - 提取酰胺基官能团的性质
    original_csv_path = "/home/yumingsu/Python/Project_Surfacia/250628_zmm/Surfacia_3.0_20250711_224420/FullOption2_44_53.csv"
    xyz1_path = "/home/yumingsu/Python/Project_Surfacia/250628_zmm/sub.xyz1"
    xyz_folder="/home/yumingsu/Python/Project_Surfacia/250628_zmm"
    run_atom_prop_extraction(original_csv_path, mode=2, xyz1_path=xyz1_path, xyz_folder=xyz_folder, threshold=1.01)
    
    # 示例3: 模式3 - 完整的loffi功能团分析def run_atom_prop_extraction
    # original_csv_path = "/home/yumingsu/Python//Project_Surfacia/250627_ShenyanS/Surfacia_2.4_20250627_151721/FullOption2_82_49.csv"    
    # run_atom_prop_extraction(original_csv_path, mode=3)


# machine_learning,multi stepreg,ChemMLanalyzer

# In[ ]:


"""
化学机器学习特征选择与分析工具包 - 完整版
适用于 Jupyter Notebook
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import json
import gc
from pathlib import Path
import shap
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')  # 设置为非交互式后端

# ================== 基础分析器类 ==================
class BaseChemMLAnalyzer:
    """
    基础化学机器学习分析器 - 包含所有共享功能
    """
    
    def __init__(self, 
                 data_file, 
                 test_sample_names=None, 
                 nan_handling='drop_columns',
                 output_dir=None):
        """
        初始化基础分析器
        """
        self.data_file = data_file
        self.test_sample_names = test_sample_names or []
        self.nan_handling = nan_handling
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.output_dir = output_dir or str(Path(data_file).parent)
        
        # XGBoost默认参数
        self.xgb_params = {
            'n_estimators': 350,
            'learning_rate': 0.03,
            'max_depth': 8,
            'verbosity': 0,
            'booster': 'gbtree',
            'reg_alpha': np.exp(-3),
            'reg_lambda': np.exp(-3),
            'gamma': np.exp(-5),
            'subsample': 0.5,
            'objective': 'reg:squarederror',
        }
        
        # 数据将在需要时加载
        self.data_loaded = False

    def _ensure_data_loaded(self):
        """确保数据已加载"""
        if not self.data_loaded:
            self._load_and_prepare_data()
            self.data_loaded = True

    def _load_and_prepare_data(self):
        """加载和预处理数据"""
        print("Loading and preprocessing data...")

        # 读取原始CSV
        original_df = pd.read_csv(self.data_file)
        print(f"Original data shape: {original_df.shape}")

        # 处理NaN
        if self.nan_handling == 'drop_rows':
            original_df = original_df.dropna()
            print(f"After dropping NaN rows: {original_df.shape}")
        elif self.nan_handling == 'drop_columns':
            original_df = original_df.dropna(axis=1)
            print(f"After dropping NaN columns: {original_df.shape}")

        # 重置索引
        original_df = original_df.reset_index(drop=True)

        # 生成特征矩阵文件
        feature_files = self._generate_feature_matrix(original_df)

        # 加载数据
        self.X = np.loadtxt(feature_files['features'], delimiter=',')
        self.y = np.loadtxt(feature_files['values'])
        self.feature_names = np.loadtxt(feature_files['titles'], dtype=str, delimiter=',', comments='!')

        # 加载SMILES和原始数据
        self.smiles_data = np.loadtxt(feature_files['smiles'], dtype=str, delimiter=',', comments='!')
        self.sample_names = original_df['Sample Name'].values

        # 根据样本名称确定测试集
        test_mask = np.zeros(len(self.y), dtype=bool)
        test_indices_found = []
        test_names_found = []
        test_names_not_found = []

        if self.test_sample_names:
            for test_name in self.test_sample_names:
                matching_indices = np.where(self.sample_names == str(test_name))[0]
                if len(matching_indices) == 0:
                    try:
                        matching_indices = np.where(self.sample_names.astype(float) == float(test_name))[0]
                    except:
                        pass

                if len(matching_indices) > 0:
                    idx = matching_indices[0]
                    test_mask[idx] = True
                    test_indices_found.append(idx + 1)
                    test_names_found.append(str(test_name))
                else:
                    test_names_not_found.append(str(test_name))

            if test_names_found:
                print(f"Found {len(test_names_found)} test samples:")
                for i, (name, idx) in enumerate(zip(test_names_found, test_indices_found)):
                    print(f"  {i+1}. Sample '{name}' at position {idx}")

            if test_names_not_found:
                print(f"Warning: {len(test_names_not_found)} test samples not found:")
                for name in test_names_not_found:
                    print(f"  - Sample '{name}'")
        
        # 分割数据
        self.X_train = self.X[~test_mask]
        self.y_train = self.y[~test_mask]
        self.X_realtest = self.X[test_mask]
        self.y_realtest = self.y[test_mask]
        self.smiles_train = self.smiles_data[~test_mask]
        self.smiles_realtest = self.smiles_data[test_mask]
        self.sample_names_train = self.sample_names[~test_mask]
        self.sample_names_realtest = self.sample_names[test_mask]
        
        # 记录数据集信息
        self.dataset_info = {
            'original_samples': len(self.y),
            'original_features': len(self.feature_names),
            'train_samples': len(self.y_train),
            'test_samples': len(self.y_realtest),
            'test_sample_names_input': self.test_sample_names,
            'test_sample_names_found': test_names_found,
            'test_sample_names_not_found': test_names_not_found,
            'test_indices_1based': test_indices_found,
            'nan_handling_method': self.nan_handling,
            'train_test_split_method': 'Sample name specification' if self.test_sample_names else 'No test set specified'
        }
        
        print(f"Data loaded: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        print(f"Train set: {len(self.y_train)} samples")
        print(f"Test set: {len(self.y_realtest)} samples")

    def _generate_feature_matrix(self, merged_df):
        """生成特征矩阵文件 - 子类应该重写此方法"""
        raise NotImplementedError("Subclasses must implement _generate_feature_matrix")

    def _setup_output_directory(self, dir_prefix):
        """设置输出目录"""
        DIR_name = f'{dir_prefix}_{self.timestamp}'
        self.ML_DIR = Path(self.output_dir, DIR_name)
        os.makedirs(self.ML_DIR, exist_ok=True)
        return self.ML_DIR

    def _save_feature_files(self, merged_df):
        """保存特征文件"""
        S_N = merged_df.shape[0]
        F_N = merged_df.shape[1] - 3
        
        files = {}
        files['smiles'] = str(self.ML_DIR / f'Smiles_{S_N}.csv')
        files['values'] = str(self.ML_DIR / f'Values_True_{S_N}.csv')
        files['features'] = str(self.ML_DIR / f'Features_{S_N}_{F_N}.csv')
        files['titles'] = str(self.ML_DIR / f'Title_{F_N}.csv')
        
        merged_df[['smiles']].to_csv(files['smiles'], index=False, header=False)
        merged_df[['target']].to_csv(files['values'], index=False, header=False)
        merged_df.drop(['Sample Name', 'smiles', 'target'], axis=1)\
                 .to_csv(files['features'], index=False, float_format='%.6f', header=False)
        
        with open(files['titles'], 'w') as f:
            for col in merged_df.columns[1:-2]:
                formatted_col = col.replace(' ', '_').replace('/', '_')
                f.write(formatted_col + '\n')
        
        return files

    def XGB_Fit(self, X, y, X_train, y_train, X_test, y_test, test_idx, paras):
        """单次XGBoost训练"""
        clf_new = XGBRegressor()
        for k, v in paras.items():
            clf_new.set_params(**{k: v})
        
        clf_new.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        y_pred = clf_new.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        shap_values = shap.TreeExplainer(clf_new).shap_values(X_test, check_additivity=False)
        
        return ([mse, mae, r2, shap_values, test_idx], clf_new)

    def _single_fit_wrapper(self, X, y, train_idx, test_idx, paras, X_realtest=None):
        """单次训练的包装函数"""
        X_train = X[train_idx, :]
        y_train = y[train_idx]
        X_test = X[test_idx, :]
        y_test = y[test_idx]
        
        results = self.XGB_Fit(X, y, X_train, y_train, X_test, y_test, test_idx, paras)
        
        return results, train_idx, test_idx

    def poolfit_optimized(self, TRAIN_TEST_SPLIT, EPOCH, CORE_NUM, X, y, paras, X_realtest=None):
        """优化的并行训练函数，支持测试集SHAP值计算"""
        print(f"Starting optimized training: {EPOCH} epochs, {CORE_NUM} cores")
        
        point = round(X.shape[0] * TRAIN_TEST_SPLIT)
        
        # 初始化存储
        sample_shap_values = {}
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                sample_shap_values[(i, j)] = []
    
        # 生成所有的训练-测试划分
        all_splits = []
        for epoch in range(EPOCH):
            permutation = np.random.permutation(y.shape[0])
            train_idx = permutation[:point]
            test_idx = permutation[point:]
            all_splits.append((train_idx, test_idx))
    
        # 分批处理
        batch_size = max(1, CORE_NUM)
        all_results = []
        
        for batch_start in range(0, EPOCH, batch_size):
            batch_end = min(batch_start + batch_size, EPOCH)
            batch_splits = all_splits[batch_start:batch_end]
            
            try:
                batch_results = Parallel(
                    n_jobs=CORE_NUM, 
                    backend='threading',
                    timeout=300,
                    verbose=0
                )(
                    delayed(self._single_fit_wrapper)(X, y, train_idx, test_idx, paras, X_realtest)
                    for train_idx, test_idx in batch_splits
                )
                
                all_results.extend(batch_results)
                gc.collect()
                
            except Exception as e:
                print(f"Batch processing failed: {e}")
                print("Falling back to single-threaded processing...")
                
                for train_idx, test_idx in batch_splits:
                    try:
                        result = self._single_fit_wrapper(X, y, train_idx, test_idx, paras, X_realtest)
                        all_results.append(result)
                    except Exception as e2:
                        print(f"Single fit also failed: {e2}")
                        continue
    
        if not all_results:
            raise RuntimeError("All training attempts failed")
    
        print(f"Successfully completed {len(all_results)} training rounds")
    
        # 处理结果
        mse_list = []
        mae_list = []
        r2_list = []
        full_m = np.zeros((len(all_results), X.shape[0]))
        y_realtest_pred_list = []
        realtest_shap_list = []  # 新增：存储测试集SHAP值
        split_l = []
        test_idx_m = []
        
        sample_counts = np.zeros((X.shape[0], X.shape[1]))
    
        for i, (results_tuple, train_idx, test_idx) in enumerate(all_results):
            temp = results_tuple[0]
            mse_list.append(temp[0])
            mae_list.append(temp[1])
            r2_list.append(temp[2])
            
            shap_values = temp[3]
            test_indices = temp[4]
            
            for j, test_idx_val in enumerate(test_indices):
                for k in range(X.shape[1]):
                    sample_shap_values[(test_idx_val, k)].append(shap_values[j, k])
                    sample_counts[test_idx_val, k] += 1
            
            clf_new = results_tuple[1]
            y_full_pred = clf_new.predict(X)
            full_m[i] = y_full_pred
            
            split_l.append(train_idx)
            test_idx_m.append(test_idx)
    
            # 计算真实测试集的预测和SHAP值
            if X_realtest is not None:
                y_realtest_pred = clf_new.predict(X_realtest)
                y_realtest_pred_list.append(y_realtest_pred)
                
                # 计算测试集SHAP值
                realtest_shap = shap.TreeExplainer(clf_new).shap_values(X_realtest, check_additivity=False)
                realtest_shap_list.append(realtest_shap)
    
        # 计算平均SHAP值和标准差（训练集）
        shap_m2 = np.zeros((X.shape[0], X.shape[1]))
        shap_std = np.zeros((X.shape[0], X.shape[1]))
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if sample_counts[i, j] > 0:
                    values = sample_shap_values[(i, j)]
                    shap_m2[i, j] = np.mean(values) if values else 0
                    shap_std[i, j] = np.std(values) if len(values) > 1 else 0
    
        # 计算测试集的平均SHAP值和标准差
        realtest_shap_mean = None
        realtest_shap_std = None
        if realtest_shap_list:
            realtest_shap_mean = np.mean(realtest_shap_list, axis=0)
            realtest_shap_std = np.std(realtest_shap_list, axis=0)
    
        mse1 = np.mean(mse_list)
        mae1 = np.mean(mae_list)
        r21 = np.mean(r2_list)
        mse2 = np.std(mse_list)
    
        y_realtest_pred_std = np.std(y_realtest_pred_list, axis=0) if y_realtest_pred_list else None
        y_realtest_pred_mean = np.mean(y_realtest_pred_list, axis=0) if y_realtest_pred_list else None
    
        print(f"Training completed: MSE={mse1:.4f}±{mse2:.4f}, MAE={mae1:.4f}, R²={r21:.4f}")
    
        return {
            'mse': mse1,
            'mae': mae1,
            'r2': r21,
            'mse_std': mse2,
            'full_predictions': full_m,
            'test_indices': test_idx_m,
            'shap_values': shap_m2,
            'shap_std': shap_std,
            'y_realtest_pred': y_realtest_pred_mean,
            'y_realtest_pred_std': y_realtest_pred_std,
            'realtest_shap_mean': realtest_shap_mean,  # 新增
            'realtest_shap_std': realtest_shap_std     # 新增
        }

    def save_comprehensive_results(self, save_dir, prefix, feature_names, feature_indices, 
                                  X_selected, perf_dict):
        """保存综合结果，包含测试集SHAP值"""
        print(f"Saving comprehensive results...")
        
        # 保存训练集详细CSV
        train_data = []
        
        # 添加特征列
        for i, feat_name in enumerate(feature_names):
            train_data.append(X_selected[:, i])
        
        # 添加SHAP列
        shap_values = perf_dict['shap_values']
        shap_std = perf_dict['shap_std']
        
        for i, feat_name in enumerate(feature_names):
            train_data.append(shap_values[:, i])
        
        # 添加SHAP标准差列
        for i, feat_name in enumerate(feature_names):
            train_data.append(shap_std[:, i])
        
        # 添加其他信息
        train_data.append(self.smiles_train)
        train_data.append(self.y_train)
        train_data.append(self.sample_names_train)
        
        # 转置以便按行保存
        train_data = np.column_stack(train_data)
        
        # 创建列名
        feature_cols = [f"Feature_{feat}" for feat in feature_names]
        shap_cols = [f"SHAP_{feat}" for feat in feature_names]
        shap_std_cols = [f"SHAP_Std_{feat}" for feat in feature_names]
        other_cols = ["SMILES", "Target", "Sample_Name"]
        column_names = feature_cols + shap_cols + shap_std_cols + other_cols
        
        # 保存训练集详细CSV
        train_df = pd.DataFrame(train_data, columns=column_names)
        save_path = save_dir / f'Training_Set_Detailed_{prefix}_{self.timestamp}.csv'
        train_df.to_csv(save_path, index=False)
        
        # 保存测试集CSV（包含SHAP值）
        if len(self.X_realtest) > 0:
            X_realtest_selected = self.X_realtest[:, feature_indices]
            
            test_data = []
            
            # 添加特征列
            for i, feat_name in enumerate(feature_names):
                test_data.append(X_realtest_selected[:, i])
            
            # 添加测试集SHAP列（如果有）
            if perf_dict.get('realtest_shap_mean') is not None:
                for i, feat_name in enumerate(feature_names):
                    test_data.append(perf_dict['realtest_shap_mean'][:, i])
                
                # 添加测试集SHAP标准差列
                for i, feat_name in enumerate(feature_names):
                    test_data.append(perf_dict['realtest_shap_std'][:, i])
                
                shap_test_cols = [f"SHAP_{feat}" for feat in feature_names]
                shap_test_std_cols = [f"SHAP_Std_{feat}" for feat in feature_names]
            else:
                shap_test_cols = []
                shap_test_std_cols = []
            
            # 添加其他信息
            test_data.append(self.smiles_realtest)
            test_data.append(self.y_realtest)
            test_data.append(self.sample_names_realtest)
            
            # 添加预测值
            y_realtest_pred = perf_dict['y_realtest_pred']
            y_realtest_pred_std = perf_dict['y_realtest_pred_std']
            if y_realtest_pred is not None:
                test_data.append(y_realtest_pred)
                test_data.append(y_realtest_pred_std)
                prediction_cols = ["Realtest_Pred", "Realtest_Pred_Std"]
            else:
                prediction_cols = []
            
            test_data = np.column_stack(test_data)
            
            # 创建列名
            test_feature_cols = [f"Feature_{feat}" for feat in feature_names]
            test_other_cols = ["SMILES", "Target", "Sample_Name"]
            test_column_names = test_feature_cols + shap_test_cols + shap_test_std_cols + test_other_cols + prediction_cols
            
            test_df = pd.DataFrame(test_data, columns=test_column_names)
            save_path = save_dir / f'Test_Set_Detailed_{prefix}_{self.timestamp}.csv'
            test_df.to_csv(save_path, index=False)

    def generate_prediction_scatter(self, perf_dict, feature_names, title_prefix="", save_dir=None):
        """生成预测散点图并保存原始数据"""
        if save_dir is None:
            save_dir = self.ML_DIR

        n_features = len(feature_names)
        full_m = np.array(perf_dict['full_predictions'])
        test_idx_m = np.array(perf_dict['test_indices'])

        # 计算测试数据的统计量
        test_data_m = [[] for _ in range(len(self.y_train))]
        for i in range(test_idx_m.shape[0]):
            for k in range(test_idx_m.shape[1]):
                test_data_m[test_idx_m[i, k]].append(full_m[i, test_idx_m[i, k]])

        test_mean_l = [np.mean(item) if item else 0 for item in test_data_m]
        test_std_l = [np.std(item) if item else 0 for item in test_data_m]
        true_y = self.y_train.flatten().tolist()

        # 绘制散点图
        plt.figure(figsize=(12, 10), dpi=300)
        sc = plt.scatter(true_y, test_mean_l, alpha=0.55, c=test_std_l, cmap='viridis', marker='o', s=60)
        left_limit = min(min(true_y) - 1, min(test_mean_l) - 1)
        right_limit = max(max(true_y) + 1, max(test_mean_l) + 1)
        plt.plot([left_limit, right_limit], [left_limit, right_limit],
                 color='#B22222', linestyle=':', linewidth=3)
        plt.plot([left_limit, right_limit], [left_limit + 1, right_limit + 1],
                 color='#FFA500', linestyle=':', linewidth=3)
        plt.plot([left_limit, right_limit], [left_limit - 1, right_limit - 1],
                 color='#FFA500', linestyle=':', linewidth=3)
        plt.legend(['Perfect Fit', '+1', '-1', 'Cross-Validation Predictions'],
                   loc='upper left', fontsize=20, shadow=True)
        plt.xlabel('True Values', fontsize=24)
        plt.ylabel('Mean Predicted Values', fontsize=24)
        plt.title(f'{title_prefix} Model: Predicted vs True Values ({n_features} Features)\n'
                  f'MSE: {perf_dict["mse"]:.4f}±{perf_dict["mse_std"]:.4f}  MAE: {perf_dict["mae"]:.4f}  R²: {perf_dict["r2"]:.4f}',
                  fontsize=26)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        cb = plt.colorbar(sc)
        cb.set_label('Std of Predictions', fontsize=20)
        cb.ax.tick_params(labelsize=16)
        plt.grid(which='major', color='#D5D5D5', alpha=0.5)
        plt.tight_layout()

        # 保存图片
        save_name = save_dir / f'{title_prefix}_Prediction_Scatter_{n_features}feats_{self.timestamp}.png'
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        plt.close()

        # 保存散点图原始数据（True vs Predicted）
        xgbscatterdata = np.column_stack((true_y, test_mean_l, test_std_l))
        data_save_name = save_dir / f'{title_prefix}_Prediction_Data_{n_features}_Features_MSE{round(perf_dict["mse"],3)}_MSEstd{round(perf_dict["mse_std"],3)}_MAE{round(perf_dict["mae"],3)}_R2{round(perf_dict["r2"],3)}.txt'
        np.savetxt(data_save_name, xgbscatterdata, fmt='%.6f', delimiter=',', 
                   header='True_Value,Mean_Prediction,Std_Prediction', comments='')

        print(f"  Scatter plot saved: {save_name.name}")
        print(f"  Raw data saved: {data_save_name.name}")

    def generate_shap_plots(self, feat_vals, shap_vals, shap_std_vals, feat_name, save_dir):
        """生成SHAP分析图，包含拟合曲线"""
        # 创建SHAP图目录
        shap_plots_dir = save_dir / 'SHAP_Plots'
        os.makedirs(shap_plots_dir, exist_ok=True)

        # 创建原始数据目录
        shap_data_dir = save_dir / 'SHAP_Raw_Data'
        os.makedirs(shap_data_dir, exist_ok=True)

        # 保存原始数据
        raw_data = np.column_stack((feat_vals, shap_vals, shap_std_vals))
        data_save_path = shap_data_dir / f'{feat_name}_raw_data_{self.timestamp}.csv'
        np.savetxt(data_save_path, raw_data, fmt='%.6f', delimiter=',',
                  header='Feature_Value,SHAP_Value,SHAP_Std', comments='')

        # 生成SHAP散点图
        plt.figure(figsize=(10, 8))
        plt.scatter(feat_vals, shap_vals, alpha=0.6, s=40, c='steelblue')
        plt.errorbar(feat_vals, shap_vals, yerr=shap_std_vals, 
                    fmt='none', ecolor='lightgray', alpha=0.3, capsize=0)

        # 尝试拟合
        try:
            fits, best_fit_type, best_params, best_r2 = self.fit_various_functions(feat_vals, shap_vals)

            if best_r2 > 0.1:  # 只有R²大于0.1时才画拟合线
                x_plot = np.linspace(min(feat_vals), max(feat_vals), 100)
                y_plot = self._eval_fit_func(x_plot, best_fit_type, best_params)

                # 获取拟合公式
                if best_fit_type in fits:
                    formula = fits[best_fit_type][2]
                else:
                    formula = f'{best_fit_type} fit'

                plt.plot(x_plot, y_plot, 'r-', linewidth=2, alpha=0.7,
                        label=f'{formula}\n(R²={best_r2:.3f})')
                plt.legend(fontsize=12, loc='best')

                # 保存拟合信息
                fit_info_path = shap_data_dir / f'{feat_name}_fit_info_{self.timestamp}.txt'
                with open(fit_info_path, 'w') as f:
                    f.write(f'Feature: {feat_name}\n')
                    f.write(f'Best fit type: {best_fit_type}\n')
                    f.write(f'R²: {best_r2:.4f}\n')
                    f.write(f'Formula: {formula}\n')
                    f.write('\nAll fits tested:\n')
                    for fit_type, (params, r2, formula) in fits.items():
                        f.write(f'  {fit_type}: R²={r2:.4f}, {formula}\n')
        except Exception as e:
            print(f"    Warning: Fitting failed for {feat_name}: {str(e)}")

        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        plt.xlabel(f'{feat_name} Value', fontsize=16)
        plt.ylabel('SHAP Value', fontsize=16)
        plt.title(f'SHAP Analysis: {feat_name}', fontsize=18)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = shap_plots_dir / f'SHAP_{feat_name}_{self.timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    def fit_various_functions(self, x, y, min_points=10, r2_threshold=0.01):
        """对SHAP散点图进行多种函数拟合并进行质量评估"""
        def calc_r2(y_true, y_pred):
            """计算R²值，处理边界情况"""
            if len(y_true) < 2:
                return 0
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            if ss_tot < 1e-10:  # 几乎为常数
                return 0
            r2 = 1 - (ss_res / ss_tot)
            return max(0, min(1, r2))  # 确保R²在[0,1]范围内

        def format_number(num):
            """格式化数字，处理科学计数法"""
            if abs(num) < 1e-3 or abs(num) > 1e3:
                return f"{num:.2e}"
            return f"{num:.3f}"

        def is_categorical(x, max_unique=5):
            """判断是否为分类变量"""
            unique_vals = np.unique(x)
            return len(unique_vals) <= max_unique

        def fit_categorical(x, y):
            """对分类变量进行拟合"""
            unique_vals = np.unique(x)
            means = {}
            for val in unique_vals:
                mask = x == val
                if np.sum(mask) > 0:
                    means[val] = np.mean(y[mask])

            # 创建预测值数组
            y_pred = np.zeros_like(y)
            for val, mean_val in means.items():
                y_pred[x == val] = mean_val

            # 计算R²
            r2 = calc_r2(y, y_pred)

            # 创建公式字符串
            formula_parts = [f"y = {format_number(v)} (x = {format_number(k)})" 
                            for k, v in means.items()]
            formula = ", ".join(formula_parts)

            return means, r2, formula

        # 初始化
        fits = {}
        best_r2 = -np.inf
        best_fit_type = None
        best_params = None

        # 检查数据量
        if len(x) < min_points:
            return {'constant': ([], 0, 'y = const')}, 'constant', [], 0

        # 检查是否为分类变量
        if is_categorical(x):
            params, r2, formula = fit_categorical(x, y)
            if r2 > r2_threshold:
                fits['categorical'] = (params, r2, formula)
                return fits, 'categorical', params, r2

        # 对于连续变量，进行标准化
        x_scale = np.std(x)
        y_scale = np.std(y)
        if x_scale < 1e-10 or y_scale < 1e-10:
            return {'constant': ([], 0, 'y = const')}, 'constant', [], 0

        try:
            # 1. 多项式拟合
            for deg in range(1, 4):
                try:
                    coeffs = np.polyfit(x, y, deg)
                    y_pred = np.polyval(coeffs, x)
                    r2 = calc_r2(y, y_pred)

                    if r2 > r2_threshold:
                        terms = []
                        for i, c in enumerate(coeffs[::-1]):
                            if abs(c) < 1e-10:
                                continue
                            if i == 0:
                                terms.append(format_number(c))
                            elif i == 1:
                                terms.append(f"{format_number(c)}x")
                            else:
                                terms.append(f"{format_number(c)}x^{i}")
                        formula = "y = " + " + ".join(reversed(terms))

                        fits[f'poly_{deg}'] = (coeffs, r2, formula)

                        if r2 > best_r2:
                            best_r2 = r2
                            best_fit_type = f'poly_{deg}'
                            best_params = coeffs
                except:
                    continue

            # 2. 指数函数拟合
            def exp_func(x_, a, b, c):
                return a * np.exp(b * x_) + c

            try:
                p0 = [(np.max(y) - np.min(y)), 1.0 / x_scale, np.min(y)]
                popt, _ = curve_fit(exp_func, x, y, p0=p0, maxfev=5000)
                y_pred = exp_func(x, *popt)
                r2 = calc_r2(y, y_pred)

                if r2 > r2_threshold:
                    formula = f"y = {format_number(popt[0])}*exp({format_number(popt[1])}x) + {format_number(popt[2])}"
                    fits['exp'] = (popt, r2, formula)

                    if r2 > best_r2:
                        best_r2 = r2
                        best_fit_type = 'exp'
                        best_params = popt
            except:
                pass

            # 3. 对数函数拟合
            def log_func(x_, a, b, c):
                return a * np.log(x_ + b) + c

            try:
                min_x_shift = max(0, -np.min(x)) + 1e-3
                p0 = [(np.max(y) - np.min(y)) / np.log(10), min_x_shift, np.min(y)]
                popt, _ = curve_fit(log_func, x, y, p0=p0, maxfev=5000)
                y_pred = log_func(x, *popt)
                r2 = calc_r2(y, y_pred)

                if r2 > r2_threshold:
                    formula = f"y = {format_number(popt[0])}*ln(x + {format_number(popt[1])}) + {format_number(popt[2])}"
                    fits['log'] = (popt, r2, formula)

                    if r2 > best_r2:
                        best_r2 = r2
                        best_fit_type = 'log'
                        best_params = popt
            except:
                pass

            # 4. Sigmoid函数拟合
            def sigmoid_func(x_, a, b, c, d):
                return a / (1 + np.exp(-b * (x_ - c))) + d

            try:
                p0 = [max(y) - min(y), 4 / x_scale, np.median(x), np.min(y)]
                popt, _ = curve_fit(sigmoid_func, x, y, p0=p0, maxfev=8000)
                y_pred = sigmoid_func(x, *popt)
                r2 = calc_r2(y, y_pred)

                if r2 > r2_threshold:
                    formula = f"y = {format_number(popt[0])}/(1 + e^(-{format_number(popt[1])}*(x - {format_number(popt[2])}))) + {format_number(popt[3])}"
                    fits['sigmoid'] = (popt, r2, formula)

                    if r2 > best_r2:
                        best_r2 = r2
                        best_fit_type = 'sigmoid'
                        best_params = popt
            except:
                pass

            # 5. 分段线性函数拟合
            percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
            best_threshold_r2 = -np.inf
            best_threshold_params = None

            for p in percentiles:
                threshold = np.percentile(x, p)
                mask = x >= threshold

                if np.sum(mask) < min_points or np.sum(~mask) < min_points:
                    continue

                try:
                    left_value = np.mean(y[~mask]) if np.any(~mask) else 0
                    if np.sum(mask) >= 2:
                        right_coeffs = np.polyfit(x[mask], y[mask], 1)
                        y_pred = np.full_like(y, left_value)
                        y_pred[mask] = np.polyval(right_coeffs, x[mask])
                        r2 = calc_r2(y, y_pred)

                        if r2 > best_threshold_r2:
                            best_threshold_r2 = r2
                            best_threshold_params = (threshold, left_value, right_coeffs)
                except:
                    continue

            if best_threshold_r2 > r2_threshold:
                threshold, left_value, right_coeffs = best_threshold_params
                formula = f"y = {format_number(left_value)} (x < {format_number(threshold)}), {format_number(right_coeffs[0])}x + {format_number(right_coeffs[1])} (x >= {format_number(threshold)})"
                fits['piecewise'] = (best_threshold_params, best_threshold_r2, formula)

                if best_threshold_r2 > best_r2:
                    best_r2 = best_threshold_r2
                    best_fit_type = 'piecewise'
                    best_params = best_threshold_params

        except Exception as e:
            print(f"Warning: Fitting error occurred: {str(e)}")
            return {'error': ([], 0, 'fitting failed')}, 'error', [], 0

        if best_r2 < r2_threshold:
            return {'constant': ([], 0, 'y = const')}, 'constant', [], 0

        return fits, best_fit_type, best_params, best_r2
    
    def _eval_fit_func(self, x, fit_type, params):
        """评估拟合函数，处理所有拟合类型"""
        if fit_type.startswith('poly_'):
            return np.polyval(params, x)
        elif fit_type == 'exp':
            a, b, c = params
            return a * np.exp(b * x) + c
        elif fit_type == 'log':
            a, b, c = params
            return a * np.log(x + b) + c
        elif fit_type == 'sigmoid':
            a, b, c, d = params
            return a / (1 + np.exp(-b * (x - c))) + d
        elif fit_type == 'categorical':
            y_pred = np.zeros_like(x)
            for val, mean_val in params.items():
                y_pred[x == val] = mean_val
            return y_pred
        elif fit_type == 'piecewise':
            threshold, left_value, right_coeffs = params
            y_pred = np.full_like(x, left_value)
            mask = x >= threshold
            y_pred[mask] = np.polyval(right_coeffs, x[mask])
            return y_pred
        return np.zeros_like(x)
    def generate_record_file(self, save_dir, analysis_type, results, mode="auto"):
        """
        生成详细的记录文件

        Args:
            save_dir: 保存目录
            analysis_type: 分析类型名称
            results: 结果字典
            mode: "auto", "manual", "workflow", "baseline"
        """
        record_path = save_dir / f'Record_{analysis_type}_{self.timestamp}.txt'

        with open(record_path, 'w', encoding='utf-8') as f:
            f.write('='*80 + '\n')
            f.write(f'CHEMICAL MACHINE LEARNING - {analysis_type.upper()} ANALYSIS RECORD\n')
            f.write('='*80 + '\n\n')

            # 基本信息
            f.write('BASIC INFORMATION:\n')
            f.write('-'*40 + '\n')
            f.write(f'Analysis start time: {self.start_time}\n')
            f.write(f'Analysis end time: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'Timestamp: {self.timestamp}\n')
            f.write(f'Input data file: {self.data_file}\n')
            f.write(f'Output directory: {save_dir}\n\n')

            # 数据集信息
            f.write('DATASET INFORMATION:\n')
            f.write('-'*40 + '\n')
            f.write(f'Original samples: {self.dataset_info["original_samples"]}\n')
            f.write(f'Original features: {self.dataset_info["original_features"]}\n')
            f.write(f'Training samples: {self.dataset_info["train_samples"]}\n')
            f.write(f'Test samples: {self.dataset_info["test_samples"]}\n')
            f.write(f'NaN handling method: {self.dataset_info["nan_handling_method"]}\n\n')

            # 测试集信息
            if self.test_sample_names:
                f.write('TEST SET SPECIFICATION:\n')
                f.write('-'*40 + '\n')
                f.write(f'Input test sample names: {self.dataset_info["test_sample_names_input"]}\n')
                f.write(f'Successfully found {len(self.dataset_info["test_sample_names_found"])} samples:\n')
                for i, (name, idx) in enumerate(zip(self.dataset_info["test_sample_names_found"], 
                                                   self.dataset_info["test_indices_1based"])):
                    f.write(f'  {i+1}. "{name}" at position {idx}\n')

                if self.dataset_info["test_sample_names_not_found"]:
                    f.write(f'Warning: {len(self.dataset_info["test_sample_names_not_found"])} samples not found:\n')
                    for name in self.dataset_info["test_sample_names_not_found"]:
                        f.write(f'  - "{name}"\n')
                f.write('\n')

            # 特征选择信息
            if 'selected_features' in results:
                f.write('FEATURE SELECTION:\n')
                f.write('-'*40 + '\n')
                f.write(f'Number of selected features: {len(results["selected_features"])}\n')
                f.write('Selected features:\n')
                for i, feat in enumerate(results["selected_features"][:20]):  # 最多显示20个
                    f.write(f'  {i+1}. {feat}\n')
                if len(results["selected_features"]) > 20:
                    f.write(f'  ... and {len(results["selected_features"]) - 20} more features\n')
                f.write('\n')

            # 模型参数
            f.write('MODEL PARAMETERS:\n')
            f.write('-'*40 + '\n')
            f.write('XGBoost parameters:\n')
            for key, value in self.xgb_params.items():
                f.write(f'  {key}: {value}\n')
            f.write('\n')

            # 性能结果
            f.write('PERFORMANCE RESULTS:\n')
            f.write('-'*40 + '\n')
            f.write(f'Cross-validation performance:\n')
            f.write(f'  MSE: {results["mse"]:.6f} ± {results["mse_std"]:.6f}\n')
            f.write(f'  MAE: {results["mae"]:.6f}\n')
            f.write(f'  R²: {results["r2"]:.6f}\n\n')

            if 'test_performance' in results and results['test_performance']:
                f.write('Test set performance:\n')
                f.write(f'  Test MSE: {results["test_performance"]["mse"]:.6f}\n')
                f.write(f'  Test MAE: {results["test_performance"]["mae"]:.6f}\n')
                f.write(f'  Test R²: {results["test_performance"]["r2"]:.6f}\n\n')

                # 详细预测结果（如果测试集较小）
                if hasattr(self, 'y_realtest') and len(self.y_realtest) <= 20:
                    f.write('DETAILED TEST PREDICTIONS:\n')
                    f.write('-'*40 + '\n')
                    if 'y_realtest_pred' in results:
                        for i, (true_val, sample_name) in enumerate(zip(self.y_realtest, self.sample_names_realtest)):
                            f.write(f'  {i+1}. {sample_name}:\n')
                            f.write(f'     True value: {true_val:.6f}\n')
                            if results.get('y_realtest_pred') is not None:
                                pred_val = results['y_realtest_pred'][i]
                                pred_std = results['y_realtest_pred_std'][i] if results.get('y_realtest_pred_std') is not None else 0
                                f.write(f'     Predicted: {pred_val:.6f} ± {pred_std:.6f}\n')
                                f.write(f'     Error: {abs(true_val - pred_val):.6f}\n')
                            f.write('\n')

            # 输出文件信息
            f.write('OUTPUT FILES:\n')
            f.write('-'*40 + '\n')

            if mode == "workflow":
                f.write('1. Baseline analysis:\n')
                f.write('   - Baseline_Analysis/ folder\n')
                f.write('2. Stepwise regression:\n')
                f.write('   - Run_1/, Run_2/, Run_3/ folders\n')
                f.write('3. SHAP analysis:\n')
                f.write('   - SHAP_Plots/ and SHAP_Raw_Data/ in each Run folder\n')
                f.write('4. Final analysis:\n')
                f.write('   - Final_Manual_Analysis/ folder\n')
                f.write('5. Recommendations:\n')
                f.write(f'   - feature_recommendations_{self.timestamp}.csv\n')
                f.write(f'   - workflow_summary_{self.timestamp}.txt\n')
            else:
                n_features = len(results.get("selected_features", self.feature_names))
                f.write(f'1. Prediction scatter plot and data\n')
                f.write(f'2. Training set details CSV\n')
                f.write(f'3. Test set details CSV (with SHAP values)\n')
                f.write(f'4. SHAP plots in SHAP_Plots/ folder\n')
                f.write(f'5. SHAP raw data in SHAP_Raw_Data/ folder\n')
                f.write(f'6. This record file\n')

            f.write('\n')
            f.write('='*80 + '\n')
            f.write(f'{analysis_type.upper()} ANALYSIS COMPLETED SUCCESSFULLY\n')
            f.write('='*80 + '\n')
# ================== 手动特征分析器 ==================
class ManualFeatureAnalyzer(BaseChemMLAnalyzer):
    """
    手动特征分析器 - 分析用户指定的特征
    """
    
    def __init__(self, data_file, test_sample_names=None, nan_handling='drop_columns', 
                 output_dir=None, features=None):
        """
        初始化手动特征分析器
        
        Args:
            features: 要分析的特征名称列表，或 'Full' 表示所有特征
        """
        self.manual_features = features or []
        super().__init__(data_file, test_sample_names, nan_handling, output_dir)
        
    def _generate_feature_matrix(self, merged_df):
        """生成特征矩阵文件"""
        if self.manual_features == 'Full':
            self._setup_output_directory('Full_Feature_Analysis')
        else:
            self._setup_output_directory('Manual_Feature_Analysis')
        return self._save_feature_files(merged_df)
    
    def run(self, epoch=64, core_num=32, train_test_split=0.85, generate_fitting=True):
        """
        运行手动特征分析
        
        Returns:
            dict: 包含分析结果的字典
        """
        self._ensure_data_loaded()
        
        print("\n" + "="*60)
        print("MANUAL FEATURE ANALYSIS")
        print("="*60)
        
        # 确定要使用的特征
        if self.manual_features == 'Full':
            print("Using ALL features for analysis")
            selected_indices = list(range(len(self.feature_names)))
            selected_features = self.feature_names.tolist()
        else:
            # 验证特征
            selected_indices = []
            selected_features = []
            
            print(f"Validating {len(self.manual_features)} specified features...")
            for feat_name in self.manual_features:
                feat_idx_array = np.where(self.feature_names == feat_name)[0]
                if len(feat_idx_array) == 0:
                    print(f"Warning: Feature '{feat_name}' not found!")
                    raise ValueError(f"Feature '{feat_name}' not found")
                selected_indices.append(feat_idx_array[0])
                selected_features.append(feat_name)
        
        print(f"\nAnalyzing {len(selected_features)} features")
        
        # 准备数据
        X_selected = self.X_train[:, selected_indices]
        X_realtest_selected = self.X_realtest[:, selected_indices] if len(self.X_realtest) > 0 else None
        
        # 训练模型
        print(f"\nTraining with {len(selected_features)} features...")
        training_start = time.strftime("%Y-%m-%d %H:%M:%S")
        
        perf = self.poolfit_optimized(train_test_split, epoch, core_num, 
                                     X_selected, self.y_train, self.xgb_params, 
                                     X_realtest_selected)
        
        training_end = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 计算测试集性能
        test_performance = None
        if len(self.y_realtest) > 0 and perf['y_realtest_pred'] is not None:
            test_mae = mean_absolute_error(self.y_realtest, perf['y_realtest_pred'])
            test_mse = mean_squared_error(self.y_realtest, perf['y_realtest_pred'])
            test_r2 = r2_score(self.y_realtest, perf['y_realtest_pred'])
            test_performance = {
                'mae': test_mae,
                'mse': test_mse,
                'r2': test_r2
            }
            print(f"\nTest Set Performance:")
            print(f"  Test MAE: {test_mae:.4f}")
            print(f"  Test MSE: {test_mse:.4f}")
            print(f"  Test R²: {test_r2:.4f}")
        
        # 生成可视化
        self.generate_prediction_scatter(perf, selected_features, "Manual")
        
        # 生成SHAP分析（即使是Full模式也生成）
        if generate_fitting:
            print("Generating SHAP analysis...")
            # 限制SHAP图的数量，但Full模式也生成
            max_shap_plots = 30 if self.manual_features == 'Full' else 20
            for i, feat_name in enumerate(selected_features[:max_shap_plots]):
                self.generate_shap_plots(
                    X_selected[:, i],
                    perf['shap_values'][:, i],
                    perf['shap_std'][:, i],
                    feat_name,
                    self.ML_DIR
                )
        
        # 保存数据
        self.save_comprehensive_results(self.ML_DIR, f"Manual_{len(selected_features)}feats",
                                       selected_features, selected_indices, 
                                       X_selected, perf)
        
        # 组装结果
        results = {
            'selected_features': selected_features,
            'selected_indices': selected_indices,
            'mse': perf['mse'],
            'mae': perf['mae'],
            'r2': perf['r2'],
            'mse_std': perf['mse_std'],
            'shap_values': perf['shap_values'],
            'shap_std': perf['shap_std'],
            'test_performance': test_performance,
            'training_start': training_start,
            'training_end': training_end,
            'output_dir': str(self.ML_DIR)
        }
        
        print(f"\nResults: MSE={perf['mse']:.4f}±{perf['mse_std']:.4f}, MAE={perf['mae']:.4f}, R²={perf['r2']:.4f}")
        print(f"Results saved in: {self.ML_DIR}")
        
        return results

# ================== 工作流分析器 ==================
class WorkflowAnalyzer(BaseChemMLAnalyzer):
    """
    完整工作流分析器 - 包含基线分析、逐步回归、SHAP分析和最终推荐
    """
    
    def _generate_feature_matrix(self, merged_df):
        """生成特征矩阵文件"""
        self._setup_output_directory('Workflow_Analysis')
        return self._save_feature_files(merged_df)
    
    def run(self, max_features=5, n_runs=3, epoch=64, core_num=32, train_test_split=0.85):
        """
        运行完整工作流
        """
        self._ensure_data_loaded()
        
        print("\n" + "="*60)
        print("COMPLETE WORKFLOW ANALYSIS")
        print("="*60)
        
        all_results = {}
        
        # Step 1: 基线分析（全特征）
        print("\n" + "="*50)
        print("STEP 1: BASELINE ANALYSIS WITH ALL FEATURES")
        print("="*50)
        baseline_results = self._run_baseline_analysis(epoch, core_num, train_test_split)
        all_results['baseline'] = baseline_results
        
        # 清理内存
        gc.collect()
        
        # Step 2: 多轮逐步回归
        print("\n" + "="*50)
        print(f"STEP 2: STEPWISE REGRESSION ({n_runs} runs)")
        print("="*50)
        stepwise_results = self._run_stepwise_regression(max_features, n_runs, epoch, core_num, train_test_split)
        all_results['stepwise'] = stepwise_results
        
        # Step 3: SHAP拟合分析（对每个Run）
        print("\n" + "="*50)
        print("STEP 3: SHAP FITTING ANALYSIS")
        print("="*50)
        shap_analysis_results = self._run_shap_analysis(stepwise_results, epoch, core_num, train_test_split)
        all_results['shap_analysis'] = shap_analysis_results
        
        # Step 4: 智能推荐
        print("\n" + "="*50)
        print("STEP 4: INTELLIGENT RECOMMENDATION")
        print("="*50)
        recommendations = self._intelligent_recommendation(stepwise_results, baseline_results)
        all_results['recommendations'] = recommendations
        
        # Step 5: 使用推荐特征进行最终分析
        print("\n" + "="*50)
        print("STEP 5: FINAL ANALYSIS WITH RECOMMENDED FEATURES")
        print("="*50)
        
        top_features = [rec['feature_name'] for rec in recommendations[:max_features]]
        final_results = self._run_final_analysis(top_features, epoch, core_num, train_test_split)
        all_results['final'] = final_results
        
        # 保存工作流摘要
        # 保存工作流摘要，传入所有必需的参数
        self._save_workflow_summary(all_results, epoch, core_num, train_test_split, n_runs, max_features)
        
        print("\n" + "="*60)
        print("WORKFLOW ANALYSIS COMPLETED")
        print("="*60)
        print(f"Baseline MSE (all features): {baseline_results['mse']:.4f}")
        print(f"Final MSE ({len(top_features)} features): {final_results['mse']:.4f}")
        print(f"Recommended features: {top_features}")
        print(f"All results saved in: {self.ML_DIR}")
        
        return all_results
    
    def _run_baseline_analysis(self, epoch, core_num, train_test_split):
        """运行基线分析（全特征）"""
        print(f"Running baseline analysis with all {len(self.feature_names)} features...")
        
        # 创建Baseline目录
        baseline_dir = self.ML_DIR / 'Baseline_Analysis'
        os.makedirs(baseline_dir, exist_ok=True)
        
        # 训练
        perf = self.poolfit_optimized(train_test_split, epoch, core_num,
                                     self.X_train, self.y_train, self.xgb_params,
                                     self.X_realtest)
        
        # 计算特征重要性
        clf_baseline = XGBRegressor(**self.xgb_params)
        clf_baseline.fit(self.X_train, self.y_train)
        feature_importance = clf_baseline.feature_importances_
        
        # 生成可视化
        self.generate_prediction_scatter(perf, self.feature_names, "Baseline", baseline_dir)
        
        # 生成SHAP图（选择重要的特征）
        importance_indices = np.argsort(feature_importance)[-20:]  # Top 20特征
        for idx in importance_indices:
            self.generate_shap_plots(
                self.X_train[:, idx],
                perf['shap_values'][:, idx],
                perf['shap_std'][:, idx],
                self.feature_names[idx],
                baseline_dir
            )
        
        # 保存结果
        self.save_comprehensive_results(baseline_dir, "Baseline_All",
                                       self.feature_names.tolist(),
                                       list(range(len(self.feature_names))),
                                       self.X_train, perf)
        
        results = {
            'mse': perf['mse'],
            'mae': perf['mae'],
            'r2': perf['r2'],
            'mse_std': perf['mse_std'],
            'feature_importance': feature_importance,
            'feature_names': self.feature_names
        }
        
        print(f"Baseline Results: MSE={perf['mse']:.4f}±{perf['mse_std']:.4f}, R²={perf['r2']:.4f}")
        
        return results
    
    def _run_stepwise_regression(self, max_features, n_runs, epoch, core_num, train_test_split):
        """运行多轮逐步回归"""
        all_results = []
        
        for run in range(n_runs):
            print(f"\nRun {run + 1}/{n_runs}...")
            
            # 创建Run目录
            run_dir = self.ML_DIR / f'Run_{run + 1}'
            os.makedirs(run_dir, exist_ok=True)
            
            # 单轮逐步回归
            result = self._single_stepwise_regression(max_features, epoch, core_num, 
                                                     train_test_split, run_dir, run + 1)
            result['run_id'] = run + 1
            all_results.append(result)
            
            print(f"  Run {run + 1} Final MSE: {result['final_mse']:.4f}")
        
        # 按MSE排序
        all_results.sort(key=lambda x: x['final_mse'])
        
        return all_results
    
    def _single_stepwise_regression(self, max_features, epoch, core_num, train_test_split, run_dir, run_id):
        """单次逐步回归，生成详细输出"""
        selected_indices = []
        mse_history = []
        feature_names_history = []
        
        # 找最佳单特征
        best_single_mse = np.inf
        best_single_idx = None
        
        print(f"  Finding best single feature...")
        for j in range(len(self.feature_names)):
            X_temp = self.X_train[:, [j]]
            perf = self.poolfit_optimized(train_test_split, epoch, core_num,
                                         X_temp, self.y_train, self.xgb_params)
            if perf['mse'] < best_single_mse:
                best_single_mse = perf['mse']
                best_single_idx = j
        
        selected_indices.append(best_single_idx)
        mse_history.append(best_single_mse)
        feature_names_history.append([self.feature_names[best_single_idx]])
        
        print(f"    Best single feature: {self.feature_names[best_single_idx]} (MSE: {best_single_mse:.4f})")
        
        # 逐步添加特征
        for step in range(1, max_features):
            print(f"  Adding feature {step + 1}/{max_features}...")
            best_mse = np.inf
            best_feature = None
            
            remaining = [i for i in range(len(self.feature_names)) if i not in selected_indices]
            
            for feat_idx in remaining:
                candidate = selected_indices + [feat_idx]
                X_temp = self.X_train[:, candidate]
                perf = self.poolfit_optimized(train_test_split, epoch, core_num,
                                             X_temp, self.y_train, self.xgb_params)
                
                if perf['mse'] < best_mse:
                    best_mse = perf['mse']
                    best_feature = feat_idx
            
            if best_feature is not None:
                selected_indices.append(best_feature)
                mse_history.append(best_mse)
                feature_names_history.append([self.feature_names[i] for i in selected_indices])
                print(f"    Added {self.feature_names[best_feature]} (MSE: {best_mse:.4f})")
        
        # 为每个特征数量生成散点图
        for i, (mse, features) in enumerate(zip(mse_history, feature_names_history)):
            indices = selected_indices[:i+1]
            X_temp = self.X_train[:, indices]
            X_realtest_temp = self.X_realtest[:, indices] if len(self.X_realtest) > 0 else None
            
            # 完整训练以获得准确结果
            perf = self.poolfit_optimized(train_test_split, epoch, core_num,
                                         X_temp, self.y_train, self.xgb_params, X_realtest_temp)
            
            # 生成散点图
            self.generate_prediction_scatter(perf, features, f"Run_{run_id}_{i+1}feats", run_dir)
        
        # 绘制MSE曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(mse_history) + 1), mse_history, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Number of Features', fontsize=14)
        plt.ylabel('MSE', fontsize=14)
        plt.title(f'Run {run_id}: MSE vs Number of Features', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(run_dir / f'Run_{run_id}_MSE_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'selected_indices': selected_indices,
            'feature_names': [self.feature_names[i] for i in selected_indices],
            'mse_history': mse_history,
            'final_mse': mse_history[-1],
            'run_dir': run_dir
        }
    
    def _run_shap_analysis(self, stepwise_results, epoch, core_num, train_test_split):
        """对每个Run进行SHAP分析"""
        shap_results = []
        
        for result in stepwise_results:
            run_id = result['run_id']
            run_dir = result['run_dir']
            selected_indices = result['selected_indices']
            selected_features = result['feature_names']
            
            print(f"\nAnalyzing SHAP for Run {run_id}...")
            
            # 创建SHAP分析目录
            shap_dir = run_dir / 'SHAP_Analysis'
            os.makedirs(shap_dir, exist_ok=True)
            
            # 使用选择的特征进行训练
            X_selected = self.X_train[:, selected_indices]
            X_realtest_selected = self.X_realtest[:, selected_indices] if len(self.X_realtest) > 0 else None
            
            perf = self.poolfit_optimized(train_test_split, epoch, core_num,
                                         X_selected, self.y_train, self.xgb_params,
                                         X_realtest_selected)
            
            # 生成SHAP图
            for i, feat_name in enumerate(selected_features):
                self.generate_shap_plots(
                    X_selected[:, i],
                    perf['shap_values'][:, i],
                    perf['shap_std'][:, i],
                    feat_name,
                    shap_dir
                )
            
            # 保存详细结果
            self.save_comprehensive_results(shap_dir, f"Run_{run_id}_SHAP",
                                          selected_features, selected_indices,
                                          X_selected, perf)
            
            shap_results.append({
                'run_id': run_id,
                'features': selected_features,
                'mse': perf['mse'],
                'r2': perf['r2']
            })
        
        return shap_results
    
    def _intelligent_recommendation(self, stepwise_results, baseline_results):
        """智能推荐算法"""
        feature_scores = {}
        
        # 从逐步回归中收集分数
        for result in stepwise_results:
            for i, feat_name in enumerate(result['feature_names']):
                if feat_name not in feature_scores:
                    feature_scores[feat_name] = {'stepwise': 0, 'importance': 0, 'count': 0}
                
                feature_scores[feat_name]['stepwise'] += 1.0 / (i + 1)
                feature_scores[feat_name]['count'] += 1
        
        # 加入基线特征重要性
        if baseline_results:
            importance = baseline_results['feature_importance']
            max_importance = np.max(importance) if np.max(importance) > 0 else 1
            
            for i, feat_name in enumerate(baseline_results['feature_names']):
                if feat_name in feature_scores:
                    feature_scores[feat_name]['importance'] = importance[i] / max_importance
        
        # 计算综合分数
        recommendations = []
        for feat_name, scores in feature_scores.items():
            # 权重：逐步回归40%，特征重要性40%，出现次数20%
            stepwise_score = scores['stepwise'] / len(stepwise_results)
            importance_score = scores['importance']
            appearance_score = scores['count'] / len(stepwise_results)
            
            final_score = 0.4 * stepwise_score + 0.4 * importance_score + 0.2 * appearance_score
            
            feat_idx = np.where(self.feature_names == feat_name)[0][0]
            
            recommendations.append({
                'feature_name': feat_name,
                'feature_index': feat_idx,
                'score': final_score,
                'stepwise_score': stepwise_score,
                'importance_score': importance_score,
                'appearance_count': scores['count']
            })
        
        # 按分数排序
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # 保存推荐结果
        rec_df = pd.DataFrame(recommendations)
        rec_df.to_csv(self.ML_DIR / f'feature_recommendations_{self.timestamp}.csv', index=False)
        
        print(f"\nTop recommended features:")
        for i, rec in enumerate(recommendations[:10]):
            print(f"  {i+1}. {rec['feature_name']} (score: {rec['score']:.3f})")
        
        return recommendations
    
    def _run_final_analysis(self, top_features, epoch, core_num, train_test_split):
        """使用推荐特征进行最终分析"""
        # 创建最终分析目录
        final_dir = self.ML_DIR / 'Final_Manual_Analysis'
        os.makedirs(final_dir, exist_ok=True)
        
        # 获取特征索引
        selected_indices = []
        for feat_name in top_features:
            idx = np.where(self.feature_names == feat_name)[0][0]
            selected_indices.append(idx)
        
        print(f"Running final analysis with {len(top_features)} recommended features...")
        
        # 准备数据
        X_selected = self.X_train[:, selected_indices]
        X_realtest_selected = self.X_realtest[:, selected_indices] if len(self.X_realtest) > 0 else None
        
        # 训练
        perf = self.poolfit_optimized(train_test_split, epoch, core_num,
                                     X_selected, self.y_train, self.xgb_params,
                                     X_realtest_selected)
        
        # 生成可视化
        self.generate_prediction_scatter(perf, top_features, "Final_Manual", final_dir)
        
        # 生成SHAP分析
        for i, feat_name in enumerate(top_features):
            self.generate_shap_plots(
                X_selected[:, i],
                perf['shap_values'][:, i],
                perf['shap_std'][:, i],
                feat_name,
                final_dir
            )
        
        # 保存详细结果
        self.save_comprehensive_results(final_dir, f"Final_{len(top_features)}feats",
                                       top_features, selected_indices,
                                       X_selected, perf)
        
        # 计算测试集性能
        test_performance = None
        if len(self.y_realtest) > 0 and perf['y_realtest_pred'] is not None:
            test_mae = mean_absolute_error(self.y_realtest, perf['y_realtest_pred'])
            test_mse = mean_squared_error(self.y_realtest, perf['y_realtest_pred'])
            test_r2 = r2_score(self.y_realtest, perf['y_realtest_pred'])
            test_performance = {
                'mae': test_mae,
                'mse': test_mse,
                'r2': test_r2
            }
        
        results = {
            'selected_features': top_features,
            'mse': perf['mse'],
            'mae': perf['mae'],
            'r2': perf['r2'],
            'mse_std': perf['mse_std'],
            'test_performance': test_performance
        }
        
        print(f"Final Results: MSE={perf['mse']:.4f}±{perf['mse_std']:.4f}, R²={perf['r2']:.4f}")
        
        return results
    
    def _save_workflow_summary(self, all_results, epoch, core_num, train_test_split, n_runs, max_features):
        """保存包含详细运行参数的工作流摘要"""
        summary_path = self.ML_DIR / f'workflow_summary_{self.timestamp}.txt'

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('='*80 + '\n')
            f.write('WORKFLOW ANALYSIS SUMMARY\n')
            f.write('='*80 + '\n\n')

            # --- 新增部分：常规信息和运行参数 ---
            f.write('GENERAL & RUN PARAMETERS:\n')
            f.write('-'*50 + '\n')
            f.write(f'Analysis Timestamp : {self.timestamp}\n')
            f.write(f'Input Data File    : {self.data_file}\n')
            f.write(f'Output Directory   : {self.ML_DIR}\n')
            f.write(f'Epochs per Training: {epoch}\n')
            f.write(f'CPU Cores Used     : {core_num}\n\n')

            # --- 新增部分：数据集和特征选择参数 ---
            f.write('DATASET & FEATURE SELECTION PARAMETERS:\n')
            f.write('-'*50 + '\n')
            f.write(f'Train/Test Split Ratio : {train_test_split}\n')
            f.write(f'Training Samples       : {self.dataset_info["train_samples"]}\n')
            f.write(f'Test Samples           : {self.dataset_info["test_samples"]}\n')
            if self.test_sample_names:
                # 只显示找到的样本，更简洁
                f.write(f'Specified Test Samples : {self.dataset_info["test_sample_names_found"]}\n')
            else:
                f.write('Specified Test Samples : None (Random Split)\n')
            f.write(f'Stepwise Reg. Runs (n_runs)      : {n_runs}\n')
            f.write(f'Max Features per Run (max_features): {max_features}\n\n')

            # --- 原有部分：分析结果 ---
            f.write('BASELINE ANALYSIS (ALL FEATURES):\n')
            f.write('-'*50 + '\n')
            f.write(f"MSE: {all_results['baseline']['mse']:.4f}\n")
            f.write(f"R²: {all_results['baseline']['r2']:.4f}\n\n")

            f.write('STEPWISE REGRESSION RESULTS:\n')
            f.write('-'*50 + '\n')
            for result in all_results['stepwise']:
                f.write(f"Run {result['run_id']}: MSE = {result['final_mse']:.4f}\n")
                f.write(f"  Features: {', '.join(result['feature_names'])}\n")
            f.write('\n')

            f.write(f'TOP {max_features} RECOMMENDED FEATURES:\n')
            f.write('-'*50 + '\n')
            # 使用 max_features 来决定显示多少个推荐特征
            for i, rec in enumerate(all_results['recommendations'][:max_features]):
                f.write(f"{i+1}. {rec['feature_name']} (score: {rec['score']:.3f})\n")
            f.write('\n')

            f.write('FINAL ANALYSIS RESULTS (WITH RECOMMENDED FEATURES):\n')
            f.write('-'*50 + '\n')
            f.write(f"Selected features: {', '.join(all_results['final']['selected_features'])}\n")
            f.write(f"Cross-Validation MSE: {all_results['final']['mse']:.4f}\n")
            f.write(f"Cross-Validation R²: {all_results['final']['r2']:.4f}\n")

            if all_results['final'].get('test_performance'):
                f.write(f"Test Set MSE: {all_results['final']['test_performance']['mse']:.4f}\n")
                f.write(f"Test Set R²: {all_results['final']['test_performance']['r2']:.4f}\n")

            f.write('\n' + '='*80 + '\n')
            f.write('WORKFLOW ANALYSIS COMPLETED SUCCESSFULLY\n')
            f.write('='*80 + '\n')

# ================== 主工作流控制器 ==================
class ChemMLWorkflow:
    """
    化学机器学习工作流控制器
    """
    
    @staticmethod
    def run_analysis(mode, data_file, test_sample_names=None, nan_handling='drop_columns',
                     output_dir=None, features=None, **kwargs):
        """
        运行分析的统一入口
        
        Args:
            mode: 'manual' 或 'workflow'
            data_file: 数据文件路径
            test_sample_names: 测试集样本名称
            nan_handling: NaN处理方式
            output_dir: 输出目录
            features: 手动模式下的特征列表，或 'Full' 表示所有特征
            **kwargs: 其他参数（epoch, core_num等）
        
        Returns:
            dict: 分析结果
        """
        
        # 提取通用参数
        epoch = kwargs.get('epoch', 64)
        core_num = kwargs.get('core_num', 32)
        train_test_split = kwargs.get('train_test_split', 0.85)
        
        if mode == 'manual':
            # 手动特征分析模式
            print("="*60)
            print("RUNNING MANUAL FEATURE ANALYSIS")
            print("="*60)
            
            if not features:
                raise ValueError("Manual mode requires 'features' parameter")
            
            analyzer = ManualFeatureAnalyzer(data_file, test_sample_names, nan_handling, 
                                            output_dir, features)
            generate_fitting = kwargs.get('generate_fitting', True)
            results = analyzer.run(epoch, core_num, train_test_split, generate_fitting)
            
            return results
            
        elif mode == 'workflow':
            # 完整工作流模式
            print("="*60)
            print("RUNNING COMPLETE WORKFLOW")
            print("="*60)
            
            max_features = kwargs.get('max_features', 5)
            n_runs = kwargs.get('n_runs', 3)
            
            analyzer = WorkflowAnalyzer(data_file, test_sample_names, nan_handling, output_dir)
            results = analyzer.run(max_features, n_runs, epoch, core_num, train_test_split)
            
            return results
            
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'manual' or 'workflow'")


# ================== 使用示例 ==================

if __name__ == "__main__":
    
#     # 示例1: 手动模式 - 分析指定的特征
#     print("\n" + "="*80)
#     print("Example 1: Manual Feature Analysis with Selected Features")
#     print("="*80)
    
#     results_manual = ChemMLWorkflow.run_analysis(
#         mode='manual',
#        data_file="/home/yumingsu/Python/Project_Surfacia/250805_ShenyanS_copy/Surfacia_3.0_20250805_093212/FinalFull_Mode1_S_82_68.csv",
#         test_sample_names=[79, 22, 82, 36, 70, 80, 14, 46],
#         nan_handling='drop_columns',
#         features=['HOMO', 'Surface_exposed_atom_num', 'LEAE_Maximal_Value', 
#                  'S_ESP_average', 'ESP_Minimal_Value'],  # 指定特征
#         epoch=128,
#         core_num=32,
#         generate_fitting=True
#     )
    
#     print(f"\nManual analysis completed.")
#     print(f"Performance: MSE={results_manual['mse']:.4f}, R²={results_manual['r2']:.4f}")
#     print(f"Results saved in: {results_manual['output_dir']}")
    
#     # 示例2: 手动模式 - 使用所有特征（基线分析）
#     print("\n" + "="*80)
#     print("Example 2: Manual Feature Analysis with ALL Features (Baseline)")
#     print("="*80)
    
#     results_baseline = ChemMLWorkflow.run_analysis(
#         mode='manual',
#          data_file="/home/yumingsu/Python/Project_Surfacia/250805_ShenyanS_copy/Surfacia_3.0_20250805_093212/FinalFull_Mode1_S_82_68.csv",
#         test_sample_names=[79, 22, 82, 36, 70, 80, 14, 46],
#         nan_handling='drop_columns',
#         features='Full',  # 使用所有特征
#         epoch=32,
#         core_num=32,
#         generate_fitting=True  # Full模式也生成SHAP图
#     )
    
#     print(f"\nBaseline analysis completed.")
#     print(f"Performance with all features: MSE={results_baseline['mse']:.4f}, R²={results_baseline['r2']:.4f}")
    
#     # 示例3: 完整工作流模式
#     print("\n" + "="*80)
#     print("Example 3: Complete Workflow Analysis")
#     print("="*80)
    
#     results_workflow = ChemMLWorkflow.run_analysis(
#         mode='workflow',
#          data_file="/home/yumingsu/Python/Project_Surfacia/250805_ShenyanS_copy/Surfacia_3.0_20250805_093212/FinalFull_Mode1_S_82_68.csv",
#         test_sample_names=[79, 22, 82, 36, 70, 80, 14, 46],
#         nan_handling='drop_columns',
#         max_features=2,  # 选择最多5个特征
#         n_runs=2,  # 运行3轮逐步回归
#         epoch=32,
#         core_num=32
#     )
    
#     print(f"\nWorkflow completed.")
#     print(f"Baseline MSE: {results_workflow['baseline']['mse']:.4f}")
#     print(f"Final MSE: {results_workflow['final']['mse']:.4f}")
#     print(f"Selected features: {results_workflow['final']['selected_features']}")


# In[ ]:





# Draw_mol_now(optional)

# In[2]:


import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

# 创建输出文件夹(如果不存在)
output_dir = 'molecule_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取CSV文件
# 替换'your_file.csv'为你的CSV文件名
df = pd.read_csv('FinalFull_42_79.csv')

# 设置绘图参数
Draw.DrawingOptions.bondLineWidth = 3.0  # 键线加粗
Draw.DrawingOptions.atomLabelFontSize = 18  # 原子标签字体大小
Draw.DrawingOptions.dotsPerAngstrom = 300  # 提高分辨率

# 遍历SMILES并生成图片
for idx, smiles in enumerate(df['smiles']):
    try:
        # 创建分子对象
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"无法处理SMILES: {smiles}")
            continue
            
        # 生成2D构象
        AllChem.Compute2DCoords(mol)
        
        # 创建图像
        img = Draw.MolToImage(mol, size=(800, 800))
        
        # 保存图片
        filename = f'molecule_{idx+1}.png'
        filepath = os.path.join(output_dir, filename)
        img.save(filepath, dpi=(300, 300))
        
        print(f"成功生成图片: {filename}")
        
    except Exception as e:
        print(f"处理SMILES时出错: {smiles}")
        print(f"错误信息: {str(e)}")

print("所有分子结构式图片生成完成！")


# In[ ]:


calculate_3_vtxpdb


# In[5]:


import os
import glob
import subprocess
import logging

def create_leae_content(sample_name):
    """生成LEAE计算的输入内容"""
    xyz_pdb_file = f"{sample_name}_xyz.pdb"
    content = f"""12
2
-4
1
1
0.01
0
5
{xyz_pdb_file}
6
-1
0
"""
    return content

def create_esp_content():
    """生成ESP计算的输入内容"""
    content = """12
1
1
0.01
0
6
-1
0
"""
    return content

def create_alie_content():
    """生成ALIE计算的输入内容"""
    content = """12
2
2
1
1
0.01
0
6
-1
0
"""
    return content

def run_multiwfn_on_fchk_files(input_path='.'):
    """
    对指定文件夹下的所有fchk文件分别进行LEAE、ESP和ALIE计算
    """
    original_dir = os.getcwd()
    os.chdir(input_path)
    fchk_files = sorted(glob.glob('*.fchk'))
    processed_files = []

    for fchk_file in fchk_files:
        sample_name = os.path.splitext(fchk_file)[0]  # e.g., '000001' if fchk_file is '000001.fchk'
        
        # 步骤1: 计算LEAE
        leae_pdb = f"{sample_name}_LEAE.pdb"
        if not os.path.exists(leae_pdb):
            leae_content = create_leae_content(sample_name)
            print(f"Running LEAE calculation for {sample_name}...")
            
            command = ["Multiwfn_noGUI", fchk_file, "-silent"]
            
            try:
                # 删除可能存在的旧vtx.pdb文件
                if os.path.exists('vtx.pdb'):
                    os.remove('vtx.pdb')
                
                result = subprocess.run(
                    command,
                    input=leae_content,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # 检查并重命名生成的vtx.pdb
                if os.path.exists('vtx.pdb'):
                    os.rename('vtx.pdb', leae_pdb)
                    print(f"Successfully created {leae_pdb}")
                    processed_files.append(leae_pdb)
                else:
                    print(f"Warning: vtx.pdb was not generated for LEAE calculation of {sample_name}")
                    print(f"Return code: {result.returncode}")
                    if result.stderr:
                        print(f"stderr: {result.stderr}")
                
            except Exception as e:
                print(f"LEAE calculation failed for {fchk_file}: {e}")
        else:
            print(f"LEAE file {leae_pdb} already exists. Skipping.")
            processed_files.append(leae_pdb)

        # 步骤2: 计算ESP
        esp_pdb = f"{sample_name}_ESP.pdb"
        if not os.path.exists(esp_pdb):
            esp_content = create_esp_content()
            print(f"Running ESP calculation for {sample_name}...")
            
            command = ["Multiwfn_noGUI", fchk_file, "-silent"]
            
            try:
                # 删除可能存在的旧vtx.pdb文件
                if os.path.exists('vtx.pdb'):
                    os.remove('vtx.pdb')
                
                result = subprocess.run(
                    command,
                    input=esp_content,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # 检查并重命名生成的vtx.pdb
                if os.path.exists('vtx.pdb'):
                    os.rename('vtx.pdb', esp_pdb)
                    print(f"Successfully created {esp_pdb}")
                    processed_files.append(esp_pdb)
                else:
                    print(f"Warning: vtx.pdb was not generated for ESP calculation of {sample_name}")
                    print(f"Return code: {result.returncode}")
                    if result.stderr:
                        print(f"stderr: {result.stderr}")
                
            except Exception as e:
                print(f"ESP calculation failed for {fchk_file}: {e}")
        else:
            print(f"ESP file {esp_pdb} already exists. Skipping.")
            processed_files.append(esp_pdb)

        # 步骤3: 计算ALIE
        alie_pdb = f"{sample_name}_ALIE.pdb"
        if not os.path.exists(alie_pdb):
            alie_content = create_alie_content()
            print(f"Running ALIE calculation for {sample_name}...")
            
            command = ["Multiwfn_noGUI", fchk_file, "-silent"]
            
            try:
                # 删除可能存在的旧vtx.pdb文件
                if os.path.exists('vtx.pdb'):
                    os.remove('vtx.pdb')
                
                result = subprocess.run(
                    command,
                    input=alie_content,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # 检查并重命名生成的vtx.pdb
                if os.path.exists('vtx.pdb'):
                    os.rename('vtx.pdb', alie_pdb)
                    print(f"Successfully created {alie_pdb}")
                    processed_files.append(alie_pdb)
                else:
                    print(f"Warning: vtx.pdb was not generated for ALIE calculation of {sample_name}")
                    print(f"Return code: {result.returncode}")
                    if result.stderr:
                        print(f"stderr: {result.stderr}")
                
            except Exception as e:
                print(f"ALIE calculation failed for {fchk_file}: {e}")
        else:
            print(f"ALIE file {alie_pdb} already exists. Skipping.")
            processed_files.append(alie_pdb)

        print(f"Completed all calculations for {sample_name}")

    os.chdir(original_dir)
    return processed_files

# 使用示例
if __name__ == "__main__":
    # 运行计算
    processed_files = run_multiwfn_on_fchk_files('.')
    print(f"Processed {len(processed_files)} files total.")


# Interactively_SHAP_Surfacia_plot

# In[4]:


import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback, State
import numpy as np
import base64
import io
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import os
import py3Dmol
import json
from datetime import datetime
from zai import ZhipuAiClient
import glob
import subprocess
import logging

class MultiwfnCalculator:
    """Multiwfn计算模块"""
    
    @staticmethod
    def create_leae_content(sample_name):
        """生成LEAE计算的输入内容"""
        xyz_pdb_file = f"{sample_name}_xyz.pdb"
        content = f"""12
2
-4
1
1
0.01
0
5
{xyz_pdb_file}
6
-1
0
"""
        return content

    @staticmethod
    def create_esp_content():
        """生成ESP计算的输入内容"""
        content = """12
1
1
0.01
0
6
-1
0
"""
        return content

    @staticmethod
    def create_alie_content():
        """生成ALIE计算的输入内容"""
        content = """12
2
2
1
1
0.01
0
6
-1
0
"""
        return content

    @staticmethod
    def run_multiwfn_calculation(input_path, surface_type, sample_name, fchk_file):
        """运行单个Multiwfn计算"""
        original_dir = os.getcwd()
        os.chdir(input_path)
        
        try:
            if surface_type == "LEAE":
                content = MultiwfnCalculator.create_leae_content(sample_name)
            elif surface_type == "ESP":
                content = MultiwfnCalculator.create_esp_content()
            elif surface_type == "ALIE":
                content = MultiwfnCalculator.create_alie_content()
            else:
                raise ValueError(f"不支持的表面类型: {surface_type}")
            
            print(f"正在运行 {surface_type} 计算 for {sample_name}...")
            
            command = ["Multiwfn_noGUI", fchk_file, "-silent"]
            
            # 删除可能存在的旧vtx.pdb文件
            if os.path.exists('vtx.pdb'):
                os.remove('vtx.pdb')
            
            result = subprocess.run(
                command,
                input=content,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            output_file = f"{sample_name}_{surface_type}.pdb"
            
            # 检查并重命名生成的vtx.pdb
            if os.path.exists('vtx.pdb'):
                os.rename('vtx.pdb', output_file)
                print(f"✅ 成功创建 {output_file}")
                return output_file
            else:
                print(f"⚠️ 警告: {surface_type} 计算未生成vtx.pdb文件")
                print(f"返回码: {result.returncode}")
                if result.stderr:
                    print(f"错误信息: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"❌ {surface_type} 计算失败: {e}")
            return None
        finally:
            os.chdir(original_dir)

    @staticmethod
    def run_multiwfn_on_fchk_files(input_path='.'):
        """对指定文件夹下的所有fchk文件分别进行LEAE、ESP和ALIE计算"""
        original_dir = os.getcwd()
        os.chdir(input_path)
        
        fchk_files = sorted(glob.glob('*.fchk'))
        processed_files = []
        
        print(f"🔍 在 {input_path} 中找到 {len(fchk_files)} 个fchk文件")
        
        for fchk_file in fchk_files:
            sample_name = os.path.splitext(fchk_file)[0]
            print(f"\n📁 处理样本: {sample_name}")
            
            # 计算所有三种表面类型
            for surface_type in ['LEAE', 'ESP', 'ALIE']:
                output_file = f"{sample_name}_{surface_type}.pdb"
                
                if not os.path.exists(output_file):
                    result = MultiwfnCalculator.run_multiwfn_calculation(
                        input_path, surface_type, sample_name, fchk_file
                    )
                    if result:
                        processed_files.append(result)
                else:
                    print(f"✅ {output_file} 已存在，跳过计算")
                    processed_files.append(output_file)
        
        os.chdir(original_dir)
        print(f"\n🎉 总共处理了 {len(processed_files)} 个文件")
        return processed_files

class InteractiveSHAPAnalyzer:
    def __init__(self, csv_file_path=None, xyz_folder_path=None, test_csv_path=None):
        """
        交互式SHAP分析器

        Parameters:
        csv_file_path: str, Training_Set_Detailed_xxx.csv文件的绝对路径
        xyz_folder_path: str, 包含xyz文件和等值面PDB文件的文件夹路径
        test_csv_path: str, Test_Set_Detailed_xxx.csv文件的绝对路径（可选）
        """
        self.csv_file_path = csv_file_path
        self.xyz_folder_path = xyz_folder_path
        self.test_csv_path = test_csv_path
        self.data = None
        self.test_data = None  # 新增：存储测试集数据
        self.feature_columns = []
        self.shap_columns = []
        self.app = None
        self.first_interaction = True

        # 支持的等值面类型
        self.surface_types = ['LEAE', 'ESP', 'ALIE']
        
        # 描述符背景知识
        self.descriptor_knowledge = """
二、 SHAP图分析通用框架

在分析任何SHAP图时，你都应遵循以下框架：

图谱基本解读：

横轴 (Feature Value)：代表数据集中每个样本（分子）的某个特定描述符的数值。（在这里要特别先把本特征的明确化学含义说出来）
纵轴 (SHAP Value)：代表该描述符的数值对模型最终预测值（Target）的贡献。正的SHAP值表示该特征将预测值推高，负值则表示推低。SHAP值的绝对大小代表其贡献强度。
颜色 (Target Value)：散点的颜色通常映射到目标变量（如活性、反应速率等）的数值大小，这有助于发现特征与目标之间的非线性关系或交互效应。
特征影响力评估：

一个特征的整体重要性可以通过其SHAP值在纵轴上的分布范围（span）来判断。分布越宽，说明该特征对模型预测的影响力越大。
理解SHAP图不同特征之间的关系， 单个特征的SHAP值只反映该特征对该样本相对于“基准（期望值）”的边际贡献，不代表该样本在其它特征上的表现。一个样本在当前特征的SHAP值可能为0或为不利方向，但它仍可能通过其他特征的强贡献获得优异的总体预测。这种“补偿/协同”是常见且重要的。 单个特征的SHAP值只反映该特征对该样本相对于“基准（期望值）”的边际贡献，不代表该样本在其它特征上的表现。一个样本在当前特征的SHAP值可能为0或为不利方向，但它仍可能通过其他特征的强贡献获得优异的总体预测。这种“补偿/协同”是常见且重要的。
趋势与规律分析：

可信度判断：当散点在某个区间内分布密集且呈现出明显、平滑的规律时（如单调递增/递减、抛物线形），我们认为该区间的结论是相对可信的。
不确定性区域：当散点在某个区间内分布非常稀疏时，这暗示模型在该区间内学习到的规律可能不可靠，或者缺乏足够的训练数据支持。对这些区域的结论应持谨慎态度。
潜在外推风险：在SHAP值绝对值很大但散点又很稀疏的区域，对应的分子可能超出了当前数据集的“化学空间”，其行为可能很特殊，值得重点关注和可视化分析。
交互式探索引导：

你的分析不应是单向的。应主动引导用户进行探索。例如，当发现一个有趣的趋势时，你可以建议：“这个趋势表明，具有较低ALIE值的分子倾向于有更高的目标值。我们可以动态选取这个区域的几个代表性分子进行三维可视化，看看是哪个官能团导致了这种强亲核性，以及它在分子中的空间位置如何。”
三、 核心知识库：综合分子描述符体系详解

这是你的核心知识库。所有分析都必须基于对以下描述符物理意义的深刻理解。

第1层：基础与全局属性
这些描述符提供了分子的基本画像。

结构与电子计数:
Atom Number: 分子中的原子总数，最基本的尺寸度量。
Molecular Weight: 分子量，衡量分子质量。
Occupied Orbitals: 被占据的分子轨道数，反映电子结构的复杂性。
表面几何属性 (基于电子密度=0.01 a.u.等值面):
Isosurface Area: 分子表面积，反映分子与外界相互作用的暴露面积。
Molecular Volume: 分子体积，影响堆积和宏观密度。
Sphericity: 球形度。值接近1代表形状趋于球形，值越低代表形状越不规则或细长。
Density: 分子密度 (MW/V)，反映分子的紧凑程度。
第2层：全局电子性质
这些描述符决定了分子的整体电子行为。

前线轨道理论:
HOMO energy: 最高占据分子轨道的能量。值越高，分子越容易失去电子，亲核性/还原性越强。
LUMO energy: 最低未占分子轨道的能量。值越低，分子越容易得到电子，亲电性/氧化性越强。
HOMO-LUMO Gap: HOMO与LUMO的能级差。能隙越大，分子越稳定，化学惰性越高；能隙越小，分子越容易被激发，反应活性越高。
电荷分布:
Dipole Moment: 偶极矩。衡量分子整体的电荷分离程度和极性。
轨道离域指数 (ODI):
ODI_HOMO, ODI_LUMO等: 单个轨道的离域指数。值越低，表示该轨道电子分布越离域（如大π体系）；值越高，表示越定域。
ODI_Mean, ODI_Standard_Deviation: 所有轨道的ODI均值和标准差，从整体上衡量分子的电子离域程度和离域模式的异质性。
第3层：几何与形状
这些描述符从不同角度量化分子的三维形状。

尺寸参数:
Farthest Distance: 分子中最远两原子间的距离，定义了分子的最大尺度。
Molecular Radius: 以质心为中心，到最远原子的距离。
Molecular Size Short/Medium/Long: 分子在三个正交方向上的最大尺寸（基于边界盒），用于描述分子的长宽高。
Long/Sum Size Ratio: L_long / (L_short + L_medium + L_long)。衡量分子的伸长程度，值从~0.33（类球形）到接近1（线形）。
平面性参数:
Molecular Planarity Parameter (MPP): 原子偏离最佳拟合平面的均方根距离。值越小，分子整体平面性越好。
Span of Deviation from Plane (SDP): 原子在拟合平面两侧偏离的最大距离差。与MPP互补，反映分子在平面两侧的“翘曲”程度。
高级形状描述符:
Shape_Asphericity: 基于惯性矩计算的非球形度。0代表完美球形，0.25代表完美圆盘，0.5代表完美线形。衡量的是质量分布的对称性。
Shape_Gyradius (Rg): 回旋半径。质量加权的原子到质心的均方根距离，经典地衡量分子的紧凑度。Rg越大，结构越伸展。
Shape_Waist_Variance: 沿主轴的截面尺寸方差。方差大表示“哑铃形”或有瓶颈的形状；方差小表示均匀的圆柱或球形。
Shape_Geometric_Asphericity: 基于分子尺寸（长宽高）的非球形度，与原子质量无关，纯粹描述几何形状。
第4层：局域电子性质 (分析核心)
这是整个描述符体系中最具化学洞察力的部分。它们描述了分子表面上不同位置的电子特性，直接关联到反应位点和分子间相互作用。所有这些性质的计算都基于将分子范德华表面（通常由电子密度等值面定义，如0.001 a.u.用于ESP/ALIE，0.004 a.u.用于LEAE）分割为各原子所属的区域。

平均局部离子化能 (Average Local Ionization Energy, ALIE):

定义: Ī(r) = Σ [ρ_i(r) * |ε_i|] / ρ(r)，对所有占据轨道求和。
化学意义: 衡量在空间点r处移走一个电子的平均难度。ALIE值越低，代表该处电子束缚越弱，能量越高，越容易失去，因此是亲核性强的位点，极易受到亲电试剂的攻击。ALIE的表面极小点（minimal value）通常对应分子中最活泼的亲核位点（如孤对电子、π电子云区域）。
局部电子附着能 (Local Electron Attachment Energy, LEAE):

定义: E_att(r) = Σ [|φ_i(r)|² * ε_i] / ρ(r)，对能量为负的未占据轨道求和。
化学意义: 衡量在空间点r处增加一个电子的能量变化。LEAE值越低（越负），代表该处接受电子的能力越强，因此是亲电性强的位点，极易受到亲核试剂的攻击。LEAE的表面极小点对应分子中最活泼的亲电位点（如σ-hole, π-hole）。
静电势 (Electrostatic Potential, ESP):

定义: 一个单位正电荷在空间点r处感受到的静电作用能。
化学意义: 反映了分子的电荷分布。ESP为负值的区域（通常在富电子区域，如孤对电子、π体系）吸引正电荷，是亲核中心。ESP为正值的区域（通常在缺电子区域，如酸性氢、σ-hole）吸引负电荷，是亲电中心。
与ALIE/LEAE的互补性: ESP主要描述长程的、经典的静电相互作用，它决定了反应物在接近过程中的初始取向。而ALIE/LEAE更多地描述短程的、与轨道相关的电子转移难易度，决定了成键反应发生的最终位点。一个完整的反应过程需要两者结合分析。例如，一个位点可能ALIE很低（化学上活泼），但如果其周围的ESP不是负的，亲电试剂可能在静电上就不会被优先吸引过来。

层级 5: 多尺度定量表面分析 (MQSA) — 选择你的分析镜头
这是本体系最具创新性的部分。它基于分子的局域电子性质，通过三种不同的“分析镜头”(Modes)，将“点”的性质提升到化学单元的层面。

核心设计哲学: 这三个模式是场景驱动、择一使用的。在向用户解释特征时，你必须首先明确当前分析是基于哪个模式，因为这决定了特征的精确计算方法和化学意义。
本描述符将局域电子性质从“点”提升到“区域”和“化学单元”的层面，实现了对分子性质的分层、定量化描述。核心思想是面积加权平均，即原子在表面暴露的面积越大，其性质对所在官能团的贡献也越大，这非常符合化学直觉。

模式1: 特定元素描述符 (Element-Specific)
适用场景: 当研究问题围绕某一特定元素展开时（例如，设计卤键相互作用、优化含硫药物的靶点亲和力）。这是一种元素中心的假设驱动分析。
格式: X_[Property]_[Stat]，例如 F_ALIE_min, O_ESP_delta。
解读: 针对分子中某一特定元素（如所有F原子），计算其所有原子相关属性的统计值。
X_area: X原子的平均表面积。
X_ALIE_min: 所有X原子中，出现的最低ALIE值。
X_ESP_average: 所有X原子局部表面的平均ESP值。
X_LEAE_delta: 所有X原子的LEAE最大值与最小值之差。这个delta值非常重要，它量化了分子内环境对同种元素电子性质的调控程度。一个大的delta值意味着虽然都是X原子，但它们所处的化学环境差异巨大，导致其亲电/亲核性呈现显著的多样性。
关键描述符的化学释义 (重点加强):
S_Area：这个值代表S原子表面积，在表面分析中其表面积越小代表其被包埋的程度越高。
化学直觉：一个低的S_area暗示这个分子的S位阻较大，不容易被接触而发生反应。
S_ALIE_min: 这个值代表了在分子中所有硫原子上能够找到的最亲核的那一个点的ALIE值，是硫原子表面区域上ALIE的全局最小值。
化学直觉: 一个极低的 S_ALIE_min 值强烈暗示，在某个特定的硫原子上，存在一个能量非常高、束缚非常弱的电子区域。这通常直接对应于一个空间上高度可及的、活泼的孤对电子。因此，这个描述符是定位和量化分子中最强硫亲核中心的直接指标，该位点极易受到亲电试剂的攻击。
C_LEAE_min: 这个值代表了在分子中所有碳原子上能够找到的最亲电的那一个点的LEAE值。
化学直觉: 一个非常负的 C_LEAE_min 值意味着，在某个特定的碳原子表面，存在一个区域，该区域有能量很低的空轨道分布，因此非常容易接受外来电子。
层级 2: 特定官能团描述符 (User-Defined Fragment, Fragment_[Property]_[Stat])
核心思想: 此模式用于检验一个特定化学假说。当研究者已经认定某个特定的官能团（例如，一个酰胺键、一个催化核心、一个反应中心）在所有分子中都至关重要时，此模式可以定量地追踪该核心官能团的电子性质是如何被分子其余部分（即周边环境的修饰）所调控的。
适用场景: 当研究者已确定一个关键功能片段（如催化核心、药效团），并希望通过修饰其周边来优化其性质时。这是一种片段中心的强假设驱动分析。
工作流程:

用户通过原子序号或结构模式（如SMARTS）预先定义一个“Fragment”。
对于数据集中的每个分子，程序会定位这个被定义的Fragment。
程序仅针对该Fragment的表面区域，通过面积加权平均等方法计算其整体的电子属性，最终为每个分子生成一个关于该Fragment的描述符值。
化学释义与实战案例分析 (重点加强):
方法: 我们使用Mode 2，将酰胺键（-C(=O)N-）定义为我们的Fragment。模型训练后发现，Fragment_ALIE_min 是一个关键特征。
SHAP图观察: SHAP图显示，Fragment_ALIE_min 的值越低，催化活性越高。
深度化学解读:
描述符的精确意义: Fragment_ALIE_min 在此案例中，代表的就是酰胺键这个特定单元自身表面上，亲核性最强的那个点的ALIE值。这个点几乎总是对应于羰基氧的孤对电子。
连接化学原理: 一个更低的Fragment_ALIE_min值，意味着酰胺键（特别是其氧原子）的孤对电子活性更强、碱性更强、给电子能力更强。SHAP图的趋势告诉我们，增强酰胺键的给电子性有利于催化反应的进行。这可能是因为它需要作为路易斯碱与底物或反应中间体相互作用。
提出分子设计原理: 这个描述符的真正力量在于它连接了“周边环境”和“核心功能”。Fragment_ALIE_min的值不是一成不变的，它被酰胺键周围的取代基所**“调控”(modulate)。因此，SHAP图揭示的设计原理是：“为了获得更高的催化活性，我们应该在酰胺键的周边修饰上更强的给电子基团（Electron-Donating Groups, EDGs）**。这些EDGs通过共轭效应或诱导效应，将电子云推向酰胺核心，从而使其表面的孤对电子更加活泼（即降低了Fragment_ALIE_min），进而提升了催化性能。
层级 3: LOFFI 全面自动化分析 (Mode 3) - 原子与官能团的双重视角
核心思想: Mode 3 (LOFFI) 是一种用于探索性分析的、标准化的、全自动的流程。它旨在提供一个关于分子表面性质的完整画像，无需用户预先指定任何官能团。其独特之处在于，它同时从两个互补的层面生成一个固定的、包含32个特征的描述符集：全局原子层面和官能团层面。

工作流程总览: LOFFI 首先对整个分子表面进行全局统计（生成 Atom_ 特征）。然后，它利用一个基于化学优先级的智能算法将分子划分为一系列不重叠的官能团，并对这些官能团的整体性质及其相互关系进行统计（生成 Fun_ 特征）。

Part A: LOFFI 的原子层面视角 (Atom_[Property]_[Stat])
这是什么?: 这一组16个特征描述了在整个分子范德华表面上所有点的电子性质的全局统计分布。它不关心一个点属于哪个原子或哪个官能团，而是将整个分子表面视为一个连续的整体。

如何解读?:

Atom_ALIE_min: 这个值代表了在分子表面任何位置能够找到的绝对最亲核的点。它是分子反应活性的“最强音”，是整个分子表面上电子最容易失去的地方。
Atom_ESP_max: 这个值代表了在分子表面任何位置能够找到的绝对最亲电的点（静电势最正）。它通常对应于最强的 σ-hole 或酸性最强的氢原子。
Atom_LEAE_delta: 这个值代表了整个分子表面上，接受电子能力最强和最弱的点之间的差异范围。它衡量了分子整体亲电性的异质性。
Atom_area_mean: 这个值是所有原子局部表面积的平均值，反映了分子表面在原子尺度上的“平滑度”或“粗糙度”。
与其它层级的关键区别: Atom_ALIE_min 是一个“点”的性质，而 S_ALIE_min (Mode 1) 是限制在所有硫原子这个子集上的“点”的性质，Fun_ALIE_min (LOFFI Part B) 则是对“区域平均值”的统计。Atom_ 提供了最基础、最全局的基准。

Part B: LOFFI 的官能团层面视角 (Fun_[Property]_[Stat])
这是什么?: 这一组16个特征旨在量化分子内部不同化学功能单元之间的关系。它的计算是一个精密的两步过程：

第一步：自动化官能团识别与表征。

识别: LOFFI 算法首先根据一套基于化学优先级的规则自动地、无冲突地将分子“切割”成多个官能团。其优先级为：芳香体系 > 高优先级官能团（如-COOH, -NO2）> 通用结构模式（如烷基链）。这确保了化学上最重要的部分被优先识别。
表征: 对于每一个被识别出的官能团，程序会计算出一个单一的、面积加权的属性值（例如 ALIE_weighted, ESP_weighted）。这个值代表了这个官能团作为一个整体的电子特性。经过这一步，一个复杂的分子就被简化为了一组代表其各个功能单元的数值列表。
第二步：对官能团之间的关系进行统计。

Fun_[Property]_[Stat] 描述符就是对上一步生成的**“官能团数值列表”**进行统计的结果。
如何解读?:

Fun_ALIE_min: 这是所有官能团的面积加权ALIE值中最小的那一个。它回答了这样一个问题：“哪一个官能团，作为一整个化学单元，是分子中最强的电子给体（最亲核）？”
Fun_ESP_delta: 这是所有官能团的面积加权ESP值的最大值与最小值之差。它衡量的是分子中最富电子的官能团和最缺电子的官能团之间的极性差异。这是一个完美的“推-拉”效应量化指标，描述了分子内部功能区域的电荷分离程度。
Fun_LEAE_mean: 这是所有官能团的面积加权LEAE值的平均值。它反映了分子整体上由各个官能团贡献的平均亲电性水平。
Fun_area_delta: 这是分子中最大官能团和最小官能团的表面积之差。它量化了分子在结构上是否由尺寸悬殊的模块构成。
四、 全新综合分析举例：解读一个复杂构效关系

场景: 我们正在建立一个预测激酶抑制剂活性的QSPR模型。经过特征筛选，XGBoost模型告诉我们，两个特征对预测IC50值（值越小活性越高）至关重要：S_ALIE_min 和 Fun_ESP_delta。

分析任务: 结合SHAP图，解释这两个特征如何影响抑制剂的活性。

第一步：独立分析 S_ALIE_min 的SHAP图
SHAP图观察: 散点图清晰地显示，随着 S_ALIE_min 的值从高到低移动（横轴从右向左），其对应的SHAP值急剧下降（变为很大的负值）。这意味着，一个更低的 S_ALIE_min 值对预测结果有强烈的负向贡献。由于我们的目标是IC50，负向贡献就意味着预测的IC50值更低，即分子活性更高。

化学解释:

回忆定义: S_ALIE_min 精准地量化了分子中最活泼的那个硫原子的亲核性。值越低，亲核性越强，代表其孤对电子越容易给出。
形成假说: 这个强烈的趋势表明，一个高活性的、亲核性强的硫中心是抑制剂发挥作用的关键。这种强亲核性可能服务于一个特定的生物学功能，例如：
与激酶活性位点中的某个金属离子（如Mg²⁺, Zn²⁺）形成强力的配位键。
作为关键的氢键受体，与活性位点中的某个重要的氢键供体（如-NH或-OH基团）形成非常稳定的氢键。
在某些情况下，甚至可能作为共价抑制剂，亲核攻击活性位点中的某个氨基酸残基（如半胱氨酸）。
引导探索: “模型强烈暗示了一个关键的硫原子相互作用。我们应该立即选取SHAP图中 S_ALIE_min 最低、SHAP值最负的几个分子，将它们与激酶的晶体结构进行对接（docking）或叠合。我们的目标是验证这个高活性的硫原子是否真的指向了已知的关键相互作用位点。”
第二步：独立分析 Fun_ESP_delta 的SHAP图
SHAP图观察: 该图显示，随着 Fun_ESP_delta 值的增大（横轴从左向右），SHAP值也呈现下降趋势（变为负值），但可能不如 S_ALIE_min 那样陡峭。这意味着，一个更大的 Fun_ESP_delta 值同样有助于提升分子活性（降低IC50）。

化学解释:

回忆定义: Fun_ESP_delta 衡量的是分子内部不同官能团之间的静电势差异。值越大，说明分子内部的“电荷分离”越明显，功能区域的极性差异越大。
形成假说: 这个趋势说明，分子的整体“药效团骨架”需要具备良好的极性分布。一个高活性的抑制剂不仅仅依赖于单个“明星原子”，还需要其整体结构能够与蛋白口袋的大环境相匹配。一个大的 Fun_ESP_delta 值可能意味着：
分子的一端（如一个带正电的胺基）能与蛋白的负电性“入口”区域形成长程静电吸引，引导分子正确进入活性位点。
分子的另一端（如一个富电子的芳环）则能与疏水性区域形成pi-pi堆积或疏水作用。
这种“功能上的两极分化”使得分子能像一把“多点钥匙”一样，与复杂的蛋白口袋实现多点、互补的相互作用，从而获得更高的亲和力。
第三步：综合两大特征进行升华分析
最终结论: 现在，我们可以将两个特征的分析结合起来，描绘出一幅完整的、具有化学洞察力的“理想抑制剂画像”。
“我们的模型揭示了一个双层面的活性机制。首先，一个高活性的抑制剂需要具备一个高 Fun_ESP_delta 值的分子骨架。这确保了分子能够通过长程的、多点的静电和疏水作用，高效地、以正确的姿态‘导航’并‘锚定’在激酶的活性口袋中。这可以看作是**‘宏观识别’和‘姿态锁定’**的步骤。”
“然而，真正决定其超高活性的‘致命一击’，来自于一个极低的 S_ALIE_min 值。这意味着，在分子骨架正确定位后，一个被环境特异性活化了的硫原子，将与活性位点的核心残基或离子发生一次极其关键的、高强度的短程相互作用。这是**‘微观结合’和‘亲和力锁定’**的决定性步骤。
"""

        # 初始化ZhipuAI API客户端
        print("🔧 [LLM] 初始化ZhipuAI API客户端...")
        try:
            self.client = ZhipuAiClient(api_key="88e3901fe0114e7b9432b94656ab738d.pAKDobPCPqxOs8x7")
            print("✅ [LLM] ZhipuAI API客户端初始化成功")
        except Exception as e:
            print(f"❌ [LLM] ZhipuAI API客户端初始化失败: {e}")
            self.client = None

        # 检查文件和运行计算
        if xyz_folder_path:
            self.check_and_generate_surface_files()

        if csv_file_path and os.path.exists(csv_file_path):
            self.load_data()

        # 加载测试集数据（如果提供）
        if test_csv_path and os.path.exists(test_csv_path):
            self.load_test_data()
    def load_test_data(self):
        """加载测试集CSV数据"""
        try:
            self.test_data = pd.read_csv(self.test_csv_path)

            print(f"测试集数据加载成功！")
            print(f"测试集样本数量: {len(self.test_data)}")

            # 检查必要的列是否存在
            required_cols = ['Sample_Name', 'Target', 'Realtest_Pred']
            missing_cols = [col for col in required_cols if col not in self.test_data.columns]
            if missing_cols:
                print(f"⚠️ 测试集缺少必要列: {missing_cols}")

        except Exception as e:
            print(f"测试集数据加载失败: {e}")
            self.test_data = None
    def check_required_files(self, sample_names):
        """检查所需的表面PDB文件是否存在"""
        missing_files = []
        
        for sample_name in sample_names:
            # 处理样本名称
            if isinstance(sample_name, (int, np.integer)):
                sample_number = str(sample_name)
            else:
                sample_number = ''.join(filter(str.isdigit, str(sample_name)))
            
            if sample_number:
                sample_number = sample_number.zfill(6)
                
                # 检查xyz.pdb文件
                xyz_pdb_file = f"{sample_number}_xyz.pdb"
                xyz_pdb_path = os.path.join(self.xyz_folder_path, xyz_pdb_file)
                if not os.path.exists(xyz_pdb_path):
                    missing_files.append(xyz_pdb_file)
                
                # 检查表面PDB文件
                for surface_type in self.surface_types:
                    surface_pdb_file = f"{sample_number}_{surface_type}.pdb"
                    surface_pdb_path = os.path.join(self.xyz_folder_path, surface_pdb_file)
                    if not os.path.exists(surface_pdb_path):
                        missing_files.append(surface_pdb_file)
        
        return missing_files

    def check_and_generate_surface_files(self):
        """检查并生成缺失的表面文件"""
        if not self.xyz_folder_path or not os.path.exists(self.xyz_folder_path):
            print("⚠️ xyz_folder_path 未设置或不存在，跳过文件检查")
            return
        
        print(f"🔍 检查 {self.xyz_folder_path} 中的文件...")
        
        # 检查是否存在fchk文件
        fchk_files = glob.glob(os.path.join(self.xyz_folder_path, "*.fchk"))
        
        if not fchk_files:
            print("ℹ️ 未找到fchk文件，跳过Multiwfn计算")
            return
        
        print(f"📋 找到 {len(fchk_files)} 个fchk文件")
        
        # 检查表面PDB文件的完整性
        sample_names = [os.path.splitext(os.path.basename(f))[0] for f in fchk_files]
        missing_files = []
        
        for sample_name in sample_names:
            sample_number = sample_name.zfill(6)
            
            # 检查xyz.pdb文件
            xyz_pdb_file = f"{sample_number}_xyz.pdb"
            xyz_pdb_path = os.path.join(self.xyz_folder_path, xyz_pdb_file)
            if not os.path.exists(xyz_pdb_path):
                missing_files.append(f"{sample_name}_xyz.pdb")
            
            # 检查表面PDB文件
            for surface_type in self.surface_types:
                surface_pdb_file = f"{sample_number}_{surface_type}.pdb"
                surface_pdb_path = os.path.join(self.xyz_folder_path, surface_pdb_file)
                if not os.path.exists(surface_pdb_path):
                    missing_files.append(f"{sample_name}_{surface_type}.pdb")
        
        if missing_files:
            print(f"⚠️ 发现 {len(missing_files)} 个缺失文件，开始Multiwfn计算...")
            print("缺失文件示例:", missing_files[:5])
            
            # 运行Multiwfn计算
            try:
                processed_files = MultiwfnCalculator.run_multiwfn_on_fchk_files(self.xyz_folder_path)
                print(f"✅ Multiwfn计算完成，生成了 {len(processed_files)} 个文件")
            except Exception as e:
                print(f"❌ Multiwfn计算失败: {e}")
        else:
            print("✅ 所有必需的表面PDB文件都已存在")

    def load_data(self):
        """加载CSV数据"""
        try:
            self.data = pd.read_csv(self.csv_file_path)
            
            # 提取特征列和SHAP列
            self.feature_columns = [col for col in self.data.columns if col.startswith('Feature_')]
            self.shap_columns = [col for col in self.data.columns if col.startswith('SHAP_')]
            
            # 提取特征名称（去掉前缀）
            self.feature_names = [col.replace('Feature_', '') for col in self.feature_columns]
            
            print(f"数据加载成功！")
            print(f"样本数量: {len(self.data)}")
            print(f"特征数量: {len(self.feature_names)}")
            
            # 如果数据加载成功，检查对应的表面文件
            if hasattr(self.data, 'Sample_Name'):
                sample_names = self.data['Sample_Name'].tolist()
                missing_files = self.check_required_files(sample_names)
                if missing_files:
                    print(f"⚠️ 数据中的样本对应 {len(missing_files)} 个缺失的表面文件")
            
        except Exception as e:
            print(f"数据加载失败: {e}")

    def call_deepseek_llm(self, user_input, images=None):
        """调用ZhipuAI LLM，加入研究背景引导和描述符知识"""
        print(f"🔧 [LLM] 开始处理用户输入: {user_input[:50]}...")
        if images:
            print(f"🔧 [LLM] 包含 {len(images)} 张图片")

        if not self.client:
            error_msg = "❌ [LLM] API客户端未初始化"
            print(error_msg)
            return error_msg

        if not user_input or not user_input.strip():
            if self.first_interaction:
                return self.get_initial_guidance()
            error_msg = "❌ [LLM] 用户输入为空"
            print(error_msg)
            return "请输入您的问题"

        try:
            print(f"🔧 [LLM] 准备发送到ZhipuAI API...")
            print(f"🔧 [LLM] 输入长度: {len(user_input)}")

            # 构建系统提示，包含描述符知识和研究背景引导
            system_prompt = f"""你是一个专业的计算化学与数据科学交叉领域的AI助手。你的核心任务是帮助研究人员深入解读基于XGBoost模型生成的SHAP (SHapley Additive exPlanations)图。该模型使用了一套独特的、基于量子化学计算的多尺度分子描述符进行训练，这些描述符旨在从基础属性、电子结构、三维形状到局域反应性等多个层面量化分子的特征。你的目标是将SHAP图所揭示的数学规律，与描述符背后深刻的物理化学原理联系起来，提供具有化学洞察力的分析。

首先，请了解用户的研究背景：
1. 用户研究的Target变量是什么？（如溶解度、毒性、活性等）
2. 研究场景和应用领域是什么？
3. 用户选取哪个mode的Surfacia特征表示方法？

以下是详细的分子描述符背景知识，请结合这些知识为用户提供专业解释：

{self.descriptor_knowledge}

请基于用户的具体研究背景和上述知识库，提供针对性的专业建议和解释。如果用户提供图片，请仔细分析图片内容并结合描述符知识提供相关解释。"""

            # 构建消息内容
            content = [{"type": "text", "text": user_input}]

            # 添加图片内容
            if images:
                for img_data in images:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": img_data
                        }
                    })

            response = self.client.chat.completions.create(
                model="glm-4.1v-thinking-flashx",
                messages=[
                    {"role": "system", "content": [
                        {"type": "text", "text": system_prompt}
                    ]},
                    {"role": "user", "content": content}
                ],
                max_tokens=2000,
                temperature=0.3
            )

            result = response.choices[0].message.content
            self.first_interaction = False
            
            print(f"✅ [LLM] API调用成功")
            print(f"🔧 [LLM] 响应长度: {len(result)}")
            print(f"🔧 [LLM] 响应预览: {result[:100]}...")

            return result

        except Exception as e:
            error_msg = f"❌ [LLM] API调用失败: {str(e)}"
            print(f"🔧 [LLM] 详细错误: {error_msg}")
            return f"抱歉，AI服务暂时不可用。错误信息：{str(e)}"

    def get_initial_guidance(self):
        """获取初始引导信息"""
        return """👋 欢迎使用分子描述符智能分析系统！

🎯 **首次使用指导**

为了为您提供更精准的分析建议，请先告诉我您的研究背景：

📊 **研究信息**
1. **Target变量**: 您研究的目标属性是什么？
   - 例如：溶解度、毒性、生物活性、药物活性、材料性能等

2. **研究领域**: 您的应用场景是什么？
   - 例如：药物发现、材料设计、环境化学、食品科学等

3. **数据特点**: 您的分子数据集有什么特点？
   - 例如：药物分子、天然产物、聚合物、小分子化合物等

💡 **我能帮您**
- 解释各种分子描述符的物理意义
- 分析SHAP值对模型预测的贡献
- 解读分子表面静电势图
- 指导特征选择和模型优化
- 提供构效关系分析建议

🔬 **描述符类型**
- **LEAE**: 局部电子亲和能 - 识别亲电位点
- **ESP**: 静电势 - 揭示电荷分布
- **ALIE**: 平均局部电离能 - 定位亲核位点

请分享您的研究背景，我将为您提供专业的个性化建议！ 🚀"""

    def create_simple_llm_interface(self):
        """创建带图片粘贴功能和研究背景引导的LLM聊天界面"""
        return html.Div([
            html.H4("🤖 AI分析助手 (GLM-4.1v) - 分子描述符专家", 
                   style={
                       'textAlign': 'center',
                       'color': '#2c3e50',
                       'marginBottom': 20,
                       'fontFamily': 'Arial Black'
                   }),
            
            # 研究背景提示卡片
            html.Div([
                html.H6("💡 首次使用提示", style={'color': '#2980b9', 'marginBottom': 10}),
                html.P("请先介绍您的研究背景：Target变量含义、研究领域、数据特点等，以获得更精准的分析建议。", 
                      style={'fontSize': 12, 'color': '#34495e', 'margin': 0})
            ], style={
                'backgroundColor': '#ebf3fd',
                'border': '1px solid #3498db',
                'borderRadius': 8,
                'padding': 12,
                'marginBottom': 15
            }),
            
            # 输入区域
            html.Div([
                # 左侧文本输入
                html.Div([
                    dcc.Textarea(
                        id='llm-input',
                        placeholder='🔬 请输入您的问题或研究背景...\n\n📋 建议首次使用时介绍：\n• Target变量是什么？(如溶解度、毒性等)\n• 研究领域？(如药物发现、材料设计等)\n• 数据集特点？(如小分子、天然产物等)\n\n💬 其他问题：\n• 解释某个分子描述符的含义\n• 分析SHAP图中的模式\n• 询问构效关系\n• 上传图片进行分析\n\n🖼️ 提示：可以使用 Ctrl+V 粘贴图片！',
                        style={
                            'width': '100%',
                            'height': 140,
                            'resize': 'none',
                            'border': '1px solid #ced4da',
                            'borderRadius': 5,
                            'padding': 10,
                            'fontSize': 14
                        }
                    )
                ], style={'width': '60%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                # 中间图片上传区域
                html.Div([
                    dcc.Upload(
                        id='image-upload',
                        children=html.Div([
                            html.I(className='fas fa-cloud-upload-alt', style={'fontSize': '24px', 'marginBottom': '8px'}),
                            html.Div('拖拽图片或点击上传', style={'fontSize': '12px'}),
                            html.Div('支持分子图、SHAP图等', style={'fontSize': '10px', 'color': '#666'})
                        ]),
                        style={
                            'width': '100%',
                            'height': '140px',
                            'lineHeight': '140px',
                            'borderWidth': '2px',
                            'borderStyle': 'dashed',
                            'borderRadius': '8px',
                            'textAlign': 'center',
                            'margin': '0',
                            'backgroundColor': '#f8f9fa',
                            'border': '2px dashed #dee2e6',
                            'cursor': 'pointer'
                        },
                        multiple=True,
                        accept='image/*'
                    )
                ], style={'width': '18%', 'marginLeft': '2%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                # 右侧发送按钮
                html.Div([
                    html.Button(
                        '发送分析',
                        id='llm-send-btn',
                        style={
                            'width': '100%',
                            'height': 140,
                            'backgroundColor': '#007bff',
                            'color': 'white',
                            'border': 'none',
                            'borderRadius': 5,
                            'cursor': 'pointer',
                            'fontSize': 16,
                            'fontWeight': 'bold'
                        }
                    )
                ], style={'width': '18%', 'marginLeft': '2%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ], style={'display': 'flex', 'marginBottom': 15}),
            
            # 图片预览区域
            html.Div(
                id='image-preview-area',
                style={
                    'minHeight': '60px',
                    'marginBottom': 15,
                    'padding': 10,
                    'border': '1px solid #dee2e6',
                    'borderRadius': 5,
                    'backgroundColor': '#f8f9fa',
                    'display': 'none'
                }
            ),
            
            # 响应显示区域
            html.Div(
                id='llm-response',
                children=[
                    html.Div(self.get_initial_guidance(),
                           style={
                               'color': '#333',
                               'padding': 20,
                               'lineHeight': '1.6',
                               'whiteSpace': 'pre-wrap'
                           })
                ],
                style={
                    'border': '1px solid #dee2e6',
                    'borderRadius': 8,
                    'padding': 15,
                    'backgroundColor': '#f8f9fa',
                    'minHeight': 300,
                    'maxHeight': 500,
                    'overflowY': 'auto'
                }
            ),
            
            # 状态显示
            html.Div(
                id='llm-status',
                children="✅ 描述符专家就绪 - 支持图片分析 & 研究背景引导",
                style={
                    'textAlign': 'center',
                    'marginTop': 10,
                    'padding': 5,
                    'fontSize': 12,
                    'color': '#666'
                }
            ),
            
            # 添加图片粘贴功能的JavaScript
            html.Script("""
                document.addEventListener('DOMContentLoaded', function() {
                    // 存储粘贴的图片数据
                    window.pastedImages = [];
                    
                    // 监听键盘粘贴事件
                    document.addEventListener('paste', function(e) {
                        const items = e.clipboardData.items;
                        const textarea = document.getElementById('llm-input');
                        
                        // 检查是否在文本输入框内
                        if (document.activeElement === textarea) {
                            for (let i = 0; i < items.length; i++) {
                                const item = items[i];
                                if (item.type.indexOf('image') !== -1) {
                                    e.preventDefault();
                                    const file = item.getAsFile();
                                    if (file) {
                                        const reader = new FileReader();
                                        reader.onload = function(event) {
                                            const base64Data = event.target.result;
                                            window.pastedImages.push(base64Data);
                                            
                                            // 显示图片预览
                                            showImagePreview(base64Data);
                                            
                                            // 在文本框中添加提示
                                            textarea.value += '\\n[已粘贴图片 - 请点击发送进行分析]';
                                            
                                            // 触发Dash回调来更新图片预览区域
                                            const event = new Event('input', { bubbles: true });
                                            textarea.dispatchEvent(event);
                                        };
                                        reader.readAsDataURL(file);
                                    }
                                }
                            }
                        }
                    });
                    
                    function showImagePreview(base64Data) {
                        const previewArea = document.getElementById('image-preview-area');
                        if (previewArea) {
                            previewArea.style.display = 'block';
                            
                            const img = document.createElement('img');
                            img.src = base64Data;
                            img.style.maxWidth = '200px';
                            img.style.maxHeight = '150px';
                            img.style.margin = '5px';
                            img.style.border = '2px solid #007bff';
                            img.style.borderRadius = '5px';
                            
                            previewArea.appendChild(img);
                        }
                    }
                });
            """),
            
            # 隐藏的存储组件用于传递图片数据
            dcc.Store(id='pasted-images-store', data=[]),
            dcc.Store(id='uploaded-images-store', data=[])
            
        ], style={
            'backgroundColor': 'white',
            'border': '2px solid #bdc3c7',
            'borderRadius': 10,
            'padding': 20,
            'maxWidth': '2340px',
            'margin': '20px auto',
            'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'
        })

    def determine_surface_type(self, feature_name):
        """根据特征名确定等值面类型"""
        feature_upper = feature_name.upper()
        
        for surface_type in self.surface_types:
            if surface_type in feature_upper:
                print(f"检测到特征 '{feature_name}' 包含 '{surface_type}'，将使用 {surface_type} 等值面")
                return surface_type
        
        # 默认使用ESP
        print(f"特征 '{feature_name}' 未包含明确的等值面类型，默认使用 ESP 等值面")
        return 'ESP'

    def load_xyz_file(self, sample_name):
        """根据样本名加载xyz文件内容"""
        if not self.xyz_folder_path:
            return None, None

        try:
            # 处理不同类型的样本名
            if isinstance(sample_name, (int, np.integer)):
                sample_number = str(sample_name)
            else:
                sample_number = ''.join(filter(str.isdigit, str(sample_name)))

            if not sample_number:
                print(f"无法从样本名中提取数字: {sample_name}")
                return None, None

            xyz_filename = f"{sample_number.zfill(6)}.xyz"
            xyz_path = os.path.join(self.xyz_folder_path, xyz_filename)

            if not os.path.exists(xyz_path):
                print(f"XYZ文件不存在: {xyz_path}")
                return None, None

            with open(xyz_path, 'r') as f:
                xyz_content = f.read()

            print(f"成功加载XYZ文件: {xyz_filename}")
            return xyz_content, xyz_filename

        except Exception as e:
            print(f"加载XYZ文件失败: {e}")
            return None, None

    def load_surface_pdb_file(self, sample_name, surface_type):
        """根据样本名和等值面类型加载对应的PDB文件"""
        if not self.xyz_folder_path:
            return None, None

        try:
            # 处理不同类型的样本名
            if isinstance(sample_name, (int, np.integer)):
                sample_number = str(sample_name)
            else:
                sample_number = ''.join(filter(str.isdigit, str(sample_name)))

            if not sample_number:
                print(f"无法从样本名中提取数字: {sample_name}")
                return None, None

            pdb_filename = f"{sample_number.zfill(6)}_{surface_type}.pdb"
            pdb_path = os.path.join(self.xyz_folder_path, pdb_filename)

            if not os.path.exists(pdb_path):
                print(f"{surface_type} PDB文件不存在: {pdb_path}")
                # 尝试自动生成
                self.auto_generate_missing_surface_file(sample_number, surface_type)
                # 重新检查
                if os.path.exists(pdb_path):
                    print(f"✅ 自动生成了 {surface_type} PDB文件")
                else:
                    return None, None

            with open(pdb_path, 'r') as f:
                pdb_content = f.read()

            print(f"成功加载 {surface_type} PDB文件: {pdb_filename}")
            return pdb_content, pdb_filename

        except Exception as e:
            print(f"加载 {surface_type} PDB文件失败: {e}")
            return None, None

    def auto_generate_missing_surface_file(self, sample_number, surface_type):
        """自动生成缺失的表面文件"""
        try:
            fchk_file = f"{sample_number}.fchk"
            fchk_path = os.path.join(self.xyz_folder_path, fchk_file)
            
            if os.path.exists(fchk_path):
                print(f"🔄 尝试自动生成 {sample_number}_{surface_type}.pdb...")
                result = MultiwfnCalculator.run_multiwfn_calculation(
                    self.xyz_folder_path, surface_type, sample_number, fchk_file
                )
                if result:
                    print(f"✅ 成功生成 {result}")
                else:
                    print(f"❌ 生成 {surface_type} 文件失败")
            else:
                print(f"⚠️ 找不到对应的fchk文件: {fchk_file}")
        except Exception as e:
            print(f"❌ 自动生成表面文件失败: {e}")

    def load_xyz_pdb_file(self, sample_name):
        """根据样本名加载xyz.pdb文件"""
        if not self.xyz_folder_path:
            return None, None

        try:
            # 处理不同类型的样本名
            if isinstance(sample_name, (int, np.integer)):
                sample_number = str(sample_name)
            else:
                sample_number = ''.join(filter(str.isdigit, str(sample_name)))

            if not sample_number:
                print(f"无法从样本名中提取数字: {sample_name}")
                return None, None

            xyz_pdb_filename = f"{sample_number.zfill(6)}_xyz.pdb"
            xyz_pdb_path = os.path.join(self.xyz_folder_path, xyz_pdb_filename)

            if not os.path.exists(xyz_pdb_path):
                print(f"XYZ PDB文件不存在: {xyz_pdb_path}")
                return None, None

            with open(xyz_pdb_path, 'r') as f:
                xyz_pdb_content = f.read()

            print(f"成功加载XYZ PDB文件: {xyz_pdb_filename}")
            return xyz_pdb_content, xyz_pdb_filename

        except Exception as e:
            print(f"加载XYZ PDB文件失败: {e}")
            return None, None

    def smiles_to_png_base64(self, smiles, width=250, height=200):
        """将SMILES转换为PNG图片的base64编码，加粗键和原子"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
            drawer.drawOptions().bondLineWidth = 4
            drawer.drawOptions().atomLabelFontSize = 20
            drawer.drawOptions().multipleBondOffset = 0.2
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            svg_string = drawer.GetDrawingText()
            
            svg_b64 = base64.b64encode(svg_string.encode()).decode()
            return f"data:image/svg+xml;base64,{svg_b64}"
            
        except Exception as e:
            print(f"分子结构生成失败 {smiles}: {e}")
            return None

    def parse_pdb_beta_values(self, pdb_content):
        """解析PDB文件中的B因子值，获取范围"""
        if not pdb_content:
            return -22, 22
        
        beta_values = []
        lines = pdb_content.split('\n')
        
        for line in lines:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    beta_value = float(line[60:66].strip())
                    beta_values.append(beta_value)
                except (ValueError, IndexError):
                    continue
        
        if beta_values:
            min_val, max_val = min(beta_values), max(beta_values)
            print(f"PDB文件B因子范围: {min_val:.3f} 到 {max_val:.3f}")
            return min_val, max_val
        else:
            print(f"未找到有效的B因子值，使用默认范围 -22 到 22")
            return -22, 22
    def create_features_shap_table(self, row):
        """创建特征值和SHAP值的表格显示"""

        # 提取所有特征列和对应的SHAP列
        table_data = []

        for i, feature_name in enumerate(self.feature_names):
            feature_col = f'Feature_{feature_name}'
            shap_col = f'SHAP_{feature_name}'

            if feature_col in row.index and shap_col in row.index:
                feature_value = row[feature_col]
                shap_value = row[shap_col]

                # 格式化数值显示
                if isinstance(feature_value, (int, float)):
                    if abs(feature_value) >= 1000:
                        feature_str = f"{feature_value:.2e}"
                    elif abs(feature_value) >= 1:
                        feature_str = f"{feature_value:.3f}"
                    else:
                        feature_str = f"{feature_value:.4f}"
                else:
                    feature_str = str(feature_value)

                if isinstance(shap_value, (int, float)):
                    if abs(shap_value) >= 1000:
                        shap_str = f"{shap_value:.2e}"
                    elif abs(shap_value) >= 1:
                        shap_str = f"{shap_value:.3f}"
                    else:
                        shap_str = f"{shap_value:.4f}"
                else:
                    shap_str = str(shap_value)

                table_data.append({
                    'index': i + 1,
                    'feature_name': feature_name,
                    'feature_value': feature_str,
                    'shap_value': shap_str,
                    'shap_numeric': shap_value  # 用于颜色编码
                })

        # 创建表格头部
        table_header = html.Thead([
            html.Tr([
                html.Th("#", style={'width': '8%', 'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '16px', 'padding': '8px', 'backgroundColor': '#f8f9fa', 'border': '1px solid #dee2e6'}),
                html.Th("特征名称", style={'width': '42%', 'textAlign': 'left', 'fontWeight': 'bold', 'fontSize': '16px', 'padding': '8px', 'backgroundColor': '#f8f9fa', 'border': '1px solid #dee2e6'}),
                html.Th("特征值", style={'width': '25%', 'textAlign': 'right', 'fontWeight': 'bold', 'fontSize': '16px', 'padding': '8px', 'backgroundColor': '#f8f9fa', 'border': '1px solid #dee2e6'}),
                html.Th("SHAP值", style={'width': '25%', 'textAlign': 'right', 'fontWeight': 'bold', 'fontSize': '16px', 'padding': '8px', 'backgroundColor': '#f8f9fa', 'border': '1px solid #dee2e6'})
            ])
        ])

        # 创建表格主体
        table_rows = []
        for item in table_data:
            # SHAP值颜色编码
            shap_numeric = item['shap_numeric']
            if isinstance(shap_numeric, (int, float)):
                if shap_numeric > 0:
                    shap_color = '#d73027'  # 红色（正贡献）
                    shap_bg_color = '#fee5e5'
                elif shap_numeric < 0:
                    shap_color = '#1a9850'  # 绿色（负贡献）
                    shap_bg_color = '#e5f5e5'
                else:
                    shap_color = '#666666'  # 灰色（零贡献）
                    shap_bg_color = '#f8f9fa'
            else:
                shap_color = '#666666'
                shap_bg_color = '#f8f9fa'

            row = html.Tr([
                html.Td(str(item['index']), style={'textAlign': 'center', 'fontSize': '22px', 'padding': '6px', 'border': '1px solid #dee2e6'}),
                html.Td(item['feature_name'], style={'textAlign': 'left', 'fontSize': '22px', 'padding': '6px', 'border': '1px solid #dee2e6', 'fontFamily': 'monospace'}),
                html.Td(item['feature_value'], style={'textAlign': 'right', 'fontSize': '22px', 'padding': '6px', 'border': '1px solid #dee2e6', 'fontFamily': 'monospace'}),
                html.Td(item['shap_value'], style={'textAlign': 'right', 'fontSize': '22px', 'padding': '6px', 'border': '1px solid #dee2e6', 'fontFamily': 'monospace', 'color': shap_color, 'backgroundColor': shap_bg_color, 'fontWeight': 'bold'})
            ])
            table_rows.append(row)

        table_body = html.Tbody(table_rows)

        # 组装完整表格
        table = html.Table([table_header, table_body], style={
            'width': '100%',
            'borderCollapse': 'collapse',
            'marginTop': '20px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'backgroundColor': 'white'
        })

        # 创建带标题的表格容器
        table_container = html.Div([
            html.H5("🔍 样本特征值与SHAP值详情", 
                   style={
                       'color': '#2c3e50',
                       'marginBottom': '10px',
                       'fontSize': '20px',
                       'fontWeight': 'bold',
                       'textAlign': 'center'
                   }),

            html.P(f"共 {len(table_data)} 个特征 | 红色: 正贡献 | 绿色: 负贡献", 
                   style={
                       'fontSize': '14px',
                       'color': '#7f8c8d',
                       'textAlign': 'center',
                       'marginBottom': '15px',
                       'fontStyle': 'italic'
                   }),

            html.Div([
                table
            ], style={
                'maxHeight': '400px',
                'overflowY': 'auto',
                'border': '2px solid #bdc3c7',
                'borderRadius': '8px'
            })
        ])

        return table_container
    def create_3d_molecule_viewer(self, xyz_content, xyz_filename, viewer_type="xyz", height="200px", show_labels=False):
        """使用py3Dmol创建三维分子查看器的HTML"""
        if not xyz_content:
            return html.Div("XYZ文件加载失败", 
                           style={
                               'color': '#e74c3c',
                               'fontSize': 18,
                               'textAlign': 'center',
                               'padding': 50
                           })
        
        viewer_id = f"mol_viewer_{viewer_type}_{hash(xyz_filename) % 10000}"
        
        label_js = ""
        if show_labels:
            label_js = """
            var atoms = viewer.getModel().selectedAtoms({});
            atoms.forEach(function(atom, index) {
                if (atom.elem && atom.elem !== 'H') {
                    viewer.addLabel(atom.elem, {
                        position: atom, 
                        backgroundColor: 'rgba(255,255,255,0.8)', 
                        fontColor: 'black',
                        fontSize: 12,
                        borderThickness: 1,
                        borderColor: 'gray'
                    });
                }
            });
            """
        
        html_content = f"""
        <div id="{viewer_id}" style="height: {height}; width: 100%; position: relative; background: white; border: 2px solid #ddd; border-radius: 8px;"></div>
        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
        <script>
            var xyz_data = `{xyz_content}`;
            var viewer = $3Dmol.createViewer('{viewer_id}', {{
                defaultcolors: $3Dmol.elementColors.Jmol,
                backgroundColor: 'white'
            }});
            
            viewer.addModel(xyz_data, 'xyz');
            viewer.setStyle({{}}, {{
                stick: {{radius: 0.15, color: 'spectrum'}},
                sphere: {{scale: 0.25}}
            }});
            
            viewer.zoomTo();
            viewer.render();
            viewer.zoom(1.2, 1000);
            
            {label_js}
        </script>
        """
        
        return html.Iframe(
            srcDoc=html_content,
            style={
                'width': '100%',
                'height': height,
                'border': 'none'
            }
        )

    def create_combined_surface_xyz_viewer(self, surface_pdb_content=None, xyz_pdb_content=None, surface_type="ESP", point_size=25, opacity=100, height="600px", use_auto_range=True):
        """创建结合等值面PDB和xyz PDB的三维查看器"""
        if not surface_pdb_content:
            return html.Div("等值面PDB文件未加载", 
                           style={
                               'color': '#e74c3c',
                               'fontSize': 18,
                               'textAlign': 'center',
                               'padding': 50
                           })
        
        # 解析实际的Beta值范围
        actual_min, actual_max = self.parse_pdb_beta_values(surface_pdb_content)
        
        # 使用实际范围或手动设置的范围
        if use_auto_range:
            color_min, color_max = actual_min, actual_max
        else:
            # 这里可以设置为VMD控制面板的值，但现在先用实际范围
            color_min, color_max = actual_min, actual_max
        
        viewer_id = "combined_surface_xyz_viewer"
        
        # 准备xyz_pdb数据
        xyz_pdb_data_js = f"var xyz_pdb_data = `{xyz_pdb_content}`;" if xyz_pdb_content else "var xyz_pdb_data = null;"
        
        # 根据等值面类型设置颜色方案和标题
        surface_info = {
            'ESP': {'title': 'Electrostatic Potential', 'unit': '(kcal/mol)', 'gradient': 'rwb'},
            'LEAE': {'title': 'Local Electron Attachment Energy', 'unit': '(eV)', 'gradient': 'rwb'},
            'ALIE': {'title': 'Average Local Ionization Energy', 'unit': '(eV)', 'gradient': 'rwb'}
        }
        
        info = surface_info.get(surface_type, surface_info['ESP'])
        
        html_content = f"""
        <div style="position: relative; height: {height}; width: 100%; background: white; border: 2px solid #ddd; border-radius: 8px;">
            <div id="{viewer_id}" style="height: {height}; width: 100%; position: relative;"></div>
            
            <!-- 颜色条 -->
            <div style="position: absolute; bottom: 30px; left: 30px; width: 350px; height: 40px; 
                        background: linear-gradient(to right, 
                        rgb(0,0,255) 0%, rgb(0,128,255) 16.67%, rgb(0,255,255) 33.33%, 
                        rgb(255,255,255) 50%, rgb(255,255,0) 66.67%, rgb(255,128,0) 83.33%, rgb(255,0,0) 100%);
                        border: 2px solid #333; border-radius: 5px; z-index: 100;">
                <div style="position: absolute; bottom: -25px; left: 0; font-size: 12px; color: black; font-weight: bold;">{color_min:.2f}</div>
                <div style="position: absolute; bottom: -25px; right: 0; font-size: 12px; color: black; font-weight: bold;">{color_max:.2f}</div>
                <div style="position: absolute; bottom: -25px; left: 50%; transform: translateX(-50%); 
                            font-size: 12px; color: black; font-weight: bold;">{((color_min + color_max) / 2):.2f}</div>
                <div style="position: absolute; top: -25px; left: 50%; transform: translateX(-50%); 
                            font-size: 14px; color: black; font-weight: bold;">{info['title']} {info['unit']}</div>
            </div>
            
            <!-- 范围信息显示 -->
            <div style="position: absolute; bottom: 15px; right: 15px; background: rgba(255,255,255,0.9); 
                        padding: 8px; border-radius: 6px; border: 1px solid #ccc; z-index: 100; font-size: 11px;">
                <div style="color: #333; font-weight: bold;">实际范围: {actual_min:.3f} ~ {actual_max:.3f}</div>
                <div style="color: #666;">显示范围: {color_min:.3f} ~ {color_max:.3f}</div>
            </div>
            
            <!-- 显示控制按钮 -->
            <div style="position: absolute; top: 15px; right: 15px; background: rgba(255,255,255,0.9); 
                        padding: 10px; border-radius: 8px; border: 1px solid #ccc; z-index: 100;">
                <div style="margin-bottom: 5px;">
                    <button id="toggleSurface_{viewer_id}" style="padding: 5px 10px; margin: 2px; 
                            background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px;">
                        Hide Surface
                    </button>
                </div>
                <div>
                    <button id="toggleSkeleton_{viewer_id}" style="padding: 5px 10px; margin: 2px; 
                            background: #e67e22; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px;">
                        Hide Skeleton
                    </button>
                </div>
            </div>
        </div>
        
        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
        <script>
            var surface_pdb_data = `{surface_pdb_content}`;
            {xyz_pdb_data_js}
            
            var viewer = $3Dmol.createViewer('{viewer_id}', {{
                defaultcolors: $3Dmol.elementColors.Jmol,
                backgroundColor: 'white'
            }});
            
            var surfaceVisible = true;
            var skeletonVisible = true;
            
            function updateVisualization() {{
                var pointSize = {point_size};
                var colorMin = {color_min};
                var colorMax = {color_max};
                var opacity = {opacity / 100.0};
                
                viewer.clear();
                
                // 添加等值面PDB（模型0）
                if (surface_pdb_data && surfaceVisible) {{
                    viewer.addModel(surface_pdb_data, 'pdb');
                    viewer.setStyle({{model: 0}}, {{
                        sphere: {{
                            radius: pointSize / 100.0,
                            opacity: opacity,
                            colorscheme: {{
                                prop: 'b',
                                gradient: '{info['gradient']}',
                                min: colorMin,
                                max: colorMax
                            }}
                        }}
                    }});
                }}
                
                // 添加XYZ骨架（模型1）
                if (xyz_pdb_data && skeletonVisible) {{
                    viewer.addModel(xyz_pdb_data, 'pdb');
                    var modelIndex = (surface_pdb_data && surfaceVisible) ? 1 : 0;
                    viewer.setStyle({{model: modelIndex}}, {{
                        stick: {{radius: 0.15, color: 'spectrum'}},
                        sphere: {{scale: 0.25, color: 'spectrum'}}
                    }});
                }}
                
                viewer.zoomTo();
                viewer.render();
                viewer.zoom(1.2, 1000);
            }}
            
            // 切换表面显示
            document.getElementById('toggleSurface_{viewer_id}').addEventListener('click', function() {{
                surfaceVisible = !surfaceVisible;
                this.textContent = surfaceVisible ? 'Hide Surface' : 'Show Surface';
                this.style.backgroundColor = surfaceVisible ? '#3498db' : '#95a5a6';
                updateVisualization();
            }});
            
            // 切换骨架显示
            document.getElementById('toggleSkeleton_{viewer_id}').addEventListener('click', function() {{
                skeletonVisible = !skeletonVisible;
                this.textContent = skeletonVisible ? 'Hide Skeleton' : 'Show Skeleton';
                this.style.backgroundColor = skeletonVisible ? '#e67e22' : '#95a5a6';
                updateVisualization();
            }});
            
            // 初始化
            updateVisualization();
            
            // 存储viewer实例以便外部控制
            window.combined_surface_xyz_viewer = viewer;
            window.updateCombinedVisualization = updateVisualization;
        </script>
        """
        
        return html.Iframe(
            srcDoc=html_content,
            style={
                'width': '100%',
                'height': height,
                'border': 'none'
            }
        )

    def create_surfacia_mode_viewer(self, sample_name, smiles, target_value, feature_value, shap_value, selected_feature, surface_type, surface_pdb_content=None, xyz_pdb_content=None, point_size=25, opacity=100):
        """创建Surfacia模式的查看器：全屏等值面+xyz组合，信息叠加在左上和右上角"""
        
        mol_img = self.smiles_to_png_base64(smiles, width=200, height=150)
        
        return html.Div([
            # 全屏等值面+xyz组合显示
            html.Div([
                self.create_combined_surface_xyz_viewer(surface_pdb_content, xyz_pdb_content, surface_type, point_size, opacity, "1040px", use_auto_range=True)
            ], style={
                'height': '1040px',
                'width': '100%',
                'position': 'relative'
            }),
            
            # 左上角叠加：二维分子图
            html.Div([
                html.Img(src=mol_img, 
                       style={
                           'width': '200px',
                           'height': '150px',
                           'border': '2px solid #333',
                           'borderRadius': 8,
                           'background': 'white'
                       }) if mol_img else 
                html.Div("分子结构生成失败", 
                       style={
                           'color': '#e74c3c',
                           'fontSize': 12,
                           'textAlign': 'center',
                           'padding': 20,
                           'background': 'white',
                           'border': '2px solid #333',
                           'borderRadius': 8
                       })
            ], style={
                'position': 'absolute',
                'top': '15px',
                'left': '15px',
                'zIndex': 200,
                'background': 'rgba(255,255,255,0.95)',
                'padding': '8px',
                'borderRadius': '10px',
                'boxShadow': '0 4px 8px rgba(0,0,0,0.3)'
            }),
            
            # 右上角叠加：样本信息
            html.Div([
                html.P(f"样本: {sample_name}", style={'fontSize': 16, 'margin': '3px 0', 'color': '#2c3e50', 'fontWeight': 'bold'}),
                html.P(f"Target: {target_value:.3f}", style={'fontSize': 16, 'margin': '3px 0', 'color': '#e74c3c', 'fontWeight': 'bold'}),
                html.P(f"{selected_feature}: {feature_value}", style={'fontSize': 16, 'margin': '3px 0', 'color': '#34495e'}),
                html.P(f"SHAP: {shap_value:.3f}", style={'fontSize': 16, 'margin': '3px 0', 'color': '#e74c3c' if shap_value > 0 else '#27ae60', 'fontWeight': 'bold'}),
                html.P(f"Surface: {surface_type}", style={'fontSize': 16, 'margin': '3px 0', 'color': '#8e44ad', 'fontWeight': 'bold'}),
            ], style={
                'position': 'absolute',
                'top': '15px',
                'right': '160px',
                'zIndex': 200,
                'background': 'rgba(255,255,255,0.95)',
                'padding': '12px',
                'borderRadius': '10px',
                'boxShadow': '0 4px 8px rgba(0,0,0,0.3)',
                'minWidth': '200px'
            })
        ], style={
            'height': '1040px',
            'width': '100%',
            'position': 'relative'
        })

    def create_feature_shap_plot(self, selected_feature, show_test_set=False):
        """创建特征-SHAP图 (Feature vs SHAP)，可选显示测试集"""
        if self.data is None or not selected_feature:
            return go.Figure()

        feature_col = f'Feature_{selected_feature}'
        shap_col = f'SHAP_{selected_feature}'

        if feature_col not in self.data.columns or shap_col not in self.data.columns:
            return go.Figure()

        # 训练集数据
        feature_values = self.data[feature_col].values
        shap_values = self.data[shap_col].values
        target_values = self.data['Target'].values

        hover_text = []
        for idx, row in self.data.iterrows():
            hover_info = (
                f"<b>训练集 - {row.get('Sample_Name', 'N/A')}</b><br>"
                f"Target: {row.get('Target', 0):.3f}<br>"
                f"{selected_feature}: {row.get(feature_col, 0)}<br>"
                f"SHAP_{selected_feature}: {row.get(shap_col, 0):.3f}"
            )
            hover_text.append(hover_info)

        fig = go.Figure()

        # 添加训练集散点
        scatter_train = go.Scatter(
            x=feature_values,
            y=shap_values,
            mode='markers',
            marker=dict(
                size=18,
                color=target_values,
                colorscale='RdYlBu_r',
                colorbar=dict(
                    title=dict(
                        text="Target Value",
                        font=dict(size=28, family="Arial Black")
                    ),
                    tickfont=dict(size=24, family="Arial Black"),
                    thickness=30,
                    len=0.8
                ),
                line=dict(width=2, color='white'),
                opacity=0.8,
                showscale=True
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            customdata=list(range(len(self.data))),
            name='Training Set',
            legendgroup='train'
        )

        fig.add_trace(scatter_train)

        # 添加测试集散点（如果启用且数据存在）
        if show_test_set and self.test_data is not None:
            if feature_col in self.test_data.columns and shap_col in self.test_data.columns:
                test_feature_values = self.test_data[feature_col].values
                test_shap_values = self.test_data[shap_col].values
                test_target_values = self.test_data['Target'].values
                test_pred_values = self.test_data.get('Realtest_Pred', test_target_values).values

                test_hover_text = []
                for idx, row in self.test_data.iterrows():
                    hover_info = (
                        f"<b>测试集 - {row.get('Sample_Name', 'N/A')}</b><br>"
                        f"Target (真实): {row.get('Target', 0):.3f}<br>"
                        f"Predicted: {row.get('Realtest_Pred', 0):.3f}<br>"
                        f"{selected_feature}: {row.get(feature_col, 0)}<br>"
                        f"SHAP_{selected_feature}: {row.get(shap_col, 0):.3f}"
                    )
                    test_hover_text.append(hover_info)

                scatter_test = go.Scatter(
                    x=test_feature_values,
                    y=test_shap_values,
                    mode='markers',
                    marker=dict(
                        size=18,
                        color=test_pred_values,  # 使用预测值着色
                        colorscale='RdYlBu_r',
                        symbol='diamond',  # 使用菱形标记区分
                        line=dict(width=2, color='black'),  # 黑色边框
                        opacity=0.9,
                        showscale=False  # 不显示第二个颜色条
                    ),
                    text=test_hover_text,
                    hovertemplate='%{text}<extra></extra>',
                    customdata=list(range(len(self.test_data))),
                    name='Test Set',
                    legendgroup='test'
                )

                fig.add_trace(scatter_test)

        # 设置坐标轴范围
        all_feature_values = feature_values
        all_shap_values = shap_values

        if show_test_set and self.test_data is not None and feature_col in self.test_data.columns:
            all_feature_values = np.concatenate([feature_values, test_feature_values])
            all_shap_values = np.concatenate([shap_values, test_shap_values])

        x_range = [all_feature_values.min(), all_feature_values.max()]
        x_margin = (x_range[1] - x_range[0]) * 0.05
        x_range = [x_range[0] - x_margin, x_range[1] + x_margin]

        # 添加SHAP=0参考线
        fig.add_shape(
            type="line",
            x0=x_range[0], y0=0, x1=x_range[1], y1=0,
            line=dict(color="red", width=5, dash="dash"),
            name="SHAP=0"
        )

        # 更新布局
        title_text = f'Feature-SHAP Analysis: {selected_feature}'
        if show_test_set and self.test_data is not None:
            title_text += ' (Train + Test)'

        fig.update_layout(
            title=dict(
                text=title_text,
                font=dict(size=36, family="Arial Black"),
                x=0.5
            ),
            xaxis=dict(
                title=dict(
                    text=f'{selected_feature} (Feature Value)',
                    font=dict(size=32, family="Arial Black")
                ),
                tickfont=dict(size=28, family="Arial Black"),
                linewidth=4,
                gridcolor='lightgray',
                gridwidth=2,
                showgrid=True
            ),
            yaxis=dict(
                title=dict(
                    text=f'SHAP_{selected_feature}',
                    font=dict(size=32, family="Arial Black")
                ),
                tickfont=dict(size=28, family="Arial Black"),
                linewidth=4,
                gridcolor='lightgray',
                gridwidth=2,
                showgrid=True,
                zeroline=True,
                zerolinecolor='red',
                zerolinewidth=5
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=1040,
            height=1040,
            showlegend=show_test_set,  # 只在显示测试集时显示图例
            legend=dict(
                x=0.02,
                y=0.98,
                font=dict(size=20, family="Arial"),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=2
            ),
            hovermode='closest',
            font=dict(family="Arial Black"),
            margin=dict(l=120, r=120, t=120, b=120)
        )

        return fig

    def create_dash_app(self):
        """创建Dash应用"""
        if self.data is None:
            print("请先加载数据！")
            return None
        
        self.app = dash.Dash(__name__)
        
        self.app.layout = html.Div([
            # 顶部标题和选择器
            html.Div([
                html.H1("🧬 交互式特征-SHAP分析 & 分子表面静电势可视化 + GLM智能助手", 
                       style={
                           'textAlign': 'center', 
                           'color': '#2c3e50', 
                           'marginBottom': 30,
                           'fontFamily': 'Arial Black',
                           'fontSize': 42,
                           'fontWeight': 'bold'
                       }),
                
                html.Div([
                    # 显示模式选择
                    html.Div([
                        html.Label("显示模式:", 
                                  style={
                                      'fontWeight': 'bold', 
                                      'fontSize': 24,
                                      'fontFamily': 'Arial Black',
                                      'marginRight': 15
                                  }),
                        dcc.RadioItems(
                            id='display-mode',
                            options=[
                                {'label': '二维分子结构', 'value': '2d'},
                                {'label': '三维分子结构', 'value': '3d'},
                                {'label': 'Surfacia模式', 'value': 'surfacia'}
                            ],
                            value='2d',
                            inline=True,
                            style={
                                'fontSize': 20,
                                'fontFamily': 'Arial'
                            }
                        )
                    ], style={'marginRight': 30}),
                    
                    # 等值面类型显示
                    html.Div([
                        html.Label("等值面类型:", 
                                  style={
                                      'fontWeight': 'bold', 
                                      'fontSize': 20,
                                      'fontFamily': 'Arial Black',
                                      'marginRight': 10,
                                      'color': '#8e44ad'
                                  }),
                        html.Span(id='surface-type-display', 
                                children="ESP",
                                style={
                                    'fontSize': 20,
                                    'fontFamily': 'Arial',
                                    'color': '#8e44ad',
                                    'fontWeight': 'bold'
                                })
                    ], style={'marginRight': 30}),
                    html.Div([
                        dcc.Checklist(
                            id='show-test-set',
                            options=[{'label': ' 显示测试集', 'value': 'show'}],
                            value=[],
                            style={
                                'fontSize': 20,
                                'fontFamily': 'Arial',
                                'marginLeft': '20px'
                            }
                        )
                    ], style={'marginRight': 30}),
                    # 特征选择
                    html.Div([
                        html.Label("选择特征:", 
                                  style={
                                      'fontWeight': 'bold', 
                                      'fontSize': 24,
                                      'fontFamily': 'Arial Black',
                                      'marginRight': 15
                                  }),
                        dcc.Dropdown(
                            id='feature-dropdown',
                            options=[{'label': name, 'value': name} for name in self.feature_names],
                            value=self.feature_names[0] if self.feature_names else None,
                            style={
                                'width': 300,
                                'fontSize': 20,
                                'fontFamily': 'Arial'
                            }
                        )
                    ])
                ], style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'marginBottom': 30
                }),
            ]),
            
            # 主要内容区域 - 左右布局
            html.Div([
                # 左侧：SHAP散点图
                html.Div([
                    dcc.Graph(
                        id='feature-shap-plot',
                        config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                        }
                    )
                ], style={
                    'flex': '1',
                    'backgroundColor': 'white',
                    'borderRadius': 15,
                    'padding': 20,
                    'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                    'marginRight': 20
                }),
                
                # 右侧：分子信息面板
                html.Div([
                    html.Div(
                        id='molecule-info-panel',
                        children=[
                            html.Div("点击散点查看分子结构", 
                                   style={
                                       'textAlign': 'center',
                                       'color': '#7f8c8d',
                                       'fontSize': 24,
                                       'fontStyle': 'italic',
                                       'padding': 100
                                   })
                        ],
                        style={
                            'backgroundColor': 'white',
                            'border': '3px solid #bdc3c7',
                            'borderRadius': 15,
                            'padding': 20,
                            'height': '1040px',
                            'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                            'overflow': 'hidden'
                        }
                    )
                ], style={
                    'flex': '1'
                })
            ], style={
                'display': 'flex',
                'maxWidth': '2340px',
                'margin': '0 auto',
                'gap': '26px'
            }),
            
            # 底部VMD控制面板
            html.Div([
                html.Div([
                    html.H4("VMD-style Controls", 
                           style={
                               'textAlign': 'center',
                               'color': '#2c3e50',
                               'marginBottom': 20,
                               'fontFamily': 'Arial Black'
                           }),
                    
                    html.Div([
                        # Point Size控制 (1-35)
                        html.Div([
                            html.Label("Point Size:", style={'fontWeight': 'bold', 'marginRight': 10}),
                            dcc.Slider(
                                id='point-size-slider',
                                min=1, max=35, step=1, value=25,
                                marks={i: str(i) for i in range(1, 36, 5)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], style={'width': '20%', 'display': 'inline-block', 'marginRight': '4%'}),
                        
                        # Opacity控制
                        html.Div([
                            html.Label("Opacity:", style={'fontWeight': 'bold', 'marginRight': 10}),
                            dcc.Slider(
                                id='opacity-slider',
                                min=0, max=100, step=5, value=90,
                                marks={i: f"{i}%" for i in range(0, 101, 25)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], style={'width': '20%', 'display': 'inline-block', 'marginRight': '4%'}),
                        
                        # 自动颜色范围开关
                        html.Div([
                            html.Label("颜色范围:", style={'fontWeight': 'bold', 'marginBottom': 10}),
                            dcc.Checklist(
                                id='auto-range-check',
                                options=[{'label': ' 自动范围', 'value': 'auto'}],
                                value=['auto'],
                                style={'marginBottom': 5}
                            ),
                            html.Div([
                                dcc.Input(id='color-min-input', type='number', value=-22, step=0.1, 
                                        style={'width': '80px', 'marginRight': '10px'}, disabled=True),
                                html.Span("to", style={'marginRight': '10px'}),
                                dcc.Input(id='color-max-input', type='number', value=22, step=0.1, 
                                        style={'width': '80px'}, disabled=True)
                            ])
                        ], style={'width': '20%', 'display': 'inline-block', 'marginRight': '4%'}),
                        
                        # 其他控制
                        html.Div([
                            html.Button("Reset View", id='reset-view-btn', 
                                      style={'padding': '8px 16px', 'marginRight': '10px',
                                             'background': '#3498db', 'color': 'white', 'border': 'none', 'borderRadius': '5px'}),
                            dcc.Checklist(
                                id='auto-rotate-check',
                                options=[{'label': ' Auto Rotate', 'value': 'rotate'}],
                                value=[],
                                style={'display': 'inline-block'}
                            )
                        ], style={'width': '25%', 'display': 'inline-block'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-around'})
                    
                ], style={
                    'backgroundColor': 'white',
                    'border': '2px solid #bdc3c7',
                    'borderRadius': 10,
                    'padding': 20,
                    'maxWidth': '2340px',
                    'margin': '20px auto',
                    'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'
                })
            ]),
            
            # 简化的LLM聊天界面（包含图片功能）
            self.create_simple_llm_interface(),
            
            # 隐藏的数据存储
            dcc.Store(id='selected-sample-store')
            
        ], style={
            'fontFamily': 'Arial',
            'backgroundColor': '#f8f9fa',
            'minHeight': '100vh',
            'padding': 20
        })
        
        # 处理图片上传的回调
        @self.app.callback(
            [Output('uploaded-images-store', 'data'),
             Output('image-preview-area', 'children'),
             Output('image-preview-area', 'style')],
            [Input('image-upload', 'contents')],
            [State('image-upload', 'filename')]
        )
        def handle_image_upload(uploaded_files, filenames):
            if not uploaded_files:
                return [], [], {'minHeight': '60px', 'marginBottom': 15, 'padding': 10, 
                              'border': '1px solid #dee2e6', 'borderRadius': 5, 
                              'backgroundColor': '#f8f9fa', 'display': 'none'}
            
            image_data = []
            preview_children = [html.H6("已上传的图片:", style={'marginBottom': 10, 'color': '#495057'})]
            
            for i, (content, filename) in enumerate(zip(uploaded_files, filenames)):
                # 存储图片数据
                image_data.append(content)
                
                # 创建预览
                preview_children.append(
                    html.Div([
                        html.Img(src=content, style={
                            'maxWidth': '200px', 
                            'maxHeight': '150px', 
                            'margin': '5px',
                            'border': '2px solid #007bff',
                            'borderRadius': '5px'
                        }),
                        html.P(filename, style={
                            'fontSize': '12px', 
                            'color': '#6c757d', 
                            'textAlign': 'center',
                            'margin': '5px'
                        })
                    ], style={'display': 'inline-block', 'textAlign': 'center', 'marginRight': '10px'})
                )
            
            style = {'minHeight': '60px', 'marginBottom': 15, 'padding': 10, 
                    'border': '1px solid #dee2e6', 'borderRadius': 5, 
                    'backgroundColor': '#f8f9fa', 'display': 'block'}
            
            return image_data, preview_children, style
        
        # 改进的LLM聊天回调，支持图片和研究背景引导
        @self.app.callback(
            [Output('llm-response', 'children'),
             Output('llm-input', 'value'),
             Output('llm-status', 'children')],
            [Input('llm-send-btn', 'n_clicks')],
            [State('llm-input', 'value'),
             State('uploaded-images-store', 'data'),
             State('pasted-images-store', 'data')]
        )
        def handle_llm_chat_with_images(n_clicks, user_input, uploaded_images, pasted_images):
            print(f"🔧 [LLM] GLM聊天回调触发 - n_clicks: {n_clicks}")
            print(f"🔧 [LLM] 上传图片数量: {len(uploaded_images) if uploaded_images else 0}")
            print(f"🔧 [LLM] 粘贴图片数量: {len(pasted_images) if pasted_images else 0}")
            
            if not n_clicks or not user_input:
                print("🔧 [LLM] 无有效输入，返回默认状态")
                return (
                    [html.Div(self.get_initial_guidance(),
                             style={'color': '#333', 'padding': 20, 'lineHeight': '1.6', 'whiteSpace': 'pre-wrap'})],
                    "",
                    "✅ 描述符专家就绪 - 支持图片分析 & 研究背景引导"
                )
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"🔧 [LLM] 开始处理用户请求 - 时间: {timestamp}")
            
            # 合并所有图片
            all_images = []
            if uploaded_images:
                all_images.extend(uploaded_images)
            if pasted_images:
                all_images.extend(pasted_images)
            
            # 调用GLM（支持图片和研究背景引导）
            try:
                ai_response = self.call_deepseek_llm(user_input, all_images)
                status = f"✅ 描述符专家完成 ({timestamp})" + (f" - 包含{len(all_images)}张图片" if all_images else "")
                print(f"🔧 [LLM] 成功获得GLM响应")
            except Exception as e:
                ai_response = f"抱歉，处理过程中出现错误：{str(e)}"
                status = f"❌ GLM错误 ({timestamp})"
                print(f"🔧 [LLM] GLM处理失败: {e}")
            
            # 构建响应显示
            response_content = []
            
            # 用户输入显示
            user_content = [html.Strong("🙋 用户: ", style={'color': '#007bff'}), html.Span(user_input, style={'color': '#333'})]
            
            # 添加图片显示
            if all_images:
                user_content.append(html.Br())
                user_content.append(html.Div("📎 附件图片:", style={'color': '#6c757d', 'fontSize': '12px', 'marginTop': '5px'}))
                for i, img in enumerate(all_images):
                    user_content.append(
                        html.Img(src=img, style={
                            'maxWidth': '100px',
                            'maxHeight': '75px',
                            'margin': '3px',
                            'border': '1px solid #007bff',
                            'borderRadius': '3px'
                        })
                    )
            
            response_content.append(
                html.Div(user_content, style={
                    'marginBottom': 15, 
                    'padding': 10, 
                    'backgroundColor': '#e3f2fd', 
                    'borderRadius': 5
                })
            )
            
            # AI响应显示
            response_content.append(
                html.Div([
                    html.Strong("🤖 描述符专家 (GLM-4.1v): ", style={'color': '#28a745'}),
                    html.Div(ai_response, style={
                        'color': '#333', 
                        'marginTop': 5,
                        'whiteSpace': 'pre-wrap',
                        'lineHeight': '1.6'
                    })
                ], style={'padding': 10, 'backgroundColor': '#f1f8e9', 'borderRadius': 5})
            )
            
            print(f"🔧 [LLM] GLM回调完成，返回响应")
            return response_content, "", status
        
        # 回调函数：更新等值面类型显示
        @self.app.callback(
            Output('surface-type-display', 'children'),
            [Input('feature-dropdown', 'value')]
        )
        def update_surface_type_display(selected_feature):
            if selected_feature:
                surface_type = self.determine_surface_type(selected_feature)
                return surface_type
            return "ESP"
        
        # 回调函数：控制颜色范围输入框的启用状态
        @self.app.callback(
            [Output('color-min-input', 'disabled'),
             Output('color-max-input', 'disabled')],
            [Input('auto-range-check', 'value')]
        )
        def toggle_color_range_inputs(auto_range_value):
            is_auto = 'auto' in auto_range_value if auto_range_value else False
            return is_auto, is_auto
        
        # 回调函数：更新散点图
# 回调函数：更新散点图
        @self.app.callback(
            Output('feature-shap-plot', 'figure'),
            [Input('feature-dropdown', 'value'),
             Input('show-test-set', 'value') if hasattr(self, 'test_data') and self.test_data is not None else Input('feature-dropdown', 'value')]
        )
        def update_scatter_plot(selected_feature, show_test_values=None):
            if selected_feature:
                # 如果有测试集数据且传入了show_test_values参数
                if hasattr(self, 'test_data') and self.test_data is not None and show_test_values is not None:
                    show_test = 'show' in show_test_values if show_test_values else False
                    return self.create_feature_shap_plot(selected_feature, show_test_set=show_test)
                else:
                    # 没有测试集数据，使用原始方法
                    return self.create_feature_shap_plot(selected_feature)
            return go.Figure()
        
        # 回调函数：更新右侧分子信息面板
        # 回调函数：更新右侧分子信息面板
        @self.app.callback(
            [Output('molecule-info-panel', 'children'),
             Output('selected-sample-store', 'data')],
            [Input('feature-shap-plot', 'clickData'),
             Input('feature-dropdown', 'value'),
             Input('display-mode', 'value'),
             Input('point-size-slider', 'value'),
             Input('opacity-slider', 'value'),
             Input('auto-range-check', 'value'),
             Input('color-min-input', 'value'),
             Input('color-max-input', 'value')]
        )
        def update_molecule_panel(clickData, selected_feature, display_mode, point_size, opacity, auto_range_value, color_min, color_max):
            if clickData is None:
                return (html.Div("点击散点查看分子结构", 
                              style={
                                  'textAlign': 'center',
                                  'color': '#7f8c8d',
                                  'fontSize': 24,
                                  'fontStyle': 'italic',
                                  'padding': 100
                              }), {})

            try:
                # 确定点击的是训练集还是测试集
                point_index = clickData['points'][0]['pointIndex']
                trace_name = clickData['points'][0].get('curveNumber', 0)

                # 根据trace判断是训练集还是测试集
                if trace_name == 0:  # 训练集
                    row = self.data.iloc[point_index]
                    data_source = "训练集"
                else:  # 测试集
                    if hasattr(self, 'test_data') and self.test_data is not None:
                        row = self.test_data.iloc[point_index]
                        data_source = "测试集"
                    else:
                        row = self.data.iloc[point_index]
                        data_source = "训练集"

                smiles = row.get('SMILES', '')
                sample_name = row.get('Sample_Name', 'Unknown')
                target_value = row.get('Target', 0)
                feature_value = row.get(f'Feature_{selected_feature}', 0)
                shap_value = row.get(f'SHAP_{selected_feature}', 0)

                # 存储选中的样本数据
                sample_data = {
                    'sample_name': sample_name,
                    'smiles': smiles,
                    'target_value': target_value,
                    'feature_value': feature_value,
                    'shap_value': shap_value,
                    'data_source': data_source
                }
                
                if display_mode == 'surfacia':
                    # 确定等值面类型
                    surface_type = self.determine_surface_type(selected_feature)
                    
                    # 加载等值面PDB文件
                    surface_pdb_content, surface_pdb_filename = self.load_surface_pdb_file(sample_name, surface_type)
                    
                    # 加载xyz PDB文件
                    xyz_pdb_content, xyz_pdb_filename = self.load_xyz_pdb_file(sample_name)
                    
                    if not surface_pdb_content:
                        return (html.Div(f"Surfacia模式不支持：缺少 {surface_type} 等值面文件", 
                                      style={
                                          'textAlign': 'center',
                                          'color': '#e74c3c',
                                          'fontSize': 24,
                                          'fontWeight': 'bold',
                                          'padding': 100
                                      }), sample_data)
                    
                    molecule_content = self.create_surfacia_mode_viewer(
                        sample_name=sample_name,
                        smiles=smiles,
                        target_value=target_value,
                        feature_value=feature_value,
                        shap_value=shap_value,
                        selected_feature=selected_feature,
                        surface_type=surface_type,
                        surface_pdb_content=surface_pdb_content,
                        xyz_pdb_content=xyz_pdb_content,
                        point_size=point_size or 25,
                        opacity=opacity or 100
                    )

                elif display_mode == '2d':
                    # 二维分子结构模式 - 重新设计布局
                    mol_img = self.smiles_to_png_base64(smiles, width=600, height=500)

                    # 创建特征值和SHAP值表格（增大字体）
                    features_table = self.create_features_shap_table(row)

                    molecule_content = html.Div([
                        # 上半部分：左侧样本信息 + 右侧分子图
                        html.Div([
                            # 左上角：样本基本信息
                            html.Div([
                                html.P(f"样本: {sample_name} ({data_source})", 
                                      style={
                                          'fontWeight': 'bold',
                                          'fontSize': 24,
                                          'margin': '8px 0',
                                          'color': '#2c3e50'
                                      }),
                                html.P(f"Target: {target_value:.3f}", 
                                      style={
                                          'fontSize': 24,
                                          'margin': '6px 0',
                                          'color': '#e74c3c',
                                          'fontWeight': 'bold'
                                      }),
                                html.P(f"{selected_feature}: {feature_value}", 
                                      style={
                                          'fontSize': 24,
                                          'margin': '6px 0'
                                      }),
                                html.P(f"SHAP: {shap_value:.3f}", 
                                      style={
                                          'fontSize': 24,
                                          'margin': '6px 0',
                                          'color': '#e74c3c' if shap_value > 0 else '#27ae60',
                                          'fontWeight': 'bold'
                                      }),
                                # SMILES显示（去掉标签）
                                html.P(smiles, 
                                      style={
                                          'fontFamily': 'Courier New',
                                          'fontSize': 20,
                                          'backgroundColor': '#ecf0f1',
                                          'padding': 10,
                                          'borderRadius': 5,
                                          'wordBreak': 'break-all',
                                          'margin': '15px 0 0 0',
                                          'maxHeight': '80px',
                                          'overflowY': 'auto'  # SMILES过长时可滚动
                                      })
                            ], style={
                                'width': '40%',
                                'paddingRight': '20px',
                                'verticalAlign': 'top',
                                'display': 'inline-block'
                            }),

                            # 右上角：分子二维图（保持原尺寸）
                            html.Div([
                                html.Img(src=mol_img, 
                                       style={
                                           'width': '600px',
                                           'height': '500px',
                                           'border': '2px solid #bdc3c7',
                                           'borderRadius': 8,
                                           'display': 'block'
                                       }) if mol_img else 
                                html.Div("分子结构生成失败", 
                                       style={
                                           'color': '#e74c3c',
                                           'fontSize': 24,
                                           'textAlign': 'center',
                                           'padding': 20,
                                           'width': '600px',
                                           'height': '500px',
                                           'border': '2px solid #bdc3c7',
                                           'borderRadius': 8,
                                           'display': 'flex',
                                           'alignItems': 'center',
                                           'justifyContent': 'center'
                                       })
                            ], style={
                                'width': '60%',
                                'display': 'inline-block',
                                'verticalAlign': 'top'
                            })
                        ], style={
                            'display': 'flex',
                            'width': '100%',
                            'marginBottom': '20px',
                            'minHeight': '500px'  # 确保上半部分有足够高度
                        }),

                        # 下半部分：特征值和SHAP值表格（可滚动）
                        html.Div([
                            features_table
                        ], style={
                            'width': '100%',
                            'maxHeight': '520px',  # 剩余空间高度
                            'overflowY': 'auto',   # 纵向滚动
                            'border': '2px solid #bdc3c7',
                            'borderRadius': '8px',
                            'backgroundColor': 'white'
                        })

                    ], style={
                        'width': '100%',
                        'height': '1040px',  # 总高度固定
                        'padding': '10px',
                        'display': 'flex',
                        'flexDirection': 'column'
                    })
                else:
                    # 三维分子结构模式
                    xyz_content, xyz_filename = self.load_xyz_file(sample_name)
                    mol_img = self.smiles_to_png_base64(smiles, width=250, height=200)
                    
                    molecule_content = html.Div([
                        html.Div([
                            html.Div([
                                html.Img(src=mol_img, 
                                       style={
                                           'width': '250px',
                                           'height': '180px',
                                           'border': '2px solid #bdc3c7',
                                           'borderRadius': 8
                                       }) if mol_img else 
                                html.Div("分子结构生成失败", 
                                       style={
                                           'color': '#e74c3c',
                                           'fontSize': 16,
                                           'textAlign': 'center',
                                           'padding': 20
                                       })
                            ], style={
                                'width': '48%',
                                'display': 'inline-block',
                                'verticalAlign': 'top'
                            }),
                            
                            html.Div([
                                html.P(f"样本: {sample_name}", 
                                      style={
                                          'fontWeight': 'bold',
                                          'fontSize': 20,
                                          'margin': '4px 0',
                                          'color': '#2c3e50'
                                      }),
                                html.P(f"Target: {target_value:.3f}", 
                                      style={
                                          'fontSize': 20,
                                          'margin': '3px 0',
                                          'color': '#e74c3c',
                                          'fontWeight': 'bold'
                                      }),
                                html.P(f"{selected_feature}: {feature_value}", 
                                      style={
                                          'fontSize': 20,
                                          'margin': '3px 0'
                                      }),
                                html.P(f"SHAP: {shap_value:.3f}", 
                                      style={
                                          'fontSize': 20,
                                          'margin': '3px 0',
                                          'color': '#e74c3c' if shap_value > 0 else '#27ae60',
                                          'fontWeight': 'bold'
                                      }),

                                html.P(f"XYZ: {xyz_filename}" if xyz_filename else "XYZ文件未找到", 
                                      style={
                                          'fontSize': 18,
                                          'color': '#7f8c8d',
                                          'fontStyle': 'italic'
                                      })
                            ], style={
                                'width': '48%',
                                'display': 'inline-block',
                                'verticalAlign': 'top',
                                'marginLeft': '4%'
                            })
                        ], style={
                            'marginBottom': 15,
                            'height': '200px'
                        }),
                        
                        html.Div([
                            html.H5("3D Molecular Structure", style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 10}),
                            self.create_3d_molecule_viewer(xyz_content, xyz_filename, "main_3d", "800px", show_labels=True)
                        ], style={'height': '820px'})
                    ])
                
                return molecule_content, sample_data
                
            except Exception as e:
                return (html.Div(f"错误: {str(e)}", 
                              style={
                                  'color': '#e74c3c', 
                                  'fontSize': 18,
                                  'textAlign': 'center',
                                  'padding': 20
                              }), {})
        
        return self.app

    def run_app(self, debug=True, port=8052, host='0.0.0.0'):
        """运行Dash应用"""
        if self.app is None:
            self.create_dash_app()

        if self.app:
            print(f"🚀 启动交互式SHAP分析应用...")
            print(f"📱 本地访问: http://localhost:{port}")
            print(f"🌐 远程访问: http://10.26.50.35:{port}")
            print(f"📊 左侧：SHAP散点图分析")
            print(f"🧬 右侧：多模式分子可视化")
            print(f"🎯 Surfacia模式：智能等值面 + XYZ骨架组合显示")
            print(f"⚡ 支持的等值面类型：{', '.join(self.surface_types)}")
            print(f"🤖 AI功能：GLM-4.1v描述符专家已集成")
            print(f"🖼️ 图片功能：支持Ctrl+V粘贴和拖拽上传图片")
            print(f"📚 知识库：内置分子描述符背景知识")
            print(f"🎓 研究引导：首次使用将引导了解研究背景")
            print(f"🔧 自动计算：自动检查并生成缺失的表面PDB文件")
            self.app.run(debug=debug, port=port, host=host)
        else:
            print("❌ 应用创建失败！")


# 使用示例
def main():
    # 指定你的文件路径
    csv_file_path = "/home/yumingsu/Python/Project_Surfacia/250805_ShenyanS_copy/Surfacia_3.0_20250805_093212/ManualFeature_Analysis_20250808_171713/Training_Set_Detailed_Manual_5feats_20250808_171713.csv"
    xyz_folder_path = "/home/yumingsu/Python/Project_Surfacia/250627_ShenyanS/"
    test_csv_path = "/home/yumingsu/Python/Project_Surfacia/250805_ShenyanS_copy/Surfacia_3.0_20250805_093212/ManualFeature_Analysis_20250808_171713/Test_Set_Detailed_Manual_5feats_20250808_171713.csv"
    # 创建分析器实例
    analyzer = InteractiveSHAPAnalyzer(
        csv_file_path=csv_file_path,
        xyz_folder_path=xyz_folder_path,
        test_csv_path=test_csv_path  # 新增参数
    )
    
    # 运行交互式应用
    analyzer.run_app(debug=True, port=8052)

if __name__ == "__main__":
    main()


# In[ ]:




