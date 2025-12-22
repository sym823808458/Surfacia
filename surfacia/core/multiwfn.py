"""
Multiwfn计算和处理模块
"""
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

# 从 descriptors 模块导入形状计算函数
from .descriptors import (
    get_atomic_mass,
    calculate_principal_moments_of_inertia,
    calculate_asphericity,
    calculate_gyradius,
    calculate_relative_gyradius,
    calculate_waist_variance,
    calculate_geometric_asphericity
)

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