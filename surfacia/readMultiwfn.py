# surfacia/readMultiwfn.py

import os
import numpy as np
import pandas as pd
import csv
import time
from pathlib import Path
import re

def extract_after(text, keyword):
    """
    提取指定关键字之后的文本。

    Args:
        text (str): 要搜索的文本。
        keyword (str): 关键字。

    Returns:
        str: 关键字之后的文本。如果未找到关键字，返回 None。
    """
    start_index = text.find(keyword)
    if start_index != -1:
        return text[start_index + len(keyword):].strip()
    return None

def extract_before(text, keyword):
    """
    提取指定关键字之前的文本。

    Args:
        text (str): 要搜索的文本。
        keyword (str): 关键字。

    Returns:
        str: 关键字之前的文本。如果未找到关键字，返回原始文本。
    """
    end_index = text.find(keyword)
    if end_index != -1:
        return text[:end_index].strip()
    return text

def extract_between(text, start_delimiter, end_delimiter):
    """
    提取两个指定分隔符之间的文本。

    Args:
        text (str): 要搜索的文本。
        start_delimiter (str): 起始分隔符。
        end_delimiter (str): 结束分隔符。

    Returns:
        str: 分隔符之间的文本。如果未找到分隔符，返回 None。
    """
    start_index = text.find(start_delimiter)
    if start_index == -1:
        return None
    subtext = text[start_index + len(start_delimiter):]
    end_index = subtext.find(end_delimiter)
    if end_index == -1:
        return None
    return subtext[:end_index].strip()

def process_txt_files(input_directory, output_directory, smiles_target_csv_path):
    """
    处理指定目录中的所有 .txt 文件，提取特征并生成特征矩阵。

    Args:
        input_directory (str): 包含 .txt 文件的目录。
        output_directory (str): 保存输出文件的目录。
        smiles_target_csv_path (str): 包含 SMILES 和 target 的 CSV 文件路径。

    Returns:
        None
    """
    # 设置版本和时间戳
    Version = '1.1'
    c_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    DIR = os.path.join(output_directory, f'Surfacia_{Version}_{c_time}')
    os.makedirs(DIR, exist_ok=True)

    # 获取当前目录中的所有 .txt 文件
    current_directory = input_directory
    file_list = [f for f in os.listdir(current_directory) if f.endswith('.txt')]

    print("Found text files:", file_list)

    # 初始化数据框
    df = pd.DataFrame()

    # 遍历每个 .txt 文件，提取特征
    for filename in file_list:
        data = {}
        with open(os.path.join(current_directory, filename), 'r') as file:
            lines_iter = iter(file.readlines())

        sample_name = filename[:-4]
        print('Processing sample:', sample_name)
        data['Sample Name'] = sample_name
        matrix = {}
        odi_values = []
        summary_count = 0  # 用于计数 Summary 标记出现的次数

        while True:
            try:
                line = next(lines_iter)
                # 基本信息
                if 'Atoms:' in line:
                    atom_num = int(extract_between(line, "Atoms: ", ","))
                    data['Atom Number'] = atom_num
                if 'Molecule weight:' in line:
                    weight = float(extract_between(line, "Molecule weight:", "Da"))
                    data['Molecule Weight'] = weight
                if 'Orbitals from 1 to' in line:
                    occupied_orbitals = float(extract_between(line, "Orbitals from 1 to", "are occupied"))
                    data['Occupied Orbitals'] = occupied_orbitals
                if 'Atom list:' in line:
                    atoms = []
                    xyz = []
                    for _ in range(atom_num):
                        line = next(lines_iter).strip()
                        if line:
                            atom_info = line.split()
                            atom = atom_info[0].split('(')[1].split(')')[0]
                            x, y, z = map(float, atom_info[-3:])
                            atoms.append(atom)
                            xyz.append([x, y, z])
                        else:
                            break
                # HOMO/LUMO 能级
                if 'is HOMO, energy:' in line:
                    homo_energy = extract_between(line, "is HOMO, energy:", "a.u.")
                    data['HOMO'] = float(homo_energy)
                if 'LUMO, energy:' in line:
                    lumo_energy = extract_between(line, "is LUMO, energy:", "a.u.")
                    data['LUMO'] = float(lumo_energy)
                if 'HOMO-LUMO gap:' in line:
                    gap_energy = extract_between(line, "gap:", "a.u.")
                    data['HOMO-LUMO Gap'] = float(gap_energy)
                # 分子形状
                if 'Farthest distance:' in line:
                    farthest_distance = float(extract_between(line, "):", "Angstrom"))
                    data['Farthest Distance'] = farthest_distance
                if 'Radius of the system: ' in line:
                    mol_radius = float(extract_between(line, ":", "Angstrom"))
                    data['Molecular Radius'] = mol_radius
                if 'Length of the three sides:' in line:
                    mol_size = list(map(float, extract_between(line, ":", "Angstrom").split()))
                    mol_size.sort()
                    data['Molecular Size Short'] = mol_size[0]
                    data['Molecular Size Medium'] = mol_size[1]
                    data['Molecular Size Long'] = mol_size[2]
                    data['Long/Sum Size Ratio'] = mol_size[2] / sum(mol_size)
                    data['Length/Diameter'] = mol_size[2] / (2 * mol_radius)
                if 'Molecular planarity parameter (MPP) is' in line:
                    mpp = float(extract_before(extract_after(line, "is"), "Angstrom"))
                    data['MPP'] = mpp
                if 'Span of deviation from plane' in line:
                    sdp = float(extract_before(extract_after(line, "is"), "Angstrom"))
                    data['SDP'] = sdp
                # 偶极矩
                if 'Magnitude of dipole moment:' in line:
                    dipole_moment = float(extract_between(line, 'Magnitude of dipole moment:', "a.u."))
                    data['Dipole Moment (a.u.)'] = dipole_moment
                if 'Magnitude: |Q_2|=' in line:
                    quadrupole_moment = float(extract_after(line, "Magnitude: |Q_2|="))
                    data['Quadrupole Moment'] = quadrupole_moment
                if 'Magnitude: |Q_3|=' in line:
                    octopole_moment = float(extract_after(line, "|Q_3|= "))
                    data['Octopole Moment'] = octopole_moment
                # ODI 指数
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
                # LEAE、ESP、ALIE
                try:
                    if '================= Summary of surface analysis =================' in line:
                        summary_count += 1
                        print('Summary of surface analysis', summary_count)
                        if summary_count == 1:
                            # LEAE 部分
                            while True:
                                line = next(lines_iter)
                                if 'Volume:' in line:
                                    data['Volume (Angstrom^3)'] = float(extract_between(line, "Bohr^3  (", 'Angstrom^3)'))
                                elif 'Estimated density according to mass and volume (M/V):' in line:
                                    data['Density (g/cm^3)'] = float(extract_between(line, "M/V):", 'g/cm^3'))
                                elif 'Minimal value:' in line:
                                    data['LEAE Minimal Value'] = float(extract_between(line, "Minimal value:", 'eV,   Maximal value:'))
                                    data['LEAE Maximal Value'] = float(extract_between(line, "eV,   Maximal value:", 'eV'))
                                if 'Note: Below minimal and maximal values are in eV' in line:
                                    next(lines_iter)
                                    matrix_data = []
                                    matrix_data0 = []
                                    for _ in range(atom_num):
                                        matrix_line = next(lines_iter).strip()
                                        if matrix_line:
                                            matrix_data.append(matrix_line)
                                        else:
                                            break
                                    next(lines_iter)
                                    line = next(lines_iter)
                                    if 'Note: Average and variance below are in eV and eV^2 respectively' in line:
                                        next(lines_iter)
                                    matrix_data2 = []
                                    for _ in range(atom_num):
                                        matrix_line = next(lines_iter).strip()
                                        if matrix_line:
                                            matrix_data2.append(matrix_line)
                                        else:
                                            break
                                    num_rows = min(len(matrix_data), len(matrix_data2))
                                    data['Surface eff atom num'] = num_rows
                                    for i in range(num_rows):
                                        matrix_data[i] += ' ' + ' '.join(matrix_data2[i].split()[-6:])
                                    matrix['Matrix Data'] = matrix_data
                                    break
                        elif summary_count == 2:
                            # ESP 部分
                            while True:
                                line = next(lines_iter)
                                if 'Minimal value:' in line:
                                    data['ESP Minimal Value'] = float(extract_between(line, "Minimal value:", 'kcal/mol   Maximal value:'))
                                    data['ESP Maximal Value'] = float(extract_between(line, "kcal/mol   Maximal value:", 'kcal/mol'))
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
                                    next(lines_iter)
                                    matrix_data3 = []
                                    for _ in range(atom_num):
                                        matrix_line = next(lines_iter).strip()
                                        if matrix_line:
                                            matrix_data3.append(matrix_line)
                                        else:
                                            break
                                    next(lines_iter)
                                    line = next(lines_iter)
                                    if 'Note: Average and variance below are in' in line:
                                        next(lines_iter)
                                    matrix_data4 = []
                                    for _ in range(atom_num):
                                        matrix_line = next(lines_iter).strip()
                                        if matrix_line:
                                            matrix_data4.append(matrix_line)
                                        else:
                                            break
                                    next(lines_iter)
                                    line = next(lines_iter)
                                    if 'Note: Internal charge separation' in line:
                                        next(lines_iter)
                                    matrix_data5 = []
                                    for _ in range(atom_num):
                                        matrix_line = next(lines_iter).strip()
                                        if matrix_line:
                                            matrix_data5.append(matrix_line)
                                        else:
                                            break
                                    for i in range(num_rows):
                                        matrix_data[i] += ' ' + ' '.join(matrix_data3[i].split()[-5:]) + ' ' + ' '.join(matrix_data4[i].split()[-6:]) + ' ' + ' '.join(matrix_data5[i].split()[-3:])
                                    matrix['Matrix Data'] = matrix_data
                                    break
                        elif summary_count == 3:
                            # ALIE 部分
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
                                    next(lines_iter)
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
                                    break
                except ValueError as e:
                    error_message = f"Error converting string to float in file {filename}, sample name {sample_name}: {e}\n"
                    print(error_message)
                    with open(c_time + 'error_log.txt', 'a') as log_file:
                        log_file.write(error_message)
                    continue
            except StopIteration:
                break

        temp_df = pd.DataFrame([data])
        df = pd.concat([df, temp_df], ignore_index=True)

        new_filename = 'AtomProp_' + sample_name + '.csv'  # 添加 '.csv' 扩展名

        max_index = max(int(row.split()[0]) for row in matrix_data)
        merged_data = [['NaN'] * (len(matrix_data[0].split()) + len(xyz[0])) for _ in range(max_index)]
        num_xyz_columns = len(xyz[0])
        num_matrix_columns = len(matrix_data[0].split())

        for row in matrix_data:
            parts = row.split()
            index = int(parts[0]) - 1  # 将 1-based 索引转换为 0-based 索引
            if index < len(xyz):
                merged_data[index] = xyz[index] + parts[:]  # 不包括 matrix_data 中的索引部分

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

        output_filename = Path(DIR, new_filename)
        with open(output_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(merged_data)
        print(f"Data successfully written to {output_filename}")

    RECORD_NAME1 = 'RawFull_' + str(df.shape[0]) + '_' + str(df.shape[1]) + '.csv'
    RECORD_NAME = Path(DIR, RECORD_NAME1)
    df.to_csv(RECORD_NAME, index=False)

    df_sorted = df.sort_values(by='Sample Name')
    df_cleaned = df_sorted.dropna(axis=1)
    RECORD_NAME_CLEAN = 'Full0_' + str(df_cleaned.shape[0]) + '_' + str(df_cleaned.shape[1]) + '.csv'
    df_cleaned.to_csv(Path(DIR, RECORD_NAME_CLEAN), index=False)

    # 初始化要提取特征的原子索引列表
    atomlist = [1]  # 列出你想要提取特征的原子索引
    featnums = [9, 11, 17, 20, 21, 22, 28, 32, 33]

    # 获取当前目录中的所有 AtomProp_*.csv 文件
    filelist = [
        f
        for f in os.listdir(DIR)
        if f.startswith("AtomProp_") and f.endswith(".csv")
    ]

    # 从第一个文件的标题行获取特征名称
    first_file = Path(DIR, filelist[0])
    titles = pd.read_csv(first_file, nrows=0).columns

    # 基于 atomlist 和 featnums 创建最终的标题
    final_titles = ['Filename']  # 以 Filename 作为第一列
    for atom_idx in atomlist:
        for featnum in featnums:
            final_titles.append(f'Atom{atom_idx}_{titles[featnum - 1]}')

    # 初始化一个数据框来存储特征数据
    t = np.zeros((len(filelist), len(final_titles) - 1))  # 调整大小以扣除 Filename 列

    # 处理文件列表中的每个文件
    for i, file in enumerate(filelist):
        # 读取文件中的数据
        df_atom = pd.read_csv(Path(DIR, file))

        # 初始化一个临时数组来保存当前文件的特征
        temp_features = np.zeros(len(final_titles) - 1)  # 不包括 Filename 列

        # 遍历 atomlist 和 featnums 的每个组合
        idx = 0
        for atom_idx in atomlist:
            for featnum in featnums:
                # 更新 temp_features 中的位置
                if atom_idx - 1 < len(df_atom) and featnum - 1 < len(df_atom.columns):
                    temp_features[idx] = df_atom.iloc[atom_idx - 1, featnum - 1]
                else:
                    temp_features[idx] = np.nan  # 如果数据不足，填充 NaN
                idx += 1

        # 将临时特征向量放入矩阵 t 的对应行
        t[i, :] = temp_features

    # 将矩阵 t 转换为 DataFrame 以便于保存为 CSV
    feature_df = pd.DataFrame(t, columns=final_titles[1:])  # 此处不包括 Filename 列
    feature_df.insert(0, 'Filename', filelist)  # 将文件名插入为第一列

    # 从文件名中提取数字并根据这些数字排序 DataFrame
    def extract_number(filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else 0

    feature_df['Sample Name'] = feature_df['Filename'].apply(extract_number)
    feature_df.sort_values('Sample Name', inplace=True)

    feature_df = feature_df[['Sample Name'] + [col for col in feature_df.columns if col != 'Sample Name']]
    feature_df.drop('Filename', axis=1, inplace=True)

    # 定义输出 CSV 文件名
    output_feature_matrix = Path(DIR, 'Atom_esp_feature_matrix.csv')

    # 将数据写入文件
    feature_df.to_csv(output_feature_matrix, index=False, float_format='%.6f')

    # 打印完成消息
    print(f'Data written to {output_feature_matrix}')

    # 读取之前保存的全局特征数据
    df1 = pd.read_csv(Path(DIR, RECORD_NAME_CLEAN))
    df2 = pd.read_csv(output_feature_matrix)

    # 合并数据
    merged_df = pd.merge(df1, df2, on='Sample Name')

    # 合并 SMILES 和 target 列并保存
    try:
        df_smiles_target = pd.read_csv(smiles_target_csv_path)
        df_smiles_target.columns = map(str.lower, df_smiles_target.columns)
        df_smiles_target = df_smiles_target[['smiles', 'target']]
    except ValueError:
        print("No 'smiles' or 'target' column found in the CSV.")
        exit()

    merged_df = pd.concat([merged_df, df_smiles_target], axis=1)

    RECORD_NAME_MERGED = 'Full_' + str(merged_df.shape[0]) + '_' + str(merged_df.shape[1]) + '.csv'
    merged_output_filename = Path(DIR, RECORD_NAME_MERGED)
    merged_df.to_csv(merged_output_filename, index=False, float_format='%.6f')

    print(f'Data written to {merged_output_filename}')
 
    print("Splitting the merged feature matrix...")   
    merged_df = pd.read_csv(merged_output_filename)
    S_N = merged_df.shape[0]
    F_N = merged_df.shape[1] - 3  # Assuming 'Sample Name', 'smiles', 'target' are three columns

    # Create a MachineLearning folder inside the main output directory
    ML_DIR = Path(DIR, "MachineLearning")
    ML_DIR.mkdir(exist_ok=True)

    # Save SMILES
    INPUT_SMILES = Path(ML_DIR, f'Smiles_{S_N}.csv')
    merged_df[['smiles']].to_csv(INPUT_SMILES, index=False, header=False)

    # Save labels (values)
    INPUT_Y = Path(ML_DIR, f'Values_True_{S_N}.csv')
    merged_df[['target']].to_csv(INPUT_Y, index=False, header=False)

    # Save feature matrix
    INPUT_X = Path(ML_DIR, f'Features_{S_N}_{F_N}.csv')
    merged_df.drop(['Sample Name', 'smiles', 'target'], axis=1).to_csv(INPUT_X, index=False, float_format='%.6f', header=False)

    # Save feature names with spaces replaced by underscores
    INPUT_TITLE = Path(ML_DIR, f'Title_{F_N}.csv')
    with open(INPUT_TITLE, 'w') as f:
        for col in merged_df.columns[1:-2]:
            # Replace spaces with underscores
            formatted_col = col.replace(' ', '_')
            formatted_col = formatted_col.replace('/', '_')
            f.write(formatted_col + '\n')

    print(f'Machine Learning data split and saved in {ML_DIR}:')
    print(f'  - SMILES: {INPUT_SMILES.name}')
    print(f'  - Values: {INPUT_Y.name}')
    print(f'  - Features: {INPUT_X.name}')
    print(f'  - Feature Titles: {INPUT_TITLE.name}')
    print("All processing completed.")
if __name__ == '__main__':
    input_directory = input("Enter the directory path containing .txt files: ")
    output_directory = input("Enter the output directory path: ")
    smiles_target_csv_path = input("Enter the path to the SMILES and target CSV file: ")

    process_txt_files(input_directory, output_directory, smiles_target_csv_path)