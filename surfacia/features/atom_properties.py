"""
原子性质提取模块 - 支持3种模式
"""
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
import logging

# 导入依赖的模块
from .loffi import apply_loffi_algorithm, LOFFI_CONTENT
from .fragment_match import (
    setup_logging, read_xyz, find_substructure,
    sort_substructure_atoms
)

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

def run_specific_atom_extraction(parent_dir, full_option_csv, target_element, final_output_csv):
    """
    模式1: 提取指定元素的13种原子性质（保持原有逻辑）
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
                print(f"找到 {len(matches)} 个匹配的官能团")
                
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
    
    print(f"特征提取统计:")
    print(f"  生成特征数: {total_features}")
    print(f"  成功匹配: {successful_matches}/{len(results)}")
    
    # 检查列名冲突并处理
    df_option_cols = set(df_option.columns)
    df_out_cols = set(df_out.columns) - {'Sample Name'}
    
    conflicting_cols = df_option_cols & df_out_cols
    if conflicting_cols:
        print(f"发现列名冲突: {conflicting_cols}")
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
    print(f"\n[Done] 写入合并的CSV{final_output_csv}")
    
    # 最终统计
    non_null_samples = df_merged[fragment_cols_final].dropna(how='all').shape[0]
    print(f"✓ {non_null_samples}/{len(df_merged)} 个样本有有效的官能团特征")

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

def run_atom_prop_extraction(original_csv_path, mode=3, target_element=None, 
                           xyz1_path=None, threshold=1.01):
    """
    三种情形的原子性质提取:
    mode=1: 只提取指定原子(如S)的性质 + 分子性质
    mode=2: 提取指定官能团(由xyz1_path定义)的性质 + 分子性质
    mode=3: 完全根据loffi代码处理(默认)
    
    Args:
        original_csv_path: 原始CSV文件路径
        mode: 处理模式 1, 2, 或 3
        target_element: 模式1中要提取的元素符号(如'S')
        xyz1_path: 模式2中官能团结构文件路径
        threshold: 模式2中结构匹配的阈值
    """
    parent_dir = os.path.dirname(original_csv_path)
    
    print(f"运行模式: {mode}")
    if mode == 1:
        print(f"目标元素: {target_element}")
    elif mode == 2:
        print(f"官能团结构文件: {xyz1_path}")
        print(f"匹配阈值: {threshold}")
    elif mode == 3:
        print("使用完整的loffi功能团分析")

    # 模式2: 运行官能团匹配
    if mode == 2:
        if xyz1_path is None:
            raise ValueError("模式2需要提供xyz1_path参数")
        
        # 设置日志
        log_file = os.path.join(parent_dir, 'substructure_matching.log')
        setup_logging(log_file)
        
        print("正在进行官能团结构匹配...")
        
        # 读取官能团结构
        if not os.path.exists(xyz1_path):
            raise FileNotFoundError(f"官能团结构文件不存在: {xyz1_path}")
        
        substructure_atoms, substructure_coords = read_xyz(xyz1_path)
        sorted_indices = sort_substructure_atoms(substructure_atoms, substructure_coords)
        
        # 查找所有xyz文件并进行匹配
        xyz_files = sorted(glob.glob(os.path.join(parent_dir, "*.xyz")))
        fragment_matches = {}
        
        for xyz_file in xyz_files:
            target_atoms, target_coords = read_xyz(xyz_file)
            matches = find_substructure(target_atoms, target_coords, substructure_atoms, substructure_coords, threshold=threshold)
            
            sample_name = os.path.splitext(os.path.basename(xyz_file))[0]
            if matches:
                # 收集所有匹配结果
                all_match_indices = []
                for match in matches:
                    sorted_match = [match[i] for i in sorted_indices]
                    all_match_indices.extend(sorted_match)
                fragment_matches[sample_name] = all_match_indices
                logging.info(f"Found {len(matches)} matches in {sample_name}: {all_match_indices}")
            else:
                logging.info(f"No match found in {sample_name}")

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
            run_functional_group_extraction(
                parent_dir=parent_dir,
                full_option_csv=os.path.basename(original_csv_path),
                fragment_matches=fragment_matches,
                substructure_atoms=substructure_atoms,
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