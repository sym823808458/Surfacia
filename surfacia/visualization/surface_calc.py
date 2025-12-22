"""
3D表面计算模块
"""
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

def run_multiwfn_surface_calculations(input_path='.'):
    """
    对指定文件夹下的所有fchk文件分别进行LEAE、ESP和ALIE计算
    
    Args:
        input_path: 包含fchk文件的目录路径
    
    Returns:
        list: 处理成功的文件列表
    """
    original_dir = os.getcwd()
    os.chdir(input_path)
    fchk_files = sorted(glob.glob('*.fchk'))
    processed_files = []

    if not fchk_files:
        print("未找到任何.fchk文件")
        os.chdir(original_dir)
        return processed_files

    print(f"找到 {len(fchk_files)} 个.fchk文件，开始计算表面性质...")

    for fchk_file in fchk_files:
        sample_name = os.path.splitext(fchk_file)[0]  # e.g., '000001' if fchk_file is '000001.fchk'
        
        print(f"正在处理样本 {sample_name}...")
        
        # 步骤1: 计算LEAE
        leae_pdb = f"{sample_name}_LEAE.pdb"
        if not os.path.exists(leae_pdb):
            leae_content = create_leae_content(sample_name)
            print(f"  计算LEAE表面...")
            
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
                    stderr=subprocess.PIPE,
                    timeout=300  # 5分钟超时
                )
                
                # 检查并重命名生成的vtx.pdb
                if os.path.exists('vtx.pdb'):
                    os.rename('vtx.pdb', leae_pdb)
                    print(f"    ✓ LEAE计算完成: {leae_pdb}")
                    processed_files.append(leae_pdb)
                else:
                    print(f"    ✗ LEAE计算失败: vtx.pdb未生成")
                    if result.stderr:
                        print(f"    错误信息: {result.stderr}")
                
            except subprocess.TimeoutExpired:
                print(f"    ✗ LEAE计算超时")
            except Exception as e:
                print(f"    ✗ LEAE计算失败: {e}")
        else:
            print(f"  LEAE文件已存在: {leae_pdb}")
            processed_files.append(leae_pdb)

        # 步骤2: 计算ESP
        esp_pdb = f"{sample_name}_ESP.pdb"
        if not os.path.exists(esp_pdb):
            esp_content = create_esp_content()
            print(f"  计算ESP表面...")
            
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
                    stderr=subprocess.PIPE,
                    timeout=300  # 5分钟超时
                )
                
                # 检查并重命名生成的vtx.pdb
                if os.path.exists('vtx.pdb'):
                    os.rename('vtx.pdb', esp_pdb)
                    print(f"    ✓ ESP计算完成: {esp_pdb}")
                    processed_files.append(esp_pdb)
                else:
                    print(f"    ✗ ESP计算失败: vtx.pdb未生成")
                    if result.stderr:
                        print(f"    错误信息: {result.stderr}")
                
            except subprocess.TimeoutExpired:
                print(f"    ✗ ESP计算超时")
            except Exception as e:
                print(f"    ✗ ESP计算失败: {e}")
        else:
            print(f"  ESP文件已存在: {esp_pdb}")
            processed_files.append(esp_pdb)

        # 步骤3: 计算ALIE
        alie_pdb = f"{sample_name}_ALIE.pdb"
        if not os.path.exists(alie_pdb):
            alie_content = create_alie_content()
            print(f"  计算ALIE表面...")
            
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
                    stderr=subprocess.PIPE,
                    timeout=300  # 5分钟超时
                )
                
                # 检查并重命名生成的vtx.pdb
                if os.path.exists('vtx.pdb'):
                    os.rename('vtx.pdb', alie_pdb)
                    print(f"    ✓ ALIE计算完成: {alie_pdb}")
                    processed_files.append(alie_pdb)
                else:
                    print(f"    ✗ ALIE计算失败: vtx.pdb未生成")
                    if result.stderr:
                        print(f"    错误信息: {result.stderr}")
                
            except subprocess.TimeoutExpired:
                print(f"    ✗ ALIE计算超时")
            except Exception as e:
                print(f"    ✗ ALIE计算失败: {e}")
        else:
            print(f"  ALIE文件已存在: {alie_pdb}")
            processed_files.append(alie_pdb)

        print(f"样本 {sample_name} 处理完成")
        print("-" * 50)

    os.chdir(original_dir)
    
    print(f"\n表面计算完成！")
    print(f"共处理 {len(fchk_files)} 个样本")
    print(f"生成 {len(processed_files)} 个表面文件")
    
    return processed_files

def check_surface_files_completeness(input_path='.'):
    """
    检查表面文件的完整性
    
    Args:
        input_path: 要检查的目录路径
    
    Returns:
        dict: 包含完整性统计的字典
    """
    original_dir = os.getcwd()
    os.chdir(input_path)
    
    fchk_files = sorted(glob.glob('*.fchk'))
    surface_types = ['LEAE', 'ESP', 'ALIE']
    
    stats = {
        'total_samples': len(fchk_files),
        'complete_samples': 0,
        'incomplete_samples': [],
        'missing_files': []
    }
    
    print("检查表面文件完整性...")
    print("=" * 60)
    
    for fchk_file in fchk_files:
        sample_name = os.path.splitext(fchk_file)[0]
        missing_surfaces = []
        
        for surface_type in surface_types:
            surface_file = f"{sample_name}_{surface_type}.pdb"
            if not os.path.exists(surface_file):
                missing_surfaces.append(surface_type)
                stats['missing_files'].append(surface_file)
        
        if missing_surfaces:
            stats['incomplete_samples'].append({
                'sample': sample_name,
                'missing': missing_surfaces
            })
            print(f"✗ {sample_name}: 缺少 {', '.join(missing_surfaces)}")
        else:
            stats['complete_samples'] += 1
            print(f"✓ {sample_name}: 完整")
    
    os.chdir(original_dir)
    
    print("=" * 60)
    print(f"统计结果:")
    print(f"  总样本数: {stats['total_samples']}")
    print(f"  完整样本: {stats['complete_samples']}")
    print(f"  不完整样本: {len(stats['incomplete_samples'])}")
    print(f"  缺失文件数: {len(stats['missing_files'])}")
    
    return stats

def cleanup_temp_files(input_path='.'):
    """
    清理临时文件
    
    Args:
        input_path: 要清理的目录路径
    """
    original_dir = os.getcwd()
    os.chdir(input_path)
    
    temp_files = ['vtx.pdb', 'Multiwfn.log', 'fort.7']
    cleaned_count = 0
    
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                cleaned_count += 1
                print(f"已删除临时文件: {temp_file}")
            except Exception as e:
                print(f"删除临时文件失败 {temp_file}: {e}")
    
    os.chdir(original_dir)
    
    if cleaned_count > 0:
        print(f"清理完成，删除了 {cleaned_count} 个临时文件")
    else:
        print("未找到需要清理的临时文件")