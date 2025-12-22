"""
2D分子绘图模块
"""
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

def draw_molecules_from_csv(csv_file, output_dir='molecule_images'):
    """
    从CSV文件读取SMILES并生成分子图片
    
    Args:
        csv_file: 包含SMILES列的CSV文件路径
        output_dir: 输出图片的目录
    """
    # 创建输出文件夹(如果不存在)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取CSV文件
    df = pd.read_csv(csv_file)

    if 'smiles' not in df.columns:
        print("错误：CSV文件中未找到'smiles'列")
        return

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

def draw_single_molecule(smiles, output_path=None, size=(800, 800)):
    """
    绘制单个分子的2D结构
    
    Args:
        smiles: 分子的SMILES字符串
        output_path: 输出文件路径，如果为None则不保存
        size: 图片尺寸
    
    Returns:
        PIL Image对象
    """
    try:
        # 创建分子对象
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"无法处理SMILES: {smiles}")
            return None
            
        # 生成2D构象
        AllChem.Compute2DCoords(mol)
        
        # 创建图像
        img = Draw.MolToImage(mol, size=size)
        
        # 保存图片（如果指定了路径）
        if output_path:
            img.save(output_path, dpi=(300, 300))
            print(f"分子图片已保存到: {output_path}")
        
        return img
        
    except Exception as e:
        print(f"处理SMILES时出错: {smiles}")
        print(f"错误信息: {str(e)}")
        return None

def batch_draw_molecules(smiles_list, output_dir='molecule_images', prefix='mol'):
    """
    批量绘制分子结构
    
    Args:
        smiles_list: SMILES字符串列表
        output_dir: 输出目录
        prefix: 文件名前缀
    """
    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 设置绘图参数
    Draw.DrawingOptions.bondLineWidth = 3.0
    Draw.DrawingOptions.atomLabelFontSize = 18
    Draw.DrawingOptions.dotsPerAngstrom = 300

    success_count = 0
    for idx, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"跳过无效SMILES: {smiles}")
                continue
                
            AllChem.Compute2DCoords(mol)
            img = Draw.MolToImage(mol, size=(800, 800))
            
            filename = f'{prefix}_{idx+1:04d}.png'
            filepath = os.path.join(output_dir, filename)
            img.save(filepath, dpi=(300, 300))
            
            success_count += 1
            
        except Exception as e:
            print(f"处理第{idx+1}个分子时出错: {smiles}")
            print(f"错误信息: {str(e)}")

    print(f"批量绘制完成！成功生成{success_count}个分子图片")