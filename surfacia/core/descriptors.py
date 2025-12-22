"""
分子描述符计算模块，包含形状描述符
"""
import numpy as np
import math

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