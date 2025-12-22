"""
LOFFI算法实现 - 功能团识别
"""
from rdkit import Chem
import numpy as np

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