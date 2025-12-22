#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive SHAP Visualizer with 3D isosurface and AI assistant
Integrated module for Surfacia CLI
"""

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
import glob
import subprocess
import logging

# 尝试导入ZhipuAI客户端
try:
    # 尝试多种可能的导入方式
    try:
        from zhipuai import ZhipuAI as ZhipuAiClient
        ZHIPUAI_AVAILABLE = True
    except ImportError:
        try:
            from zai import ZhipuAiClient
            ZHIPUAI_AVAILABLE = True
        except ImportError:
            try:
                import zhipuai
                ZhipuAiClient = zhipuai.ZhipuAI
                ZHIPUAI_AVAILABLE = True
            except ImportError:
                raise ImportError("ZhipuAI not found")
except ImportError:
    ZHIPUAI_AVAILABLE = False
    # 只在调试模式下显示警告，避免每次都打印
    import os
    if os.getenv('SURFACIA_DEBUG', '').lower() in ('1', 'true', 'yes'):
        print("⚠️ ZhipuAI not installed. AI assistant features will be disabled.")
        print("   Install with: pip install zhipuai")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os
import subprocess
import logging

# 建议使用logging模块，比print更适合在类库中使用
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import os
import glob
import shutil
import logging
import subprocess
from typing import Optional


class MultiwfnPDBGenerator:
    """
    最终健壮版 PDB 生成器（命令行版）

    主要特性:
    - 使用 os.chdir 控制工作目录，保证 Multiwfn 在 fchk 所在目录执行
    - 生成骨架 PDB ({sample}_xyz.pdb) 时，不依赖 vtx.pdb，而是比较执行前后新增的 *.pdb
    - 等值面 (ESP/LEAE/ALIE) 走 vtx.pdb 流水线，生成后移动/改名到目标文件
    - 统一加上 -silent，降低交互/缓冲干扰
    - 使用绝对路径与 shutil.move，避免跨分区/相对路径问题
    """

    # 可通过环境变量覆盖 Multiwfn 可执行文件名
    MULTIWFN_CMD = os.environ.get("MULTIWFN_CMD", "Multiwfn_noGUI")

    logger = logging.getLogger("MultiwfnPDBGenerator")
    if not logger.handlers:
        # 如果外部未配置 logging，则给出一个简易的配置
        _handler = logging.StreamHandler()
        _formatter = logging.Formatter("[%(levelname)s] %(message)s")
        _handler.setFormatter(_formatter)
        logger.addHandler(_handler)
        logger.setLevel(logging.INFO)

    # ---------- 输入内容构造（菜单序列） ----------

    @staticmethod
    def create_leae_pdb_content(xyz_pdb_abspath: str) -> str:
        """
        生成 LEAE 计算的输入内容。
        注意：这里的 xyz_pdb_abspath 使用绝对路径，避免 chdir 差异导致找不到文件。
        典型菜单序列:
            12 -> 2 (LEAE) -> -4 (build on isosurface) -> 1 (isosurface type)
            -> 1 (rho isovalue) -> 0.01 -> 0 -> 5 (color by file) -> <xyz.pdb path>
            -> 6 (output points as PDB) -> -1 -> 0
        """
        return (
            "12\n"
            "2\n"
            "-4\n"
            "1\n"
            "1\n"
            "0.01\n"
            "0\n"
            "5\n"
            f"{xyz_pdb_abspath}\n"
            "6\n"
            "-1\n"
            "0\n"
        )

    @staticmethod
    def create_esp_pdb_content() -> str:
        """
        生成 ESP 计算的输入内容。
        典型菜单序列:
            12 -> 1 (ESP) -> 1 (isosurface) -> 0.01 -> 0 -> 6 (output PDB) -> -1 -> 0
        """
        return (
            "12\n"
            "1\n"
            "1\n"
            "0.01\n"
            "0\n"
            "6\n"
            "-1\n"
            "0\n"
        )

    @staticmethod
    def create_alie_pdb_content() -> str:
        """
        生成 ALIE 计算的输入内容。
        典型菜单序列:
            12 -> 2 (ALIE/related) -> 2 (ALIE) -> 1 (isosurface) -> 1 -> 0.01 -> 0 -> 6 -> -1 -> 0
        """
        return (
            "12\n"
            "2\n"
            "2\n"
            "1\n"
            "1\n"
            "0.01\n"
            "0\n"
            "6\n"
            "-1\n"
            "0\n"
        )

    # ---------- 核心 Multiwfn 运行封装 ----------

    @classmethod
    def _run_multiwfn_core(
        cls,
        fchk_filename: str,
        input_content: str,
        target_pdb_path: str,
        add_silent: bool = True,
    ) -> bool:
        """
        核心 Multiwfn 执行逻辑（适用于“等值面导出顶点为 vtx.pdb”的流程）
        - 在当前工作目录中运行
        - 自动清理旧 vtx.pdb 并在成功后移动到 target_pdb_path
        """
        command = [cls.MULTIWFN_CMD, fchk_filename]
        if add_silent:
            command.append("-silent")

        # 预清理 vtx.pdb
        if os.path.exists("vtx.pdb"):
            try:
                os.remove("vtx.pdb")
            except Exception as e:
                cls.logger.warning(f"Cannot remove old vtx.pdb: {e}")

        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = process.communicate(input=input_content)

            # 调试信息（截断输出，避免日志过大）
            if stdout:
                cls.logger.debug(f"[Multiwfn stdout]\n{stdout[:2000]}")
            if stderr:
                cls.logger.debug(f"[Multiwfn stderr]\n{stderr[:2000]}")

            # 成功/失败都先检查 vtx.pdb
            if os.path.exists("vtx.pdb"):
                # 跨分区安全移动
                try:
                    shutil.move("vtx.pdb", target_pdb_path)
                    if process.returncode != 0:
                        cls.logger.warning(
                            f"Multiwfn returncode={process.returncode}, "
                            f"but vtx.pdb was created. Assume success: {target_pdb_path}"
                        )
                    else:
                        cls.logger.info(f"Successfully created: {target_pdb_path}")
                    return True
                except Exception as e:
                    cls.logger.error(f"Failed to move vtx.pdb -> {target_pdb_path}: {e}")
                    return False

            # 未生成 vtx.pdb -> 失败
            cls.logger.error(
                f"Failed to create PDB via vtx.pdb for {fchk_filename}. Exit code={process.returncode}"
            )
            if stdout:
                cls.logger.error(f"Stdout(head): {stdout[:2000]}")
            if stderr:
                cls.logger.error(f"Stderr(head): {stderr[:2000]}")
            return False

        except FileNotFoundError:
            cls.logger.error(
                f"Command '{cls.MULTIWFN_CMD}' not found. Is Multiwfn installed and in PATH?"
            )
            return False
        except Exception as e:
            cls.logger.error(f"A Python exception occurred during subprocess execution: {e}")
            return False

    # ---------- 骨架 PDB 生成（不依赖 vtx.pdb） ----------

    @classmethod
    def _try_generate_xyz_by_sequence(
        cls,
        fchk_filename: str,
        menu_sequence: str,
    ) -> Optional[str]:
        """
        尝试用指定菜单序列生成骨架 PDB，返回新生成的 PDB 文件名（位于当前目录）；
        若未检测到新 PDB，返回 None。
        """
        command = [cls.MULTIWFN_CMD, fchk_filename, "-silent"]

        # 记录执行前的 PDB 列表
        before = set(glob.glob("*.pdb"))

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate(input=menu_sequence)

        if stdout:
            cls.logger.debug(f"[XYZ-PDB stdout]\n{stdout[:2000]}")
        if stderr:
            cls.logger.debug(f"[XYZ-PDB stderr]\n{stderr[:2000]}")

        after = set(glob.glob("*.pdb"))
        new_pdbs = list(after - before)
        if not new_pdbs:
            return None

        # 多个新文件，选最近修改的
        if len(new_pdbs) > 1:
            cls.logger.warning(
                f"Multiple new PDB files detected: {new_pdbs}. Picking the latest modified."
            )
            new_pdbs.sort(key=lambda p: os.path.getmtime(p), reverse=True)

        return new_pdbs[0]

    @classmethod
    def generate_xyz_pdb_file(cls, fchk_file: str, output_dir: str = ".") -> bool:
        """
        生成骨架 PDB（{sample}_xyz.pdb）。
        注意：此流程不会产生 vtx.pdb，不可用 _run_multiwfn_core 判据。
        我们通过执行前后比较目录中新出现的 *.pdb 来定位输出文件。
        """
        fchk_file_abs = os.path.abspath(fchk_file)
        fchk_dir = os.path.dirname(fchk_file_abs)
        fchk_filename = os.path.basename(fchk_file_abs)
        sample_name = os.path.splitext(fchk_filename)[0]
        final_xyz_pdb_path = os.path.abspath(
            os.path.join(output_dir, f"{sample_name}_xyz.pdb")
        )

        if os.path.exists(final_xyz_pdb_path):
            cls.logger.info(f"XYZ PDB already exists, skip: {final_xyz_pdb_path}")
            return True

        original_dir = os.getcwd()
        try:
            os.chdir(fchk_dir)
            cls.logger.info(f"Generating XYZ (skeleton) PDB for {fchk_filename}...")

            # 不同 Multiwfn 版本菜单略有差异，逐个尝试
            sequences = [
                # 常见：100 -> 2 (导出分子结构为 PDB)
                "100\n2\n1\n\n",
            ]

            generated_file = None
            for seq in sequences:
                generated_file = cls._try_generate_xyz_by_sequence(fchk_filename, seq)
                if generated_file:
                    break

            if not generated_file:
                cls.logger.error(
                    "No new PDB file detected after trying '100->2' sequences."
                )
                return False

            # 移动/改名到目标
            try:
                shutil.move(os.path.abspath(generated_file), final_xyz_pdb_path)
                cls.logger.info(f"Skeleton PDB created: {final_xyz_pdb_path}")
                return True
            except Exception as e:
                cls.logger.error(
                    f"Failed to move '{generated_file}' -> '{final_xyz_pdb_path}': {e}"
                )
                return False

        except FileNotFoundError:
            cls.logger.error(
                f"Command '{cls.MULTIWFN_CMD}' not found. Is Multiwfn installed and in PATH?"
            )
            return False
        except Exception as e:
            cls.logger.error(f"Error while generating skeleton PDB: {e}")
            return False
        finally:
            os.chdir(original_dir)

    # ---------- 表面 PDB 生成（ESP/LEAE/ALIE） ----------

    @classmethod
    def generate_surface_pdb_file(cls, fchk_file: str, surface_type: str, output_dir: str = ".") -> bool:
        """
        生成等值面 PDB（ESP/LEAE/ALIE），产出重命名为 {sample}_{surface}.pdb
        - ESP/ALIE：不依赖 xyz_pdb
        - LEAE：依赖先生成 {sample}_xyz.pdb，并将其绝对路径传入
        """
        fchk_file_abs = os.path.abspath(fchk_file)
        fchk_dir = os.path.dirname(fchk_file_abs)
        fchk_filename = os.path.basename(fchk_file_abs)
        sample_name = os.path.splitext(fchk_filename)[0]
        surface_type = surface_type.upper().strip()
        final_surface_pdb_path = os.path.abspath(
            os.path.join(output_dir, f"{sample_name}_{surface_type}.pdb")
        )

        if os.path.exists(final_surface_pdb_path):
            cls.logger.info(f"{surface_type} PDB already exists, skip: {final_surface_pdb_path}")
            return True

        # 若为 LEAE，先准备 xyz_pdb 的绝对路径
        xyz_pdb_abs = None
        if surface_type == "LEAE":
            if not cls.generate_xyz_pdb_file(fchk_file, output_dir):
                cls.logger.error(f"Prerequisite XYZ PDB generation failed for {fchk_filename}.")
                return False
            xyz_pdb_abs = os.path.abspath(os.path.join(output_dir, f"{sample_name}_xyz.pdb"))

        original_dir = os.getcwd()
        try:
            os.chdir(fchk_dir)
            cls.logger.info(f"Generating {surface_type} surface for {fchk_filename}...")

            if surface_type == "LEAE":
                content = cls.create_leae_pdb_content(xyz_pdb_abs)
            elif surface_type == "ESP":
                content = cls.create_esp_pdb_content()
            elif surface_type == "ALIE":
                content = cls.create_alie_pdb_content()
            else:
                cls.logger.error(f"Unsupported surface type: {surface_type}")
                return False

            return cls._run_multiwfn_core(
                fchk_filename=fchk_filename,
                input_content=content,
                target_pdb_path=final_surface_pdb_path,
                add_silent=True,
            )
        finally:
            os.chdir(original_dir)

    # ---------- 批处理：为一个目录下的所有 fchk 生成全部 PDB ----------

    @classmethod
    def generate_all_pdb_files(cls, input_path: str = ".") -> bool:
        """
        为指定文件夹下的所有 fchk 文件生成：
        - 骨架 {sample}_xyz.pdb
        - 表面 {sample}_ESP.pdb, {sample}_LEAE.pdb, {sample}_ALIE.pdb
        """
        original_dir = os.getcwd()
        try:
            os.chdir(input_path)
            fchk_files = sorted(glob.glob("*.fchk"))

            if not fchk_files:
                cls.logger.info("No fchk files found for PDB generation")
                return True

            cls.logger.info(f"Found {len(fchk_files)} fchk files for PDB generation")

            success_count = 0
            surface_types = ["ESP", "LEAE", "ALIE"]

            for fchk_file in fchk_files:
                sample_name = os.path.splitext(fchk_file)[0]
                cls.logger.info(f"Processing {sample_name}...")

                # 1) 生成骨架 xyz.pdb
                if cls.generate_xyz_pdb_file(fchk_file, "."):
                    # 2) 依次生成等值面
                    surface_success = 0
                    for s in surface_types:
                        if cls.generate_surface_pdb_file(fchk_file, s, "."):
                            surface_success += 1

                    if surface_success == len(surface_types):
                        success_count += 1
                        cls.logger.info(f"Successfully generated all PDB files for {sample_name}")
                    else:
                        cls.logger.warning(
                            f"Partially generated PDB files for {sample_name} "
                            f"({surface_success}/{len(surface_types)} surfaces)"
                        )
                else:
                    cls.logger.error(f"Failed to generate XYZ PDB for {sample_name}")

            cls.logger.info(
                f"PDB generation completed: {success_count}/{len(fchk_files)} samples successful"
            )
            return success_count > 0

        except Exception as e:
            cls.logger.error(f"Error in PDB generation batch process: {e}")
            return False
        finally:
            os.chdir(original_dir)

class InteractiveSHAPAnalyzer:
    def __init__(self, csv_file_path=None, xyz_folder_path=None, test_csv_path=None, api_key=None):
        """
        交互式SHAP分析器
        
        Parameters:
        csv_file_path: str, Training_Set_Detailed_xxx.csv文件的绝对路径
        xyz_folder_path: str, 包含xyz文件和等值面PDB文件的文件夹路径
        test_csv_path: str, Test_Set_Detailed_xxx.csv文件的绝对路径（可选）
        api_key: str, ZhipuAI API密钥（可选）
        """
        self.csv_file_path = csv_file_path
        self.xyz_folder_path = xyz_folder_path
        self.test_csv_path = test_csv_path
        self.data = None
        self.test_data = None
        self.feature_columns = []
        self.shap_columns = []
        self.app = None
        self.first_interaction = True
        
        # 添加对话历史记录
        self.conversation_history = []
        self.user_context = {}  # 存储用户研究背景信息
        
        # 支持的等值面类型
        self.surface_types = ['LEAE', 'ESP', 'ALIE']
        
        # 描述符背景知识
        self.descriptor_knowledge = """二、 SHAP图分析通用框架

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
第4层：分子定量表面分析 (Molecular Quantitative Surface Analysis, MQSA) - 分析核心
这是整个描述符体系中最具化学洞察力的部分。局域电子性质通过对电子密度衍生的量进行基于表面的分析，为分子反应性和相互作用位点提供了深刻的见解。这些描述符揭示了电子特性在分子表面的空间分布，从而能够预测反应位点和分子间相互作用模式。

4.1 理论基础：表面区域划分
所有这些性质的计算都基于将分子的范德华表面（通常由电子密度等值面定义，如0.001 a.u.用于ESP/ALIE，0.004 a.u.用于LEAE）分割为各原子所属的区域。表面点的归属基于距离加权算法：
$w_A = 1 - \frac{|r - r_A|}{R_A}$
其中，$r$ 代表一个表面点，$r_A$ 是原子A的坐标，$R_A$ 是原子A的半径。每个表面点都被分配给具有最高权重值 $w$ 的原子，这种方法自然地考虑了原子尺寸的差异。

4.2 核心表面性质描述符
平均局部离子化能 (Average Local Ionization Energy, ALIE)
定义:
$\bar{I}(r) = \frac{\sum_{i}{\rho_i(r)|\varepsilon_i|}}{\rho(r)}$
其中 $\rho_i(r)$ 和 $\varepsilon_i$ 分别代表第i个占据分子轨道的电子密度和轨道能量，求和遍历所有占据轨道。

化学意义: 衡量在空间点r处移走一个电子的平均难度。ALIE值越低，代表该处电子束缚越弱，能量越高，越容易失去，因此是亲核性强的位点，极易受到亲电试剂的攻击。

全局统计描述符:

ALIE Minimal Value (eV): 识别分子表面上最亲核的位点（最容易失去电子的区域）。
ALIE Maximal Value (eV): 定位最不亲核的位点。
ALIE Average Value (eV): 表征分子整体的电子给出能力。
ALIE Variance (eV²): 量化整个表面上离子化能的非均一性。
局部电子附着能 (Local Electron Attachment Energy, LEAE)
定义:
$E_{att}(r) = \frac{\sum_{i=LUMO}^{\varepsilon_i<0}{|\phi_i(r)|^2 \times \varepsilon_i}}{\rho(r)}$
其中 $\rho(r)$ 是总电子密度，$|\phi_i(r)|^2$ 是第i个未占分子轨道的概率密度，$\varepsilon_i$ 是其对应的轨道能量，求和遍历所有能量为负的未占轨道。

化学意义: 衡量在空间点r处增加一个电子的能量变化。LEAE值越低（越负），代表该处接受电子的能力越强，因此是亲电性强的位点，极易受到亲核试剂的攻击。

全局统计描述符:

LEAE Minimal Value (eV): 识别分子表面上最亲电的位点（最容易接受电子的区域）。
LEAE Maximal Value (eV): 定位最难接受电子的区域。
LEAE Average Value (eV): 提供了分子整体接受电子能力的综合度量。
LEAE Variance (eV²): 量化整个表面上电子亲和性（接受电子能力）的非均一性。
静电势 (Electrostatic Potential, ESP)
定义: 分子与一个单位正测试电荷之间的相互作用能。
$V(r) = \sum_{A}\frac{Z_A}{|r-R_A|} - \int\frac{\rho(r')}{|r-r'|}dr'$
其中 $Z_A$ 代表原子核电荷，$R_A$ 代表原子核坐标。

化学意义: 反映了分子的电荷分布。ESP为负值的区域（通常在富电子区域，如孤对电子、π体系）吸引正电荷，是亲核中心。ESP为正值的区域（通常在缺电子区域，如酸性氢、σ-hole）吸引负电荷，是亲电中心。

全局统计描述符:

ESP Minimal Value (kcal/mol): 识别最富电子（最亲核）的区域。
ESP Maximal Value (kcal/mol): 定位最缺电子（最亲电）的位点。
ESP Overall Average Value (kcal/mol): 表征分子总体的静电环境。
ESP Overall Variance ((kcal/mol)²), $\sigma^2_{tot}$: 量化静电势的非均一性。
Balance of Charges (ν) (无量纲): 电荷平衡度，$\nu = \frac{\sigma_+^2 \times \sigma_-^2}{(\sigma_{tot}^2)^2}$。
Product of $\sigma^2_{tot}$ and ν ((kcal/mol)²): 提供一个复合的电荷指标。
Internal Charge Separation (Π) (kcal/mol): 内部分电荷分离度，$\Pi = \frac{1}{t}\sum_{t}{|V(r_k) - \bar{V_s}|}$。
Molecular Polarity Index (MPI) (kcal/mol): 分子极性指数，代表平均绝对静电势， $MPI = \frac{1}{t}\sum_{t}{|V(r_k)|}$。
Polar Surface Area (Å²): |ESP| > 10 kcal/mol 的区域面积。
Polar Surface Area (%): 极性表面积占总表面积的百分比。
4.3 性质的互补性与综合分析
ESP主要描述长程的、经典的静电相互作用，它决定了反应物在接近过程中的初始取向。而ALIE/LEAE更多地描述短程的、与轨道相关的电子转移难易度，决定了成键反应发生的最终位点。一个完整的反应过程需要两者结合分析。例如，一个位点可能ALIE很低（化学上活泼），但如果其周围的ESP不是负的，亲电试剂可能在静电上就不会被优先吸引过来。

‼️ 重要指令：特征层级区分说明 ‼️

请注意： 不论后续分析选择哪个场景（层级5中的元素、官能团或LOFFI分析），都可能会遇到本章节（第4层）定义的特征，例如 ESP_Minimal_Value, ALIE_Minimal_Value 等。

当这些不带任何前缀（如 S_, Fun_, Atom_）的特征出现时，它们代表的是对【整个分子表面】进行的全局统计结果。 它们描述的是分子作为一个整体的最强反应位点或总体性质。

你必须将它们与“层级5：多尺度定量表面分析(MQSA)”中的特征严格区分开：

层级5的特征是“聚焦的”： 例如 S_ALIE_min 只关心所有硫原子上的ALIE最小值；Fun_ESP_delta 则是在官能团之间进行比较。
本层级4的特征是“全局的”： 例如 ALIE_Minimal_Value 是在整个分子所有原子（无论C, H, O, S...）的表面上寻找绝对的最小值。
在解释时，绝不能将 ALIE_Minimal_Value 错误地解读为某个特定元素的性质，除非通过可视化等手段确认该最小值恰好落在该元素上。必须强调其“分子全局”的视角。

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

示例场景（下面的例子只是助你理解，不代表用户的真实案例）: 我们正在建立一个预测激酶抑制剂活性的QSPR模型。经过特征筛选，XGBoost模型告诉我们，两个特征对预测IC50值（值越小活性越高）至关重要：S_ALIE_min 和 Fun_ESP_delta。

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
独立分析 ESP_Minimal_Value 的SHAP图

SHAP图观察: 散点图清晰地显示，随着 ESP_Minimal_Value 的值从高到低移动（横轴从右向左），其对应的SHAP值急剧下降（变为很大的负值）。这意味着，一个更低的 ESP_Minimal_Value 值对预测结果有强烈的负向贡献。由于我们的目标是IC50，负向贡献就意味着预测的IC50值更低，即分子活性更高。
化学解释:
回忆定义: ESP_Minimal_Value 来自于对整个分子表面的静电势（ESP）分析。它精准地量化了在分子表面上静电势最低（最负）的那个点的数值。这个点是分子最富电子、最能吸引正电荷（如金属阳离子、质子化基团）的区域，是分子长程静电相互作用的“热点”。
形成假说: 这个强烈的趋势表明，分子的高活性依赖于一个强大的静电吸引中心。一个极度为负的 ESP_Minimal_Value 是实现高效生物功能的关键，其作用可能包括：
长程引导与初始识别： 在分子进入复杂的蛋白口袋时，这个强负电势中心就像一个“静电信标”，能够被蛋白口袋中带正电或部分正电的区域（如赖氨酸、精氨酸残基或金属辅因子）在较远距离上识别，从而引导分子以正确的姿态进入结合位点。这是结合过程的第一步。
形成关键的锚定相互作用： 一旦分子正确定位，这个位点将与靶点形成一个或多个高强度的静电相互作用，如盐桥或强氢键。这种相互作用是分子与靶点高亲和力结合的“静电锚点”，将分子牢牢固定在活性位点。
与ALIE的协同/区别： ESP描述的是长程静电吸引力，而ALIE描述的是短程电子转移的难易度。一个理想的活性分子可能同时需要一个很低的 ESP_Minimal_Value 来“吸引”靶点，以及一个很低的 ALIE 值来在吸引后“反应”或形成稳定的轨道重叠。SHAP分析揭示 ESP_Minimal_Value 是关键，说明在此构效关系中，长程的静电“拉力”可能是决定活性的首要因素。
引导探索: “模型强烈暗示了一个关键的静电吸引中心。我们应该立即选取SHAP图中 ESP_Minimal_Value 最低、SHAP值最负的几个分子，进行三维可视化并渲染其ESP表面。我们的目标是确认这个负电势中心位于哪个官能团（例如，羧酸根、磷酸根、或某个富电子的杂环），并检查它在与靶蛋白对接时，是否正对着一个已知的带正电的相互作用伙伴。”
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
        self.client = None
        if ZHIPUAI_AVAILABLE and api_key:
            try:
                self.client = ZhipuAiClient(api_key=api_key)
                logger.info("✅ ZhipuAI API client initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize ZhipuAI API client: {e}")
        
        # 检查文件和运行计算
        if xyz_folder_path:
            self.check_and_generate_surface_files()
        
        if csv_file_path and os.path.exists(csv_file_path):
            self.load_data()
        
        # 加载测试集数据（如果提供）
        if test_csv_path and os.path.exists(test_csv_path):
            self.load_test_data()
    
    def check_and_generate_surface_files(self):
        """检查并生成表面文件"""
        if not self.xyz_folder_path:
            return
        
        try:
            import glob
            fchk_files = glob.glob(os.path.join(self.xyz_folder_path, "*.fchk"))
            if fchk_files:
                logger.info(f"Found {len(fchk_files)} fchk files, generating PDB files...")
                MultiwfnPDBGenerator.generate_all_pdb_files(self.xyz_folder_path)
            else:
                logger.info("No fchk files found, skipping surface generation")
        except Exception as e:
            logger.error(f"Surface file generation failed: {e}")
    
    def load_test_data(self):
        """加载测试集CSV数据"""
        try:
            self.test_data = pd.read_csv(self.test_csv_path)

            logger.info(f"Test data loaded successfully! Samples: {len(self.test_data)}")

            # 检查必要的列是否存在
            required_cols = ['Sample_Name', 'Target', 'Realtest_Pred']
            missing_cols = [col for col in required_cols if col not in self.test_data.columns]
            if missing_cols:
                logger.warning(f"⚠️ 测试集缺少必要列: {missing_cols}")

        except Exception as e:
            logger.error(f"Test data loading failed: {e}")
            self.test_data = None
    
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
            
        except Exception as e:
            print(f"数据加载失败: {e}")
    
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
- 提供构效关系的化学信息分析建议

请分享您的研究背景，我将为您提供专业的个性化建议！ 🚀"""

    def call_deepseek_llm(self, user_input, images=None):
        """调用ZhipuAI LLM，加入研究背景引导、描述符知识和对话记忆"""
        logger.info(f"🔧 [LLM] 开始处理用户输入: {user_input[:50]}...")
        if images:
            logger.info(f"🔧 [LLM] 包含 {len(images)} 张图片")

        if not self.client:
            error_msg = "❌ [LLM] API客户端未初始化"
            logger.error(error_msg)
            return error_msg

        if not user_input or not user_input.strip():
            if self.first_interaction:
                return self.get_initial_guidance()
            error_msg = "❌ [LLM] 用户输入为空"
            logger.error(error_msg)
            return "请输入您的问题"

        try:
            logger.info(f"🔧 [LLM] 准备发送到ZhipuAI API...")
            logger.info(f"🔧 [LLM] 输入长度: {len(user_input)}")

            # 构建系统提示，包含描述符知识、研究背景引导和用户上下文
            context_info = ""
            if self.user_context:
                context_info = f"\n\n用户研究背景信息：\n"
                for key, value in self.user_context.items():
                    context_info += f"- {key}: {value}\n"

            system_prompt = f"""你是一个专业的计算化学与数据科学交叉领域的AI助手。你的核心任务是帮助研究人员深入解读基于XGBoost模型生成的SHAP (SHapley Additive exPlanations)图。该模型使用了一套独特的、基于量子化学计算的多尺度分子描述符进行训练，这些描述符旨在从基础属性、电子结构、三维形状到局域反应性等多个层面量化分子的特征。你的目标是将SHAP图所揭示的数学规律，与描述符背后深刻的物理化学原理联系起来，提供具有化学洞察力的分析。

{context_info}

以下是详细的分子描述符背景知识，请结合这些知识为用户提供专业解释：

{self.descriptor_knowledge}

请基于用户的具体研究背景和上述知识库，提供针对性的专业建议和解释。如果用户提供图片，请仔细分析图片内容并结合描述符知识提供相关解释。

重要：请记住用户之前提到的研究背景信息，在后续对话中保持一致性。"""

            # 构建消息历史（包含最近的对话）
            messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
            
            # 添加对话历史（利用64K上下文窗口，保留更多历史）
            # 估算token使用：系统提示约8K，描述符知识约15K，留约40K给对话历史
            # 保守估计每轮对话约500token，可保留约80轮对话
            max_history_rounds = 80  # 最多保留40轮用户+40轮AI的对话
            recent_history = self.conversation_history[-max_history_rounds:] if len(self.conversation_history) > max_history_rounds else self.conversation_history
            for msg in recent_history:
                messages.append(msg)

            # 构建当前用户输入
            content = [{"type": "text", "text": user_input}]
            if images:
                for img_data in images:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": img_data}
                    })

            messages.append({"role": "user", "content": content})

            response = self.client.chat.completions.create(
                model="glm-4.5v",
                messages=messages,
                max_tokens=4000,  # 增加输出token数量
                temperature=0.3
            )

            result = response.choices[0].message.content
            
            # 保存对话历史
            self.conversation_history.append({"role": "user", "content": [{"type": "text", "text": user_input}]})
            self.conversation_history.append({"role": "assistant", "content": [{"type": "text", "text": result}]})
            
            self.first_interaction = False
            
            logger.info(f"✅ [LLM] API调用成功")
            logger.info(f"🔧 [LLM] 响应长度: {len(result)}")
            logger.info(f"🔧 [LLM] 响应预览: {result[:100]}...")

            return result

        except Exception as e:
            error_msg = f"❌ [LLM] API调用失败: {str(e)}"
            logger.error(f"🔧 [LLM] 详细错误: {error_msg}")
            return f"抱歉，AI服务暂时不可用。错误信息：{str(e)}"

    def save_chat_log(self, user_input, ai_response):
        """保存聊天记录到txt文件"""
        if not self.xyz_folder_path:
            return
        
        try:
            # 创建聊天记录文件路径
            chat_log_path = os.path.join(self.xyz_folder_path, "chat_log.txt")
            
            # 获取当前时间戳
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 准备要写入的内容
            log_entry = f"\n{'='*80}\n"
            log_entry += f"时间: {timestamp}\n"
            log_entry += f"{'='*80}\n"
            log_entry += f"用户: {user_input}\n"
            log_entry += f"{'-'*80}\n"
            log_entry += f"AI助手: {ai_response}\n"
            log_entry += f"{'='*80}\n"
            
            # 追加写入文件
            with open(chat_log_path, 'a', encoding='utf-8') as f:
                f.write(log_entry)
            
            logger.info(f"✅ 聊天记录已保存到: {chat_log_path}")
            
        except Exception as e:
            logger.error(f"❌ 保存聊天记录失败: {e}")

    def create_simple_llm_interface(self):
        """创建带图片粘贴功能和研究背景引导的LLM聊天界面"""
        return html.Div([
            html.H4("🤖 AI分析助手 (GLM-4.5v) - 分子描述符专家", 
                   style={
                       'textAlign': 'center',
                       'color': '#2c3e50',
                       'marginBottom': 20,
                       'fontFamily': 'Arial Black'
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

    def create_features_shap_table(self, row):
        """创建特征值和SHAP值的表格显示，特征值使用科学计数法时保持一致性"""

        # 提取所有特征列和对应的SHAP列
        table_data = []

        for i, feature_name in enumerate(self.feature_names):
            feature_col = f'Feature_{feature_name}'
            shap_col = f'SHAP_{feature_name}'

            if feature_col in row.index and shap_col in row.index:
                feature_value = row[feature_col]
                shap_value = row[shap_col]

                # 格式化数值显示 - 修改：当特征值使用科学计数法时，两者都使用科学计数法
                if isinstance(feature_value, (int, float)):
                    if abs(feature_value) >= 1000 or abs(feature_value) < 0.001:
                        feature_str = f"{feature_value:.2e}"
                        use_scientific = True
                    elif abs(feature_value) >= 1:
                        feature_str = f"{feature_value:.3f}"
                        use_scientific = False
                    else:
                        feature_str = f"{feature_value:.4f}"
                        use_scientific = False
                else:
                    feature_str = str(feature_value)
                    use_scientific = False

                if isinstance(shap_value, (int, float)):
                    # 如果特征值使用了科学计数法，SHAP值也使用科学计数法
                    if use_scientific:
                        shap_str = f"{shap_value:.2e}"
                    elif abs(shap_value) >= 1000:
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
        """根据样本名加载xyz文件内容（修改为加载xyz.pdb文件）"""
        if not self.xyz_folder_path:
            return None, None

        try:
            # 处理不同类型的样本名
            if isinstance(sample_name, (int, np.integer)):
                sample_number = str(sample_name)
            else:
                sample_number = ''.join(filter(str.isdigit, str(sample_name)))

            if not sample_number:
                logger.warning(f"无法从样本名中提取数字: {sample_name}")
                return None, None

            # 修改为加载xyz.pdb文件而非xyz文件
            xyz_pdb_filename = f"{sample_number.zfill(6)}_xyz.pdb"
            xyz_pdb_path = os.path.join(self.xyz_folder_path, xyz_pdb_filename)

            if not os.path.exists(xyz_pdb_path):
                logger.warning(f"XYZ PDB文件不存在: {xyz_pdb_path}")
                return None, None

            with open(xyz_pdb_path, 'r') as f:
                xyz_pdb_content = f.read()

            logger.info(f"成功加载XYZ PDB文件: {xyz_pdb_filename}")
            return xyz_pdb_content, xyz_pdb_filename

        except Exception as e:
            logger.error(f"加载XYZ PDB文件失败: {e}")
            return None, None
    
    def load_surface_pdb_file(self, sample_name, surface_type):
        """根据样本名和等值面类型加载对应的PDB文件，如果不存在则尝试生成"""
        if not self.xyz_folder_path:
            return None, None

        try:
            # 处理不同类型的样本名
            if isinstance(sample_name, (int, np.integer)):
                sample_number = str(sample_name)
            else:
                sample_number = ''.join(filter(str.isdigit, str(sample_name)))

            if not sample_number:
                logger.warning(f"无法从样本名中提取数字: {sample_name}")
                return None, None

            pdb_filename = f"{sample_number.zfill(6)}_{surface_type}.pdb"
            pdb_path = os.path.join(self.xyz_folder_path, pdb_filename)

            # 如果PDB文件不存在，尝试生成
            if not os.path.exists(pdb_path):
                logger.info(f"{surface_type} PDB文件不存在，尝试生成: {pdb_path}")
                
                # 查找对应的fchk文件
                fchk_filename = f"{sample_number.zfill(6)}.fchk"
                fchk_path = os.path.join(self.xyz_folder_path, fchk_filename)
                
                if os.path.exists(fchk_path):
                    logger.info(f"找到fchk文件，开始生成PDB: {fchk_path}")
                    success = MultiwfnPDBGenerator.generate_surface_pdb_file(
                        fchk_path, surface_type, self.xyz_folder_path
                    )
                    if not success:
                        logger.error(f"生成 {surface_type} PDB文件失败")
                        return None, None
                else:
                    logger.error(f"未找到对应的fchk文件: {fchk_path}")
                    return None, None

            # 再次检查文件是否存在
            if not os.path.exists(pdb_path):
                logger.error(f"{surface_type} PDB文件仍然不存在: {pdb_path}")
                return None, None

            with open(pdb_path, 'r') as f:
                pdb_content = f.read()

            if not pdb_content.strip():
                logger.error(f"{surface_type} PDB文件为空: {pdb_path}")
                return None, None

            # 解析单位并进行必要的转换
            pdb_content, unit = self.parse_pdb_unit_and_convert(pdb_content, surface_type)

            logger.info(f"成功加载 {surface_type} PDB文件: {pdb_filename} (单位: {unit})")
            return pdb_content, pdb_filename

        except Exception as e:
            logger.error(f"加载 {surface_type} PDB文件失败: {e}")
            return None, None

    def parse_pdb_unit_and_convert(self, pdb_content, surface_type):
        """解析PDB文件单位并进行必要的转换"""
        if not pdb_content:
            return pdb_content, "kcal/mol"
        
        lines = pdb_content.split('\n')
        unit = "kcal/mol"  # 默认单位
        conversion_factor = 1.0
        
        # 查找单位信息
        for line in lines:
            if line.startswith('REMARK') and 'Unit of B-factor field' in line:
                if 'eV' in line:
                    unit = "eV"
                    # eV to kcal/mol conversion factor: 1 eV = 23.06 kcal/mol
                    conversion_factor = 23.06
                    logger.info(f"检测到单位为 eV，将转换为 kcal/mol (转换因子: {conversion_factor})")
                    break
        
        # 如果是ESP模式且单位为eV，则转换B因子值
        if surface_type == 'ESP' and unit == 'eV':
            converted_lines = []
            for line in lines:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    try:
                        # 提取B因子值并转换
                        before_b = line[:60]
                        b_factor_str = line[60:66].strip()
                        after_b = line[66:]
                        
                        if b_factor_str:
                            b_factor = float(b_factor_str)
                            converted_b_factor = b_factor * conversion_factor
                            # 保持原格式
                            converted_line = before_b + f"{converted_b_factor:6.2f}" + after_b
                            converted_lines.append(converted_line)
                        else:
                            converted_lines.append(line)
                    except (ValueError, IndexError):
                        converted_lines.append(line)
                else:
                    # 更新REMARK行中的单位信息
                    if line.startswith('REMARK') and 'Unit of B-factor field' in line and 'eV' in line:
                        converted_lines.append(line.replace('eV', 'kcal/mol'))
                    else:
                        converted_lines.append(line)
            
            pdb_content = '\n'.join(converted_lines)
            unit = "kcal/mol"
            logger.info(f"✅ 成功将ESP数据从 eV 转换为 kcal/mol")
        
        return pdb_content, unit    
    def load_xyz_pdb_file(self, sample_name):
        """根据样本名加载xyz.pdb文件，如果不存在则尝试生成"""
        if not self.xyz_folder_path:
            return None, None

        try:
            # 处理不同类型的样本名
            if isinstance(sample_name, (int, np.integer)):
                sample_number = str(sample_name)
            else:
                sample_number = ''.join(filter(str.isdigit, str(sample_name)))

            if not sample_number:
                logger.warning(f"无法从样本名中提取数字: {sample_name}")
                return None, None

            xyz_pdb_filename = f"{sample_number.zfill(6)}_xyz.pdb"
            xyz_pdb_path = os.path.join(self.xyz_folder_path, xyz_pdb_filename)

            # 如果XYZ PDB文件不存在，尝试生成
            if not os.path.exists(xyz_pdb_path):
                logger.info(f"XYZ PDB文件不存在，尝试生成: {xyz_pdb_path}")
                
                # 查找对应的fchk文件
                fchk_filename = f"{sample_number.zfill(6)}.fchk"
                fchk_path = os.path.join(self.xyz_folder_path, fchk_filename)
                
                if os.path.exists(fchk_path):
                    logger.info(f"找到fchk文件，开始生成XYZ PDB: {fchk_path}")
                    success = MultiwfnPDBGenerator.generate_xyz_pdb_file(
                        fchk_path, self.xyz_folder_path
                    )
                    if not success:
                        logger.error(f"生成XYZ PDB文件失败")
                        return None, None
                else:
                    logger.error(f"未找到对应的fchk文件: {fchk_path}")
                    return None, None

            # 再次检查文件是否存在
            if not os.path.exists(xyz_pdb_path):
                logger.error(f"XYZ PDB文件仍然不存在: {xyz_pdb_path}")
                return None, None

            with open(xyz_pdb_path, 'r') as f:
                xyz_pdb_content = f.read()

            if not xyz_pdb_content.strip():
                logger.error(f"XYZ PDB文件为空: {xyz_pdb_path}")
                return None, None

            logger.info(f"成功加载XYZ PDB文件: {xyz_pdb_filename}")
            return xyz_pdb_content, xyz_pdb_filename

        except Exception as e:
            logger.error(f"加载XYZ PDB文件失败: {e}")
            return None, None

    def smiles_to_png_base64(self, smiles, width=250, height=200):
        """将SMILES转换为PNG图片的base64编码，加粗键和原子"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
            # 修复属性名称错误
            drawer.drawOptions().bondLineWidth = 4
            drawer.drawOptions().atomLabelFontSize = 20  # 这个属性在新版本RDKit中可能不存在
            drawer.drawOptions().multipleBondOffset = 0.2
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            svg_string = drawer.GetDrawingText()
            
            svg_b64 = base64.b64encode(svg_string.encode()).decode()
            return f"data:image/svg+xml;base64,{svg_b64}"
            
        except AttributeError as e:
            # 如果属性不存在，使用简化版本
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return None
                
                drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
                drawer.drawOptions().bondLineWidth = 4
                drawer.drawOptions().multipleBondOffset = 0.2
                drawer.DrawMolecule(mol)
                drawer.FinishDrawing()
                svg_string = drawer.GetDrawingText()
                
                svg_b64 = base64.b64encode(svg_string.encode()).decode()
                return f"data:image/svg+xml;base64,{svg_b64}"
            except Exception as e2:
                print(f"分子结构生成失败 {smiles}: {e2}")
                return None
        except Exception as e:
            print(f"分子结构生成失败 {smiles}: {e}")
            return None
    
    def parse_pdb_beta_values(self, pdb_content):
        """解析PDB文件中的B因子值，获取范围，并实现以0为中心的对称范围"""
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
            # 计算最大绝对值，实现以0为中心的对称范围
            max_abs = max(abs(min_val), abs(max_val))
            symmetric_min, symmetric_max = -max_abs, max_abs
            
            logger.info(f"PDB文件B因子实际范围: {min_val:.3f} 到 {max_val:.3f}")
            logger.info(f"使用对称范围（以0为中心）: {symmetric_min:.3f} 到 {symmetric_max:.3f}")
            return symmetric_min, symmetric_max
        else:
            logger.warning(f"未找到有效的B因子值，使用默认对称范围 -22 到 22")
            return -22, 22
    
    def create_3d_molecule_viewer(self, pdb_content, pdb_filename, viewer_type="pdb", height="200px", show_labels=False):
        """使用py3Dmol创建三维分子查看器的HTML（修改为使用PDB格式）"""
        if not pdb_content:
            return html.Div("PDB文件加载失败", 
                        style={
                            'color': '#e74c3c',
                            'fontSize': 18,
                            'textAlign': 'center',
                            'padding': 50
                        })
        
        viewer_id = f"mol_viewer_{viewer_type}_{hash(pdb_filename) % 10000}"
        
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
            var pdb_data = `{pdb_content}`;
            var viewer = $3Dmol.createViewer('{viewer_id}', {{
                defaultcolors: $3Dmol.elementColors.Jmol,
                backgroundColor: 'white'
            }});
            
            viewer.addModel(pdb_data, 'pdb');
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
        """创建结合等值面PDB和xyz PDB的三维查看器，修正颜色映射"""
        if not surface_pdb_content:
            return html.Div("等值面PDB文件未加载", 
                        style={
                            'color': '#e74c3c',
                            'fontSize': 18,
                            'textAlign': 'center',
                            'padding': 50
                        })
        
        # 解析实际的Beta值范围，并使用对称范围
        actual_min, actual_max = self.parse_pdb_beta_values(surface_pdb_content)
        
        # 使用对称范围（以0为中心）
        if use_auto_range:
            color_min, color_max = actual_min, actual_max
        else:
            # 这里可以设置为VMD控制面板的值，但现在先用实际范围
            color_min, color_max = actual_min, actual_max
        
        viewer_id = "combined_surface_xyz_viewer"
        
        # 准备xyz_pdb数据
        xyz_pdb_data_js = f"var xyz_pdb_data = `{xyz_pdb_content}`;" if xyz_pdb_content else "var xyz_pdb_data = null;"
        
        # 根据等值面类型设置颜色方案和标题 - 修正颜色映射，使用3Dmol.js支持的格式
        surface_info = {
            'ESP': {'title': 'Electrostatic Potential', 'unit': '(kcal/mol)', 'gradient': 'rwb'},  # 先改回rwb测试
            'LEAE': {'title': 'Local Electron Attachment Energy', 'unit': '(eV)', 'gradient': 'rwb'},
            'ALIE': {'title': 'Average Local Ionization Energy', 'unit': '(eV)', 'gradient': 'rwb'}
        }
        
        info = surface_info.get(surface_type, surface_info['ESP'])
        
        html_content = f"""
        <div style="position: relative; height: {height}; width: 100%; background: white; border: 2px solid #ddd; border-radius: 8px;">
            <div id="{viewer_id}" style="height: {height}; width: 100%; position: relative;"></div>
            
            <!-- 颜色条 - 修正为BWR（蓝-白-红）颜色方案 -->
            <div style="position: absolute; bottom: 30px; left: 30px; width: 350px; height: 40px; 
                        background: linear-gradient(to right, 
                        rgb(0,0,255) 0%, rgb(100,149,237) 25%, rgb(255,255,255) 50%, 
                        rgb(255,100,100) 75%, rgb(255,0,0) 100%);
                        border: 2px solid #333; border-radius: 5px; z-index: 100;">
                <div style="position: absolute; bottom: -25px; left: 0; font-size: 12px; color: black; font-weight: bold;">{color_min:.2f}</div>
                <div style="position: absolute; bottom: -25px; right: 0; font-size: 12px; color: black; font-weight: bold;">{color_max:.2f}</div>
                <div style="position: absolute; bottom: -25px; left: 50%; transform: translateX(-50%); 
                            font-size: 12px; color: black; font-weight: bold;">0.00</div>
                <div style="position: absolute; top: -25px; left: 50%; transform: translateX(-50%); 
                            font-size: 14px; color: black; font-weight: bold;">{info['title']} {info['unit']}</div>
            </div>
            
            <!-- 范围信息显示 -->
            <div style="position: absolute; bottom: 15px; right: 15px; background: rgba(255,255,255,0.9); 
                        padding: 8px; border-radius: 6px; border: 1px solid #ccc; z-index: 100; font-size: 11px;">
                <div style="color: #333; font-weight: bold;">对称范围: {color_min:.3f} ~ {color_max:.3f}</div>
                <div style="color: #666;">白色对应: 0.00 kcal/mol</div>
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
                                gradient: 'rwb',
                                min: colorMax,
                                max: colorMin
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
        """创建特征-SHAP图 (Feature vs SHAP)，可选显示测试集，x轴格式化改进"""
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

        # 确定是否使用科学计数法
        use_scientific = False
        if len(feature_values) > 0:
            sample_feature = feature_values[0]
            if isinstance(sample_feature, (int, float)):
                if abs(sample_feature) >= 1000 or abs(sample_feature) < 0.001:
                    use_scientific = True

        hover_text = []
        for idx, row in self.data.iterrows():
            # 根据科学计数法设置决定特征值的显示格式
            feature_val = row.get(feature_col, 0)
            if use_scientific and isinstance(feature_val, (int, float)):
                feature_display = f"{feature_val:.2e}"
            else:
                feature_display = str(feature_val)
                
            hover_info = (
                f"<b>训练集 - {row.get('Sample_Name', 'N/A')}</b><br>"
                f"Target: {row.get('Target', 0):.3f}<br>"
                f"{selected_feature}: {feature_display}<br>"
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
        test_feature_values = None
        test_shap_values = None
        
        if show_test_set and self.test_data is not None:
            if feature_col in self.test_data.columns and shap_col in self.test_data.columns:
                test_feature_values = self.test_data[feature_col].values
                test_shap_values = self.test_data[shap_col].values
                test_target_values = self.test_data['Target'].values
                test_pred_values = self.test_data.get('Realtest_Pred', test_target_values).values

                test_hover_text = []
                for idx, row in self.test_data.iterrows():
                    # 测试集也使用相同的格式化规则
                    feature_val = row.get(feature_col, 0)
                    if use_scientific and isinstance(feature_val, (int, float)):
                        feature_display = f"{feature_val:.2e}"
                    else:
                        feature_display = str(feature_val)
                        
                    hover_info = (
                        f"<b>测试集 - {row.get('Sample_Name', 'N/A')}</b><br>"
                        f"Target (真实): {row.get('Target', 0):.3f}<br>"
                        f"Predicted: {row.get('Realtest_Pred', 0):.3f}<br>"
                        f"{selected_feature}: {feature_display}<br>"
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

        # 只有在测试集数据存在时才合并数据
        if test_feature_values is not None and test_shap_values is not None:
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

        # x轴标题格式化
        x_axis_title = f'{selected_feature} (Feature Value)'
        if use_scientific:
            x_axis_title += ' [Scientific Notation]'

        fig.update_layout(
            title=dict(
                text=title_text,
                font=dict(size=36, family="Arial Black"),
                x=0.5
            ),
            xaxis=dict(
                title=dict(
                    text=x_axis_title,
                    font=dict(size=32, family="Arial Black")
                ),
                tickfont=dict(size=28, family="Arial Black"),
                linewidth=4,
                gridcolor='lightgray',
                gridwidth=2,
                showgrid=True,
                # 设置x轴刻度格式
                tickformat='.2e' if use_scientific else None
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
                    
                    # 测试集显示选项
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
                            'height': '1040px',  # 恢复原始高度
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
        
        # 改进的LLM聊天回调，支持图片、研究背景引导和处理状态显示
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
            logger.info(f"🔧 [LLM] GLM聊天回调触发 - n_clicks: {n_clicks}")
            logger.info(f"🔧 [LLM] 上传图片数量: {len(uploaded_images) if uploaded_images else 0}")
            logger.info(f"🔧 [LLM] 粘贴图片数量: {len(pasted_images) if pasted_images else 0}")
            
            if not n_clicks or not user_input:
                logger.info("🔧 [LLM] 无有效输入，返回默认状态")
                return (
                    [html.Div(self.get_initial_guidance(),
                             style={'color': '#333', 'padding': 20, 'lineHeight': '1.6', 'whiteSpace': 'pre-wrap'})],
                    "",
                    "✅ 描述符专家就绪 - 支持图片分析 & 研究背景引导"
                )
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            logger.info(f"🔧 [LLM] 开始处理用户请求 - 时间: {timestamp}")
            
            # 合并所有图片
            all_images = []
            if uploaded_images:
                all_images.extend(uploaded_images)
            if pasted_images:
                all_images.extend(pasted_images)
            
            # 构建用户输入显示
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
            
            # 显示处理状态
            processing_status = f"🔄 正在分析您的问题，请稍候... ({timestamp})" + (f" - 包含{len(all_images)}张图片" if all_images else "")
            
            # 调用GLM（支持图片和研究背景引导）
            try:
                ai_response = self.call_deepseek_llm(user_input, all_images)
                status = f"✅ 描述符专家完成 ({timestamp})" + (f" - 包含{len(all_images)}张图片" if all_images else "")
                logger.info(f"🔧 [LLM] 成功获得GLM响应")
                
                # 保存聊天记录到txt文件
                self.save_chat_log(user_input, ai_response)
                
            except Exception as e:
                ai_response = f"抱歉，处理过程中出现错误：{str(e)}"
                status = f"❌ GLM错误 ({timestamp})"
                logger.error(f"🔧 [LLM] GLM处理失败: {e}")
            
            # 构建最终响应显示
            response_content = []
            
            # 用户输入显示
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
                    html.Strong("🤖 描述符专家 (GLM-4.5v): ", style={'color': '#28a745'}),
                    html.Div(ai_response, style={
                        'color': '#333',
                        'marginTop': 5,
                        'whiteSpace': 'pre-wrap',
                        'lineHeight': '1.6'
                    })
                ], style={'padding': 10, 'backgroundColor': '#f1f8e9', 'borderRadius': 5})
            )
            
            logger.info(f"🔧 [LLM] GLM回调完成，返回响应")
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
                                html.P(f"样本: {sample_name} ({data_source})", 
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
            print(f"🤖 AI功能：GLM-4.5v描述符专家已集成")
            print(f"🖼️ 图片功能：支持Ctrl+V粘贴和拖拽上传图片")
            print(f"📚 知识库：内置分子描述符背景知识")
            print(f"🎓 研究引导：首次使用将引导了解研究背景")
            print(f"🔧 自动计算：自动检查并生成缺失的表面PDB文件")
            print(f"📈 测试集显示：支持在SHAP图中显示测试集数据（菱形标记）")
            print(f"📋 2D视图增强：详细的特征值与SHAP值表格显示")
            print(f"💾 聊天记录：自动保存所有对话到chat_log.txt文件")
            self.app.run(debug=debug, port=port, host=host)
        else:
            print("❌ 应用创建失败！")


# 标准的命令行入口函数
def interactive_shap_viz_main(csv_path, xyz_path, test_csv_path=None, api_key=None,
                             skip_surface_gen=False, port=8052, host='0.0.0.0'):
    """
    标准的命令行入口函数 - 符合 Surfacia CLI 架构
    
    Parameters:
    csv_path: Path to Training_Set_Detailed CSV file
    xyz_path: Path to folder containing xyz files
    test_csv_path: Path to Test_Set_Detailed CSV file (optional)
    api_key: ZhipuAI API key (optional)
    skip_surface_gen: Skip surface generation if True
    port: Port for web server
    host: Host for web server
    """
    logger.info("🚀 Starting Interactive SHAP Visualization...")
    
    # 验证输入文件
    if not os.path.exists(csv_path):
        logger.error(f"❌ CSV file not found: {csv_path}")
        return False
    
    if not os.path.exists(xyz_path):
        logger.error(f"❌ XYZ folder not found: {xyz_path}")
        return False
    
    try:
        # Step 1: Generate surface files if needed
        if not skip_surface_gen:
            logger.info("🔄 Checking and generating surface PDB files if needed...")
            fchk_files = glob.glob(os.path.join(xyz_path, "*.fchk"))
            if fchk_files:
                logger.info(f"Found {len(fchk_files)} fchk files")
                MultiwfnPDBGenerator.generate_all_pdb_files(xyz_path)
            else:
                logger.info("ℹ️ No fchk files found, skipping surface generation")
        else:
            logger.info("ℹ️ Skipping surface generation")
        
        # Step 2: Launch interactive analyzer
        logger.info("🚀 Launching interactive SHAP analyzer...")
        analyzer = InteractiveSHAPAnalyzer(
            csv_file_path=csv_path,
            xyz_folder_path=xyz_path,
            test_csv_path=test_csv_path,
            api_key=api_key
        )
        
        analyzer.run_app(debug=False, port=port, host=host)
        return True
        
    except Exception as e:
        logger.error(f"❌ Interactive SHAP visualization failed: {e}")
        return False

# 保持向后兼容性的别名
def run_interactive_shap_viz(csv_path, xyz_path, test_csv_path=None, api_key=None,
                skip_surface_gen=False, port=8052, host='0.0.0.0'):
    """
    向后兼容性函数 - 调用新的标准入口函数
    """
    return interactive_shap_viz_main(csv_path, xyz_path, test_csv_path, api_key,
                                   skip_surface_gen, port, host)

if __name__ == "__main__":
    # Example usage
    csv_file_path = "Training_Set_Detailed.csv"
    xyz_folder_path = "."
    test_csv_path = "Test_Set_Detailed.csv"

    analyzer = InteractiveSHAPAnalyzer(
        csv_file_path=csv_file_path,
        xyz_folder_path=xyz_folder_path,
        test_csv_path=test_csv_path,

    )
    analyzer.run_app(debug=True, port=8052)