#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Surfacia Workflow Manager
管理完整的 Surfacia 工作流程，确保各步骤之间的文件路径正确传递
"""

import os
import sys
import glob
import time
import shutil
from pathlib import Path
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SurfaciaWorkflow:
    """Surfacia 完整工作流程管理器"""
    
    def __init__(self, input_csv, working_dir=None):
        """
        初始化工作流程管理器
        
        Args:
            input_csv: 输入的 CSV 文件路径
            working_dir: 工作目录，默认为当前目录
        """
        self.input_csv = os.path.abspath(input_csv)
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.workflow_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建工作流程目录
        self.workflow_dir = self.working_dir / f"Surfacia_Workflow_{self.workflow_id}"
        self.workflow_dir.mkdir(exist_ok=True)
        
        # 各步骤的输出目录
        self.step_dirs = {}
        self.step_outputs = {}
        
        logger.info(f"Initialized Surfacia Workflow: {self.workflow_dir}")
        
    def run_full_workflow(self, **kwargs):
        """
        运行完整的工作流程
        
        Args:
            skip_xtb: 是否跳过 XTB 优化
            gaussian_keywords: Gaussian 关键词
            charge: 分子电荷
            multiplicity: 自旋多重度
            nproc: 处理器数量
            memory: 内存分配
            test_samples: 测试样本
            max_features: 最大特征数
            stepreg_runs: 逐步回归运行次数
            initial_features: 初始特征
            train_test_split: 训练测试分割比例
            shap_fit_threshold: SHAP 拟合阈值
            generate_fitting: 是否生成拟合图
            epoch: 训练轮数
            cores: CPU 核心数
        """
        try:
            # 切换到工作流程目录
            original_dir = os.getcwd()
            os.chdir(self.workflow_dir)
            
            # Step 1: SMILES to XYZ
            self._step1_smi2xyz()
            
            # Step 2: XTB optimization (optional)
            if not kwargs.get('skip_xtb', False):
                self._step2_xtb_opt()
            
            # Step 3: Generate Gaussian input
            self._step3_xyz2gaussian(**kwargs)
            
            # Step 4: Run Gaussian
            self._step4_run_gaussian()
            
            # Step 5: Multiwfn analysis
            self._step5_multiwfn_analysis()
            
            # Step 6: Extract features
            self._step6_extract_features()
            
            # Step 7: Machine learning analysis
            self._step7_ml_analysis(**kwargs)
            
            # Step 8: Interactive SHAP visualization
            self._step8_shap_visualization()
            
            logger.info("🎉 Complete workflow finished successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Workflow failed: {e}")
            return False
        finally:
            # 返回原始目录
            os.chdir(original_dir)
    
    def _step1_smi2xyz(self):
        """Step 1: SMILES to XYZ conversion"""
        logger.info("=== Step 1: Converting SMILES to XYZ ===")
        
        from ..core.smi2xyz import smi2xyz_main
        
        # 复制输入文件到工作目录
        input_copy = self.workflow_dir / "input.csv"
        shutil.copy2(self.input_csv, input_copy)
        
        # 运行转换
        smi2xyz_main(str(input_copy), ['.fchk'])
        
        # 记录输出
        self.step_outputs['step1'] = {
            'xyz_files': list(self.workflow_dir.glob("*.xyz")),
            'mapping_file': self.workflow_dir / "sample_mapping.csv"
        }
        
        logger.info(f"✅ Step 1 completed. Generated {len(self.step_outputs['step1']['xyz_files'])} XYZ files")
    
    def _step2_xtb_opt(self):
        """Step 2: XTB optimization"""
        logger.info("=== Step 2: XTB Optimization ===")
        
        from ..core.xtb_opt import run_xtb_opt
        
        run_xtb_opt()
        
        logger.info("✅ Step 2 completed. XTB optimization finished")
    
    def _step3_xyz2gaussian(self, **kwargs):
        """Step 3: Generate Gaussian input files"""
        logger.info("=== Step 3: Generating Gaussian Input ===")
        
        from ..core import gaussian
        from ..core.gaussian import xyz2gaussian_main
        
        # 设置 Gaussian 参数
        gaussian.GAUSSIAN_KEYWORD_LINE = kwargs.get('gaussian_keywords', 
                                                   "# PBE1PBE/6-311g* scrf(SMD,solvent=Water) em=GD3")
        gaussian.DEFAULT_CHARGE = kwargs.get('charge', 0)
        gaussian.DEFAULT_MULTIPLICITY = kwargs.get('multiplicity', 1)
        gaussian.DEFAULT_NPROC = kwargs.get('nproc', 32)
        gaussian.DEFAULT_MEMORY = kwargs.get('memory', "30GB")
        
        xyz2gaussian_main()
        
        # 记录输出
        self.step_outputs['step3'] = {
            'gjf_files': list(self.workflow_dir.glob("*.gjf"))
        }
        
        logger.info(f"✅ Step 3 completed. Generated {len(self.step_outputs['step3']['gjf_files'])} Gaussian input files")
    
    def _step4_run_gaussian(self):
        """Step 4: Run Gaussian calculations"""
        logger.info("=== Step 4: Running Gaussian ===")
        
        from ..core.gaussian import run_gaussian
        
        run_gaussian()
        
        # 记录输出
        self.step_outputs['step4'] = {
            'fchk_files': list(self.workflow_dir.glob("*.fchk")),
            'log_files': list(self.workflow_dir.glob("*.log"))
        }
        
        logger.info(f"✅ Step 4 completed. Generated {len(self.step_outputs['step4']['fchk_files'])} FCHK files")
    
    def _step5_multiwfn_analysis(self):
        """Step 5: Multiwfn analysis"""
        logger.info("=== Step 5: Multiwfn Analysis ===")
        
        from ..core.multiwfn import run_multiwfn_on_fchk_files, process_txt_files
        
        # 运行 Multiwfn 计算
        run_multiwfn_on_fchk_files(str(self.workflow_dir))
        
        # 处理输出文件
        process_txt_files(str(self.workflow_dir), str(self.workflow_dir))
        
        # 找到生成的 Surfacia 文件夹
        surfacia_folders = list(self.workflow_dir.glob("Surfacia_*_*"))
        if surfacia_folders:
            # 选择最新的文件夹
            latest_folder = max(surfacia_folders, key=os.path.getmtime)
            self.step_dirs['multiwfn_output'] = latest_folder
            
            # 记录输出
            self.step_outputs['step5'] = {
                'output_dir': latest_folder,
                'fulloption_files': list(latest_folder.glob("FullOption*.csv")),
                'rawfull_files': list(latest_folder.glob("RawFull*.csv"))
            }
            
            logger.info(f"✅ Step 5 completed. Output in: {latest_folder}")
        else:
            raise Exception("No Surfacia output folder found after Multiwfn analysis")
    
    def _step6_extract_features(self):
        """Step 6: Extract atomic features"""
        logger.info("=== Step 6: Extracting Features ===")
        
        from ..features.atom_properties import run_atom_prop_extraction
        
        # 找到 FullOption2 文件
        output_dir = self.step_outputs['step5']['output_dir']
        fulloption_files = list(output_dir.glob("FullOption*.csv"))
        
        if not fulloption_files:
            raise Exception("No FullOption CSV file found")
        
        # 使用第一个找到的文件
        fulloption_file = fulloption_files[0]
        
        # 运行特征提取（模式3：完整的LOFFI分析）
        run_atom_prop_extraction(str(fulloption_file), mode=3)
        
        # 记录输出
        finalfull_files = list(output_dir.glob("FinalFull*.csv"))
        self.step_outputs['step6'] = {
            'finalfull_files': finalfull_files
        }
        
        logger.info(f"✅ Step 6 completed. Generated {len(finalfull_files)} FinalFull files")
    
    def _step7_ml_analysis(self, **kwargs):
        """Step 7: Machine learning analysis"""
        logger.info("=== Step 7: ML Analysis ===")
        
        from ..ml.chem_ml_analyzer import ChemMLAnalyzer
        
        # 找到 FinalFull 文件
        finalfull_files = self.step_outputs['step6']['finalfull_files']
        if not finalfull_files:
            raise Exception("No FinalFull CSV file found")
        
        finalfull_file = finalfull_files[0]
        
        # 准备测试样本
        test_samples = []
        if kwargs.get('test_samples'):
            for s in kwargs['test_samples'].split(','):
                try:
                    test_samples.append(int(s.strip()))
                except ValueError:
                    test_samples.append(s.strip())
        
        # 准备初始特征
        initial_features = None
        if kwargs.get('initial_features'):
            initial_features = [f.strip() for f in kwargs['initial_features'].split(',')]
        
        # 创建分析器
        analyzer = ChemMLAnalyzer(
            data_file=str(finalfull_file),
            test_sample_names=test_samples,
            nan_handling='drop_columns'
        )
        
        # 运行分析
        results = analyzer.run_full_analysis(
            max_features=kwargs.get('max_features', 5),
            stepreg_runs=kwargs.get('stepreg_runs', 3),
            initial_features=initial_features,
            epoch=kwargs.get('epoch', 64),
            core_num=kwargs.get('cores', 32),
            train_test_split=kwargs.get('train_test_split', 0.85),
            shap_fit_threshold=kwargs.get('shap_fit_threshold', 0.3),
            generate_fitting=kwargs.get('generate_fitting', True)
        )
        
        # 找到生成的分析文件夹
        output_dir = self.step_outputs['step5']['output_dir']
        analysis_folders = list(output_dir.glob("*Analysis*"))
        if analysis_folders:
            latest_analysis = max(analysis_folders, key=os.path.getmtime)
            self.step_dirs['ml_analysis'] = latest_analysis
            
            # 记录输出
            training_files = list(latest_analysis.glob("**/Training_Set_Detailed*.csv"))
            test_files = list(latest_analysis.glob("**/Test_Set_Detailed*.csv"))
            
            self.step_outputs['step7'] = {
                'analysis_dir': latest_analysis,
                'training_files': training_files,
                'test_files': test_files,
                'results': results
            }
            
            logger.info(f"✅ Step 7 completed. Analysis in: {latest_analysis}")
        else:
            logger.warning("No ML analysis folder found, but analysis completed")
            self.step_outputs['step7'] = {'results': results}
    
    def _step8_shap_visualization(self):
        """Step 8: Interactive SHAP visualization"""
        logger.info("=== Step 8: Interactive SHAP Visualization ===")
        
        try:
            from ..visualization.interactive_shap_viz import interactive_shap_viz_main, MultiwfnCalculator
            
            # 找到训练和测试文件
            if 'training_files' in self.step_outputs['step7'] and self.step_outputs['step7']['training_files']:
                training_files = self.step_outputs['step7']['training_files']
                test_files = self.step_outputs['step7'].get('test_files', [])
                
                latest_training = max(training_files, key=os.path.getmtime)
                latest_test = max(test_files, key=os.path.getmtime) if test_files else None
                
                logger.info(f"Using training file: {latest_training}")
                if latest_test:
                    logger.info(f"Using test file: {latest_test}")
                
                # 生成表面文件
                logger.info("Generating surface PDB files...")
                MultiwfnCalculator.run_multiwfn_on_fchk_files(str(self.workflow_dir))
                
                # 启动可视化
                success = interactive_shap_viz_main(
                    csv_path=str(latest_training),
                    xyz_path=str(self.workflow_dir),
                    test_csv_path=str(latest_test) if latest_test else None,
                    api_key=None,  # 用户可以通过环境变量设置
                    skip_surface_gen=False,
                    port=8052,
                    host='0.0.0.0'
                )
                
                if success:
                    logger.info("✅ Step 8 completed. Interactive visualization launched")
                else:
                    logger.warning("⚠️ Step 8 completed with warnings")
            else:
                logger.warning("No training files found for visualization")
                
        except ImportError:
            logger.warning("Interactive SHAP visualization module not available")
    
    def get_workflow_summary(self):
        """获取工作流程摘要"""
        summary = {
            'workflow_id': self.workflow_id,
            'workflow_dir': str(self.workflow_dir),
            'input_csv': self.input_csv,
            'step_outputs': {}
        }
        
        for step, outputs in self.step_outputs.items():
            summary['step_outputs'][step] = {}
            for key, value in outputs.items():
                if isinstance(value, list):
                    summary['step_outputs'][step][key] = [str(v) for v in value]
                else:
                    summary['step_outputs'][step][key] = str(value)
        
        return summary

def workflow_main(input_csv, **kwargs):
    """
    工作流程主函数 - 符合 Surfacia CLI 架构
    
    Args:
        input_csv: 输入 CSV 文件路径
        **kwargs: 其他工作流程参数
    
    Returns:
        bool: 成功返回 True，失败返回 False
    """
    try:
        workflow = SurfaciaWorkflow(input_csv)
        success = workflow.run_full_workflow(**kwargs)
        
        if success:
            summary = workflow.get_workflow_summary()
            logger.info("📊 Workflow Summary:")
            logger.info(f"   Workflow ID: {summary['workflow_id']}")
            logger.info(f"   Output Directory: {summary['workflow_dir']}")
            
        return success
        
    except Exception as e:
        logger.error(f"❌ Workflow execution failed: {e}")
        return False