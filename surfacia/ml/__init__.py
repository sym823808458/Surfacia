"""
Machine learning modules for Surfacia
"""



# 导入新的分析器
from .chem_ml_analyzer_v2 import (
    BaseChemMLAnalyzer,
    ManualFeatureAnalyzer,
    WorkflowAnalyzer,
    ChemMLWorkflow
)

# 为了兼容性，将ChemMLWorkflow作为默认的ChemMLAnalyzer
ChemMLAnalyzer = ChemMLWorkflow

__all__ = [
    'ChemMLAnalyzer',
    'ChemMLAnalyzerLegacy',
    'BaseChemMLAnalyzer',
    'ManualFeatureAnalyzer', 
    'WorkflowAnalyzer',
    'ChemMLWorkflow'
]