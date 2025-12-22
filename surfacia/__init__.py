"""
Surfacia - Surface Atomic Chemical Interaction Analyzer
Version: 3.0.0
"""

__version__ = "3.0.1"
__author__ = "YumingSu"

from .core.smi2xyz import smi2xyz_main
from .core.xtb_opt import run_xtb_opt
from .core.gaussian import xyz2gaussian_main, run_gaussian
from .core.multiwfn import run_multiwfn_on_fchk_files, process_txt_files
from .features.atom_properties import run_atom_prop_extraction

# �����µ�MLģ��
from .ml.chem_ml_analyzer_v2 import ChemMLWorkflow
from .ml import ChemMLAnalyzer  # �����Ա���

from .core.workflow import workflow_main
from .visualization.interactive_shap_viz import InteractiveSHAPAnalyzer, interactive_shap_viz_main

__all__ = [
    'smi2xyz_main',
    'run_xtb_opt',
    'xyz2gaussian_main',
    'run_gaussian',
    'run_multiwfn_on_fchk_files',
    'process_txt_files',
    'run_atom_prop_extraction',
    'ChemMLAnalyzer',  # ������
    'ChemMLWorkflow',  # �µ�
    'InteractiveSHAPAnalyzer',
    'workflow_main',
    'interactive_shap_viz_main'
]