"""
Visualization modules for Surfacia
"""

from .mol_viewer import see_xyz_interactive
from .mol_drawer import draw_molecules_from_csv, draw_single_molecule, batch_draw_molecules
from .surface_calc import run_multiwfn_surface_calculations, check_surface_files_completeness
# from .shap_analyzer import InteractiveSHAPAnalyzer

# 导入新的交互式SHAP可视化模块
try:
    from .interactive_shap_viz import (
        InteractiveSHAPAnalyzer,
        interactive_shap_viz_main,
        run_interactive_shap_viz,
        MultiwfnPDBGenerator
    )
    INTERACTIVE_SHAP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Interactive SHAP visualization not available: {e}")
    INTERACTIVE_SHAP_AVAILABLE = False
    InteractiveSHAPAnalyzer = None
    interactive_shap_viz_main = None
    run_interactive_shap_viz = None
    MultiwfnPDBGenerator = None

__all__ = [
    'see_xyz_interactive',
    'draw_molecules_from_csv',
    'draw_single_molecule',
    'batch_draw_molecules',
    'run_multiwfn_surface_calculations',
    'check_surface_files_completeness',
    'InteractiveSHAPAnalyzer',
    'interactive_shap_viz_main',
    'run_interactive_shap_viz',
    'MultiwfnPDBGenerator',
    'INTERACTIVE_SHAP_AVAILABLE'
]