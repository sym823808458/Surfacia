"""
Surfacia - Surface Atomic Chemical Interaction Analyzer.
"""

from importlib import import_module

__version__ = "3.0.2"
__author__ = "YumingSu"


def _optional_attr(module_name: str, attr_name: str):
    try:
        module = import_module(module_name, package=__name__)
        return getattr(module, attr_name)
    except Exception:
        return None


smi2xyz_main = _optional_attr(".core.smi2xyz", "smi2xyz_main")
run_xtb_opt = _optional_attr(".core.xtb_opt", "run_xtb_opt")
xyz2gaussian_main = _optional_attr(".core.gaussian", "xyz2gaussian_main")
run_gaussian = _optional_attr(".core.gaussian", "run_gaussian")
run_multiwfn_on_fchk_files = _optional_attr(".core.multiwfn", "run_multiwfn_on_fchk_files")
process_txt_files = _optional_attr(".core.multiwfn", "process_txt_files")
run_atom_prop_extraction = _optional_attr(".features.atom_properties", "run_atom_prop_extraction")
ChemMLAnalyzer = _optional_attr(".ml", "ChemMLAnalyzer")
ChemMLWorkflow = _optional_attr(".ml.chem_ml_analyzer_v2", "ChemMLWorkflow")
workflow_main = _optional_attr(".core.workflow", "workflow_main")
InteractiveSHAPAnalyzer = _optional_attr(
    ".visualization.interactive_shap_viz",
    "InteractiveSHAPAnalyzer",
)
interactive_shap_viz_main = _optional_attr(
    ".visualization.interactive_shap_viz",
    "interactive_shap_viz_main",
)

__all__ = [
    "smi2xyz_main",
    "run_xtb_opt",
    "xyz2gaussian_main",
    "run_gaussian",
    "run_multiwfn_on_fchk_files",
    "process_txt_files",
    "run_atom_prop_extraction",
    "ChemMLAnalyzer",
    "ChemMLWorkflow",
    "InteractiveSHAPAnalyzer",
    "workflow_main",
    "interactive_shap_viz_main",
]
