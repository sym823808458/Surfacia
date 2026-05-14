"""Core computation modules for Surfacia.

This module uses lazy attribute loading so optional dependencies are only
required when related functions are actually used.
"""

from importlib import import_module

_EXPORTS = {
    "smi2xyz_main": ("smi2xyz", "smi2xyz_main"),
    "run_xtb_opt": ("xtb_opt", "run_xtb_opt"),
    "xyz2gaussian_main": ("gaussian", "xyz2gaussian_main"),
    "run_gaussian": ("gaussian", "run_gaussian"),
    "rerun_failed_calculations": ("gaussian", "rerun_failed_calculations"),
    "run_multiwfn_on_fchk_files": ("multiwfn", "run_multiwfn_on_fchk_files"),
    "process_txt_files": ("multiwfn", "process_txt_files"),
    "SurfaciaWorkflow": ("workflow", "SurfaciaWorkflow"),
    "workflow_main": ("workflow", "workflow_main"),
    "get_atomic_mass": ("descriptors", "get_atomic_mass"),
    "calculate_principal_moments_of_inertia": (
        "descriptors",
        "calculate_principal_moments_of_inertia",
    ),
    "calculate_asphericity": ("descriptors", "calculate_asphericity"),
    "calculate_gyradius": ("descriptors", "calculate_gyradius"),
    "calculate_relative_gyradius": ("descriptors", "calculate_relative_gyradius"),
    "calculate_waist_variance": ("descriptors", "calculate_waist_variance"),
    "calculate_geometric_asphericity": (
        "descriptors",
        "calculate_geometric_asphericity",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
