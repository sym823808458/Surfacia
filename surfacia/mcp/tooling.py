from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .utils import (
    ToolExecutionError,
    capture_python_output,
    command_available,
    compact_log,
    ensure_directory,
    find_latest_path,
    import_available,
    normalize_sample_names,
    result_payload,
    serialize_jsonable,
    working_directory,
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]


def _require_argument(arguments: dict[str, Any], key: str) -> Any:
    value = arguments.get(key)
    if value in (None, ""):
        raise ToolExecutionError(f"Missing required argument: {key}")
    return value


def _working_dir_from(arguments: dict[str, Any], *, default: str | None = None) -> Path:
    working_dir = arguments.get("working_dir") or default or "."
    return ensure_directory(working_dir)


def _preview_path_list(paths: list[Path], *, limit: int = 20) -> list[str]:
    preview = [str(path.resolve()) for path in paths[:limit]]
    if len(paths) > limit:
        preview.append(f"... ({len(paths) - limit} more)")
    return preview


def _latest_matrix_shape(csv_path: Path) -> tuple[int, int] | None:
    try:
        import pandas as pd

        dataframe = pd.read_csv(csv_path)
        return dataframe.shape
    except Exception:
        return None


def _call_in_directory(
    working_dir: Path,
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, str, str]:
    with working_directory(working_dir):
        return capture_python_output(func, *args, **kwargs)


def surfacia_check_environment(arguments: dict[str, Any]) -> dict[str, Any]:
    working_dir = _working_dir_from(arguments)
    python_modules = arguments.get(
        "python_modules",
        ["openbabel", "rdkit", "xgboost", "shap", "dash", "plotly"],
    )
    external_commands = arguments.get(
        "external_commands",
        ["xtb", "g16", "formchk", os.getenv("MULTIWFN_CMD", "Multiwfn_noGUI")],
    )

    module_status = {name: import_available(name) for name in python_modules}
    command_status = {name: command_available(name) for name in external_commands}
    ready = all(module_status.values()) and all(command_status.values())

    summary = (
        "Environment ready for the full Surfacia pipeline."
        if ready
        else "Environment check completed; some required dependencies are missing."
    )
    return result_payload(
        "surfacia_check_environment",
        summary,
        working_dir=str(working_dir),
        metrics={
            "python_modules": module_status,
            "external_commands": command_status,
            "ready": ready,
        },
    )


def surfacia_detect_workflow_state(arguments: dict[str, Any]) -> dict[str, Any]:
    working_dir = _working_dir_from(arguments)
    xyz_files = sorted(working_dir.glob("*.xyz"))
    com_files = sorted(working_dir.glob("*.com"))
    fchk_files = sorted(working_dir.glob("*.fchk"))
    empty_fchk = [path for path in fchk_files if path.stat().st_size == 0]
    complete_fchk = [path for path in fchk_files if path.stat().st_size > 0]
    missing_fchk = []
    for xyz_file in xyz_files:
        candidate = working_dir / f"{xyz_file.stem}.fchk"
        if not candidate.exists():
            missing_fchk.append(candidate)

    latest_surfacia_output = find_latest_path(working_dir, ["Surfacia_*"], recursive=False)
    latest_fulloption = find_latest_path(working_dir, ["FullOption*.csv"], recursive=True)
    latest_finalfull = find_latest_path(working_dir, ["FinalFull*.csv"], recursive=True)
    latest_training = find_latest_path(working_dir, ["Training_Set_Detailed*.csv"], recursive=True)
    latest_test = find_latest_path(working_dir, ["Test_Set_Detailed*.csv"], recursive=True)

    if latest_training:
        recommended_next_step = "launch_shap_visualizer"
    elif latest_finalfull:
        recommended_next_step = "run_ml_analysis"
    elif latest_fulloption:
        recommended_next_step = "extract_features"
    elif complete_fchk and not empty_fchk and not missing_fchk:
        recommended_next_step = "run_multiwfn_analysis"
    elif com_files:
        recommended_next_step = "run_gaussian_jobs"
    elif xyz_files:
        recommended_next_step = "generate_gaussian_inputs"
    else:
        recommended_next_step = "smi_to_xyz"

    summary = (
        f"Detected {len(xyz_files)} xyz, {len(com_files)} com, and {len(fchk_files)} fchk files. "
        f"Recommended next step: {recommended_next_step}."
    )
    return result_payload(
        "surfacia_detect_workflow_state",
        summary,
        working_dir=str(working_dir),
        metrics={
            "xyz_count": len(xyz_files),
            "com_count": len(com_files),
            "fchk_count": len(fchk_files),
            "empty_fchk_count": len(empty_fchk),
            "complete_fchk_count": len(complete_fchk),
            "missing_fchk_count": len(missing_fchk),
            "recommended_next_step": recommended_next_step,
        },
        artifacts={
            "latest_surfacia_output": str(latest_surfacia_output.resolve()) if latest_surfacia_output else None,
            "latest_fulloption": str(latest_fulloption.resolve()) if latest_fulloption else None,
            "latest_finalfull": str(latest_finalfull.resolve()) if latest_finalfull else None,
            "latest_training_csv": str(latest_training.resolve()) if latest_training else None,
            "latest_test_csv": str(latest_test.resolve()) if latest_test else None,
        },
    )


def surfacia_generate_benchmark_matrices(arguments: dict[str, Any]) -> dict[str, Any]:
    input_csv = Path(_require_argument(arguments, "input_csv")).expanduser().resolve()
    output_dir = arguments.get("output_dir")
    include_mordred = bool(arguments.get("include_mordred", False))

    from ..benchmark.descriptor_matrices import BenchmarkDescriptorMatrixGenerator

    generator = BenchmarkDescriptorMatrixGenerator(
        input_csv=str(input_csv),
        output_dir=output_dir,
        dataset_label=arguments.get("dataset_label"),
        ecfp_bits=int(arguments.get("ecfp_bits", 1024)),
        ecfp_radius=int(arguments.get("ecfp_radius", 2)),
    )
    result, stdout_text, stderr_text = capture_python_output(
        generator.save_all,
        include_mordred=include_mordred,
    )
    saved = {name: str(path.resolve()) for name, path in result.items()}
    summary = f"Generated {len(saved)} benchmark descriptor matrices from {input_csv.name}."
    return result_payload(
        "surfacia_generate_benchmark_matrices",
        summary,
        working_dir=str(input_csv.parent),
        artifacts={"saved_matrices": saved},
        logs={
            "stdout": compact_log(stdout_text),
            "stderr": compact_log(stderr_text),
        },
    )


def surfacia_smi_to_xyz(arguments: dict[str, Any]) -> dict[str, Any]:
    input_csv = Path(_require_argument(arguments, "input_csv")).expanduser().resolve()
    working_dir = _working_dir_from(arguments, default=str(input_csv.parent))
    check_extensions = arguments.get("check_extensions", [".fchk"])

    from ..core.smi2xyz import smi2xyz_main

    _, stdout_text, stderr_text = _call_in_directory(
        working_dir,
        smi2xyz_main,
        str(input_csv),
        check_extensions,
    )
    xyz_files = sorted(working_dir.glob("*.xyz"))
    summary = f"Generated or refreshed xyz structures for {len(xyz_files)} samples."
    return result_payload(
        "surfacia_smi_to_xyz",
        summary,
        working_dir=str(working_dir),
        artifacts={
            "xyz_files": _preview_path_list(xyz_files),
            "sample_mapping": str((working_dir / "sample_mapping.csv").resolve())
            if (working_dir / "sample_mapping.csv").exists()
            else None,
        },
        metrics={"xyz_count": len(xyz_files)},
        logs={
            "stdout": compact_log(stdout_text),
            "stderr": compact_log(stderr_text),
        },
    )


def surfacia_run_xtb_opt(arguments: dict[str, Any]) -> dict[str, Any]:
    working_dir = _working_dir_from(arguments)
    param_file = arguments.get("param_file")
    options_text = arguments.get("options_text")
    temp_param_path: Path | None = None

    from ..core.xtb_opt import run_xtb_opt

    try:
        if options_text:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".txt",
                delete=False,
                dir=working_dir,
                encoding="utf-8",
            ) as handle:
                handle.write(str(options_text).strip())
                temp_param_path = Path(handle.name)
            param_file = str(temp_param_path)

        _, stdout_text, stderr_text = _call_in_directory(
            working_dir,
            run_xtb_opt,
            param_file,
        )
    finally:
        if temp_param_path and temp_param_path.exists():
            temp_param_path.unlink()

    xyz_files = sorted(working_dir.glob("*.xyz"))
    out_logs = sorted(working_dir.glob("*.out"))
    summary = f"Completed XTB optimization pass in {working_dir}."
    return result_payload(
        "surfacia_run_xtb_opt",
        summary,
        working_dir=str(working_dir),
        artifacts={
            "optimized_xyz_files": _preview_path_list(xyz_files),
            "xtb_logs": _preview_path_list(out_logs),
        },
        metrics={"xyz_count": len(xyz_files), "log_count": len(out_logs)},
        logs={
            "stdout": compact_log(stdout_text),
            "stderr": compact_log(stderr_text),
        },
    )


def surfacia_generate_gaussian_inputs(arguments: dict[str, Any]) -> dict[str, Any]:
    working_dir = _working_dir_from(arguments)

    from ..core import gaussian
    from ..core.gaussian import xyz2gaussian_main

    gaussian.GAUSSIAN_KEYWORD_LINE = arguments.get(
        "keywords",
        "# PBE1PBE/6-311g* scrf(SMD,solvent=Water) em=GD3",
    )
    gaussian.DEFAULT_CHARGE = int(arguments.get("charge", 0))
    gaussian.DEFAULT_MULTIPLICITY = int(arguments.get("multiplicity", 1))
    gaussian.DEFAULT_NPROC = int(arguments.get("nproc", 32))
    gaussian.DEFAULT_MEMORY = str(arguments.get("memory", "30GB"))

    _, stdout_text, stderr_text = _call_in_directory(working_dir, xyz2gaussian_main)
    com_files = sorted(working_dir.glob("*.com"))
    summary = f"Generated {len(com_files)} Gaussian input files."
    return result_payload(
        "surfacia_generate_gaussian_inputs",
        summary,
        working_dir=str(working_dir),
        artifacts={"com_files": _preview_path_list(com_files)},
        metrics={"com_count": len(com_files)},
        logs={
            "stdout": compact_log(stdout_text),
            "stderr": compact_log(stderr_text),
        },
    )


def _execute_gaussian_job(com_file: Path, working_dir: Path) -> dict[str, Any]:
    if not command_available("g16"):
        raise ToolExecutionError("Command 'g16' was not found in PATH.")
    if not command_available("formchk"):
        raise ToolExecutionError("Command 'formchk' was not found in PATH.")

    g16_log = working_dir / f"{com_file.stem}.g16.stdout.log"
    formchk_log = working_dir / f"{com_file.stem}.formchk.log"
    chk_file = working_dir / f"{com_file.stem}.chk"
    fchk_file = working_dir / f"{com_file.stem}.fchk"

    with open(g16_log, "w", encoding="utf-8") as handle:
        g16_result = subprocess.run(
            ["g16", str(com_file.name)],
            cwd=str(working_dir),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    formchk_return_code = None
    if g16_result.returncode == 0 and chk_file.exists():
        with open(formchk_log, "w", encoding="utf-8") as handle:
            formchk_result = subprocess.run(
                ["formchk", str(chk_file.name)],
                cwd=str(working_dir),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        formchk_return_code = formchk_result.returncode

    return {
        "com_file": str(com_file.resolve()),
        "g16_return_code": g16_result.returncode,
        "formchk_return_code": formchk_return_code,
        "chk_exists": chk_file.exists(),
        "fchk_exists": fchk_file.exists(),
        "g16_log": str(g16_log.resolve()),
        "formchk_log": str(formchk_log.resolve()) if formchk_log.exists() else None,
    }


def surfacia_run_gaussian_jobs(arguments: dict[str, Any]) -> dict[str, Any]:
    working_dir = _working_dir_from(arguments)
    com_files = sorted(working_dir.glob(arguments.get("com_pattern", "*.com")))
    if not com_files:
        raise ToolExecutionError(f"No Gaussian input files found in {working_dir}.")

    job_results = [_execute_gaussian_job(path, working_dir) for path in com_files]
    success_count = sum(1 for item in job_results if item["g16_return_code"] == 0 and item["fchk_exists"])
    summary = f"Finished Gaussian execution for {len(job_results)} jobs; {success_count} produced fchk files."
    return result_payload(
        "surfacia_run_gaussian_jobs",
        summary,
        working_dir=str(working_dir),
        artifacts={"job_results": job_results},
        metrics={
            "job_count": len(job_results),
            "successful_fchk_jobs": success_count,
        },
    )


def surfacia_rerun_failed_gaussian_jobs(arguments: dict[str, Any]) -> dict[str, Any]:
    working_dir = _working_dir_from(arguments)
    failed_jobs: list[Path] = []
    removed_files: list[str] = []

    for fchk_file in sorted(working_dir.glob("*.fchk")):
        if fchk_file.stat().st_size == 0:
            com_file = working_dir / f"{fchk_file.stem}.com"
            if com_file.exists():
                failed_jobs.append(com_file)
                fchk_file.unlink()
                removed_files.append(str(fchk_file.resolve()))
                chk_file = working_dir / f"{fchk_file.stem}.chk"
                if chk_file.exists():
                    chk_file.unlink()
                    removed_files.append(str(chk_file.resolve()))

    for xyz_file in sorted(working_dir.glob("*.xyz")):
        com_file = working_dir / f"{xyz_file.stem}.com"
        fchk_file = working_dir / f"{xyz_file.stem}.fchk"
        if com_file.exists() and not fchk_file.exists() and com_file not in failed_jobs:
            failed_jobs.append(com_file)

    if not failed_jobs:
        return result_payload(
            "surfacia_rerun_failed_gaussian_jobs",
            "No failed Gaussian jobs were detected.",
            working_dir=str(working_dir),
            artifacts={"removed_files": removed_files, "job_results": []},
            metrics={"job_count": 0},
        )

    job_results = [_execute_gaussian_job(path, working_dir) for path in sorted(failed_jobs)]
    success_count = sum(1 for item in job_results if item["g16_return_code"] == 0 and item["fchk_exists"])
    summary = f"Reran {len(job_results)} Gaussian jobs; {success_count} regenerated fchk files."
    return result_payload(
        "surfacia_rerun_failed_gaussian_jobs",
        summary,
        working_dir=str(working_dir),
        artifacts={"removed_files": removed_files, "job_results": job_results},
        metrics={"job_count": len(job_results), "successful_fchk_jobs": success_count},
    )


def surfacia_run_multiwfn_analysis(arguments: dict[str, Any]) -> dict[str, Any]:
    input_dir = Path(_require_argument(arguments, "input_dir")).expanduser().resolve()
    output_dir = ensure_directory(arguments.get("output_dir", str(input_dir)))

    from ..core.multiwfn import process_txt_files, run_multiwfn_on_fchk_files

    processed_files, stdout_step1, stderr_step1 = capture_python_output(
        run_multiwfn_on_fchk_files,
        str(input_dir),
    )
    _, stdout_step2, stderr_step2 = capture_python_output(
        process_txt_files,
        str(input_dir),
        str(output_dir),
    )

    latest_fulloption = find_latest_path(output_dir, ["FullOption*.csv"], recursive=True)
    latest_rawfull = find_latest_path(output_dir, ["RawFull*.csv"], recursive=True)
    summary = f"Completed Multiwfn processing for {len(processed_files or [])} fchk files."
    return result_payload(
        "surfacia_run_multiwfn_analysis",
        summary,
        working_dir=str(output_dir),
        artifacts={
            "processed_fchk_files": serialize_jsonable(processed_files or []),
            "latest_fulloption": str(latest_fulloption.resolve()) if latest_fulloption else None,
            "latest_rawfull": str(latest_rawfull.resolve()) if latest_rawfull else None,
        },
        metrics={"processed_count": len(processed_files or [])},
        logs={
            "stdout": compact_log("\n".join([stdout_step1, stdout_step2])),
            "stderr": compact_log("\n".join([stderr_step1, stderr_step2])),
        },
    )


def surfacia_extract_features(arguments: dict[str, Any]) -> dict[str, Any]:
    input_csv = Path(_require_argument(arguments, "input_csv")).expanduser().resolve()
    mode = int(_require_argument(arguments, "mode"))
    if mode == 1 and not arguments.get("target_element"):
        raise ToolExecutionError("Mode 1 requires 'target_element'.")
    if mode == 2 and not arguments.get("xyz1_path"):
        raise ToolExecutionError("Mode 2 requires 'xyz1_path'.")

    from ..features.atom_properties import run_atom_prop_extraction

    _, stdout_text, stderr_text = capture_python_output(
        run_atom_prop_extraction,
        str(input_csv),
        mode=mode,
        target_element=arguments.get("target_element"),
        xyz1_path=arguments.get("xyz1_path"),
        threshold=float(arguments.get("threshold", 1.01)),
    )
    latest_finalfull = find_latest_path(input_csv.parent, ["FinalFull*.csv"], recursive=False)
    matrix_shape = _latest_matrix_shape(latest_finalfull) if latest_finalfull else None
    summary = (
        f"Completed Surfacia Mode {mode} feature extraction."
        if latest_finalfull
        else f"Feature extraction finished, but no FinalFull CSV was found in {input_csv.parent}."
    )
    return result_payload(
        "surfacia_extract_features",
        summary,
        working_dir=str(input_csv.parent),
        artifacts={
            "latest_finalfull": str(latest_finalfull.resolve()) if latest_finalfull else None,
        },
        metrics={
            "mode": mode,
            "rows": matrix_shape[0] if matrix_shape else None,
            "columns": matrix_shape[1] if matrix_shape else None,
        },
        logs={
            "stdout": compact_log(stdout_text),
            "stderr": compact_log(stderr_text),
        },
    )


def surfacia_run_ml_analysis(arguments: dict[str, Any]) -> dict[str, Any]:
    data_file = Path(_require_argument(arguments, "data_file")).expanduser().resolve()
    mode = arguments.get("mode", "workflow")
    output_dir = arguments.get("output_dir")
    test_sample_names = normalize_sample_names(arguments.get("test_sample_names"))
    features = arguments.get("features")
    if isinstance(features, str) and mode == "manual":
        features_value: Any = [item.strip() for item in features.split(",") if item.strip()]
    else:
        features_value = features

    from ..ml.chem_ml_analyzer_v2 import ChemMLWorkflow

    result, stdout_text, stderr_text = capture_python_output(
        ChemMLWorkflow.run_analysis,
        mode=mode,
        data_file=str(data_file),
        test_sample_names=test_sample_names,
        nan_handling=arguments.get("nan_handling", "drop_columns"),
        output_dir=output_dir,
        features=features_value,
        max_features=int(arguments.get("max_features", 5)),
        n_runs=int(arguments.get("n_runs", 3)),
        epoch=int(arguments.get("epoch", 64)),
        core_num=int(arguments.get("core_num", 32)),
        train_test_split=float(arguments.get("train_test_split", 0.85)),
        generate_fitting=bool(arguments.get("generate_fitting", True)),
    )

    analysis_root = Path(output_dir).expanduser().resolve() if output_dir else data_file.parent
    latest_analysis_dir = find_latest_path(analysis_root, ["*Analysis*"], recursive=False)
    summary = f"Completed {mode}-mode ML analysis for {data_file.name}."
    selected_features = None
    if mode == "workflow" and isinstance(result, dict):
        selected_features = result.get("final", {}).get("selected_features")
    elif isinstance(result, dict):
        selected_features = result.get("selected_features")

    return result_payload(
        "surfacia_run_ml_analysis",
        summary,
        working_dir=str(analysis_root),
        artifacts={
            "latest_analysis_dir": str(latest_analysis_dir.resolve()) if latest_analysis_dir else None,
            "selected_features": selected_features,
        },
        metrics=serialize_jsonable(result),
        logs={
            "stdout": compact_log(stdout_text),
            "stderr": compact_log(stderr_text),
        },
    )


def surfacia_launch_shap_visualizer(arguments: dict[str, Any]) -> dict[str, Any]:
    training_csv = Path(_require_argument(arguments, "training_csv")).expanduser().resolve()
    xyz_dir = Path(_require_argument(arguments, "xyz_dir")).expanduser().resolve()
    working_dir = _working_dir_from(arguments, default=str(training_csv.parent))
    log_dir = ensure_directory(working_dir / "surfacia_mcp_logs")
    stdout_log = log_dir / "shap_visualizer.stdout.log"
    stderr_log = log_dir / "shap_visualizer.stderr.log"

    command = [
        sys.executable,
        "-m",
        "surfacia.mcp.shap_launcher",
        "--training-csv",
        str(training_csv),
        "--xyz-dir",
        str(xyz_dir),
        "--port",
        str(int(arguments.get("port", 8052))),
        "--host",
        str(arguments.get("host", "127.0.0.1")),
    ]
    test_csv = arguments.get("test_csv")
    if test_csv:
        command.extend(["--test-csv", str(Path(test_csv).expanduser().resolve())])
    if arguments.get("skip_surface_gen", False):
        command.append("--skip-surface-gen")

    env = os.environ.copy()
    api_key = arguments.get("api_key")
    if api_key:
        env["SURFACIA_ZHIPUAI_API_KEY"] = str(api_key)

    creation_flags = 0
    if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

    with open(stdout_log, "w", encoding="utf-8") as stdout_handle, open(
        stderr_log, "w", encoding="utf-8"
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=str(working_dir),
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=env,
            creationflags=creation_flags,
        )

    url = f"http://{arguments.get('host', '127.0.0.1')}:{int(arguments.get('port', 8052))}"
    summary = f"Started Surfacia SHAP visualizer in the background at {url}."
    return result_payload(
        "surfacia_launch_shap_visualizer",
        summary,
        working_dir=str(working_dir),
        artifacts={
            "stdout_log": str(stdout_log.resolve()),
            "stderr_log": str(stderr_log.resolve()),
            "url": url,
        },
        metrics={"pid": process.pid},
        extra={"command": command},
    )


def surfacia_run_full_pipeline(arguments: dict[str, Any]) -> dict[str, Any]:
    input_csv = Path(_require_argument(arguments, "input_csv")).expanduser().resolve()
    working_dir = _working_dir_from(arguments, default=str(input_csv.parent))
    steps: list[dict[str, Any]] = []

    steps.append(
        surfacia_smi_to_xyz(
            {
                "input_csv": str(input_csv),
                "working_dir": str(working_dir),
                "check_extensions": arguments.get("check_extensions", [".fchk"]),
            }
        )
    )

    if not arguments.get("skip_xtb", False):
        steps.append(
            surfacia_run_xtb_opt(
                {
                    "working_dir": str(working_dir),
                    "param_file": arguments.get("xtb_param_file"),
                    "options_text": arguments.get("xtb_options_text"),
                }
            )
        )

    steps.append(
        surfacia_generate_gaussian_inputs(
            {
                "working_dir": str(working_dir),
                "keywords": arguments.get(
                    "keywords",
                    "# PBE1PBE/6-311g* scrf(SMD,solvent=Water) em=GD3",
                ),
                "charge": arguments.get("charge", 0),
                "multiplicity": arguments.get("multiplicity", 1),
                "nproc": arguments.get("nproc", 32),
                "memory": arguments.get("memory", "30GB"),
            }
        )
    )
    steps.append(surfacia_run_gaussian_jobs({"working_dir": str(working_dir)}))
    steps.append(
        surfacia_run_multiwfn_analysis(
            {
                "input_dir": str(working_dir),
                "output_dir": str(working_dir),
            }
        )
    )

    latest_fulloption = find_latest_path(working_dir, ["FullOption*.csv"], recursive=True)
    if latest_fulloption is None:
        raise ToolExecutionError("Full pipeline stopped because no FullOption CSV was generated.")

    steps.append(
        surfacia_extract_features(
            {
                "input_csv": str(latest_fulloption),
                "mode": int(arguments.get("extract_mode", 3)),
                "target_element": arguments.get("extract_element"),
                "xyz1_path": arguments.get("extract_xyz1"),
                "threshold": float(arguments.get("extract_threshold", 1.01)),
            }
        )
    )

    latest_finalfull = find_latest_path(working_dir, ["FinalFull*.csv"], recursive=True)
    if latest_finalfull is None:
        raise ToolExecutionError("Full pipeline stopped because no FinalFull CSV was generated.")

    steps.append(
        surfacia_run_ml_analysis(
            {
                "data_file": str(latest_finalfull),
                "mode": arguments.get("ml_mode", "workflow"),
                "test_sample_names": arguments.get("test_sample_names"),
                "output_dir": arguments.get("ml_output_dir"),
                "features": arguments.get("features"),
                "max_features": arguments.get("max_features", 5),
                "n_runs": arguments.get("n_runs", 3),
                "epoch": arguments.get("epoch", 64),
                "core_num": arguments.get("core_num", 32),
                "train_test_split": arguments.get("train_test_split", 0.85),
                "generate_fitting": arguments.get("generate_fitting", True),
                "nan_handling": arguments.get("nan_handling", "drop_columns"),
            }
        )
    )

    shap_result = None
    if arguments.get("launch_shap_visualizer", False):
        latest_training = find_latest_path(working_dir, ["Training_Set_Detailed*.csv"], recursive=True)
        latest_test = find_latest_path(working_dir, ["Test_Set_Detailed*.csv"], recursive=True)
        if latest_training:
            shap_arguments = {
                "training_csv": str(latest_training),
                "xyz_dir": str(working_dir),
                "working_dir": str(working_dir),
                "test_csv": str(latest_test) if latest_test else None,
                "port": arguments.get("port", 8052),
                "host": arguments.get("host", "127.0.0.1"),
                "skip_surface_gen": arguments.get("skip_surface_gen", False),
                "api_key": arguments.get("api_key"),
            }
            shap_result = surfacia_launch_shap_visualizer(shap_arguments)
            steps.append(shap_result)

    summary = f"Completed Surfacia full pipeline in {working_dir}."
    return result_payload(
        "surfacia_run_full_pipeline",
        summary,
        working_dir=str(working_dir),
        artifacts={
            "latest_fulloption": str(latest_fulloption.resolve()) if latest_fulloption else None,
            "latest_finalfull": str(latest_finalfull.resolve()) if latest_finalfull else None,
            "shap_visualizer": shap_result,
        },
        metrics={"step_count": len(steps)},
        extra={"steps": steps},
    )


TOOL_SPECS = [
    ToolSpec(
        name="surfacia_check_environment",
        description="Check whether Surfacia dependencies and external executables are available.",
        input_schema={
            "type": "object",
            "properties": {
                "working_dir": {"type": "string"},
                "python_modules": {"type": "array", "items": {"type": "string"}},
                "external_commands": {"type": "array", "items": {"type": "string"}},
            },
        },
        handler=surfacia_check_environment,
    ),
    ToolSpec(
        name="surfacia_detect_workflow_state",
        description="Inspect a directory and report the current Surfacia workflow state.",
        input_schema={
            "type": "object",
            "required": ["working_dir"],
            "properties": {
                "working_dir": {"type": "string"},
            },
        },
        handler=surfacia_detect_workflow_state,
    ),
    ToolSpec(
        name="surfacia_generate_benchmark_matrices",
        description="Generate RDKit/ECFP benchmark descriptor matrices from smiles and target.",
        input_schema={
            "type": "object",
            "required": ["input_csv"],
            "properties": {
                "input_csv": {"type": "string"},
                "output_dir": {"type": "string"},
                "dataset_label": {"type": "string"},
                "include_mordred": {"type": "boolean"},
                "ecfp_bits": {"type": "integer"},
                "ecfp_radius": {"type": "integer"},
            },
        },
        handler=surfacia_generate_benchmark_matrices,
    ),
    ToolSpec(
        name="surfacia_smi_to_xyz",
        description="Convert a smiles CSV into xyz structures and Surfacia sample mapping files.",
        input_schema={
            "type": "object",
            "required": ["input_csv"],
            "properties": {
                "input_csv": {"type": "string"},
                "working_dir": {"type": "string"},
                "check_extensions": {"type": "array", "items": {"type": "string"}},
            },
        },
        handler=surfacia_smi_to_xyz,
    ),
    ToolSpec(
        name="surfacia_run_xtb_opt",
        description="Run XTB geometry optimization on xyz files in a working directory.",
        input_schema={
            "type": "object",
            "required": ["working_dir"],
            "properties": {
                "working_dir": {"type": "string"},
                "param_file": {"type": "string"},
                "options_text": {"type": "string"},
            },
        },
        handler=surfacia_run_xtb_opt,
    ),
    ToolSpec(
        name="surfacia_generate_gaussian_inputs",
        description="Generate Gaussian .com input files from xyz structures.",
        input_schema={
            "type": "object",
            "required": ["working_dir"],
            "properties": {
                "working_dir": {"type": "string"},
                "keywords": {"type": "string"},
                "charge": {"type": "integer"},
                "multiplicity": {"type": "integer"},
                "nproc": {"type": "integer"},
                "memory": {"type": "string"},
            },
        },
        handler=surfacia_generate_gaussian_inputs,
    ),
    ToolSpec(
        name="surfacia_run_gaussian_jobs",
        description="Execute Gaussian and formchk safely for all .com files in a directory.",
        input_schema={
            "type": "object",
            "required": ["working_dir"],
            "properties": {
                "working_dir": {"type": "string"},
                "com_pattern": {"type": "string"},
            },
        },
        handler=surfacia_run_gaussian_jobs,
    ),
    ToolSpec(
        name="surfacia_rerun_failed_gaussian_jobs",
        description="Detect empty or missing fchk files and rerun the corresponding Gaussian jobs.",
        input_schema={
            "type": "object",
            "required": ["working_dir"],
            "properties": {
                "working_dir": {"type": "string"},
            },
        },
        handler=surfacia_rerun_failed_gaussian_jobs,
    ),
    ToolSpec(
        name="surfacia_run_multiwfn_analysis",
        description="Run Multiwfn analysis and generate RawFull/FullOption outputs.",
        input_schema={
            "type": "object",
            "required": ["input_dir"],
            "properties": {
                "input_dir": {"type": "string"},
                "output_dir": {"type": "string"},
            },
        },
        handler=surfacia_run_multiwfn_analysis,
    ),
    ToolSpec(
        name="surfacia_extract_features",
        description="Run Surfacia Mode 1/2/3 feature extraction on a FullOption CSV.",
        input_schema={
            "type": "object",
            "required": ["input_csv", "mode"],
            "properties": {
                "input_csv": {"type": "string"},
                "mode": {"type": "integer", "enum": [1, 2, 3]},
                "target_element": {"type": "string"},
                "xyz1_path": {"type": "string"},
                "threshold": {"type": "number"},
            },
        },
        handler=surfacia_extract_features,
    ),
    ToolSpec(
        name="surfacia_run_ml_analysis",
        description="Run Surfacia ML analysis in workflow or manual mode and return structured metrics.",
        input_schema={
            "type": "object",
            "required": ["data_file"],
            "properties": {
                "data_file": {"type": "string"},
                "mode": {"type": "string", "enum": ["workflow", "manual"]},
                "test_sample_names": {},
                "nan_handling": {"type": "string", "enum": ["drop_rows", "drop_columns"]},
                "output_dir": {"type": "string"},
                "features": {},
                "max_features": {"type": "integer"},
                "n_runs": {"type": "integer"},
                "epoch": {"type": "integer"},
                "core_num": {"type": "integer"},
                "train_test_split": {"type": "number"},
                "generate_fitting": {"type": "boolean"},
            },
        },
        handler=surfacia_run_ml_analysis,
    ),
    ToolSpec(
        name="surfacia_launch_shap_visualizer",
        description="Launch the Surfacia Dash SHAP visualizer as a detached subprocess.",
        input_schema={
            "type": "object",
            "required": ["training_csv", "xyz_dir"],
            "properties": {
                "training_csv": {"type": "string"},
                "xyz_dir": {"type": "string"},
                "test_csv": {"type": "string"},
                "working_dir": {"type": "string"},
                "port": {"type": "integer"},
                "host": {"type": "string"},
                "skip_surface_gen": {"type": "boolean"},
                "api_key": {"type": "string"},
            },
        },
        handler=surfacia_launch_shap_visualizer,
    ),
    ToolSpec(
        name="surfacia_run_full_pipeline",
        description="Run the full Surfacia pipeline from smiles CSV to ML outputs, with optional SHAP launch.",
        input_schema={
            "type": "object",
            "required": ["input_csv"],
            "properties": {
                "input_csv": {"type": "string"},
                "working_dir": {"type": "string"},
                "check_extensions": {"type": "array", "items": {"type": "string"}},
                "skip_xtb": {"type": "boolean"},
                "xtb_param_file": {"type": "string"},
                "xtb_options_text": {"type": "string"},
                "keywords": {"type": "string"},
                "charge": {"type": "integer"},
                "multiplicity": {"type": "integer"},
                "nproc": {"type": "integer"},
                "memory": {"type": "string"},
                "extract_mode": {"type": "integer", "enum": [1, 2, 3]},
                "extract_element": {"type": "string"},
                "extract_xyz1": {"type": "string"},
                "extract_threshold": {"type": "number"},
                "ml_mode": {"type": "string", "enum": ["workflow", "manual"]},
                "test_sample_names": {},
                "features": {},
                "max_features": {"type": "integer"},
                "n_runs": {"type": "integer"},
                "epoch": {"type": "integer"},
                "core_num": {"type": "integer"},
                "train_test_split": {"type": "number"},
                "generate_fitting": {"type": "boolean"},
                "nan_handling": {"type": "string", "enum": ["drop_rows", "drop_columns"]},
                "launch_shap_visualizer": {"type": "boolean"},
                "port": {"type": "integer"},
                "host": {"type": "string"},
                "skip_surface_gen": {"type": "boolean"},
                "api_key": {"type": "string"},
            },
        },
        handler=surfacia_run_full_pipeline,
    ),
]


def get_tool_specs() -> list[ToolSpec]:
    return TOOL_SPECS
