#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate benchmark descriptor matrices from a CSV containing smiles and target.

This module is intentionally independent from the Surfacia quantum-chemistry
workflow so that conventional descriptor baselines can be created directly from
raw molecular structures.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Lipinski, Crippen

try:
    from mordred import Calculator, descriptors as mordred_descriptors
except ImportError:  # pragma: no cover - optional dependency
    Calculator = None
    mordred_descriptors = None


@dataclass
class MatrixSpec:
    name: str
    frame: pd.DataFrame


class BenchmarkDescriptorMatrixGenerator:
    """
    Build conventional descriptor matrices from a source CSV.

    Required input columns:
    - smiles
    - target

    Optional input column:
    - Sample Name
    """

    RDKIT_2D_FEATURES = [
        ("RDKit_MolWt", Descriptors.MolWt),
        ("RDKit_ExactMolWt", Descriptors.ExactMolWt),
        ("RDKit_HeavyAtomMolWt", Descriptors.HeavyAtomMolWt),
        ("RDKit_MolLogP", Crippen.MolLogP),
        ("RDKit_MolMR", Crippen.MolMR),
        ("RDKit_TPSA", Descriptors.TPSA),
        ("RDKit_FractionCSP3", Descriptors.FractionCSP3),
        ("RDKit_NumValenceElectrons", Descriptors.NumValenceElectrons),
        ("RDKit_NumRadicalElectrons", Descriptors.NumRadicalElectrons),
        ("RDKit_HeavyAtomCount", Lipinski.HeavyAtomCount),
        ("RDKit_NHOHCount", Lipinski.NHOHCount),
        ("RDKit_NOCount", Lipinski.NOCount),
        ("RDKit_NumHAcceptors", Lipinski.NumHAcceptors),
        ("RDKit_NumHDonors", Lipinski.NumHDonors),
        ("RDKit_NumHeteroatoms", Lipinski.NumHeteroatoms),
        ("RDKit_NumRotatableBonds", Lipinski.NumRotatableBonds),
        ("RDKit_NumAromaticRings", Lipinski.NumAromaticRings),
        ("RDKit_NumAliphaticRings", Lipinski.NumAliphaticRings),
        ("RDKit_NumSaturatedRings", Lipinski.NumSaturatedRings),
        ("RDKit_RingCount", Lipinski.RingCount),
        ("RDKit_BalabanJ", Descriptors.BalabanJ),
        ("RDKit_BertzCT", Descriptors.BertzCT),
        ("RDKit_Chi0", Descriptors.Chi0),
        ("RDKit_Chi0n", Descriptors.Chi0n),
        ("RDKit_Chi0v", Descriptors.Chi0v),
        ("RDKit_Chi1", Descriptors.Chi1),
        ("RDKit_Chi1n", Descriptors.Chi1n),
        ("RDKit_Chi1v", Descriptors.Chi1v),
        ("RDKit_Chi2n", Descriptors.Chi2n),
        ("RDKit_Chi2v", Descriptors.Chi2v),
        ("RDKit_Chi3n", Descriptors.Chi3n),
        ("RDKit_Chi3v", Descriptors.Chi3v),
        ("RDKit_Chi4n", Descriptors.Chi4n),
        ("RDKit_Chi4v", Descriptors.Chi4v),
        ("RDKit_HallKierAlpha", Descriptors.HallKierAlpha),
        ("RDKit_Kappa1", Descriptors.Kappa1),
        ("RDKit_Kappa2", Descriptors.Kappa2),
        ("RDKit_Kappa3", Descriptors.Kappa3),
    ]

    def __init__(
        self,
        input_csv: str,
        output_dir: Optional[str] = None,
        dataset_label: Optional[str] = None,
        ecfp_bits: int = 1024,
        ecfp_radius: int = 2,
        drop_constant_columns: bool = True,
        drop_all_nan_columns: bool = True,
    ) -> None:
        self.input_csv = Path(input_csv)
        self.output_dir = Path(output_dir) if output_dir else self.input_csv.parent / "benchmark_matrices"
        self.dataset_label = dataset_label or self.input_csv.stem
        self.ecfp_bits = ecfp_bits
        self.ecfp_radius = ecfp_radius
        self.drop_constant_columns = drop_constant_columns
        self.drop_all_nan_columns = drop_all_nan_columns
        self.df: Optional[pd.DataFrame] = None
        self.mols: Optional[List[Chem.Mol]] = None

    def load_input(self) -> None:
        self.df = pd.read_csv(self.input_csv)
        required = {"smiles", "target"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required column(s): {sorted(missing)}")

        if "Sample Name" not in self.df.columns:
            self.df.insert(0, "Sample Name", np.arange(1, len(self.df) + 1))

        self.df["smiles"] = self.df["smiles"].astype(str)
        self.mols = []
        invalid = []
        for idx, smiles in enumerate(self.df["smiles"], start=1):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid.append((idx, smiles))
            self.mols.append(mol)

        if invalid:
            preview = ", ".join([f"row {row}: {smi}" for row, smi in invalid[:5]])
            raise ValueError(f"Invalid SMILES detected ({len(invalid)} total). Examples: {preview}")

    def _ensure_loaded(self) -> None:
        if self.df is None or self.mols is None:
            self.load_input()

    @staticmethod
    def _safe_float(value) -> float:
        try:
            result = float(value)
            if np.isnan(result) or np.isinf(result):
                return np.nan
            return result
        except Exception:
            return np.nan

    def _cleanup_descriptor_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        cleaned = frame.copy()
        if self.drop_all_nan_columns:
            cleaned = cleaned.dropna(axis=1, how="all")
        if self.drop_constant_columns:
            nunique = cleaned.nunique(dropna=False)
            cleaned = cleaned.loc[:, nunique > 1]
        return cleaned

    def _compose_output_frame(self, descriptor_frame: pd.DataFrame) -> pd.DataFrame:
        self._ensure_loaded()
        assert self.df is not None
        meta = self.df[["Sample Name", "smiles", "target"]].copy()
        descriptor_frame = self._cleanup_descriptor_frame(descriptor_frame)
        return pd.concat(
            [meta[["Sample Name"]], descriptor_frame.reset_index(drop=True), meta[["smiles", "target"]]],
            axis=1,
        )

    def build_rdkit_2d(self) -> pd.DataFrame:
        self._ensure_loaded()
        rows = []
        assert self.mols is not None
        for mol in self.mols:
            row = {name: self._safe_float(func(mol)) for name, func in self.RDKIT_2D_FEATURES}
            rows.append(row)
        return pd.DataFrame(rows)

    def build_mordred(self) -> pd.DataFrame:
        self._ensure_loaded()
        if Calculator is None or mordred_descriptors is None:
            raise ImportError(
                "Mordred is not installed. Install it in your environment before generating this matrix."
            )
        assert self.mols is not None
        calc = Calculator(mordred_descriptors, ignore_3D=True)
        mordred_df = calc.pandas(self.mols)
        mordred_df = mordred_df.apply(pd.to_numeric, errors="coerce")
        mordred_df.columns = [f"Mordred_{col}" for col in mordred_df.columns]
        return mordred_df

    def build_ecfp4(self) -> pd.DataFrame:
        self._ensure_loaded()
        assert self.mols is not None
        rows = []
        names = [f"ECFP4_{i:04d}" for i in range(self.ecfp_bits)]
        for mol in self.mols:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.ecfp_radius, nBits=self.ecfp_bits)
            arr = np.zeros((self.ecfp_bits,), dtype=int)
            DataStructs.ConvertToNumpyArray(fp, arr)
            rows.append(arr)
        return pd.DataFrame(rows, columns=names)

    def build_rdkit_plus_ecfp4(self) -> pd.DataFrame:
        rdkit_df = self.build_rdkit_2d()
        ecfp_df = self.build_ecfp4()
        return pd.concat([rdkit_df, ecfp_df], axis=1)

    def generate_all(self, include_mordred: bool = True) -> Dict[str, pd.DataFrame]:
        matrices = {
            "RDKit2D": self._compose_output_frame(self.build_rdkit_2d()),
            "ECFP4": self._compose_output_frame(self.build_ecfp4()),
            "RDKit2D_ECFP4": self._compose_output_frame(self.build_rdkit_plus_ecfp4()),
        }
        if include_mordred:
            matrices["Mordred"] = self._compose_output_frame(self.build_mordred())
        return matrices

    def save_all(self, include_mordred: bool = True) -> Dict[str, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        matrices = self.generate_all(include_mordred=include_mordred)
        saved_paths: Dict[str, Path] = {}
        for method_name, df in matrices.items():
            n_samples = len(df)
            n_features = df.shape[1] - 3
            filename = f"FinalFull_{self.dataset_label}_{method_name}_{n_samples}_{n_features}.csv"
            save_path = self.output_dir / filename
            df.to_csv(save_path, index=False)
            saved_paths[method_name] = save_path
        return saved_paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate conventional benchmark descriptor matrices from smiles and target."
    )
    parser.add_argument("-i", "--input", required=True, help="Input CSV containing smiles and target")
    parser.add_argument("-o", "--output-dir", help="Directory for generated matrices")
    parser.add_argument("--dataset-label", help="Label inserted into output filenames")
    parser.add_argument("--ecfp-bits", type=int, default=1024, help="Number of ECFP bits (default: 1024)")
    parser.add_argument("--ecfp-radius", type=int, default=2, help="Morgan radius (default: 2, i.e. ECFP4)")
    parser.add_argument(
        "--skip-mordred",
        action="store_true",
        help="Skip Mordred matrix generation even if Mordred is installed",
    )
    parser.add_argument(
        "--keep-constant-columns",
        action="store_true",
        help="Keep columns that are constant across the full dataset",
    )
    parser.add_argument(
        "--keep-all-nan-columns",
        action="store_true",
        help="Keep columns that are entirely NaN",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    generator = BenchmarkDescriptorMatrixGenerator(
        input_csv=args.input,
        output_dir=args.output_dir,
        dataset_label=args.dataset_label,
        ecfp_bits=args.ecfp_bits,
        ecfp_radius=args.ecfp_radius,
        drop_constant_columns=not args.keep_constant_columns,
        drop_all_nan_columns=not args.keep_all_nan_columns,
    )
    saved = generator.save_all(include_mordred=not args.skip_mordred)
    print("Generated matrices:")
    for name, path in saved.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
