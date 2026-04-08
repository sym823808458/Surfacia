from __future__ import annotations

import argparse
import os

from ..visualization.interactive_shap_viz import interactive_shap_viz_main


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch Surfacia SHAP visualizer.")
    parser.add_argument("--training-csv", required=True, help="Training_Set_Detailed CSV path")
    parser.add_argument("--xyz-dir", required=True, help="Directory containing xyz/fchk files")
    parser.add_argument("--test-csv", help="Optional Test_Set_Detailed CSV path")
    parser.add_argument("--port", type=int, default=8052, help="Server port")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument(
        "--skip-surface-gen",
        action="store_true",
        help="Skip PDB/surface generation before starting the Dash app",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    api_key = os.getenv("SURFACIA_ZHIPUAI_API_KEY")
    success = interactive_shap_viz_main(
        csv_path=args.training_csv,
        xyz_path=args.xyz_dir,
        test_csv_path=args.test_csv,
        api_key=api_key,
        skip_surface_gen=args.skip_surface_gen,
        port=args.port,
        host=args.host,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
