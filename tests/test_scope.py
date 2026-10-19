"""Regression checks for removing ranking without damaging ordinary SHAP outputs."""
import argparse
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from update_cli_reference import parser


class ScopeRegressionTests(unittest.TestCase):
    def test_cli_keeps_raw_test_overlay_and_rejects_removed_option(self):
        cli = parser()
        args = cli.parse_args(['shap-viz', '-i', 'train.csv', '-x', '.', '--test-csv', 'test.csv'])
        self.assertEqual(args.test_csv, 'test.csv')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.parse_args(['shap-viz', '-i', 'train.csv', '-x', '.', '--spes-csv', 'old.csv'])

    def test_original_test_predictions_and_shap_are_exported(self):
        import numpy as np
        import pandas as pd
        from surfacia.ml.chem_ml_analyzer_v2 import BaseChemMLAnalyzer
        model = BaseChemMLAnalyzer.__new__(BaseChemMLAnalyzer)
        model.timestamp = 'regression'
        model.X_realtest = np.array([[3., 4.], [5., 6.]])
        model.smiles_train = np.array(['CC', 'CO'])
        model.y_train = np.array([1., 2.])
        model.sample_names_train = np.array(['train-a', 'train-b'])
        model.smiles_realtest = np.array(['CCC', 'CCO'])
        model.y_realtest = np.array([3., 4.])
        model.sample_names_realtest = np.array(['hold-a', 'hold-b'])
        perf = {'shap_values': np.array([[.2], [.4]]), 'shap_std': np.zeros((2, 1)),
                'realtest_shap_mean': np.array([[.5], [.6]]), 'realtest_shap_std': np.zeros((2, 1)),
                'y_realtest_pred': np.array([2.8, 4.2]), 'y_realtest_pred_std': np.array([.1, .2])}
        with tempfile.TemporaryDirectory() as folder:
            model.save_comprehensive_results(Path(folder), 'test', ['local'], [1], np.array([[1.], [2.]]), perf)
            files = sorted(p.name for p in Path(folder).iterdir())
            self.assertEqual(len(files), 2)
            self.assertFalse(any('SPES' in name for name in files))
            test = pd.read_csv(Path(folder) / 'Test_Set_Detailed_test_regression.csv')
            np.testing.assert_allclose(test.Realtest_Pred, [2.8, 4.2])
            np.testing.assert_allclose(test.SHAP_local, [.5, .6])
            np.testing.assert_allclose(test.Feature_local, [4., 6.])

    def test_dashboard_keeps_only_original_test_points(self):
        import pandas as pd
        from surfacia.visualization.interactive_shap_viz import InteractiveSHAPAnalyzer
        analyzer = InteractiveSHAPAnalyzer.__new__(InteractiveSHAPAnalyzer)
        analyzer.test_data = pd.DataFrame({'Target': [1.], 'Realtest_Pred': [1.1]})
        self.assertEqual([item['value'] for item in analyzer.get_overlay_options()], ['none', 'test'])
        self.assertEqual(analyzer.get_overlay_dataset('test')['color_column'], 'Realtest_Pred')
        self.assertIsNone(analyzer.get_overlay_dataset('spes'))


if __name__ == '__main__':
    unittest.main()
