"""Build a small reading exercise from the existing, recorded public demo CSVs.

No model fitting, quantum calculations, API calls or manuscript assets are used.
"""
import hashlib
import json
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/source'
DATA = SOURCE / '_static/examples'
IMAGES = SOURCE / '_static/images'


def main():
    train_path = DATA / 'Training_Set_Detailed_demo.csv'
    test_path = DATA / 'Test_Set_Detailed_demo.csv'
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    row = test.loc[test.Sample_Name == 6].iloc[0]
    assert row.SMILES == 'CCCCCO'
    features = ['MolWt', 'TPSA', 'HBD', 'HBA', 'RotB', 'Rings']
    fig, ax = plt.subplots(figsize=(7, 4.2), layout='constrained')
    sc = ax.scatter(train.Feature_TPSA, train.SHAP_TPSA, c=train.Feature_MolWt,
                    cmap='viridis', s=45, edgecolors='white', linewidths=.4)
    ax.axhline(0, color='#839098', lw=.8)
    ax.set(xlabel='TPSA (angstrom squared)', ylabel='Mean validation SHAP for TPSA (logP units)',
           title='Read the direction, then inspect possible confounding')
    fig.colorbar(sc, ax=ax, label='Molecular weight (g/mol)')
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(IMAGES / 'demo_tpsa_direction.png', dpi=180)
    plt.close(fig)
    ordered = sorted(features, key=lambda f: abs(row['SHAP_'+f]))
    vals = [float(row['SHAP_'+f]) for f in ordered]
    fig, ax = plt.subplots(figsize=(7, 4.2), layout='constrained')
    ax.barh(ordered, vals, color=['#d9541a' if v > 0 else '#00478c' for v in vals])
    ax.axvline(0, color='#839098', lw=.8)
    ax.set(xlabel='Mean held-out SHAP contribution (logP units)',
           title='Sample 6: 1-pentanol (CCCCCO)')
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(IMAGES / 'demo_sample6_shap.png', dpi=180)
    plt.close(fig)
    record = {
        'source': 'Recorded public 36-molecule RDKit-computed logP teaching run',
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [train_path, test_path]},
        'sample': 6, 'smiles': row.SMILES, 'computed_target': float(row.Target),
        'prediction': float(row.Realtest_Pred),
        'features': {f: {'value': float(row['Feature_'+f]), 'shap': float(row['SHAP_'+f])} for f in features},
        'not_a_surface_benchmark': True,
        'no_quantum_rerun_or_llm_call': True,
    }
    (DATA / 'explanation_card.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    for z in [False, True]:
        lines = ['.. list-table:: '+('一个真实输出的解释卡：样本 6（1-戊醇）' if z else 'An observed output card: sample 6 (1-pentanol)'),
                  '   :header-rows: 1', '', '   * - '+('字段' if z else 'Field'), '     - '+('数值与含义' if z else 'Value and meaning')]
        pairs = [('SMILES', row.SMILES),
                 ('RDKit Crippen logP', f'{row.Target:.4f}'),
                 ('预测' if z else 'Prediction', f'{row.Realtest_Pred:.4f}'),
                 ('MolWt', f'{row.Feature_MolWt:.2f} g/mol; SHAP {row.SHAP_MolWt:+.4f}'),
                 ('TPSA', f'{row.Feature_TPSA:.2f} A²; SHAP {row.SHAP_TPSA:+.4f}')]
        for key, val in pairs:
            lines += ['   * - '+key, '     - '+val]
        (SOURCE / ('zh/examples' if z else 'examples') / 'explanation_card.rstinc').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        labels = ([('化学问题', '元素、共同片段，或自动局部环境'), ('结构与表面', 'SMILES → xTB → Gaussian → Multiwfn'),
                   ('物理描述符', '明确区域 + ESP / ALIE / LEAE + 统计量'), ('模型归因', '先看预测，再看 SHAP 大小与方向'),
                   ('化学证据卡', '定义、映射、稳健性、可检验的假设')]
                  if z else [('Chemical question', 'Element, conserved fragment, or automated local region'),
                  ('Structure and surface', 'SMILES → xTB → Gaussian → Multiwfn'),
                  ('Physical descriptors', 'Region + ESP / ALIE / LEAE + aggregation rule'),
                  ('Model attribution', 'Prediction quality, then SHAP magnitude and direction'),
                  ('Chemical evidence card', 'Definition, mapping, robustness and a testable hypothesis')])
        svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 660 660" role="img">',
             '<title>'+('工作流示意；不是计算结果' if z else 'Workflow schematic; not a calculated result')+'</title>',
             '<rect width="660" height="660" rx="14" fill="#f4f7f9"/>',
             '<g font-family="Georgia,serif" fill="#183544">']
        for i,(a,b) in enumerate(labels):
            y=18+i*128
            svg += [f'<rect x="18" y="{y}" width="624" height="104" rx="10" fill="white" stroke="#b5cad5"/>',
                    f'<text x="38" y="{y+36}" font-size="23" font-weight="bold">{i+1:02d} · {escape(a)}</text>',
                    f'<text x="38" y="{y+73}" font-size="19">{escape(b)}</text>']
            if i<4: svg += [f'<path d="M330 {y+107}v15m-5-5 5 5 5-5" fill="none" stroke="#d9541a" stroke-width="2"/>']
        svg+=['</g></svg>']
        (IMAGES / ('explanation_workflow_zh.svg' if z else 'explanation_workflow.svg')).write_text('\n'.join(svg), encoding='utf-8')
    print('Built two observed SHAP figures, an auditable card and paired workflow schematics.')


if __name__ == '__main__':
    main()
