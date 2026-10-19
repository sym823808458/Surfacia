"""Regenerate public teaching outputs using the current package.
Run from the repository root after installing surfacia[viz].
"""
import csv,hashlib,json,platform,shutil,subprocess,sys
from pathlib import Path
import importlib.metadata as metadata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.metrics import mean_absolute_error,r2_score

ROOT=Path(__file__).resolve().parents[1]
DEMO=ROOT/'examples/descriptor_table'
STATIC=ROOT/'docs/source/_static/examples'
IMAGES=ROOT/'docs/source/_static/images'

def regenerate(run=True):
 STATIC.mkdir(parents=True,exist_ok=True)
 subprocess.run([sys.executable,str(DEMO/'make_data.py')],check=True,cwd=ROOT)
 command=['surfacia','ml-analysis','-i','examples/descriptor_table/descriptors.csv','--manual','--manual-features','Full','--test-samples','6,12,18,24,30,36','--epoch','32','--cores','1','--train-test-split','0.85','--no-generate-fitting']
 if run:
  subprocess.run([str(Path(sys.executable).with_name('surfacia.exe')) if sys.platform=='win32' else 'surfacia',*command[1:]],check=True,cwd=ROOT)
 folder=max(DEMO.glob('Full_Feature_Analysis_*'),key=lambda p:p.stat().st_mtime)
 training=next(folder.glob('Training_Set_Detailed*.csv'))
 test=next(folder.glob('Test_Set_Detailed*.csv'))
 train=pd.read_csv(training); held=pd.read_csv(test)
 assert len(train)==30 and len(held)==6
 assert not set(train.Sample_Name)&set(held.Sample_Name)
 assert not list(folder.glob('SPES*'))
 for src,name in [(DEMO/'descriptors.csv','descriptors.csv'),(DEMO/'molecules.csv','molecules.csv'),(DEMO/'make_data.py','make_data.py'),(training,'Training_Set_Detailed_demo.csv'),(test,'Test_Set_Detailed_demo.csv')]:shutil.copyfile(src,STATIC/name)
 raw=next(folder.glob('Manual_Prediction_Data*.txt'))
 validation=np.loadtxt(raw,skiprows=1,delimiter=',')
 print('prediction data shape',validation.shape)
 # Prediction data columns: target, out-of-fold prediction, between-split spread.
 target=validation[:,0];pred=validation[:,1]
 fig,ax=plt.subplots(figsize=(7.0,5.0),layout='constrained')
 ax.scatter(target,pred,c='#256b82',alpha=.8,label='Modeling pool · validation',s=36)
 ax.scatter(held.Target,held.Realtest_Pred,facecolors='none',edgecolors='#b86635',marker='D',s=64,label='Held-out · 6 samples',linewidths=1.6)
 lo=min(float(target.min()),float(held.Target.min()))-.2;hi=max(float(target.max()),float(held.Target.max()))+.2
 ax.plot([lo,hi],[lo,hi],color='#9aaeb5',ls='--',lw=1)
 ax.set(xlabel='RDKit-calculated Crippen logP',ylabel='Original model prediction',title='Descriptor-table tutorial · computed target')
 ax.legend(frameon=False,fontsize=9);ax.spines[['top','right']].set_visible(False)
 fig.savefig(IMAGES/'demo_predictions.png',dpi=180);plt.close(fig)
 shap_cols=[c for c in train if c.startswith('SHAP_') and not c.startswith('SHAP_Std_')]
 values=train[shap_cols].abs().mean().sort_values()
 fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
 ax.barh([c.removeprefix('SHAP_') for c in values.index],values.values,color='#256b82')
 ax.set(xlabel='Mean absolute SHAP attribution',title='Model influence · not causal evidence')
 ax.spines[['top','right']].set_visible(False)
 fig.savefig(IMAGES/'demo_shap.png',dpi=180);plt.close(fig)
 sulfur=[('Methanethiol','CS'),('Ethanethiol','CCS'),('Dimethyl sulfide','CSC'),('Dimethyl disulfide','CSSC'),('Thiophene','c1ccsc1'),('Thioanisole','CSc1ccccc1')]
 with (STATIC/'sulfur_molecules.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.writer(f);w.writerow(['Sample Name','smiles']);w.writerows([[i,s] for i,(_,s) in enumerate(sulfur,1)])
 image=Draw.MolsToGridImage([Chem.MolFromSmiles(s) for _,s in sulfur],molsPerRow=2,subImgSize=(350,210),legends=[n for n,_ in sulfur],useSVG=False)
 image.save(str(IMAGES/'sulfur_molecules.png'))
 np.random.seed(42);coverage=np.zeros(len(train),dtype=int);point=round(len(train)*.85)
 for _ in range(32):coverage[np.random.permutation(len(train))[point:]]+=1
 summary={'date':'2026-10-19','source_version':metadata.version('surfacia'),'target':'RDKit-computed Crippen logP; not experimental','features':'RDKit MolWt TPSA HBD HBA RotB Rings; not Surfacia quantum descriptors','sample_count':36,'modeling_count':30,'held_out_names':[6,12,18,24,30,36],'command':command,'validation':{'r2':r2_score(target,pred),'mae':mean_absolute_error(target,pred),'minimum_validation_appearances':int(coverage.min())},'held_out':{'r2':r2_score(held.Target,held.Realtest_Pred),'mae':mean_absolute_error(held.Target,held.Realtest_Pred)},'environment':{n:metadata.version(n) for n in ['surfacia','rdkit','numpy','pandas','xgboost','shap','scikit-learn']},'python':platform.python_version(),'quantum_workflow_rerun':False,'unpublished_experiment_data_included':False,'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in STATIC.glob('*.csv')}}
 (STATIC/'run_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':regenerate(run='--reuse-run' not in sys.argv)
