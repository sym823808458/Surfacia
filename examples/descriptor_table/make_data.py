"""Public teaching data: RDKit-computed target and features, not experimental data.
Run with Python and RDKit to regenerate descriptors.csv beside this script.
"""
import csv
from pathlib import Path
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, Crippen, Lipinski

SMILES = [
 'CCO','CCCO','CCCCO','CC(C)O','CC(C)CO','CCCCCO',
 'CC','CCC','CCCC','CCCCC','CCCCCC','CC(C)C',
 'CC(=O)C','CCC(=O)C','CC(=O)O','CCC(=O)O','CCOC','CCOCC',
 'CCN','CCCN','CC(C)N','CC(=O)N','CC#N','CCC#N',
 'c1ccccc1','Cc1ccccc1','CCc1ccccc1','Oc1ccccc1','Nc1ccccc1','COc1ccccc1',
 'c1ccncc1','c1ccoc1','c1ccsc1','C1CCCCC1','C1CCNCC1','CCOC(=O)C',
]
FIELDS = ['Sample Name','MolWt','TPSA','HBD','HBA','RotB','Rings','smiles','target']
def generate(folder=None):
 folder=Path(folder) if folder else Path(__file__).resolve().parent
 folder.mkdir(parents=True,exist_ok=True)
 rows=[]
 for i,smi in enumerate(SMILES,1):
  mol=Chem.MolFromSmiles(smi)
  assert mol is not None,smi
  rows.append([i,Descriptors.MolWt(mol),Descriptors.TPSA(mol),Lipinski.NumHDonors(mol),Lipinski.NumHAcceptors(mol),Lipinski.NumRotatableBonds(mol),Lipinski.RingCount(mol),Chem.MolToSmiles(mol),Crippen.MolLogP(mol)])
 with (folder/'descriptors.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.writer(f);w.writerow(FIELDS);w.writerows(rows)
 with (folder/'molecules.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.writer(f);w.writerow(['Sample Name','smiles']);w.writerows([[r[0],r[-2]] for r in rows])
 print('Generated',len(rows),'molecules using RDKit',rdBase.rdkitVersion)
if __name__=='__main__':generate()
