#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Molecular properties calculation utilities
"""

import pandas as pd
import os

def calculate_molecular_properties(smiles):
    """Calculate basic molecular properties from SMILES"""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Crippen, Lipinski
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        properties = {
            'SMILES': smiles,
            'Molecular_Weight': round(Descriptors.MolWt(mol), 2),
            'LogP': round(Crippen.MolLogP(mol), 2),
            'HBD': Lipinski.NumHDonors(mol),
            'HBA': Lipinski.NumHAcceptors(mol),
            'TPSA': round(Descriptors.TPSA(mol), 2),
            'Rotatable_Bonds': Descriptors.NumRotatableBonds(mol),
            'Aromatic_Rings': Descriptors.NumAromaticRings(mol),
            'Heavy_Atoms': mol.GetNumHeavyAtoms()
        }
        return properties
    except ImportError:
        print("❌ Error: RDKit not installed. Please install with: pip install rdkit")
        return None
    except Exception as e:
        print(f"❌ Error calculating properties for {smiles}: {e}")
        return None

def analyze_molecules_from_csv(csv_file, output_file=None, properties=None):
    """Analyze molecules from CSV file and calculate properties"""
    if not os.path.exists(csv_file):
        print(f"❌ Error: Input file '{csv_file}' not found!")
        return False
    
    try:
        df = pd.read_csv(csv_file)
        
        if 'smiles' not in df.columns:
            print("❌ Error: CSV file must contain a 'smiles' column")
            return False
        
        print(f"📊 Analyzing {len(df)} molecules...")
        
        results = []
        for idx, smiles in enumerate(df['smiles']):
            props = calculate_molecular_properties(smiles)
            if props:
                props['Index'] = idx + 1
                results.append(props)
                if (idx + 1) % 10 == 0:
                    print(f"   Processed {idx + 1}/{len(df)} molecules...")
        
        if not results:
            print("❌ No valid molecules found!")
            return False
        
        results_df = pd.DataFrame(results)
        
        # Reorder columns
        cols = ['Index', 'SMILES', 'Molecular_Weight', 'LogP', 'HBD', 'HBA', 
                'TPSA', 'Rotatable_Bonds', 'Aromatic_Rings', 'Heavy_Atoms']
        results_df = results_df[cols]
        
        if output_file:
            results_df.to_csv(output_file, index=False)
            print(f"✅ Results saved to: {output_file}")
        else:
            print("\n📋 Molecular Properties Summary:")
            print(results_df.to_string(index=False))
        
        print(f"\n📈 Statistics:")
        print(f"   Total molecules: {len(results_df)}")
        print(f"   Average MW: {results_df['Molecular_Weight'].mean():.2f}")
        print(f"   Average LogP: {results_df['LogP'].mean():.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing CSV file: {e}")
        return False

def analyze_single_molecule(smiles):
    """Analyze a single molecule and display properties"""
    props = calculate_molecular_properties(smiles)
    if not props:
        return False
    
    print(f"\n🧪 Molecular Properties for: {smiles}")
    print("=" * 50)
    for key, value in props.items():
        if key != 'SMILES':
            print(f"   {key.replace('_', ' ')}: {value}")
    
    return True