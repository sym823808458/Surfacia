import os
import pandas as pd

merged_df_path = input("Full.csv path: ")
try:
    merged_df = pd.read_csv(merged_df_path)
except Exception as e:
    print(f"error{e}")
    exit()
S_N = merged_df.shape[0]
F_N = merged_df.shape[1] - 3  

RECORD_NAME = 'Full_'+str(S_N) +'_'+str(merged_df.shape[1]) +'.csv'
merged_df.to_csv(RECORD_NAME, index=False, float_format='%.6f')
print(f'Data written to {RECORD_NAME}')



# Smiles
INPUT_SMILES = 'Smiles_' + str(S_N) + '.csv'
merged_df[['smiles']].to_csv(INPUT_SMILES, index=False, header=False)

# Value
INPUT_Y = 'Values_True_' + str(S_N) + '.csv'
merged_df[['target']].to_csv(INPUT_Y, index=False, header=False)

# Feature
INPUT_X = 'Features_' + str(S_N) + '_' + str(F_N) + '.csv'
merged_df.drop(['Sample Name', 'smiles', 'target'], axis=1).to_csv(INPUT_X, index=False, float_format='%.6f', header=False)

# Title
INPUT_TITLE = 'Title_' + str(F_N) + '.csv'
with open(INPUT_TITLE, 'w') as f:
    for col in merged_df.columns[3:]:  
        if col !=0:
            f.write(col + '\n')


print(f'Data split into {INPUT_SMILES}, {INPUT_Y}, {INPUT_X}, and {INPUT_TITLE}')