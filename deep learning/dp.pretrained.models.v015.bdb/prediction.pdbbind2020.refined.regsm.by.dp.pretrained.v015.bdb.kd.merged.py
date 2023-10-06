#/bin/python

##### Prediction of PDBbind-2020-refined-regsm by DeepPurpose-v0.1.5 BindingDB-Kd-pretrained models #####

from time import ctime, time, sleep

print(ctime())
print()

st = time()

import os
import pickle
import pandas as pd
import numpy as np

import sys
sys.path.append('../')
from DeepPurpose import models

import matplotlib.pyplot as plt

MODEL_DIR = '../dp/save_folder/pretrained_models/'

DATA_FILE = '../data/pdbbind2020.reg.smiles.processed.csv'

OUTPUT_FILENAME1 = '../result/pdbbind2020/prediction.pdbbind2020.refined.regsm.by.dp.pretrained.v015.bdb.kd.merged.csv'
OUTPUT_FILENAME2 = 'prediction.pdbbind2020.refined.regsm.by.dp.pretrained.v015.bdb.kd.merged.scatter.png'


### 1. Load the pretrained models
tmpdir = os.listdir(MODEL_DIR)
MODELS = sorted([x for x in tmpdir if x.endswith('_bindingdb')])
MODELS_names = [x.replace('model_', '').replace('_bindingdb', '').replace('aac', 'AAC').replace('cnn', 'CNN').replace('mpnn', 'MPNN').replace('daylight', 'Daylight').replace('morgan', 'Morgan').replace('transformer', 'Transformer') for x in MODELS]

PRTMOD = {}
for k,m in zip(MODELS_names, MODELS):
    PRTMOD[k] = models.model_pretrained(MODEL_DIR + m)

for k,m in PRTMOD.items():
    print(k)
    print(m.config)
    print()
print()

MODELS = MODELS_names


### 2. PDBbind data
pdbbind = pd.read_csv(DATA_FILE)

lignames = list(pdbbind.Ligand) #lignames = ['L_'+x for x in pdbbind.Ligand]
smiles = list(pdbbind.SMILES)

target_names = [x+'_'+y for x,y in zip(pdbbind.ID, pdbbind.TargetChain)]
targets = list(pdbbind.AASeq)

print(len(lignames), len(target_names))
print()


### 3. Predictions
y_pred_all = []
for i, (lig_name, smile, target_name, target) in enumerate(zip(lignames, smiles, target_names, targets)):
    print('====================')
    print(str(i) + ': ' + lig_name + ', ' + target_name)
    print('====================')

    y_pred = {}
    y_pred['Ligand'] = lig_name
    y_pred['Target'] = target_name
    
    for k,m in PRTMOD.items():
        print(k)
        y_pred[k] = models.repurpose([smile], target, m, 
                                     [lig_name], target_name, 
                                     BindingDB=False, verbose=False)
        print('\n##########\n')
    
    y_pred_all.append(pd.DataFrame(y_pred))
    
    #if i == 9: break # for testing/debugging
    
    print('====================\n')
    print()

os.system("find ../result -maxdepth 1 -name 'repurposing*.output.txt' -type f -delete")
print()


### 4. Results
pred_merged = pd.concat(y_pred_all, axis=0).reset_index(drop=True)
pred_merged['BA_score'] = -np.log10(np.exp(pdbbind.Label))
print(pred_merged)
print()

pred_merged.to_csv(OUTPUT_FILENAME1, index=False)

pcorr_all = pred_merged.drop(['Ligand', 'Target'], axis=1).corr()
print(pcorr_all)
print()

pcorr = {m:pcorr_all['BA_score'][m] for m in MODELS}
mse = {x:sum((pred_merged.BA_score - pred_merged[x])**2)/pred_merged.shape[0] for x in MODELS}
rmse = {x:np.sqrt(sum((pred_merged.BA_score - pred_merged[x])**2)/pred_merged.shape[0]) for x in MODELS}

print(pd.DataFrame([pcorr, mse, rmse], index=['PCC', 'MSE', 'RMSE']))
print()

_,ax = plt.subplots(1,len(MODELS), figsize=(5*len(MODELS),5*1))
plt.subplots_adjust()

for i,m in enumerate(MODELS):
    title = 'PDBbind2020-refined (' + str(pred_merged.shape[0]) + ')\n' + \
    'r = ' + str(round(pcorr[m],2)) + '; MSE = ' + str(round(mse[m],2))

    ax[i].plot(pred_merged.BA_score, pred_merged[m], 'o', color='k', alpha=.2)
    ax[i].set_xlabel('-log10(KdKi[M])', fontsize=18)
    ax[i].set_ylabel(m, fontsize=18)
    ax[i].set_title(title, fontsize=18)
    ax[i].axis('equal')

plt.tight_layout(w_pad=4)
plt.savefig(OUTPUT_FILENAME2)

sleep(10)


### 5. Runtime
time_taken = time() - st
if time_taken < 60:
    print('Analysis time taken: %.2f sec.' %time_taken)
elif time_taken < 3600:
    print('Analysis time taken: %.2f min.' %(time_taken / 60))
else:
    print('Analysis time taken: %.2f hr.' %(time_taken / 3600))
print()

print(ctime())
print()
