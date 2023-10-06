#!/bin/python

##### Training of MPNN-Transformer with batch_size = 256, n_epochs = 100, and LR = 0.001 using full BindingDB #####

from time import time
st = time()

import os
import pickle

import sys
sys.path.append('/home/hl796/gibbs/tools/DeepPurpose')
from DeepPurpose import utils, models


###### 1. Parameters #####
ENC_L = 'MPNN'
ENC_T = 'Transformer'

BATCH = 256
EPOCHS = 100
LR = 0.001
N_WORKERS = 0

INPUT_FILE1 = '/home/hl796/gibbs/tools/DeepPurpose/data/bindingdb.Kd.processed.pckl'

DIR_BASE = '/home/hl796/gibbs/tools/DeepPurpose/training/regression/BindingDB-fulldata/'
DIR_RES = os.path.join(DIR_BASE + ENC_L + '_' + ENC_T)

print('##### Parameters are all set. #####\n')
print()


##### 2. Load the data #####
with open(INPUT_FILE1, 'rb') as f:
    bddb = pickle.load(f)

X_drugs, X_targets, y_kd = bddb['Drugs'], bddb['Targets'], bddb['Kd']
print(X_drugs.shape, X_targets.shape, y_kd.shape)

print('##### BindingDB is loaded. #####\n')
print()


##### 3. Data processing #####
train, _, _ = utils.data_process(X_drugs, X_targets, y_kd, 
                                 drug_encoding=ENC_L, target_encoding=ENC_T, 
                                 frac=[1,0,0], BindingDB=True)
train.shape

print()
print('##### Data processing is done. #####\n')
print()


##### 4. Model configuration #####
config = utils.generate_config(drug_encoding=ENC_L, target_encoding=ENC_T, 
                               batch_size=BATCH, train_epoch=EPOCHS, result_folder=DIR_RES, 
                               LR=LR, num_workers=N_WORKERS)

print('##### Model configuration is all set. #####\n')
print()


##### 5. Model training #####
model = models.model_initialize(**config)
model.train(train, None, None)

print()
print('##### Model training is done. #####\n')
print()


##### 6. Save the model #####
model.save_model(DIR_RES)

print('##### Model is saved. #####\n')
print()


##### 7. Time taken #####
time_taken = time() - st
if time_taken < 60:
    print('Analysis time taken: %.2f sec.' %time_taken)
elif time_taken < 3600:
    print('Analysis time taken: %.2f min.' %(time_taken / 60))
else:
    print('Analysis time taken: %.2f hr.' %(time_taken / 3600))

print('##### Done. #####')
print()
