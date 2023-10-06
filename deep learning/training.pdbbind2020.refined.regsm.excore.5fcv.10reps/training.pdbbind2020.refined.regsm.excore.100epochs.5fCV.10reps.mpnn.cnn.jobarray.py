#!/bin/python

##### MPNN-CNN #####
##### 100-epoch training with 5f-CV 10-reps PDBBind2020-refined-regsm-excore (jobarray) #####

##### Libraries #####
import os
import pickle
import pandas as pd
import copy

dir_dp = '/home/hl796/gibbs/tools/DeepPurpose/'

import sys
sys.path.append(dir_dp)
from DeepPurpose import utils, models

import torch


##### Training Function #####
def TrainingUnit(i):
    for j in range(N_FOLDS):
        print('##### Rep', i, '; CV-fold', j, '#####')
        
        ### Data processing ###
        tmp_train = pd.DataFrame({
            'drug_encoding': data_cv[i]['drugs'][j][0],
            'target_encoding': data_cv[i]['targets'][j][0],
            'Label': data_cv[i]['kd'][j][0]
        }).reset_index(drop=True)

        tmp_val = pd.DataFrame({
            'drug_encoding': data_cv[i]['drugs'][j][1],
            'target_encoding': data_cv[i]['targets'][j][1],
            'Label': data_cv[i]['kd'][j][1]
        }).reset_index(drop=True)

        tmp_test = pd.DataFrame({
            'drug_encoding': data_cv[i]['drugs'][j][2],
            'target_encoding': data_cv[i]['targets'][j][2],
            'Label': data_cv[i]['kd'][j][2]
        }).reset_index(drop=True)
        
        config = utils.generate_config(drug_encoding=drug_encoding, target_encoding=target_encoding, 
                                       batch_size=BATCH, train_epoch=EPOCHS, 
                                       LR=LR, num_workers=N_WORKERS, 
                                       result_folder=DIR_OUTPUT + 'R' + str(i) + 'F' + str(j))

        model = models.model_initialize(**config)

        model.train(tmp_train, tmp_val, tmp_test)
        model.save_model(DIR_OUTPUT + 'R' + str(i) + 'F' + str(j))

        print('##### Done. #####\n')


if __name__ == "__main__":

    from sys import argv
    i = int(argv[1])
    print(f"CV-5f rep {i}")
    print()

    ##### START #####
    from time import time
    st = time()

    print('> torch version:', torch.__version__)
    print('> pandas version:', pd.__version__)
    print()


    ##### Params and file names #####
    drug_encoding = 'MPNN'
    target_encoding = 'CNN'

    INPUT_FILE1 = dir_dp + 'data/pdbbind2020.regsm.excore.5fCV.10rep.encoding.' + \
    drug_encoding.lower() + '.' + target_encoding.lower() + '.pkl'

    DIR_OUTPUT = dir_dp + 'training/regression/PDBbind2020-refined-regsm-excore/'
    DIR_OUTPUT = os.path.join(DIR_OUTPUT + drug_encoding + '_' + target_encoding + '/jobarray-RCV-')
    print(DIR_OUTPUT)
    print()

    N_FOLDS = 5
    EPOCHS = 100
    BATCH = 256
    LR = 0.001
    N_WORKERS = 1


    ##### Data loading #####
    with open(INPUT_FILE1, 'rb') as f:
        data_cv = pickle.load(f)


    ##### Run training for each 5f-CV rep #####
    TrainingUnit(i)


    ##### END #####
    time_taken = time() - st
    if time_taken < 60:
        print('Analysis time taken: %.2f sec.' %time_taken)
    elif time_taken < 3600:
        print('Analysis time taken: %.2f min.' %(time_taken / 60))
    else:
        print('Analysis time taken: %.2f hr.' %(time_taken / 3600))
