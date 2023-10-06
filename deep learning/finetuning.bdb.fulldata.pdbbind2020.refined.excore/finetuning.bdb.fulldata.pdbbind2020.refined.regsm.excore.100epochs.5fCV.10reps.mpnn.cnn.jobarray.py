#!/bin/python

##### MPNN-CNN #####
##### 100-epoch finetuning on the BindingDB-fulldata-trained model with 5f-CV 10-reps PDBbind2020-refined-regsm-excore (jobarray) #####

##### Libraries #####
import os
import pickle
import pandas as pd
import copy

dir_dp = '/home/hl796/gibbs/tools/DeepPurpose/'

import sys
sys.path.append(dir_dp)
from DeepPurpose import models

import torch


##### Training Function #####
def TrainingUnit(i):
    for j in range(N_FOLDS):
        print('##### Rep', i, '; CV-fold', j, '#####')

        modelcp = copy.deepcopy(model)
        
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

        ### Model configuration ###
        tmp_dir = DIR_OUTPUT + 'R' + str(i) + 'F' + str(j)
        if not os.path.exists(tmp_dir): os.makedirs(tmp_dir)

        modelcp.config['result_folder'] = tmp_dir
        print(modelcp.config)
        print()
    
        ### Model training and saving ###
        modelcp.result_folder = tmp_dir
        modelcp.train(tmp_train, tmp_val, tmp_test)
        modelcp.save_model(tmp_dir)

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

    DIR_PRETRAINED = dir_dp + 'training/regression/BindingDB-fulldata/' + drug_encoding + '_' + target_encoding

    INPUT_FILE1 = dir_dp + 'data/pdbbind2020.regsm.excore.5fCV.10rep.encoding.mpnn.cnn.pkl'

    DIR_OUTPUT = dir_dp + 'training/regression/finetuned/BDB-full-PDBbind2020-refined-regsm-excore/'
    DIR_OUTPUT = os.path.join(DIR_OUTPUT + drug_encoding + '_' + target_encoding + '/jobarray-RCV-')
    print(DIR_OUTPUT)
    print()

    N_FOLDS = 5
    EPOCHS = 100
    BATCH = 256
    LR = 0.001
    N_WORKERS = 1


    ##### BindingDB-trained models #####
    model = models.model_pretrained(path_dir=DIR_PRETRAINED)

    model.config['train_epoch'] = EPOCHS
    model.config['batch_size'] = BATCH
    model.config['LR'] = LR
    model.config['num_workers'] = N_WORKERS


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

