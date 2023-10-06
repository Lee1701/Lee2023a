rmseimport pandas as pd
import numpy as np
import os
import sys
from sklearn.linear_model import LassoCV, LinearRegression, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,RandomizedSearchCV
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from scipy.stats import uniform, randint, spearmanr
import argparse
import math
import glob
from copy import deepcopy
from joblib import dump, load
from joblib import parallel_backend

#### Calculate the RMSE between two vectors
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))
#### Code to extract top results from the XGBoost model
def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
    return results['mean_test_score'][np.flatnonzero(results['rank_test_score'] == i)[0]]


#### Extract the Output metrics for the docking scores only: Pearson's and Spearman correlations, MSE and RMSE
def docking_metrics(scoredf,corepdb, core_score,outfilename):
    total_dim = scoredf.loc[:,'Log Dissociation'].shape
    y_check = scoredf.loc[:,'Log Dissociation'].to_numpy().flatten()
    x_check = scoredf.loc[:,'SMINA score'].to_numpy().flatten()
    smina_total_pearson = np.corrcoef(x_check,y_check)[0,1]
    smina_total_spearman,pval = spearmanr(x_check,y_check)
    smina_total_mse = np.square(np.subtract(x_check,y_check)).mean()
    smina_total_rmse = math.sqrt(smina_total_mse)

    y_check = scoredf.loc[:,'Log Dissociation'].to_numpy().flatten()
    x_check = scoredf.loc[:,'Vinardo score'].to_numpy().flatten()
    vinardo_total_pearson = np.corrcoef(x_check,y_check)[0,1]
    vinardo_total_spearman,pval = spearmanr(x_check,y_check)
    vinardo_total_mse = np.square(np.subtract(x_check,y_check)).mean()
    vinardo_total_rmse = math.sqrt(vinardo_total_mse)

    scoredf = scoredf[~scoredf['PDB ID'].isin(corepdb)]

    #### Docking metrics for refined
    refined_dim = scoredf.loc[:,'Log Dissociation'].shape
    y_check = scoredf.loc[:,'Log Dissociation'].to_numpy().flatten()
    x_check = scoredf.loc[:,'SMINA score'].to_numpy().flatten()
    smina_refined_pearson = np.corrcoef(x_check,y_check)[0,1]
    smina_refined_spearman,pval = spearmanr(x_check,y_check)
    smina_refined_mse = np.square(np.subtract(x_check,y_check)).mean()
    smina_refined_rmse = math.sqrt(smina_refined_mse)

    y_check = scoredf.loc[:,'Log Dissociation'].to_numpy().flatten()
    x_check = scoredf.loc[:,'Vinardo score'].to_numpy().flatten()
    vinardo_refined_pearson = np.corrcoef(x_check,y_check)[0,1]
    vinardo_refined_spearman,pval = spearmanr(x_check,y_check)
    vinardo_refined_mse = np.square(np.subtract(x_check,y_check)).mean()
    vinardo_refined_rmse = math.sqrt(vinardo_refined_mse)

    #### Docking metrics for core set
    core_dim = core_score.loc[:,'Log Dissociation'].shape
    y_check = core_score.loc[:,'Log Dissociation'].to_numpy().flatten()
    x_check = core_score.loc[:,'SMINA score'].to_numpy().flatten()
    smina_core_pearson = np.corrcoef(x_check,y_check)[0,1]
    smina_core_spearman,pval = spearmanr(x_check,y_check)
    smina_core_mse = np.square(np.subtract(x_check,y_check)).mean()
    smina_core_rmse = math.sqrt(smina_core_mse)

    y_check = core_score.loc[:,'Log Dissociation'].to_numpy().flatten()
    x_check = core_score.loc[:,'Vinardo score'].to_numpy().flatten()
    vinardo_core_pearson = np.corrcoef(x_check,y_check)[0,1]
    vinardo_core_spearman,pval = spearmanr(x_check,y_check)
    vinardo_core_mse = np.square(np.subtract(x_check,y_check)).mean()
    vinardo_core_rmse = math.sqrt(vinardo_core_mse)



    with open(outfilename,'w') as outfile:
        outfile.write(f"Total dimensions = {total_dim}\n")
        outfile.write("===============================\n")
        outfile.write(f"Total SMINA Pearson's Correlation = {smina_total_pearson}\n")
        outfile.write(f"Total SMINA Spearman Rank Correlation = {smina_total_spearman}\n")
        outfile.write(f"Total SMINA MSE = {smina_total_mse}\n")
        outfile.write(f"Total SMINA RMSE = {smina_total_rmse}\n")
        outfile.write("===============================\n")
        outfile.write(f"Total Vinardo Pearson's Correlation = {vinardo_total_pearson}\n")
        outfile.write(f"Total Vinardo Spearman Rank Correlation = {vinardo_total_spearman}\n")
        outfile.write(f"Total Vinardo MSE = {vinardo_total_mse}\n")
        outfile.write(f"Total Vinardo RMSE = {vinardo_total_rmse}\n")
        outfile.write("===============================\n")

        outfile.write(f"Refined dimensions = {refined_dim}\n")
        outfile.write("===============================\n")
        outfile.write(f"Refined SMINA Pearson's Correlation = {smina_refined_pearson}\n")
        outfile.write(f"Refined SMINA Spearman Rank Correlation = {smina_refined_spearman}\n")
        outfile.write(f"Refined SMINA MSE = {smina_refined_mse}\n")
        outfile.write(f"Refined SMINA RMSE = {smina_refined_rmse}\n")
        outfile.write("===============================\n")
        outfile.write(f"Refined Vinardo Pearson's Correlation = {vinardo_refined_pearson}\n")
        outfile.write(f"Refined Vinardo Spearman Rank Correlation = {vinardo_refined_spearman}\n")
        outfile.write(f"Refined Vinardo MSE = {vinardo_refined_mse}\n")
        outfile.write(f"Refined Vinardo RMSE = {vinardo_refined_rmse}\n")
        outfile.write("===============================\n")

        outfile.write(f"Core dimensions = {core_dim}\n")
        outfile.write("===============================\n")
        outfile.write(f"Core SMINA Pearson's Correlation = {smina_core_pearson}\n")
        outfile.write(f"Core SMINA Spearman Rank Correlation = {smina_core_spearman}\n")
        outfile.write(f"Core SMINA MSE = {smina_core_mse}\n")
        outfile.write(f"Core SMINA RMSE = {smina_core_rmse}\n")
        outfile.write("===============================\n")
        outfile.write(f"Core Vinardo Pearson's Correlation = {vinardo_core_pearson}\n")
        outfile.write(f"Core Vinardo Spearman Rank Correlation = {vinardo_core_spearman}\n")
        outfile.write(f"Core Vinardo MSE = {vinardo_core_mse}\n")
        outfile.write(f"Core Vinardo RMSE = {vinardo_core_rmse}\n")
        outfile.write("===============================\n")

    return

#### Train the ML models and predict the leave-out test values for the best-fit models in each ML category
def run_ml_models(numproc,X_train,y_train,X_ext_val,y_ext_val):

    niter = 100
    ####LASSO
    minscore = 10**6
    for i in range(niter):
        X_train,y_train = shuffle(X_train,y_train,random_state=0)
        lasso_classifier = LassoCV(cv=5,fit_intercept=True,selection='cyclic',random_state=0).fit(X_train,y_train)
        curr_dual = lasso_classifier.dual_gap_
        if curr_dual < minscore:
            minscore = curr_dual
            best_class = lasso_classifier

    lasso_pred = best_class.predict(X_ext_val)
    lasso_model = best_class

    ####ElasticNet
    minscore = 10**6
    l1_list = [.1, .5, .7, .9, .95, .99, 1]
    for l1_val in l1_list:
        for i in range(10):
            X_train,y_train = shuffle(X_train,y_train,random_state=0)
            lasso_classifier = ElasticNetCV(cv=5,l1_ratio=l1_val,fit_intercept=True,selection='cyclic',random_state=0).fit(X_train,y_train)
            curr_dual = lasso_classifier.dual_gap_
            if curr_dual < minscore:
                minscore = curr_dual
                best_class = lasso_classifier
    print("2. ElasticNet results")
    en_pred = best_class.predict(X_ext_val)
    en_model = best_class


    ####Linear Regression
    reg = LinearRegression().fit(X_train, y_train)
    lr_pred = reg.predict(X_ext_val)
    lr_model = reg


    ####XGBoost
    print("3. XGBoost results")
    xgb_model = XGBRegressor(objective="reg:squarederror", random_state=0)
    params = {
        "colsample_bytree": uniform(0.8, 0.2),
        "gamma": uniform(0, 0.5),
        "learning_rate": uniform(0.02, 0.3), # default 0.1
        "max_depth": randint(2, 6), # default 3
        "n_estimators": randint(100, 150), # default 100
        "subsample": uniform(0.7, 0.3)
    }
    niter = 1
    maxscore = 0
    for i in range(niter):
        search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=0, n_iter=100, cv=5, verbose=1, n_jobs=numproc, return_train_score=True)
        X_train,y_train = shuffle(X_train,y_train,random_state=0)
        with parallel_backend('threading', n_jobs=numproc):
            search.fit(X_train, y_train)

        score = report_best_scores(search.cv_results_, 1)

        if score>maxscore:
            maxscore = score
            maxsearch = search
    bdb_maxsearch = maxsearch
    xgb_pred = maxsearch.predict(X_ext_val)
    xgb_model = maxsearch
    xgb_pred = xgb_model.predict(X_ext_val)
    return lasso_model,en_model,lr_model,xgb_model,xgb_pred
def run_en_models(numproc,X_train,y_train,X_ext_val,y_ext_val):

    ####ElasticNet
    minscore = 10**6
    l1_list = [.1, .5, .7, .9, .95, .99, 1]
    for l1_val in l1_list:
        for i in range(10):
            X_train,y_train = shuffle(X_train,y_train,random_state=0)
            en_classifier = ElasticNetCV(cv=5,l1_ratio=l1_val,fit_intercept=True,selection='cyclic',random_state=0).fit(X_train,y_train)
            curr_dual = en_classifier.dual_gap_
            if curr_dual < minscore:
                minscore = curr_dual
                best_class = en_classifier
    print("2. ElasticNet results")
    en_pred = best_class.predict(X_ext_val)
    en_model = best_class
    en_pred = en_model.predict(X_ext_val)
    return en_model,en_pred
def run_lasso_models(numproc,X_train,y_train,X_ext_val,y_ext_val):

    niter = 100
    ####LASSO
    minscore = 10**6
    for i in range(niter):
        X_train,y_train = shuffle(X_train,y_train,random_state=0)
        lasso_classifier = LassoCV(cv=5,fit_intercept=True,selection='cyclic',random_state=0).fit(X_train,y_train)
        curr_dual = lasso_classifier.dual_gap_
        if curr_dual < minscore:
            minscore = curr_dual
            best_class = lasso_classifier

    lasso_pred = best_class.predict(X_ext_val)
    lasso_model = best_class
    lasso_pred = lasso_model.predict(X_ext_val)
    return lasso_model,lasso_pred
def run_lr_models(numproc,X_train,y_train,X_ext_val,y_ext_val):

    ####Linear Regression
    reg = LinearRegression().fit(X_train, y_train)
    lr_pred = reg.predict(X_ext_val)
    lr_model = reg
    lr_pred = lr_model.predict(X_ext_val)
    return lr_model,lr_pred
def run_xgb_models(numproc,X_train,y_train,X_ext_val,y_ext_val):
    ####XGBoost
    print("3. XGBoost results")
    xgb_model = XGBRegressor(objective="reg:squarederror", random_state=0)
    params = {
        "colsample_bytree": uniform(0.8, 0.2),
        "gamma": uniform(0, 0.5),
        "learning_rate": uniform(0.02, 0.3), # default 0.1
        "max_depth": randint(2, 6), # default 3
        "n_estimators": randint(100, 150), # default 100
        "subsample": uniform(0.7, 0.3)
    }
    niter = 1
    maxscore = 0
    for i in range(niter):
        search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=0, n_iter=100, cv=5, verbose=1, n_jobs=numproc, return_train_score=True)
        X_train,y_train = shuffle(X_train,y_train,random_state=0)
        with parallel_backend('threading', n_jobs=numproc):
            search.fit(X_train, y_train)

        score = report_best_scores(search.cv_results_, 1)

        if score>maxscore:
            maxscore = score
            maxsearch = search
    bdb_maxsearch = maxsearch
    xgb_pred = maxsearch.predict(X_ext_val)
    xgb_model = maxsearch
    xgb_pred = xgb_model.predict(X_ext_val)

    return xgb_model,xgb_pred

#### LASSO: Output the Pearson's and Spearman correlations, MSE and RMSE, as well as the feature importance values
def lasso_output_metrics(X_ext_val,y_ext_val,lasso_model):
    subdata = []

    lasso_pred = lasso_model.predict(X_ext_val)

    subdata.append(np.corrcoef(lasso_pred,y_ext_val)[0,1])

    spearman_val,pval = spearmanr(lasso_pred,y_ext_val)
    mse = np.square(np.subtract(lasso_pred,y_ext_val)).mean()
    rmse = math.sqrt(mse)
    subdata.append(spearman_val)
    subdata.append(mse)
    subdata.append(rmse)

    lasso_coef = lasso_model.coef_
    lasso_intercept = lasso_model.intercept_
    lasso_importance = [lasso_intercept, lasso_coef]

    return subdata,lasso_pred,lasso_importance

#### ElasticNet: Output the Pearson's and Spearman correlations, MSE and RMSE, as well as the feature importance values
def en_output_metrics(X_ext_val,y_ext_val,en_model):
    subdata = []

    ####ElasticNet
    en_pred = en_model.predict(X_ext_val)

    subdata.append(np.corrcoef(en_pred,y_ext_val)[0,1])
    spearman_val,pval = spearmanr(en_pred,y_ext_val)
    mse = np.square(np.subtract(en_pred,y_ext_val)).mean()
    rmse = math.sqrt(mse)
    subdata.append(spearman_val)
    subdata.append(mse)
    subdata.append(rmse)

    en_coef = en_model.coef_
    en_intercept = en_model.intercept_
    en_importance = [en_intercept, en_coef]
    return subdata,en_pred,en_importance

#### Linear Regression: Output the Pearson's and Spearman correlations, MSE and RMSE, as well as the feature importance values
def lr_output_metrics(X_ext_val,y_ext_val,lr_model):
    subdata = []

    ####Linear Regression

    lr_pred = lr_model.predict(X_ext_val)

    subdata.append(np.corrcoef(lr_pred,y_ext_val)[0,1])
    spearman_val,pval = spearmanr(lr_pred,y_ext_val)
    mse = np.square(np.subtract(lr_pred,y_ext_val)).mean()
    rmse = math.sqrt(mse)
    subdata.append(spearman_val)
    subdata.append(mse)
    subdata.append(rmse)

    lr_coef = lr_model.coef_
    lr_intercept = lr_model.intercept_
    lr_importance = [lr_intercept, lr_coef]

    return subdata,lr_pred,lr_importance

#### XGBoost: Output the Pearson's and Spearman correlations, MSE and RMSE, as well as the feature importance values
def xgb_output_metrics(X_ext_val,y_ext_val,xgb_model):
    subdata = []

    ####XGBoost

    xgb_pred = xgb_model.predict(X_ext_val)
    xgb_importance = xgb_model.best_estimator_.feature_importances_
    xgb_importance = [(i,v) for i, v in enumerate(xgb_importance)]
    subdata.append(np.corrcoef(xgb_pred,y_ext_val)[0,1])
    spearman_val,pval = spearmanr(xgb_pred,y_ext_val)
    mse = np.square(np.subtract(xgb_pred,y_ext_val)).mean()
    rmse = math.sqrt(mse)
    subdata.append(spearman_val)
    subdata.append(mse)
    subdata.append(rmse)

    return subdata,xgb_pred,xgb_importance
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--cutoff', type = float, required=False, default = 100.0)
    parser.add_argument('-f','--filtertype', required=False, default = 'Exptl')
    parser.add_argument('-n','--numproc', type = int, required=False, default = 10)
    parser.add_argument('--pcs', required=False, type = int, default = 22)
    args = parser.parse_args()

    cutoff = args.cutoff
    filtertype = args.filtertype
    numproc = args.numproc
    pcs = args.pcs
    data = {}


    #### Setting up the feature sets for all the Meta-models####################
    feat_docking = ['SMINA score','Vinardo score']
    feat_mw = ['MW'] + feat_docking
    feat_dld = feat_mw + ['CNN_AAC','MPNN_CNN','CNN_CNN','Daylight_CNN','MPNN_AAC','MPNN_Transformer']
    feat_dld_pretrained = feat_mw + ['CNN_CNN','Daylight_AAC','Morgan_AAC','Morgan_CNN','MPNN_CNN','Transformer_CNN']
    score_folder = "/gpfs/slayman/pi/gerstein/pse5/AutoDock/smina/DL_Docking_folder"
    model_dict = {'E':[None,feat_docking,None],'EW':[None,feat_mw,None],'ED1':[f"{score_folder}/prediction.pdbbind2020.refined.regsm.by.dp.pretrained.v015.bdb.kd.merged.csv",feat_dld_pretrained,None],'ED2':[f"{score_folder}/prediction.pdbbind2020.refined.regsm.by.bdb.2020m2.trained.merged.csv",feat_dld,None],'ED1-F':[f"{score_folder}/prediction.full.pdbbind2020.regsm.by.pdbbind2020.refined.regsm.excore.finetuned.trained.100epochs.5fCV.10reps.models.merged.mean.d220806.csv",feat_dld_pretrained,'M1'],'ED2-F':[f"{score_folder}/prediction.full.pdbbind2020.regsm.by.pdbbind2020.refined.regsm.excore.finetuned.trained.100epochs.5fCV.10reps.models.merged.mean.d220806.csv",feat_dld,'M2'],'ED3':[f"{score_folder}/prediction.full.pdbbind2020.regsm.by.pdbbind2020.refined.regsm.excore.finetuned.trained.100epochs.5fCV.10reps.models.merged.mean.d220806.csv",feat_dld,'M3']}
    #'DLPC-MW':[f"{score_folder}/prediction.pdbbind2020.regsm.excore.by.pdbbind2020.refined.regsm.excore.finetuned.trained.100epochs.5fCV.10reps.models.all.pca.no.centering.no.scaling.projection.d220806.csv",['MW'],f"{score_folder}/prediction.pdbbind2020.regsm.excore.by.pdbbind2020.refined.regsm.excore.finetuned.trained.100epochs.5fCV.10reps.models.all.pca.no.centering.no.scaling.projection.coreset.d220806.csv"],
    pc_model_dict = {'ED1-F-P':[f"{score_folder}/prediction.pdbbind2020.regsm.excore.by.bdb.pretrained.v015.pdbbind2020.refined.regsm.excore.finetuned.100epochs.5fCV.10reps.models.all.pca.no.centering.no.scaling.projection.d220726.csv",feat_mw,f"{score_folder}/prediction.pdbbind2020.regsm.excore.by.bdb.pretrained.v015.pdbbind2020.refined.regsm.excore.finetuned.100epochs.5fCV.10reps.models.all.pca.no.centering.no.scaling.projection.coreset.d220726.csv"],'ED2-F-P':[f"{score_folder}/prediction.pdbbind2020.regsm.excore.by.bdb.fulldata.trained.pdbbind2020.refined.regsm.excore.finetuned.100epochs.5fCV.10reps.models.all.pca.no.centering.no.scaling.projection.d220726.csv",feat_mw,f"{score_folder}/prediction.pdbbind2020.regsm.excore.by.bdb.fulldata.trained.pdbbind2020.refined.regsm.excore.finetuned.100epochs.5fCV.10reps.models.all.pca.no.centering.no.scaling.projection.coreset.d220726.csv"],'ED3-P':[f"{score_folder}/prediction.pdbbind2020.regsm.excore.by.pdbbind2020.refined.regsm.excore.trained.100epochs.5fCV.10reps.models.all.pca.no.centering.no.scaling.projection.d220806.csv",feat_mw,f"{score_folder}/prediction.pdbbind2020.regsm.excore.by.pdbbind2020.refined.regsm.excore.trained.100epochs.5fCV.10reps.models.all.pca.no.centering.no.scaling.projection.coreset.d220806.csv"],'ED-A-P':[f"{score_folder}/prediction.pdbbind2020.regsm.excore.by.pdbbind2020.refined.regsm.excore.finetuned.trained.100epochs.5fCV.10reps.models.all.pca.no.centering.no.scaling.projection.d220806.csv",feat_mw,f"{score_folder}/prediction.pdbbind2020.regsm.excore.by.pdbbind2020.refined.regsm.excore.finetuned.trained.100epochs.5fCV.10reps.models.all.pca.no.centering.no.scaling.projection.coreset.d220806.csv"]}

    model_folder = "Trained_Models"
    ############################################################################

    ####Extract the Docking scores and merge with the RMSD information##########
    if filtertype=="Exptl":
        ur_scoredf = pd.read_csv("/home/pse5/RMSD_filt_PDBBind_v2020_Top_binders_Vinardo+SMINA_combined_LogDissoc.txt",sep="\t",header=0)
        rmsd_df = pd.read_csv("/home/pse5/Docking_vs_Exptl_RMSD.txt",sep="\t",header=0)
        rmsd_df = rmsd_df.rename(columns = {"Unnamed: 0":"PDB ID"})
    elif filtertype=="VvS":
        ur_scoredf = pd.read_csv("/home/pse5/VvS_RMSD_filt_PDBBind_v2020_Top_binders_Vinardo+SMINA_combined_LogDissoc.txt",sep="\t",header=0)
        rmsd_df = pd.read_csv("/home/pse5/Total_Vinardo_vs_SMINA_RMSD.txt",sep="\t",header=0)
        rmsd_df = rmsd_df.rename(columns = {"Unnamed: 0.1":"PDB ID"})
    ur_scoredf = ur_scoredf[~ur_scoredf["Ligand Name"].str.contains('|'.join(["&","/"]))]
    molw = pd.read_csv("/gpfs/slayman/pi/gerstein/pse5/AutoDock/PDBbind/Molecular_wts.txt",sep="\t",header=None)
    molw.columns = ["PDB ID","MW"]
    ur_scoredf = ur_scoredf.merge(molw,how='left',on='PDB ID')
    ############################################################################

    ####Apply the RMSD filters to the data #####################################
    rmsd_df = rmsd_df[(rmsd_df["Vinardo_RMSD"] <cutoff) & (rmsd_df["SMINA_RMSD"] <cutoff)]
    ur_scoredf = ur_scoredf.merge(rmsd_df,how='left',on='PDB ID')
    ur_scoredf = ur_scoredf.dropna()
    ############################################################################

    ####Read in the CoreSet data (to help in the creation of the leave-out test set) #
    coreset = pd.read_csv("/home/pse5/CoreSet.dat",header=0,delim_whitespace=True)
    coreset = coreset.rename(columns={"#code":"PDB ID"})
    corepdb = coreset["PDB ID"].tolist()

    dockfilename = f"/gpfs/slayman/pi/gerstein/pse5/AutoDock/smina/{filtertype}_{cutoff}_Docking_metrics.txt"

    core_score = ur_scoredf[ur_scoredf['PDB ID'].isin(corepdb)]
    ############################################################################



    outfile = open(f"{filtertype}_{cutoff}_All_feature_importance_scores.tsv",'w')

    ####Loop over the meta-models that do not require Principal component hyperparameter optimization #
    for model, val in model_dict.items():
        dl_file = val[0]
        features = val[1]
        filters = val[2]
        scoredf = deepcopy(ur_scoredf)
        print(f"Currently processing: {model}")
        print(f"Features: {features}")
        print(f"Filters: {filters}")

        if dl_file is not None:
            deepdf = pd.read_csv(dl_file,sep=",",header=0)

            if 'Target' in deepdf.columns:
                deepdf[['PDB ID','Chain']] = deepdf.Target.str.split("_",expand=True)
            else:
                deepdf = deepdf.rename(columns = {"Unnamed: 0":"Target"})
                deepdf[['PDB ID','Ligand']] = deepdf.Target.str.split("|",expand=True)

            if filters is not None:
                deepdf = pd.concat([deepdf[['PDB ID','Ligand']],deepdf.filter(like=f'|{filters}')],axis=1)

            dl_features = features[-6:]
            for feat in dl_features:
                deepdf[feat] = deepdf.filter(like=feat)
            scoredf = scoredf.merge(deepdf,how='left',on='PDB ID')

        core_score = scoredf[scoredf['PDB ID'].isin(corepdb)]
        core_score = core_score.merge(coreset,how='left',on='PDB ID')
        scoredf = scoredf.dropna()
        scoredf = scoredf.set_index('Ligand Name')

        ####Docking metrics for the full set
        if model == 'E':
            docking_metrics(scoredf,corepdb,core_score,dockfilename)
        ############################################################################

        scoredf = scoredf[~scoredf['PDB ID'].isin(corepdb)]

        X = scoredf.loc[:,features].to_numpy()
        y = scoredf.loc[:,'Log Dissociation'].to_numpy().flatten()
        X_ext_val = core_score.loc[:,features].to_numpy()
        y_ext_val = core_score.loc[:,'Log Dissociation'].to_numpy().flatten()

        X_train = X
        y_train = y

        #### Run ML Models if the corresponding XGBoost model doesn't already exist #
        if not os.path.exists(f"{model_folder}/{model}_{filtertype}_{cutoff}_XGBoost.joblib"):
            lasso_model,en_model,lr_model,xgb_model,xgb_pred = run_ml_models(numproc,X_train,y_train,X_ext_val,y_ext_val)
        ############################################################################
            dump(lasso_model,f"{model_folder}/{model}_{filtertype}_{cutoff}_LASSO.joblib")
            dump(en_model,f"{model_folder}/{model}_{filtertype}_{cutoff}_ElasticNet.joblib")
            dump(lr_model,f"{model_folder}/{model}_{filtertype}_{cutoff}_LinReg.joblib")
            dump(xgb_model,f"{model_folder}/{model}_{filtertype}_{cutoff}_XGBoost.joblib")


        ############################################################################

        subdata = []

        #### Step 1: Get the output metrics for the CoreSet (test predictions) for each ML model
        #### Step 2: Get the output metrics for the RefinedSet without the Coreset (training predictions) for each ML model

        ####LASSO
        lasso_model = load(f"{model_folder}/{model}_{filtertype}_{cutoff}_LASSO.joblib")
        lasso_subdata,lasso_pred,lasso_importance = lasso_output_metrics(X_ext_val,y_ext_val,lasso_model)
        subdata.extend(lasso_subdata)
        outfile.write(f"{model}\tLASSO\t{features}\t{list(map(str,lasso_importance))}\n")
        core_score['LASSO'] = lasso_pred.tolist()
        lasso_subdata,lasso_pred,lasso_importance = lasso_output_metrics(X,y,lasso_model)
        scoredf['LASSO'] = lasso_pred.tolist()


        ####ElasticNet
        en_model = load(f"{model_folder}/{model}_{filtertype}_{cutoff}_ElasticNet.joblib")
        en_subdata,en_pred,en_importance = en_output_metrics(X_ext_val,y_ext_val,en_model)
        subdata.extend(en_subdata)
        outfile.write(f"{model}\tElasticNet\t{features}\t{list(map(str,en_importance))}\n")
        core_score['ElasticNet'] = en_pred.tolist()
        en_subdata,en_pred,en_importance = en_output_metrics(X,y,en_model)
        scoredf['ElasticNet'] = en_pred.tolist()

        ####Linear Regression
        lr_model = load(f"{model_folder}/{model}_{filtertype}_{cutoff}_LinReg.joblib")
        lr_subdata,lr_pred,lr_importance = lr_output_metrics(X_ext_val,y_ext_val,lr_model)
        subdata.extend(lr_subdata)
        outfile.write(f"{model}\tLinReg\t{features}\t{list(map(str,lr_importance))}\n")
        core_score['LinReg'] = lr_pred.tolist()
        lr_subdata,lr_pred,lr_importance = lr_output_metrics(X,y,lr_model)
        scoredf['LinReg'] = lr_pred.tolist()

        ####XGBoost
        xgb_model = load(f"{model_folder}/{model}_{filtertype}_{cutoff}_XGBoost.joblib")
        xgb_subdata,xgb_pred,xgb_importance = xgb_output_metrics(X_ext_val,y_ext_val,xgb_model)
        subdata.extend(xgb_subdata)
        outfile.write(f"{model}\tXGBoost\t{features}\t{list(map(str,xgb_importance))}\n")
        core_score['XGBoost'] = xgb_pred.tolist()
        xgb_subdata,xgb_pred,xgb_importance = xgb_output_metrics(X,y,xgb_model)
        scoredf['XGBoost'] = xgb_pred.tolist()

        core_score.to_csv(f"Predictions/{model}_{filtertype}_{cutoff}_coreset_predictions.txt",sep="\t")
        data[model] = subdata

        scoredf.to_csv(f"Predictions/{model}_{filtertype}_{cutoff}_trainset_predictions.txt",sep="\t")

    ############################################################################

    ####Loop over the meta-models that require Principal component hyperparameter optimization #
    for model, val in pc_model_dict.items():
        dl_file = val[0]
        core_file = val[2]
        print(f"Currently processing: {model}")
        print(f"Features: {val[1]}")
        subdata = []
        scoredf = deepcopy(ur_scoredf)
        deepdf = pd.read_csv(dl_file,sep=",",header=0)
        deepdf = deepdf.rename(columns = {"Unnamed: 0":"Target"})
        corepred = pd.read_csv(core_file,sep=",",header=0)
        corepred = corepred.rename(columns = {"Unnamed: 0":"Target"})
        deepdf = pd.concat([deepdf,corepred])

        deepdf[['PDB ID','Chain']] = deepdf.Target.str.split("|",expand=True)
        scoredf = scoredf.merge(deepdf,how='left',on='PDB ID')

        core_score = scoredf[scoredf['PDB ID'].isin(corepdb)]
        core_score = core_score.merge(coreset,how='left',on='PDB ID')
        scoredf = scoredf.dropna()
        scoredf = scoredf.set_index('Ligand Name')
        scoredf = scoredf[~scoredf['PDB ID'].isin(corepdb)]

        #### Run ML Models if none of the ML models corresponding to the Meta-model configuration exist #
        outputfile = glob.glob(f"{model_folder}/{model}_{filtertype}_{cutoff}_*.joblib")
        if len(outputfile) == 0:
            max_xgb_score = 0.0
            max_lasso_score = 0.0
            max_en_score = 0.0
            max_lr_score = 0.0

            #### Optimize over the number of PCs, separately for each of he 4 ML models
            for ipc in range(1,pcs+1,1):

                PClist = [f"PC{i}" for i in range(1,ipc+1,1)]

                features = val[1] + PClist

                X = scoredf.loc[:,features].to_numpy()
                y = scoredf.loc[:,'Log Dissociation'].to_numpy().flatten()
                X_ext_val = core_score.loc[:,features].to_numpy()
                y_ext_val = core_score.loc[:,'Log Dissociation'].to_numpy().flatten()

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
                ############################################################################

                #### Separate functions for each of the ML models here, due to the need for independent hyperparameter optimization
                lasso_model,lasso_pred = run_lasso_models(numproc,X_train,y_train,X_test,y_test)
                en_model,en_pred = run_en_models(numproc,X_train,y_train,X_test,y_test)
                lr_model,lr_pred = run_lr_models(numproc,X_train,y_train,X_test,y_test)
                xgb_model,xgb_pred = run_xgb_models(numproc,X_train,y_train,X_test,y_test)

                lasso_score = np.corrcoef(lasso_pred,y_test)[0,1]
                en_score = np.corrcoef(en_pred,y_test)[0,1]
                lr_score = np.corrcoef(lr_pred,y_test)[0,1]
                xgb_score = np.corrcoef(xgb_pred,y_test)[0,1]
                if xgb_score > max_xgb_score:
                    max_xgb_score = xgb_score
                    best_xgb_pcs = ipc
                    best_xgb_model = xgb_model
                    best_xgb_features = features
                if lasso_score > max_lasso_score:
                    max_lasso_score = lasso_score
                    best_lasso_pcs = ipc
                    best_lasso_features = features
                    best_lasso_model = lasso_model
                if en_score > max_en_score:
                    max_en_score = en_score
                    best_en_pcs = ipc
                    best_en_features = features
                    best_en_model = en_model
                if lr_score > max_lr_score:
                    max_lr_score = lr_score
                    best_lr_pcs = ipc
                    best_lr_features = features
                    best_lr_model = lr_model

            dump(best_lasso_model,f"{model_folder}/{model}_{filtertype}_{cutoff}_PCs{best_lasso_pcs}_LASSO.joblib")
            dump(best_en_model,f"{model_folder}/{model}_{filtertype}_{cutoff}_PCs{best_en_pcs}_ElasticNet.joblib")
            dump(best_lr_model,f"{model_folder}/{model}_{filtertype}_{cutoff}_PCs{best_lr_pcs}_LinReg.joblib")
            dump(best_xgb_model,f"{model_folder}/{model}_{filtertype}_{cutoff}_PCs{best_xgb_pcs}_XGBoost.joblib")

        outputfile = glob.glob(f"{model_folder}/{model}_{filtertype}_{cutoff}_*_XGBoost.joblib")
        xgbfile = os.path.basename(outputfile[0])
        best_xgb_pcs = xgbfile.split("_")[3].replace("PCs","")

        outputfile = glob.glob(f"{model_folder}/{model}_{filtertype}_{cutoff}_*_LASSO.joblib")
        lassofile = os.path.basename(outputfile[0])
        best_lasso_pcs = lassofile.split("_")[3].replace("PCs","")

        outputfile = glob.glob(f"{model_folder}/{model}_{filtertype}_{cutoff}_*_ElasticNet.joblib")
        enfile = os.path.basename(outputfile[0])
        best_en_pcs = enfile.split("_")[3].replace("PCs","")

        outputfile = glob.glob(f"{model_folder}/{model}_{filtertype}_{cutoff}_*_LinReg.joblib")
        lrfile = os.path.basename(outputfile[0])
        best_lr_pcs = lrfile.split("_")[3].replace("PCs","")

        lasso_PClist = [f"PC{i}" for i in range(1,int(best_lasso_pcs)+1,1)]
        lasso_features = val[1] + lasso_PClist

        en_PClist = [f"PC{i}" for i in range(1,int(best_en_pcs)+1,1)]
        en_features = val[1] + en_PClist

        lr_PClist = [f"PC{i}" for i in range(1,int(best_lr_pcs)+1,1)]
        lr_features = val[1] + lr_PClist

        xgb_PClist = [f"PC{i}" for i in range(1,int(best_xgb_pcs)+1,1)]
        xgb_features = val[1] + xgb_PClist

        subdata = []

        #### Output metrics for the CoreSet (test) and RefinedSet without the CoreSet (training)
        ####LASSO
        X = scoredf.loc[:,lasso_features].to_numpy()
        y = scoredf.loc[:,'Log Dissociation'].to_numpy().flatten()
        X_ext_val = core_score.loc[:,lasso_features].to_numpy()
        y_ext_val = core_score.loc[:,'Log Dissociation'].to_numpy().flatten()
        lasso_model = load(f"{model_folder}/{model}_{filtertype}_{cutoff}_PCs{best_lasso_pcs}_LASSO.joblib")
        lasso_subdata,lasso_pred,lasso_importance = lasso_output_metrics(X_ext_val,y_ext_val,lasso_model)
        subdata.extend(lasso_subdata)
        outfile.write(f"{model}\tLASSO\t{lasso_features}\t{list(map(str,lasso_importance))}\n")
        core_score['LASSO'] = lasso_pred.tolist()
        lasso_subdata,lasso_pred,lasso_importance = lasso_output_metrics(X,y,lasso_model)
        scoredf['LASSO'] = lasso_pred.tolist()


        ####ElasticNet
        X = scoredf.loc[:,en_features].to_numpy()
        y = scoredf.loc[:,'Log Dissociation'].to_numpy().flatten()
        X_ext_val = core_score.loc[:,en_features].to_numpy()
        y_ext_val = core_score.loc[:,'Log Dissociation'].to_numpy().flatten()
        en_model = load(f"{model_folder}/{model}_{filtertype}_{cutoff}_PCs{best_en_pcs}_ElasticNet.joblib")
        en_subdata,en_pred,en_importance = en_output_metrics(X_ext_val,y_ext_val,en_model)
        subdata.extend(en_subdata)
        outfile.write(f"{model}\tElasticNet\t{en_features}\t{list(map(str,en_importance))}\n")
        core_score['ElasticNet'] = en_pred.tolist()
        en_subdata,en_pred,en_importance = en_output_metrics(X,y,en_model)
        scoredf['ElasticNet'] = en_pred.tolist()
        ####Linear Regression
        X = scoredf.loc[:,lr_features].to_numpy()
        y = scoredf.loc[:,'Log Dissociation'].to_numpy().flatten()
        X_ext_val = core_score.loc[:,lr_features].to_numpy()
        y_ext_val = core_score.loc[:,'Log Dissociation'].to_numpy().flatten()
        lr_model = load(f"{model_folder}/{model}_{filtertype}_{cutoff}_PCs{best_lr_pcs}_LinReg.joblib")
        lr_subdata,lr_pred,lr_importance = lr_output_metrics(X_ext_val,y_ext_val,lr_model)
        subdata.extend(lr_subdata)
        outfile.write(f"{model}\tLinReg\t{lr_features}\t{list(map(str,lr_importance))}\n")
        core_score['LinReg'] = lr_pred.tolist()
        lr_subdata,lr_pred,lr_importance = lr_output_metrics(X,y,lr_model)
        scoredf['LinReg'] = lr_pred.tolist()
        ####XGBoost
        X = scoredf.loc[:,xgb_features].to_numpy()
        y = scoredf.loc[:,'Log Dissociation'].to_numpy().flatten()
        X_ext_val = core_score.loc[:,xgb_features].to_numpy()
        y_ext_val = core_score.loc[:,'Log Dissociation'].to_numpy().flatten()
        xgb_model = load(f"{model_folder}/{model}_{filtertype}_{cutoff}_PCs{best_xgb_pcs}_XGBoost.joblib")
        xgb_subdata,xgb_pred,xgb_importance = xgb_output_metrics(X_ext_val,y_ext_val,xgb_model)
        subdata.extend(xgb_subdata)
        outfile.write(f"{model}\tXGBoost\t{xgb_features}\t{list(map(str,xgb_importance))}\n")
        core_score['XGBoost'] = xgb_pred.tolist()
        xgb_subdata,xgb_pred,xgb_importance = xgb_output_metrics(X,y,xgb_model)
        scoredf['XGBoost'] = xgb_pred.tolist()

        core_score.to_csv(f"Predictions/{model}_{filtertype}_{cutoff}_coreset_predictions.txt",sep="\t")
        data[model] = subdata

        scoredf.to_csv(f"Predictions/{model}_{filtertype}_{cutoff}_trainset_predictions.txt",sep="\t")
        #############################################################################
    outfile.close()
    data_df = pd.DataFrame.from_dict(data, orient='index',columns = ['LASSO_Pearson','LASSO_Spearman','LASSO_MSE','LASSO_RMSE','ElasticNet_Pearson','ElasticNet_Spearman','ElasticNet_MSE','ElasticNet_RMSE','LinearRegression_Pearson','LinearRegression_Spearman','LinearRegression_MSE','LinearRegression_RMSE','XGBoost_Pearson','XGBoost_Spearman','XGBoost_MSE','XGBoost_RMSE'])

    #### Compile the scores across all the models
    data_df.to_csv(f"/gpfs/slayman/pi/gerstein/pse5/AutoDock/smina/All_models_{filtertype}_{cutoff}_final_scores.txt",sep="\t")
