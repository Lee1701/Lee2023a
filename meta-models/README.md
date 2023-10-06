# Meta-models
Here we provide the Python code used to train the meta-models, as well as all the trained meta-models considered in the paper.
1. **Meta-model_training_pipeline.py** 
 
    The code is hard-coded with the names of the requisite PDBbind-based data files needed to run the training, as well as the feature sets considered for each model. The following aspects are included:
    - Training of the four machine-learning models - LASSO, Linear Regression, ElasticNet and XGBoost
    - Hyperparameter optimization over the number of Principal Components (PCs) in the relevant models
    - The type of filter and RMSD restriction applied to the empirical docking scores are input to the code, and models are generated with these input parameters in the title (along with the optimal PC hyperparameters, if relevant).
2. **Trained_Models**

    This folder contains all the .joblib files (see [joblib usage](https://joblib.readthedocs.io/en/latest/generated/joblib.load.html) for more information) corresponding to the meta-models trained on the PDBbind as used in our study. The model abbreviations and their corresponding descriptions can be found in our paper.

