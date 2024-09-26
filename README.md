# Improved Prediction Of Ligand-Protein Binding Affinities By Meta-Modeling
<p align="center">
  <img src="https://github.com/Lee1701/Lee2023a/blob/main/images/Fig1.revised.png" width="600">
</p>

- This repository contains scripts, codes, and data for the following study by Lee, Emani, and Gerstein:
<https://arxiv.org/abs/2310.03946>

- The scripts and codes are shared without optimization in this repository, which may include analyses and results that were not reported in the study above.

## File description
1. `README.md`
  : This current page
2. `SupplementaryTables.revision2.xlsx`
  : Supplementary Tables S1 to S11 associated with our article (v3) above
3. `data`
  : Data for model training and evaluation
4. `docking`
  : Empirical docking tools
5. `deep learning`
  : Deep learning models
6. `meta-models`
  : Meta-models
7. `LICENSE`
  : GNU General Public License v3.0

## Overall workflow
1. **Docking tools:** The docking tools, SMINA and Vinardo, have to be run on complexes with 3D structures. The code provided in the `empirical docking` folder provides Python-based tools to convert a file into the 3D SDF format required to run `smina`. Subsequently, code is provided therein to run the docking tools and output a docked complex. The log files can then be parsed to obtain the lowest-energy binding affinities predicted, which are part of meta-features for the meta-models.
2. **Deep learning models:** Our deep learning models are based on the DeepPurpose library in Python. The input data are ligand SMILES and protein amino acid sequences. We developed 4 families of de novo-trained and fine-tuned models using BindingDB and PDBbind. Together with pre-trained models from DeepPurpose, we built up to 1,100 model variants from cross validations. Predicted binding affinities are part of meta-features for the meta-models with or without dimensionality reduction by PCA.
3. **Molecular weight:** The molecular weights of the ligands may be used as a meta-feature for the meta-models. One way to programmatically extract them is to use `OpenBabel`'s `obprop` function.
4. **Meta-models:** The code for the meta-model prediction task is provided in the `meta-models` folder. It takes a folder containing deep-learning predictions and a spreadsheet containing the docking tool scores and molecular weights as inputs.

## Contacts
- Ho-Joon Lee, Ph.D.: **ho-joon.lee[_at_]yale.edu**
- Prashant Emani, Ph.D.: **prashant.emani[_at_]yale.edu**

## License
Released under the GNU General Public License v3.0. See LICENSE.
