# Empirical Docking
Here we provide the Python code used to predict the empirical docking scores, based on the [PDBbind](http://www.pdbbind.org.cn/) complexes.

**PDBbind_docking_analyses.py** 
 
 The code does the following:
 - Pre-processes the ligand sdf file 
 - Generates a PDBQT file for the receptor based on the PDB file using [OpenBabel](http://openbabel.org/wiki/Main_Page)
 - Runs the docking software [smina](https://sourceforge.net/projects/smina/) with either of the two docking score functions: SMINA, Vinardo
 - Filters the resulting scores based on consensus or experimental RMSD filters
