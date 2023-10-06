import sys
import os
import subprocess
import numpy as np
import pandas as pd
import math
import time
import random
from copy import deepcopy
import argparse
import multiprocessing as mp
from functools import partial
from collections import Counter
import gzip
import re
from pymol import cmd
from Bio.PDB import *
import glob
from rdkit import Chem
from rdkit.Chem import AllChem

def parse_mw(group_folder,complex_ID):
    with open(f"{group_folder+complex_ID}/{complex_ID}_ligand_properties.txt",'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            terms = line.strip().split()
            if terms[0] == "mol_weight":
                mw = terms[1]
                break

    return mw
def calculate_mw(group_folder,complex_ID):
    obcall = f"obprop {group_folder+complex_ID}/{complex_ID}_ligand.sdf > {group_folder+complex_ID}/{complex_ID}_ligand_properties.txt"
    subprocess.call(obcall,shell=True)
    return
def create_config(group_folder,complex_ID):

    cmd.load(f"{group_folder+complex_ID}/{complex_ID}_ligand.sdf")
    coords = cmd.centerofmass()
    cmd.reinitialize()

    config_file = f"{group_folder+complex_ID}/{complex_ID}_config.txt"
    outfile = open(config_file,'w')
    outfile.write(f"receptor = {group_folder+complex_ID}/{complex_ID}_protein.pdbqt\nligand = {group_folder+complex_ID}/{complex_ID}_ligand.sdf\ncenter_x = {coords[0]}\ncenter_y = {coords[1]}\ncenter_z = {coords[2]}\nsize_x = 40\nsize_y = 40\nsize_z = 40")
    outfile.close()
    return config_file
def create_receptor_pdbqt(group_folder,complex_ID):

    cmd.load(f"{group_folder+complex_ID}/{complex_ID}_protein.pdb")
    cmd.remove("solvent")
    cmd.remove("hetatm")
    cmd.save(f"{group_folder+complex_ID}/{complex_ID}_nosolv_protein.pdb")
    cmd.reinitialize()
    obabelcall = f"obabel -ipdb {group_folder+complex_ID}/{complex_ID}_nosolv_protein.pdb -opdbqt -O {group_folder+complex_ID}/{complex_ID}_protein.pdbqt -p 7.0 -e -xrp --partialcharge gasteiger"
    subprocess.call(obabelcall,shell=True)

    return
def run_smina(group_folder, outfolder, indexdf, config_file, suffix, complex_ID):
    ligand_sdf = f"{group_folder+complex_ID}/{complex_ID}_ligand.sdf"
    ligandname = indexdf[indexdf['PDB ID']==complex_ID]['Ligand Name'].tolist()[0]
    complex_pdb = f"{outfolder+complex_ID}/{complex_ID}_ligand_{ligandname}.pdb"
    if os.path.exists(ligand_sdf) and not os.path.exists(complex_pdb):
        activatecall = "source activate local"
        subprocess.call(activatecall,shell=True)
        complex_pdbqt = f"{outfolder+complex_ID}_ligand_{ligandname}.pdbqt"
        logfile = f"{outfolder}Logfile_{complex_ID}_ligand_{ligandname}.txt"

        if suffix == "SMINA":
            sminacall = f"./smina --config {config_file} --out {complex_pdbqt} --log {logfile} --exhaustiveness 32"
            subprocess.call(sminacall,shell=True)
        if suffix == "Vinardo":
            sminacall = f"./smina --config {config_file} --out {complex_pdbqt} --log {logfile} --exhaustiveness 32 --scoring vinardo"
            subprocess.call(sminacall,shell=True)

        complex_pdb = f"{outfolder+complex_ID}_ligand_{ligandname}.pdb"
        cutcall = f"cut -c-66 {complex_pdbqt} > {complex_pdb}"
        subprocess.call(cutcall,shell=True)
        final_pdb = f"{outfolder}Complex_{complex_ID}_ligand_{ligandname}.pdb"
        catcall = f"cat {group_folder+complex_ID}/{complex_ID}_protein.pdb {complex_pdb} | grep -v '^END   ' | grep -v '^END$' > {final_pdb}"
        subprocess.call(catcall,shell=True)
    return
def parse_vina_log(outfolder, indexdf, complex_ID):
    ligandname = indexdf[indexdf['PDB ID']==complex_ID]['Ligand Name'].tolist()[0]
    logfile = f"{outfolder}Logfile_{complex_ID}_ligand_{ligandname}.txt"
    binding = 0
    if os.path.exists(logfile):
        with open(logfile,'r') as infile:
            for line in infile:
                pattern = re.compile('0.000      0.000')
                match = re.search(pattern,line)
                if match:
                    binding = float(line.strip().split()[1])
    return (complex_ID,ligandname,binding)
def parse_vina_log_rmsd_filt(suffix, rmsd_df, outfolder, indexdf, complex_ID):
    ligandname = indexdf[indexdf['PDB ID']==complex_ID]['Ligand Name'].tolist()[0]
    logfile = f"{outfolder}Logfile_{complex_ID}_ligand_{ligandname}.txt"
    binding = 0
    if os.path.exists(logfile):

        if suffix == "Vinardo":
            log_df = pd.read_csv(logfile,delim_whitespace=True,header=None,skiprows=24)
        if suffix == "SMINA":
            log_df = pd.read_csv(logfile,delim_whitespace=True,header=None,skiprows=25)
        log_df.columns = ["Pose","Affinity","rmsd1","rmsd2"]
        log_dict = dict(zip(log_df["Pose"],log_df["Affinity"]))

        rmsd_subset = rmsd_df[rmsd_df['PDB ID']==complex_ID]
        print(rmsd_subset)
        pose = rmsd_subset[f"{suffix}_BestPose"].tolist()[0]
        binding = log_dict[pose+1]
    return (complex_ID,ligandname,binding)
def run_docking(group_folder,outfolder,indexdf,suffix,complex_ID):
    config_file = create_config(group_folder,complex_ID)
    create_receptor_pdbqt(group_folder,complex_ID)
    run_smina(group_folder, outfolder, indexdf, config_file, suffix, complex_ID)

def ad_rmsd(pdb1,pdb2):
    cmd.reinitialize()
    cmd.load(pdb1,"mol1")
    cmd.load(pdb2,"mol2")
    cmd.create("mol1_heavy",selection="mol1 and not elem H")
    cmd.create("mol2_heavy",selection="mol2 and not elem H")
    rmsd12 = 0
    for at1 in cmd.index("mol1_heavy"):
        temp = []
        cmd.select(name="sele2",selection=f'mol2_heavy like (mol1_heavy and index {at1[1]})')
        for at2 in cmd.index("sele2"):

            try:
                rij = cmd.distance(None, f"mol1_heavy and index {at1[1]}", f"mol2_heavy and index {at2[1]}")
            except:
                print("%s`%d"%at1)
                print("%s`%d"%at2)
            temp.append(rij*rij)
        rmsd12 += np.min(np.array(temp))
    rmsd12 /= len(cmd.index("mol1_heavy"))
    rmsd12 = np.sqrt(rmsd12)

    rmsd21 = 0
    for at1 in cmd.index("mol2_heavy"):
        temp = []
        cmd.select(name="sele1",selection=f'mol1_heavy like (mol2_heavy and index {at1[1]})')
        for at2 in cmd.index("sele1"):

            try:
                rij = cmd.distance(None,f"mol2_heavy and index {at1[1]}", f"mol1_heavy and index {at2[1]}")
            except:
                print("%s`%d"%at1)
                print("%s`%d"%at2)

            temp.append(rij*rij)
        rmsd21 += np.min(np.array(temp))
    rmsd21 /= len(cmd.index("mol2_heavy"))
    rmsd21 = np.sqrt(rmsd21)

    rmsd = max(rmsd12,rmsd21)
    cmd.reinitialize()
    return rmsd

def compare_poses(group_folder,outfolderlist,indexdf,complex_ID):
    ligandname = indexdf[indexdf['PDB ID']==complex_ID]['Ligand Name'].tolist()[0]
    npairs = int(len(outfolderlist)*(len(outfolderlist)-1)/2)
    if("&" in ligandname) or ("/" in ligandname):
        return [20]*npairs
    rmsdlist = []
    for i,outfolder1 in enumerate(outfolderlist):
        complex_pdb_1 = f"{outfolder1+complex_ID}_ligand_{ligandname}.pdb"

        endmdl = "'/^ENDMDL/+1' '{'$b'}'"
        folder1_call = f"a=`grep ENDMDL {complex_pdb_1} | wc -l`;b=`expr $a - 1`;csplit -z -k -s -n 3 -b '%03d.pdb' -f {complex_pdb_1.split('.')[0]}. {complex_pdb_1} {endmdl}"
        
        subprocess.call(folder1_call,shell=True)
        for j in range(i+1,len(outfolderlist)):
            complex_pdb_2 = f"{outfolderlist[j]+complex_ID}_ligand_{ligandname}.pdb"
            folder2_call = f"a=`grep ENDMDL {complex_pdb_2} | wc -l`;b=`expr $a - 1`;csplit -z -k -s -n 3 -b '%03d.pdb' -f {complex_pdb_2.split('.')[0]}. {complex_pdb_2} {endmdl}"

            subprocess.call(folder2_call,shell=True)
            rmsd = ad_rmsd(f"{complex_pdb_1.split('.')[0]}.000.pdb",f"{complex_pdb_2.split('.')[0]}.000.pdb")
            print(rmsd)
            rmsdlist.append(rmsd)
            rmcall=f"rm {complex_pdb_2.split('.')[0]}.0*"
            subprocess.call(rmcall,shell=True)
        rmcall=f"rm {complex_pdb_1.split('.')[0]}.0*"
        subprocess.call(rmcall,shell=True)
    return rmsdlist
def compare_poses_filter(group_folder,outfolderlist,indexdf,complex_ID):
    ligandname = indexdf[indexdf['PDB ID']==complex_ID]['Ligand Name'].tolist()[0]
    npairs = int(len(outfolderlist)*(len(outfolderlist)-1)/2)
    if("&" in ligandname) or ("/" in ligandname):
        return [(0,0,100.0)]*npairs*81
    rmsdlist = []
    for i,outfolder1 in enumerate(outfolderlist):
        complex_pdb_1 = f"{outfolder1+complex_ID}_ligand_{ligandname}.pdb"


        endmdl = "'/^ENDMDL/+1' '{'$b'}'"
        folder1_call = f"a=`grep ENDMDL {complex_pdb_1} | wc -l`;b=`expr $a - 1`;csplit -z -k -s -n 3 -b '%03d.pdb' -f {complex_pdb_1.split('.')[0]}. {complex_pdb_1} {endmdl}"

        subprocess.call(folder1_call,shell=True)
        molfiles1 = glob.glob(f"{complex_pdb_1.split('.')[0]}.0*.pdb")

        for j in range(i+1,len(outfolderlist)):
            complex_pdb_2 = f"{outfolderlist[j]+complex_ID}_ligand_{ligandname}.pdb"
            folder2_call = f"a=`grep ENDMDL {complex_pdb_2} | wc -l`;b=`expr $a - 1`;csplit -z -k -s -n 3 -b '%03d.pdb' -f {complex_pdb_2.split('.')[0]}. {complex_pdb_2} {endmdl}"

            subprocess.call(folder2_call,shell=True)
            molfiles2 = glob.glob(f"{complex_pdb_2.split('.')[0]}.0*.pdb")

            for molfile1 in molfiles1:
                mol_id1 = int(molfile1.split('.')[-2])
                for molfile2 in molfiles2:
                    mol_id2 = int(molfile2.split('.')[-2])
                    rmsd = ad_rmsd(molfile1,molfile2)
                    rmsdlist.append((mol_id1,mol_id2,rmsd))

            rmcall=f"rm {complex_pdb_2.split('.')[0]}.0*"
            subprocess.call(rmcall,shell=True)
        rmcall=f"rm {complex_pdb_1.split('.')[0]}.0*"
        subprocess.call(rmcall,shell=True)
    return rmsdlist
if __name__=="__main__":
    numproc = int(os.getenv("SLURM_CPUS_ON_NODE"))
    parser = argparse.ArgumentParser(description='Docking PDBBind complexes with SMINA')
    parser.add_argument('-b','--batchnum', required=True, type = int, help='Batch number')
    parser.add_argument('-d','--dataset', required=True, help='Dataset (general/refined)')
    args = parser.parse_args()
    batchnum = args.batchnum
    dataset = args.dataset
    if dataset == "refined":
        indexfile = "refined-set/index/INDEX_refined_data.2020"
        group_folder = "refined-set/"


        indexdf = pd.read_csv(indexfile, delim_whitespace=True,header=None,skiprows=6)

        indexdf.columns = ['PDB ID','Resolution','Year of Release','-logKd/Ki','Kd or Ki', 'Spacer','PDF file','Annotation']

        indexdf[['Dissociation Type','Dissociation/Inhibition Rate']] = indexdf['Kd or Ki'].str.split("=",expand=True)
        indexdf['Ligand Name'] = indexdf['Annotation'].str.extract(r'\((.*)\)')
        print(indexdf)
    elif dataset == "general":
        group_folder = "PDBbind/GeneralSet_v2020/"
        indexdf = pd.read_csv("PDBbind/Selected_general_set.tsv",sep="\t",header=0)


    outfolderlist = ["Results_v2020_Vinardo/","Results_v2020_SMINA/"]
    suffixlist = ["Vinardo","SMINA"]
    for outfolder in outfolderlist:
        if not os.path.exists(outfolder):
           os.mkdir(outfolder)
    folder_names = indexdf['PDB ID'].tolist()
    numbatch = 10
    batch_folder_names = [name for i, name in enumerate(folder_names) if i%numbatch==batchnum]
    result = [0]*len(outfolderlist)
    for i,outfolder in enumerate(outfolderlist):
        pool = mp.Pool(processes=numproc)
        fix_dock=partial(run_docking,group_folder,outfolder,indexdf,suffixlist[i])
        result = pool.map(fix_dock, batch_folder_names)


        pool.close()
        pool.join()
