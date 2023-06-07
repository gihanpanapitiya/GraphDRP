import os
import csv
from pubchempy import *
import numpy as np
import numbers
import h5py
import math
import pandas as pd
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *
import random
import pickle
import sys
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import train_test_split
import improve_utils
import urllib


def is_not_float(string_list):
    try:
        for string in string_list:
            float(string)
        return False
    except:
        return True

"""
The following 4 function is used to preprocess the drug data. We download the drug list manually, 
and download the SMILES format using pubchempy. Since this part is time consuming, I write the cids and SMILES into a csv file. 
"""

folder = "data/"
#folder = ""

def load_drug_list():
    filename = folder + "Druglist.csv"
    csvfile = open(filename, "rb")
    reader = csv.reader(csvfile)
    next(reader, None)
    drugs = []
    for line in reader:
        drugs.append(line[0])
    drugs = list(set(drugs))
    return drugs

def write_drug_cid():
    drugs = load_drug_list()
    drug_id = []
    datas = []
    outputfile = open(folder + 'pychem_cid.csv', 'wb')
    wr = csv.writer(outputfile)
    unknow_drug = []
    for drug in drugs:
        c = get_compounds(drug, 'name')
        if drug.isdigit():
            cid = int(drug)
        elif len(c) == 0:
            unknow_drug.append(drug)
            continue
        else:
            cid = c[0].cid
        print(drug, cid)
        drug_id.append(cid)
        row = [drug, str(cid)]
        wr.writerow(row)
    outputfile.close()
    outputfile = open(folder + "unknow_drug_by_pychem.csv", 'wb')
    wr = csv.writer(outputfile)
    wr.writerow(unknow_drug)

def cid_from_other_source():
    """
    some drug can not be found in pychem, so I try to find some cid manually.
    the small_molecule.csv is downloaded from http://lincs.hms.harvard.edu/db/sm/
    """
    f = open(folder + "small_molecule.csv", 'r')
    reader = csv.reader(f)
    reader.next()
    cid_dict = {}
    for item in reader:
        name = item[1]
        cid = item[4]
        if not name in cid_dict: 
            cid_dict[name] = str(cid)

    unknow_drug = open(folder + "unknow_drug_by_pychem.csv").readline().split(",")
    drug_cid_dict = {k:v for k,v in cid_dict.iteritems() if k in unknow_drug and not is_not_float([v])}
    return drug_cid_dict

def load_cid_dict():
    reader = csv.reader(open(folder + "pychem_cid.csv"))
    pychem_dict = {}
    for item in reader:
        pychem_dict[item[0]] = item[1]
    pychem_dict.update(cid_from_other_source())
    return pychem_dict


def download_smiles():
    cids_dict = load_cid_dict()
    cids = [v for k,v in cids_dict.iteritems()]
    inv_cids_dict = {v:k for k,v in cids_dict.iteritems()}
    download('CSV', folder + 'drug_smiles.csv', cids, operation='property/CanonicalSMILES,IsomericSMILES', overwrite=True)
    f = open(folder + 'drug_smiles.csv')
    reader = csv.reader(f)
    header = ['name'] + reader.next()
    content = []
    for line in reader:
        content.append([inv_cids_dict[line[0]]] + line)
    f.close()
    f = open(folder + "drug_smiles.csv", "w")
    writer = csv.writer(f)
    writer.writerow(header)
    for item in content:
        writer.writerow(item)
    f.close()

"""
The following code will convert the SMILES format into onehot format
"""

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br',
     'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
      'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index

def load_drug_smile(folder):
    reader = csv.reader(open(folder + "/drug_smiles.csv"))
    next(reader, None)

    drug_dict = {}
    drug_smile = []

    for item in reader:
        name = item[0]
        smile = item[2]

        if name in drug_dict:
            pos = drug_dict[name]
        else:
            pos = len(drug_dict)
            drug_dict[name] = pos
        drug_smile.append(smile)
    
    smile_graph = {}
    for smile in drug_smile:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
    
    return drug_dict, drug_smile, smile_graph

def save_cell_mut_matrix(folder):
    f = open(folder + "/PANCANCER_Genetic_feature.csv")
    reader = csv.reader(f)
    next(reader)
    features = {}
    cell_dict = {}
    mut_dict = {}
    matrix_list = []

    for item in reader:
        cell_id = item[1]
        mut = item[5]
        is_mutated = int(item[6])

        if mut in mut_dict:
            col = mut_dict[mut]
        else:
            col = len(mut_dict)
            mut_dict[mut] = col

        if cell_id in cell_dict:
            row = cell_dict[cell_id]
        else:
            row = len(cell_dict)
            cell_dict[cell_id] = row
        if is_mutated == 1:
            matrix_list.append((row, col))
    
    cell_feature = np.zeros((len(cell_dict), len(mut_dict)))

    for item in matrix_list:
        cell_feature[item[0], item[1]] = 1

    with open(folder+'/mut_dict', 'wb') as fp:
        pickle.dump(mut_dict, fp)
    
    return cell_dict, cell_feature


"""
This part is used to extract the drug - cell interaction strength. it contains IC50, AUC, Max conc, RMSE, Z_score
"""
def save_mix_drug_cell_matrix(data_path, random_seed):
    folder=data_path

    f = open(folder + "/PANCANCER_IC.csv")
    reader = csv.reader(f)
    next(reader)

    cell_dict, cell_feature = save_cell_mut_matrix(folder)
    drug_dict, drug_smile, smile_graph = load_drug_smile(folder)

    temp_data = []
    bExist = np.zeros((len(drug_dict), len(cell_dict)))

    for item in reader:
        drug = item[0]
        cell = item[3]
        ic50 = item[8]
        ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
        temp_data.append((drug, cell, ic50))

    xd = []
    xc = []
    y = []
    lst_drug = []
    lst_cell = []
    random.shuffle(temp_data)
    for data in temp_data:
        drug, cell, ic50 = data
        if drug in drug_dict and cell in cell_dict:
            xd.append(drug_smile[drug_dict[drug]])
            xc.append(cell_feature[cell_dict[cell]])
            y.append(ic50)
            bExist[drug_dict[drug], cell_dict[cell]] = 1
            lst_drug.append(drug)
            lst_cell.append(cell)
        
    with open(data_path+'/drug_dict', 'wb') as fp:
        pickle.dump(drug_dict, fp)

    xd, xc, y = np.asarray(xd), np.asarray(xc), np.asarray(y)

    size = int(xd.shape[0] * 0.8)
    size1 = int(xd.shape[0] * 0.9)

    all_ids = np.arange(xd.shape[0])
    print('random_seed: ', random_seed)
    train_ids, test_ids = train_test_split(all_ids, test_size=.2, random_state=random_seed)
    val_ids, test_ids = train_test_split(test_ids, test_size=.5, random_state=random_seed)
    

    with open(data_path+'/list_drug_mix_test', 'wb') as fp:
        # pickle.dump(lst_drug[size1:], fp)
        pickle.dump([lst_drug[i] for i in test_ids], fp)
        
    with open(data_path+'/list_cell_mix_test', 'wb') as fp:
        # pickle.dump(lst_cell[size1:], fp)
        pickle.dump([lst_cell[i] for i in test_ids], fp)

    # xd_train = xd[:size]
    xd_train = xd[train_ids]
    # xd_val = xd[size:size1]
    xd_val = xd[val_ids]
    # xd_test = xd[size1:]
    xd_test = xd[test_ids]

    pd.DataFrame(xd_test, columns=['smiles']).to_csv(data_path+'/test_smiles.csv', index=False)

    # xc_train = xc[:size]
    xc_train = xc[train_ids]
    # xc_val = xc[size:size1]
    xc_val = xc[val_ids]
    # xc_test = xc[size1:]
    xc_test = xc[test_ids]

    # y_train = y[:size]
    y_train = y[train_ids]
    # y_val = y[size:size1]
    y_val = y[val_ids]
    # y_test = y[size1:]
    y_test = y[test_ids]

    dataset = 'GDSC'
    print('preparing ', dataset + '_train.pt in pytorch format!')

    train_data = TestbedDataset(root=data_path+'/data', dataset=dataset+'_train_mix', xd=xd_train, xt=xc_train, y=y_train, smile_graph=smile_graph)
    val_data = TestbedDataset(root=data_path+'/data', dataset=dataset+'_val_mix', xd=xd_val, xt=xc_val, y=y_val, smile_graph=smile_graph)
    test_data = TestbedDataset(root=data_path+'/data', dataset=dataset+'_test_mix', xd=xd_test, xt=xc_test, y=y_test, smile_graph=smile_graph)


def save_blind_drug_matrix():
    f = open(folder + "PANCANCER_IC.csv")
    reader = csv.reader(f)
    next(reader)

    cell_dict, cell_feature = save_cell_mut_matrix()
    drug_dict, drug_smile, smile_graph = load_drug_smile()

    matrix_list = []

    temp_data = []

    xd_train = []
    xc_train = []
    y_train = []

    xd_val = []
    xc_val = []
    y_val = []

    xd_test = []
    xc_test = []
    y_test = []

    xd_unknown = []
    xc_unknown = []
    y_unknown = []

    dict_drug_cell = {}

    bExist = np.zeros((len(drug_dict), len(cell_dict)))

    for item in reader:
        drug = item[0]
        cell = item[3]
        ic50 = item[8]
        ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
        
        temp_data.append((drug, cell, ic50))

    random.shuffle(temp_data)
    
    for data in temp_data:
        drug, cell, ic50 = data
        if drug in drug_dict and cell in cell_dict:
            if drug in dict_drug_cell:
                dict_drug_cell[drug].append((cell, ic50))
            else:
                dict_drug_cell[drug] = [(cell, ic50)]
            
            bExist[drug_dict[drug], cell_dict[cell]] = 1

    lstDrugTest = []

    size = int(len(dict_drug_cell) * 0.8)
    size1 = int(len(dict_drug_cell) * 0.9)
    pos = 0
    for drug,values in dict_drug_cell.items():
        pos += 1
        for v in values:
            cell, ic50 = v
            if pos < size:
                xd_train.append(drug_smile[drug_dict[drug]])
                xc_train.append(cell_feature[cell_dict[cell]])
                y_train.append(ic50)
            elif pos < size1:
                xd_val.append(drug_smile[drug_dict[drug]])
                xc_val.append(cell_feature[cell_dict[cell]])
                y_val.append(ic50)
            else:
                xd_test.append(drug_smile[drug_dict[drug]])
                xc_test.append(cell_feature[cell_dict[cell]])
                y_test.append(ic50)
                lstDrugTest.append(drug)

    with open('drug_bind_test', 'wb') as fp:
        pickle.dump(lstDrugTest, fp)
    
    print(len(y_train), len(y_val), len(y_test))

    xd_train, xc_train, y_train = np.asarray(xd_train), np.asarray(xc_train), np.asarray(y_train)
    xd_val, xc_val, y_val = np.asarray(xd_val), np.asarray(xc_val), np.asarray(y_val)
    xd_test, xc_test, y_test = np.asarray(xd_test), np.asarray(xc_test), np.asarray(y_test)

    dataset = 'GDSC'
    print('preparing ', dataset + '_train.pt in pytorch format!')
    train_data = TestbedDataset(root='data', dataset=dataset+'_train_blind', xd=xd_train, xt=xc_train, y=y_train, smile_graph=smile_graph)
    val_data = TestbedDataset(root='data', dataset=dataset+'_val_blind', xd=xd_val, xt=xc_val, y=y_val, smile_graph=smile_graph)
    test_data = TestbedDataset(root='data', dataset=dataset+'_test_blind', xd=xd_test, xt=xc_test, y=y_test, smile_graph=smile_graph)


def save_blind_cell_matrix():
    f = open(folder + "PANCANCER_IC.csv")
    reader = csv.reader(f)
    next(reader)

    cell_dict, cell_feature = save_cell_mut_matrix()
    drug_dict, drug_smile, smile_graph = load_drug_smile()

    matrix_list = []

    temp_data = []

    xd_train = []
    xc_train = []
    y_train = []

    xd_val = []
    xc_val = []
    y_val = []

    xd_test = []
    xc_test = []
    y_test = []

    xd_unknown = []
    xc_unknown = []
    y_unknown = []

    dict_drug_cell = {}

    bExist = np.zeros((len(drug_dict), len(cell_dict)))

    for item in reader:
        drug = item[0]
        cell = item[3]
        ic50 = item[8]
        ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
        
        temp_data.append((drug, cell, ic50))

    random.shuffle(temp_data)
    
    for data in temp_data:
        drug, cell, ic50 = data
        if drug in drug_dict and cell in cell_dict:
            if cell in dict_drug_cell:
                dict_drug_cell[cell].append((drug, ic50))
            else:
                dict_drug_cell[cell] = [(drug, ic50)]
            
            bExist[drug_dict[drug], cell_dict[cell]] = 1

    lstCellTest = []

    size = int(len(dict_drug_cell) * 0.8)
    size1 = int(len(dict_drug_cell) * 0.9)
    pos = 0
    for cell,values in dict_drug_cell.items():
        pos += 1
        for v in values:
            drug, ic50 = v
            if pos < size:
                xd_train.append(drug_smile[drug_dict[drug]])
                xc_train.append(cell_feature[cell_dict[cell]])
                y_train.append(ic50)
            elif pos < size1:
                xd_val.append(drug_smile[drug_dict[drug]])
                xc_val.append(cell_feature[cell_dict[cell]])
                y_val.append(ic50)
            else:
                xd_test.append(drug_smile[drug_dict[drug]])
                xc_test.append(cell_feature[cell_dict[cell]])
                y_test.append(ic50)
                lstCellTest.append(cell)

    with open('cell_bind_test', 'wb') as fp:
        pickle.dump(lstCellTest, fp)
    
    print(len(y_train), len(y_val), len(y_test))

    xd_train, xc_train, y_train = np.asarray(xd_train), np.asarray(xc_train), np.asarray(y_train)
    xd_val, xc_val, y_val = np.asarray(xd_val), np.asarray(xc_val), np.asarray(y_val)
    xd_test, xc_test, y_test = np.asarray(xd_test), np.asarray(xc_test), np.asarray(y_test)

    dataset = 'GDSC'
    print('preparing ', dataset + '_train.pt in pytorch format!')
    train_data = TestbedDataset(root='data', dataset=dataset+'_train_cell_blind', xd=xd_train, 
    xt=xc_train, y=y_train, smile_graph=smile_graph)
    val_data = TestbedDataset(root='data', dataset=dataset+'_val_cell_blind', xd=xd_val, 
    xt=xc_val, y=y_val, smile_graph=smile_graph)
    test_data = TestbedDataset(root='data', dataset=dataset+'_test_cell_blind', xd=xd_test, 
    xt=xc_test, y=y_test, smile_graph=smile_graph)

def save_best_individual_drug_cell_matrix():
    f = open(folder + "PANCANCER_IC.csv")
    reader = csv.reader(f)
    next(reader)

    cell_dict, cell_feature = save_cell_mut_matrix()
    drug_dict, drug_smile, smile_graph = load_drug_smile()

    matrix_list = []

    temp_data = []

    xd_train = []
    xc_train = []
    y_train = []

    dict_drug_cell = {}

    bExist = np.zeros((len(drug_dict), len(cell_dict)))
    i=0
    for item in reader:
        drug = item[0]
        cell = item[3]
        ic50 = item[8]
        ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
        
        if drug == "Bortezomib":
            temp_data.append((drug, cell, ic50))
    random.shuffle(temp_data)
    
    for data in temp_data:
        drug, cell, ic50 = data
        if drug in drug_dict and cell in cell_dict:
            if drug in dict_drug_cell:
                dict_drug_cell[drug].append((cell, ic50))
            else:
                dict_drug_cell[drug] = [(cell, ic50)]
            
            bExist[drug_dict[drug], cell_dict[cell]] = 1
    cells = []
    for drug,values in dict_drug_cell.items():
        for v in values:
            cell, ic50 = v
            xd_train.append(drug_smile[drug_dict[drug]])
            xc_train.append(cell_feature[cell_dict[cell]])
            y_train.append(ic50)
            cells.append(cell)

    xd_train, xc_train, y_train = np.asarray(xd_train), np.asarray(xc_train), np.asarray(y_train)
    with open('cell_blind_sal', 'wb') as fp:
        pickle.dump(cells, fp)
    dataset = 'GDSC'
    print('preparing ', dataset + '_train.pt in pytorch format!')
    train_data = TestbedDataset(root='data', dataset=dataset+'_bortezomib', xd=xd_train, 
    xt=xc_train, y=y_train, smile_graph=smile_graph, saliency_map=True)




# CADNDLE FUNCTIONS

def get_drug_response_data(df, metric):
    
    # df = rs_train.copy()
    smiles_df = improve_utils.load_smiles_data()
    data_smiles_df = pd.merge(df, smiles_df, on = "improve_chem_id", how='left') 
    data_smiles_df = data_smiles_df.dropna(subset=[metric])
    data_smiles_df = data_smiles_df[['improve_sample_id', 'smiles', metric]]
    data_smiles_df = data_smiles_df.drop_duplicates()
    data_smiles_df = data_smiles_df.reset_index(drop=True)

    return data_smiles_df


def load_drug_smile_candle(split=0):

    data_type='CCLE'
    metric='ic50'

    rs_all = improve_utils.load_single_drug_response_data(source=data_type, split=split,
                                                          split_type=["train", "test", 'val'],
                                                          y_col_name=metric)

    se = improve_utils.load_smiles_data()
    smiles_df = pd.merge(rs_all, se, on='improve_chem_id', how='left')
    smiles_df = smiles_df.drop_duplicates(subset=['smiles'])


    smiles_df.reset_index(drop=True, inplace=True)

    drug_smile = smiles_df.smiles.tolist()
    drug_dict = {v:i for i,v in enumerate(smiles_df.improve_chem_id.values)}
    drug_smile_dict = {i:v for i, v in  smiles_df[['improve_chem_id', 'smiles']].values}


    smile_graph = {}
    for smile in drug_smile:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
        
    return drug_dict, drug_smile, smile_graph, drug_smile_dict, smiles_df

def get_cell_dict_feature_candle(model_data, gene_list=None, data_path=None):

    mutation_data = improve_utils.load_cell_mutation_data(gene_system_identifier="Entrez")
    # expr_data = improve_utils.load_gene_expression_data(gene_system_identifier="Gene_Symbol")
    mutation_data = mutation_data.reset_index()
    gene_data = mutation_data.columns[1:]

    if gene_list:
        gene_list = list(set(gene_list))
        common_genes = list(set(gene_data).intersection(gene_list))
    else:
        urllib.request.urlretrieve('https://raw.githubusercontent.com/gihanpanapitiya/GraphDRP/to_candle/landmark_genes', data_path+'/landmark_genes')
        lmgenes = pd.read_csv(data_path+'/landmark_genes', header=None).values.ravel().tolist()
        common_genes = list(set(lmgenes).intersection(gene_data))


    mutation_data = mutation_data[mutation_data.improve_sample_id.isin(model_data.improve_sample_id)]

    tmp = mutation_data.loc[:, ['improve_sample_id']+common_genes]
    tmp.reset_index(drop=True, inplace=True)
    tmp2 = tmp.iloc[:, 1:].gt(0).astype(int)
    tmp2 = pd.concat([tmp[['improve_sample_id']], tmp2], axis=1)


    cell_dict = {v:i for i,v in enumerate(tmp2.improve_sample_id.values)}
    cell_feature = tmp2.iloc[:, 1:].values
    
    return cell_dict, cell_feature


def get_input_data_candle(df, cell_feature, cell_dict):
    
    xd, xc, y=[],[],[]
    for i in df.index:

        cell_id = df.loc[i, 'improve_sample_id']   
        cf = cell_feature[  cell_dict[cell_id] ]
        # cell_id = train_df.loc[i, 'improve_sample_id']
        smiles = df.loc[i, 'smiles']
        ic50 = df.loc[i, 'ic50']

        xd.append(smiles)
        xc.append(cf)
        y.append(ic50)
        
    return xd, xc, y

def save_mix_drug_cell_matrix_candle(data_path=None, data_type='CCLE', metric='ic50', data_split_seed=-1):
    
    rs_all = improve_utils.load_single_drug_response_data(source=data_type, split=0,
                                                          split_type=["train", "test", 'val'],
                                                          y_col_name=metric)
    rs_train = improve_utils.load_single_drug_response_data(source=data_type,
                                                            split=0, split_type=["train"],
                                                            y_col_name=metric)
    rs_test = improve_utils.load_single_drug_response_data(source=data_type,
                                                           split=0,
                                                           split_type=["test"],
                                                           y_col_name=metric)
    rs_val = improve_utils.load_single_drug_response_data(source=data_type,
                                                          split=0,
                                                          split_type=["val"],
                                                          y_col_name=metric)

    train_df = get_drug_response_data(rs_train, metric)
    val_df = get_drug_response_data(rs_val, metric)
    test_df = get_drug_response_data(rs_test, metric)


    all_df = pd.concat([train_df, val_df, test_df], axis=0)
    all_df = all_df.sort_values(by='improve_sample_id')
    all_df.reset_index(drop=True, inplace=True)

    if data_split_seed > -1:
        train_df, val_df = train_test_split(all_df, test_size=0.2, random_state=data_split_seed)
        test_df, val_df = train_test_split(val_df, test_size=0.5, random_state=data_split_seed)
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)
    test_df.to_csv(os.path.join(data_path, 'test_smiles2.csv'), index=False)

    drug_dict, drug_smile, smile_graph, drug_smile_dict, smiles_df = load_drug_smile_candle()

    cell_dict, cell_feature = get_cell_dict_feature_candle(all_df, data_path=data_path)

    all_df=pd.merge(all_df, smiles_df[['improve_chem_id', 'smiles']], on='smiles', how='left')

    xd_train, xc_train, y_train = get_input_data_candle(train_df, cell_feature, cell_dict)
    xd_val, xc_val, y_val = get_input_data_candle(val_df, cell_feature, cell_dict)
    xd_test, xc_test, y_test = get_input_data_candle(test_df, cell_feature, cell_dict)

  

    train_data = TestbedDataset(root=data_path+'/data', dataset=data_type+'_train_mix', xd=xd_train, xt=xc_train,
                                y=y_train, smile_graph=smile_graph)
    val_data = TestbedDataset(root=data_path+'/data', dataset=data_type+'_val_mix', xd=xd_val, xt=xc_val, 
                              y=y_val, smile_graph=smile_graph)
    test_data = TestbedDataset(root=data_path+'/data', dataset=data_type+'_test_mix', xd=xd_test, xt=xc_test,
                               y=y_test, smile_graph=smile_graph)
    
    return train_data, val_data, test_data





# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='prepare dataset to train model')
#     parser.add_argument('--choice', type=int, required=False, default=0, help='0.mix test, 1.saliency value, 2.drug blind, 3.cell blind')
#     args = parser.parse_args()
#     choice = args.choice
#     if choice == 0:
#         # save mix test dataset
#         save_mix_drug_cell_matrix()
#     elif choice == 1:
#         # save saliency map dataset
#         save_best_individual_drug_cell_matrix()
#     elif choice == 2:
#         # save blind drug dataset
#         save_blind_drug_matrix()
#     elif choice == 3:
#         # save blind cell dataset
#         save_blind_cell_matrix()
#     else:
#         print("Invalide option, choose 0 -> 4")