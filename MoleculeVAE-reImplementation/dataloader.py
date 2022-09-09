import torch
import numpy as np
import deepchem as dc
from rdkit import Chem
from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, Descriptors
print(rdBase.rdkitVersion)
import pandas as pd
import seaborn as sns
import os
import sys
from rdkit.Chem import QED
import hyperparameters
sys.path.append(os.path.join(Chem.RDConfig.RDContribDir, 'SA_Score'))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import sascorer
import argparse
from collections import OrderedDict
import json
import h5py
import collections
import smilite
from collections import OrderedDict
import urllib
import requests
from bs4 import BeautifulSoup
import re
#defualt=Fingerprint dataset
import numpy as np
from sklearn.model_selection import train_test_split
from rdkit.Chem import QED

def loadDataset(dataset=None,from_deepchem=True,featurizer='ECFP'):
    """Loads dataset, either from from deepchem or from local file system.

    - Inputs:
        - dataset:

    could also split up further as:
    def load_dataset_from_file():
        ...

    def load_dataset_from_deepchem():
        ...

    """
    if from_deepchem:
        # TODO: move this import out
        import deepchem as dc
        tasks, datasets, transformers = dc.molnet.load_qm8(featurizer=featurizer)

        return datasets
    else:
        Datapath=dataset
        return pd.read_csv(Datapath)

#calculate Y.dataset for latenspace predictor
def get_zinc_smile(zinc_id):
    url = "https://zinc.docking.org/substances/"+str(zinc_id)+"/"
    req = requests.get(url)
    soup = BeautifulSoup(req.content, 'html.parser')
    div=soup.find(id="substance-smiles-field")
    print(str(div))
    smile_str = re.search('[^value]+$', str(div)).group().split("\"")[1]
    return smile_str


def caculate_target_values(smiles):
    """Calculates logP, QED and SAS for a smiles input;

    - Inputs:
        - smiles:
          str. The smiles of the molecule for which we want to caluclate the values.

    - Returns:
        - logP
          ...
        - QED
          ...
        - SAS
          ...

    If the molecule cannot be calculated by RDkit, returns 0,0,0
    These values calculated here sometimes disagree with the values given in the dataset from the original
    dataset.
    """

    m = Chem.MolFromSmiles(smiles)
    logP = Chem.Descriptors.MolLogP(m)
    QED = QED.default(m)
    SAS = sascorer.calculateScore(m)

    return logP, QED, SAS

# TODO: fill in by generating from current data set and update later if required for new datasets.
CHARACTERS_IN_SMILES = ['c','-','b',...]

def one_hotencoder(SmiData,normalizeSize):
    """Fill in"""

    character_index_lookup_dict = load_character_index_lookup_dict()
    x = [make_encoding_for_smiles(vec, character_index_lookup_dict, normalizeSize) for vec
            in SmiData]

    return x,character_index_lookup_dict

def make_encoding_for_smiles(vec, character_index_lookup_dict, normalizeSize):
    # TODO: split this up into separate steps so it is more undestandable.
    return np.concatenate((torch.nn.functional.one_hot(torch.tensor(list(map(lambda a: character_index_lookup_dict[a], [*vec]))), num_classes=len(character_index_lookup_dict)), torch.zeros(normalizeSize - len(vec), len(character_index_lookup_dict))), axis=0)

def load_character_index_lookup_dict():
    """This is a dictionary that maps character to a unique index"""
    if os.path.exists('Dicdata.json'):
        with open('Dicdata.json') as json_file:
            character_index_lookup_dict = json.load(json_file)
    else:
        character_index_lookup_dict = collections.OrderedDict([(a, i) for i,a in enumerate(CHARACTERS_IN_SMILES)])
        with open("Dicdata.json", "w") as outfile:
            json.dump(character_index_lookup_dict, outfile)

    return character_index_lookup_dict


def oneHotdecoder(onehotData,dic):
    dic_swap = {v: k for k, v in dic.items()}
    return["".join(map(str, list(map(lambda a: dic_swap[a], ids.argmax(-1))))) for ids in onehotData]

#task='do_prop_pred'or'AE_only'
def Smiles2dataset(params):
    if params['FromDeepchem']:
        datasets=loadDataset(params['Data_file'],params['FromDeepchem'],params['featurizer'])
        train_dataset, valid_dataset, test_dataset = datasets
        TRSmiOnehot,od=one_hotencoder( train_dataset.ids,params['normalizeSize'])
        TSmiOnehot,od= one_hotencoder(test_dataset.ids, params['normalizeSize'])
        X_train, X_test= TRSmiOnehot, TSmiOnehot
        if params['do_prop_pred']:
            X_train_final = []
            Y_train = []
            for x, pre  in zip(X_train, train_dataset.ids):
                try:
                    y = caculate_target_values(pre)
                except:
                    continue
                X_train_final.append(x)
                Y_train.append(y)
            # TODO: do the same for X_test
            Y_test=[caculate_target_values(pre) for pre in test_dataset.ids]
            return X_train_final, X_test, Y_train, Y_test,od
        else:
            return X_train, X_test,od
    else:

        df = loadDataset(params['Data_file'], params['FromDeepchem'], params['featurizer'])
        train_docs = df['smiles'].tolist()
        logP = df["logP"].tolist()
        QED = df["qed"].tolist()
        SAS = df["SAS"].tolist()
        train_docsy=np.stack(( logP, QED ,SAS), axis=-1)
        X_train_s,X_test_s, Y_train, Y_test=train_test_split(train_docs,train_docsy)

        X_train, od = one_hotencoder(X_train_s, params['normalizeSize'])
        X_test, od = one_hotencoder( X_test_s, params['normalizeSize'])
        if params['do_prop_pred']:

            return X_train, X_test, Y_train, Y_test,od
        else:
            return X_train, X_test,od


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_file',
                        help='experiment file', default='expParam.json')
    parser.add_argument('-d', '--directory',
                        help='exp directory', default=None)
    args = vars(parser.parse_args())
    if args['directory'] is not None:
        args['exp_file'] = os.path.join(args['directory'], args['exp_file'])

    params = hyperparameters.load_params(args['exp_file'])




