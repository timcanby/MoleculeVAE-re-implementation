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
def loadDataset(Dataset=None,FromDeepchem=True,featurizer='ECFP'):
    if FromDeepchem:
        import deepchem as dc
        tasks, datasets, transformers = dc.molnet.load_qm8(featurizer=featurizer)

        return datasets
    else:
        Datapath=Dataset
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
def caculateLSvalue(Smiles):
    try:
        if "ZIN" in Smiles:
            m = Chem.MolFromSmiles(str(get_zinc_smile(Smiles)))
        else:m = Chem.MolFromSmiles(Smiles)
    except:return 0,0,0
    logP = Chem.Descriptors.MolLogP(m)
    QED = QED.default(m)
    SAS = sascorer.calculateScore(m)

    return logP, QED, SAS

def oneHotencoder(SmiData,normalizeSize):

    dicset=list(set([*''.join(SmiData)]))
    if os.path.exists('Dicdata.json'):
        with open('Dicdata.json') as json_file:
            od = json.load(json_file)
        x = [np.concatenate((torch.nn.functional.one_hot(torch.tensor(list(map(lambda a: od[a], [*vec]))), num_classes=len(od)), torch.zeros(normalizeSize - len(vec), len(od))), axis=0) for vec
             in SmiData]
    else:
        od = collections.OrderedDict([(a, list(dicset).index(a)) for a in dicset])
        with open("Dicdata.json", "w") as outfile:
            json.dump(od, outfile)

        x = [np.concatenate((torch.nn.functional.one_hot(torch.tensor(list(map(lambda a: od[a], [*vec]))), num_classes=len(od)), torch.zeros(normalizeSize - len(vec), len(od))), axis=0) for vec
             in SmiData]

    return x,od
def oneHotdecoder(onehotData,dic):
    dic_swap = {v: k for k, v in dic.items()}
    return["".join(map(str, list(map(lambda a: dic_swap[a], ids.argmax(-1))))) for ids in onehotData]

#task='do_prop_pred'or'AE_only'
def Smiles2dataset(params):
    if params['FromDeepchem']:
        datasets=loadDataset(params['Data_file'],params['FromDeepchem'],params['featurizer'])
        train_dataset, valid_dataset, test_dataset = datasets
        TRSmiOnehot,od=oneHotencoder( train_dataset.ids,params['normalizeSize'])
        TSmiOnehot,od= oneHotencoder(test_dataset.ids, params['normalizeSize'])
        X_train, X_test= TRSmiOnehot, TSmiOnehot
        if params['do_prop_pred']:
            Y_train=[caculateLSvalue(pre) for pre in train_dataset.ids]
            Y_test=[caculateLSvalue(pre) for pre in test_dataset.ids]
            return X_train, X_test, Y_train, Y_test,od
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

        X_train, od = oneHotencoder(X_train_s, params['normalizeSize'])
        X_test, od = oneHotencoder( X_test_s, params['normalizeSize'])
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



