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
#defualt=Fingerprint dataset
def loadDataset(Dataset=None,FromDeepchem=True,featurizer='ECFP'):
    if FromDeepchem:
        import deepchem as dc
        tasks, datasets, transformers = dc.molnet.load_qm9(featurizer=featurizer)

        return datasets
    else:
        Datapath=Dataset
        return Dataset

#calculate Y.dataset for latenspace predictor
def caculateLSvalue(Smiles):
    from rdkit.Chem import QED
    m = Chem.MolFromSmiles(Smiles)
    logP=Chem.Descriptors.MolLogP(m)
    QED=QED.default(m)
    SAS=sascorer.calculateScore(m)
    return logP, QED, SAS

def oneHotencoder(SmiData,normalizeSize):
    import collections
    from collections import OrderedDict
    dicset=list(set([*''.join(SmiData)]))
    od = collections.OrderedDict([(a,list(dicset).index(a)) for a in dicset])
    #x=[torch.nn.functional.one_hot(torch.tensor(list(map(lambda a: od[a], [*vec]))), num_classes=len(dicset)) for vec in SmiData]
    x = [np.concatenate((torch.nn.functional.one_hot(torch.tensor(list(map(lambda a: od[a], [*vec]))), num_classes=len(dicset)), torch.zeros(normalizeSize - len(vec), len(dicset))), axis=0) for vec
         in SmiData]
    return x,od
def oneHotdecoder():
    return
#task='do_prop_pred'or'AE_only'
def Smiles2dataset(params):
    datasets=loadDataset(params['Dataname'],params['FromDeepchem'],params['featurizer'])
    train_dataset, valid_dataset, test_dataset = datasets
    TRSmiOnehot,TRSmiDic=oneHotencoder( train_dataset.ids,params['normalizeSize'])
    TSmiOnehot, TSmiDic = oneHotencoder(test_dataset.ids[:1000], params['normalizeSize'])
    X_train, X_test= TRSmiOnehot, TSmiOnehot
    if params['do_prop_pred']:
        Y_train=[caculateLSvalue(pre) for pre in train_dataset.ids]
        Y_test=[caculateLSvalue(pre) for pre in test_dataset.ids[:1000]]
        #print(Y_train)
        return X_train, X_test, Y_train, Y_test
    else:
        return X_train, X_test

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
    #print("All params:", params)
    #Smiles2dataset(params)



