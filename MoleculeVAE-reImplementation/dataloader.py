
import numpy as np
from rdkit import rdBase, Chem
print(rdBase.rdkitVersion)
import os
import sys
import hyperparameters
sys.path.append(os.path.join(Chem.RDConfig.RDContribDir, 'SA_Score'))
import torch
import torch.utils.data
import sascorer
import argparse

import json
import collections

od={}
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

    global od

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
    datasets=loadDataset(params['Dataname'],params['FromDeepchem'],params['featurizer'])
    train_dataset, valid_dataset, test_dataset = datasets

    TRSmiOnehot,od=oneHotencoder( train_dataset.ids,params['normalizeSize'])
    TSmiOnehot,od= oneHotencoder(test_dataset.ids[:100], params['normalizeSize'])

    X_train, X_test= TRSmiOnehot, TSmiOnehot

    if params['do_prop_pred']:
        Y_train=[caculateLSvalue(pre) for pre in train_dataset.ids]
        Y_test=[caculateLSvalue(pre) for pre in test_dataset.ids[:100]]
        #Y_test=0
        #print(Y_train)
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




