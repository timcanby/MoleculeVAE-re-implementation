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
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn import model_selection
import zipfile
import torch
from sklearn.preprocessing import OneHotEncoder
import h5py
import pickle

from model import MolecularVAE
def oneHotencoder(SmiData,normalizeSize):
    import collections
    from collections import OrderedDict
    dicset=list(set([*''.join(SmiData)]))
    od = collections.OrderedDict([(a,list(dicset).index(a)) for a in dicset])
    #x=[torch.nn.functional.one_hot(torch.tensor(list(map(lambda a: od[a], [*vec]))), num_classes=len(dicset)) for vec in SmiData]
    x = [np.concatenate((torch.nn.functional.one_hot(torch.tensor(list(map(lambda a: od[a], [*vec]))), num_classes=len(dicset)+1), torch.zeros(normalizeSize - len(vec), len(dicset)+1)), axis=0) for vec
         in SmiData]
    return x,od
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tasks, datasets, transformers = dc.molnet.load_qm9(featurizer="ECFP")
train_dataset, valid_dataset, test_dataset = datasets
TSmiOnehot, TSmiDic = oneHotencoder(test_dataset.ids,120)

model = MolecularVAE().to(device)

model.load_state_dict(torch.load('param.pth'))

output, mean, logvar,pre = model(torch.from_numpy((np.array(TSmiOnehot[:100]))).to(dtype=torch.float32, device=device))
from dataloader import  oneHotdecoder
from dataloader import od



print(oneHotdecoder(output.cpu().detach().numpy(),od))
#from sklearn.decomposition import PCA
#pca = PCA(n_components=2)
#mol_data=pca.fit_transform(output.cpu().detach().numpy()[0])