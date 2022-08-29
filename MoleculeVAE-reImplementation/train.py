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
from dataloader import  oneHotdecoder
from dataloader import od
import h5py
import pickle

def vae_loss(x_decoded_mean, x, z_mean, z_logvar):
    xent_loss = nn.MSELoss()
    vecloss=xent_loss (x_decoded_mean,x)
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return vecloss + kl_loss

def train(epochs):
    model.train()
    train_loss = 0

    if params['do_prop_pred']:
        for batch_idx, (data,label) in enumerate(train_loader):
            data = data.to(dtype=torch.float32, device=device)
            label=torch.from_numpy(np.array(label)).to(device=device)
            #label=torch.tensor(label).to(dtype=torch.float32, device=device)
            optimizer.zero_grad()
            output, mean, logvar,pre = model(data.float())
            pre_loss_ca=nn.MSELoss()
            pre_loss=pre_loss_ca(pre,label)
            loss = vae_loss(output.to(torch.float32), data, mean, logvar)+pre_loss
            loss.backward()
            train_loss += loss
            optimizer.step()
            if batch_idx % 1000 == 0:
                torch.save(model.state_dict(), 'param.pth')
                print(f'{epochs} / {batch_idx}\t{loss:.4f}')
                with torch.no_grad():
                    for data in test_loader:
                        smidata, labels = data
                        output, mean, logvar,pre = model(smidata.to(dtype=torch.float32, device=device))
                        a_file = open("Dicdata.pkl", "rb")
                        od = pickle.load(a_file)
                        a_file.close()
                        print(oneHotdecoder(smidata[:1].cpu().detach().numpy(), od))
                        print(oneHotdecoder(output[:1].cpu().detach().numpy(), od))


    else:
        for batch_idx, data in enumerate(train_loader):
            data = data.to(dtype=torch.float32, device=device)
            optimizer.zero_grad()
            output, mean, logvar,_= model(data.float())
            loss = vae_loss(output.to(torch.float32), data, mean, logvar)
            loss.backward()
            train_loss += loss
            optimizer.step()
            if batch_idx % 1000 == 0:
             print(f'{epochs} / {batch_idx}\t{loss:.4f}')
        print('train', train_loss / len(train_loader.dataset))
        return train_loss / len(train_loader.dataset)

from torch.utils.data import Dataset, DataLoader
class CustomTextDataset(torch.utils.data.Dataset):
    def __init__(self,x,y):
        self.X = x
        self.y = torch.tensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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
    from dataloader import Smiles2dataset
    torch.manual_seed(params['RAND_SEED'])
    epochs = params['epochs']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from model import MolecularVAE
    model = MolecularVAE().to(device)
    optimizer = optim.Adam(model.parameters())
    if params['do_prop_pred']:
        X_train, X_test, Y_train, Y_test = Smiles2dataset(params)
        text_labels_df = pd.DataFrame({'Smi': X_train, 'PreLabel': Y_train})
        test_labels_df = pd.DataFrame({'Smi': X_test, 'PreLabel': Y_test})
        dataset= CustomTextDataset(text_labels_df['Smi'],text_labels_df['PreLabel'])
        testDataset=CustomTextDataset(test_labels_df['Smi'],test_labels_df['PreLabel'])
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'])
        test_loader=torch.utils.data.DataLoader(testDataset)

    else:
        X_train, X_test = Smiles2dataset(params)
        train_loader = torch.utils.data.DataLoader(X_train, batch_size=params['batch_size'])
    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)







