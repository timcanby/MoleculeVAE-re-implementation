import numpy as np
from rdkit import rdBase, Chem
print(rdBase.rdkitVersion)
import pandas as pd
import os
import sys
import hyperparameters
sys.path.append(os.path.join(Chem.RDConfig.RDContribDir, 'SA_Score'))
import torch.nn as nn
import torch.utils.data
import argparse
import torch.optim as optim
import torch
from dataloader import oneHotdecoder
from model import MolecularVAE

list=[]
class CustomTextDataset(torch.utils.data.Dataset):
    def __init__(self,x,y):
        self.X = torch.tensor(np.squeeze(x))
        self.y = torch.tensor(np.squeeze(y))
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def vae_loss(x_decoded_mean, x, z_mean, z_logvar):
    xent_loss = nn.MSELoss()
    vecloss=xent_loss (x_decoded_mean,x)
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return vecloss + kl_loss

def train(epochs):

    model.train()
    train_loss = 0
    if params['do_prop_pred']:
        X_train, X_test, Y_train, Y_test,od = Smiles2dataset(params)
        text_labels_df = pd.DataFrame({'Smi': [X_train], 'PreLabel': [Y_train]})
        test_labels_df = pd.DataFrame({'Smi': [X_test], 'PreLabel': [Y_test]})
        dataset = CustomTextDataset(text_labels_df['Smi'], text_labels_df['PreLabel'])
        testDataset = CustomTextDataset(test_labels_df['Smi'], test_labels_df['PreLabel'])
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'])
        test_loader = torch.utils.data.DataLoader(testDataset)
        for batch_idx, (data,label) in enumerate(train_loader):
            data = data.to(dtype=torch.float32, device=device)
            label=torch.from_numpy(np.array(label)).to(device=device,dtype=torch.float32)
            optimizer.zero_grad()
            output, mean, logvar,pre = model(data)
            pre_loss_ca=nn.MSELoss()
            pre_loss=pre_loss_ca(pre,label)
            loss = vae_loss(output.to(torch.float32), data, mean, logvar)+pre_loss
            loss.backward()
            train_loss += loss
            optimizer.step()
            if batch_idx % 1000 == 0:
                torch.save(model.state_dict(), 'Weights/param.pth')
                print(f'{epochs} / {batch_idx}\t{loss:.4f}')
                with torch.no_grad():
                    for data in test_loader:
                        smidata, labels = data
                        output, mean, logvar,pre = model(smidata.to(dtype=torch.float32, device=device))
                        mae = nn.L1Loss()
                        print(oneHotdecoder(smidata[:1].cpu().detach().numpy(), od))
                        print(oneHotdecoder(output[:1].cpu().detach().numpy(), od))

                        mae_logP=mae(labels[:, 0].to(device),pre[:, 0].to(device))
                        mae_qed=mae(labels[:, 1].to(device),pre[:, 1].to(device))
                        mae_SAS=mae(labels[:, 2].to(device),pre[:, 2].to(device))

                        list.append([epochs,torch.mean((pre[:, 0])),torch.mean((pre[:, 1])),torch.mean((pre[:, 2])),mae_logP, mae_qed,mae_SAS])

    else:
        X_train, X_test,od = Smiles2dataset(params)
        train_loader = torch.utils.data.DataLoader(X_train, batch_size=params['batch_size'])
        for batch_idx, data in enumerate(train_loader):
            data = data.to(dtype=torch.float32, device=device)
            optimizer.zero_grad()
            output, mean, logvar,_= model(data.float())
            loss = vae_loss(output.to(torch.float32), data, mean, logvar)
            loss.backward()
            train_loss += loss
            optimizer.step()
            if batch_idx % 1000 == 0:
                torch.save(model.state_dict(), 'Weights/param(Without_pro-pretiction).pth')
                print(f'{epochs} / {batch_idx}\t{loss:.4f}')
        print('train', train_loss / len(train_loader.dataset))
        return train_loss / len(train_loader.dataset)




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
    model = MolecularVAE().to(device)
    optimizer = optim.Adam(model.parameters())
    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        df = pd.DataFrame(list, columns=['epoch','logP', 'qed', 'SAS', 'mae_logP', 'mae_qed', 'mae_SAS'])
        df.to_csv('trainingprocess.csv')


