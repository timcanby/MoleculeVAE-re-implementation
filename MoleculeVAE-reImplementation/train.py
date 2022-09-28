import numpy as np
from rdkit import rdBase, Chem
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
from dataloader import one_hot_encoder,one_hot_decoder
from model import MolecularVAE
from model import CustomMoleculeDataset
from torch.utils.data import Dataset, DataLoader
from dataloader import load_dataset
import torch.utils.data as data
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import gif
import warnings
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA, IncrementalPCA
from scipy.spatial import geometric_slerp
import time
import os
from datetime import datetime

# dd/mm/YY H:M:S
dt_string = datetime.utcnow().strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)
#record_list for recording the loss value
record_list=[]


def split_validation_dataset(train_dataset,percentage_train):
    '''
     This is for building the train and validation datasets
         - train_dataset:torch.utils.data.Dataset object
         - percentage_train:percentage of training data default=0.9
     - Returns:
        -train_set_final: dataset for training
        -valid_set_final: dataset for validation
     '''
    # use 20% of training data for validation
    train_set_size = int(len(train_dataset) * percentage_train)
    valid_set_size = len(train_dataset) - train_set_size
    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = data.random_split(train_dataset, [train_set_size, valid_set_size], generator=seed)
    train_set_final = torch.utils.data.DataLoader(train_set, batch_size=params['batch_size'])
    valid_set_final = torch.utils.data.DataLoader(valid_set, batch_size=params['batch_size'])

    return train_set_final,valid_set_final


def main_property_run(params):
    '''
    This is for training process include property prediction
        - Inputs: params (Use hyperparameters.load_params() to load params from expParam.json)
    - Returns:
        -train_set_final,
        -valid_set_final,
        -test_dataset,
    '''
    # load data
    X_train, X_test, Y_train, Y_test, character_index_lookup_dict = load_dataset(params)
    train_dataset=CustomMoleculeDataset(X_train,Y_train)
    train_set_final, valid_set_final=split_validation_dataset(train_dataset,percentage_train=0.9)

    test_dataset=CustomMoleculeDataset(X_test,Y_test)
    return train_set_final, valid_set_final,test_dataset

def main_no_prop(params):
    '''
    This is for training process which not include property prediction
        - Inputs: params (Use hyperparameters.load_params() to load params from expParam.json)
    - Returns:
        -train_set_final,
        -valid_set_final,
        -test_dataset,
    '''
    X_train, X_test, character_index_lookup_dict = load_dataset(params)
    train_dataset = CustomMoleculeDataset(X_train, Y_train)
    train_set_final, valid_set_final = split_validation_dataset(train_dataset, percentage_train=0.9)

    test_dataset = CustomMoleculeDataset(X_test, Y_test)
    return train_set_final, valid_set_final,test_dataset

def vae_loss(x_decoded_mean, x, z_mean, z_logvar):
    xent_loss = nn.MSELoss()
    vecloss=xent_loss (x_decoded_mean.to(dtype=torch.float32, device=device),x.to(dtype=torch.float32, device=device))
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return vecloss + kl_loss

def vae_loss_with_annealing(x_decoded_mean, x, z_mean, z_logvar,epoch):
    '''
    This is for calculating the loss value with added annealer
        - Inputs: params (Use hyperparameters.load_params() to load params from expParam.json)
        - Default :cycle sigmoid schedule

    '''
    xent_loss = nn.MSELoss()
    vecloss=xent_loss (x_decoded_mean.to(dtype=torch.float32, device=device),x.to(dtype=torch.float32, device=device))
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

    if epoch>=params['vae_annealer_start']:
        return vecloss + frange_cycle_sigmoid(0.0, 1.0, epoch, 1, 0.25)[0]*kl_loss
    else:
        return vecloss + kl_loss


'''============cyclical_annealing for KL vanishing problem==================
from paper: Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing (NAACL 2019)
paper link:https://arxiv.org/abs/1903.10145
Github:https://github.com/haofuml/cyclical_annealing

KL vanishing:https://alibabatech.medium.com/next-gen-text-generation-alibaba-makes-progress-on-the-kl-vanishing-problem-77ec35f2afa2
'''

def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)  # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_epoch):
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L


def frange_cycle_sigmoid(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)  # step is in [0,1]

    # transform into [-6, 6] for plots: v*12.-6.

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop:
            L[int(i + c * period)] = 1.0 / (1.0 + np.exp(- (v * 12. - 6.)))
            v += step
            i += 1
    return L


#  function  = 1 − cos(a), where a scans from 0 to pi/2

def frange_cycle_cosine(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)  # step is in [0,1]

    # transform into [0, pi] for plots:

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop:
            L[int(i + c * period)] = 0.5 - .5 * math.cos(v * math.pi)
            v += step
            i += 1
    return L

'''====End of cyclical_annealing module================================'''
def calculate_loss_by_type(decoder_output_data,input_string_data,z_mean,z_logvar,property_prediction_loss,epochs):
    '''To calculate the loss value '''
    pre_loss=property_prediction_loss
    loss_type= params['loss_type']
    if loss_type== "Variance_only":
        loss = vae_loss(decoder_output_data, input_string_data, z_mean, z_logvar)
    elif loss_type== "pro_prediction_only":
        loss = -pre_loss
    elif loss_type== "vae_pre_no_annealing":
        loss = vae_loss(decoder_output_data, input_string_data, z_mean, z_logvar) + pre_loss
    elif loss_type== "vae_pre_with_annealing":
        loss = vae_loss_with_annealing(decoder_output_data, input_string_data, z_mean, z_logvar, epochs) + pre_loss
    else:
        loss=None
    return loss

def train(epochs,train_set_final, valid_set_final):
    model.train()
    train_loss = 0
    valid_loss=0
    epoch_save_update=True
    if params['do_prop_pred']:
        for batch_idx, (data,label) in enumerate(train_set_final):
            data = data.to(dtype=torch.float32, device=device)
            label=label.to(dtype=torch.float32, device=device)
            optimizer.zero_grad()
            output, mean, logvar,pre,z = model(data)
            pre_loss_ca=nn.MSELoss()
            pre_loss=pre_loss_ca(pre,label)
            loss=calculate_loss_by_type(output,data, mean, logvar,pre_loss,epochs)
            loss.backward()
            train_loss += loss
            optimizer.step()
            if batch_idx % len(train_set_final)== 0:

                with torch.no_grad():
                    for data in valid_set_final:
                        valid_smidata, valid_labels = data
                        valid_output, valid_mean, valid_logvar,valid_pre,valid_z = model(valid_smidata.to(dtype=torch.float32, device=device))
                        valid_pre_loss_ca = nn.MSELoss()
                        valid_pre_loss = valid_pre_loss_ca(valid_pre, valid_labels.to(dtype=torch.float32, device=device))
                        valid_loss += calculate_loss_by_type(valid_output,valid_smidata,valid_mean,valid_logvar,valid_pre_loss,epochs)
                        if valid_loss>=10:
                            epoch_save_update=False
                    if epoch_save_update:
                        torch.save(model.state_dict(), 'Weights/' + str(dt_string.replace("/", '_')) + str(epochs)+'_total120_epochs_with_stop_full_vae_pre_annealing_param.pth')

                    print(f'{epochs} / {batch_idx}\t{loss:.4f}')
    return (train_loss.cpu().detach().numpy()/len(train_set_final)),(valid_loss.cpu().detach().numpy()/len(valid_set_final))


def test_do_pro(test_set,epochs):
    dataset=torch.utils.data.DataLoader(test_set, batch_size=params['batch_size'])
    test_loss=0
    with torch.no_grad():
        for data in dataset:
            test_smidata, test_labels = data
            test_output, test_mean, test_logvar, test_pre, test_z = model(
                test_smidata.to(dtype=torch.float32, device=device))
            test_pre_loss_ca = nn.MSELoss()
            test_pre_loss = test_pre_loss_ca(test_pre, test_labels.to(dtype=torch.float32, device=device))
            test_loss+=calculate_loss_by_type(test_output,test_smidata,test_mean,test_logvar,test_pre_loss,epochs)
    return test_loss.cpu().detach().numpy()/len(dataset)


if __name__ == "__main__":
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
    torch.manual_seed(params['RAND_SEED'])
    epochs = params['epochs']
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = nn.DataParallel(MolecularVAE().to(device))
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1, epochs + 1):
        if params['do_prop_pred']:
            train_set_final, valid_set_final, test_dataset = main_property_run(params)
        else:
            train_set_final, valid_set_final, test_dataset = main_no_prop(params)
        train_loss,valid_loss=train(epoch, train_set_final, valid_set_final)
        test_loss=test_do_pro(test_dataset,epoch)
        print([epoch,train_loss,valid_loss,test_loss])
        list.append([epoch,train_loss,valid_loss,test_loss])
    df = pd.DataFrame(list, columns=['epoch', 'Train_loss', 'Valid_loss', 'Test_loss'])
    df.to_csv(''+str(dt_string.replace("/",'_'))+'train_process.csv')

