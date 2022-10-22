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

from model import MolecularVAE
from model import CustomMoleculeDataset

from dataloader import load_dataset
import torch.utils.data as data

import matplotlib.pyplot as plt

import gif
import warnings
warnings.filterwarnings("ignore")

import os
from datetime import datetime

# dd/mm/YY H:M:S
dt_string = datetime.utcnow().strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)


"---To record the Loss log---"


class LogFile(object):
    def __init__(self):
        # loss_record_list for recording the loss terms [epoch,vae_loss, reconstruction_loss, prediction_loss,process_type]
        self.record_list = []
        # performance on separate prediction tasks [epoch,'logP_pre', 'qed_pre', 'SAS_pre', 'mae_logP', 'mae_qed', 'mae_SAS']
        self.separate_prediction_list = []
        # list for record annealing performance [epoch,test_total_loss_with_annealing,test_total_loss_without_annealing,annealing_weights,process_type]
        self.list_for_record_annealing_performance = []

    def add_record_list(self, log):
        self.record_list.append(log)

    def add_separate_prediction(self, log):
        self.separate_prediction_list.append(log)

    def add_list_for_record_annealing_performance(self, log):
        self.list_for_record_annealing_performance.append(log)

    def save_log(self,params):
        if not os.path.exists('Loss_record'):
            os.mkdir('Loss_record')
        '''separate_prediction_list For evaluate the prediction results on test set of each epoch'''
        # performance on separate prediction tasks [epoch,'logP_pre', 'qed_pre', 'SAS_pre', 'mae_logP', 'mae_qed', 'mae_SAS']
        pd.DataFrame(self.separate_prediction_list,
                     columns=['epoch', 'logP_pre', 'qed_pre', 'SAS_pre', 'mae_logP', 'mae_qed', 'mae_SAS']).to_csv(
            'Loss_record/' + str(dt_string.replace("/", '_')) + '_separate_prediction_list.csv')

        '''list for record annealing performance '''
        # [epoch,loss_with_annealing,loss_without_annealing,annealing_weights,process_type]
        pd.DataFrame(self.list_for_record_annealing_performance,
                     columns=['epoch', 'kl_loss_with_annealing', 'kl_without_annealing', 'annealing_weights',
                              'process_type']).to_csv(
            'Loss_record/' + str(dt_string.replace("/", '_')) + '_list_for_record_annealing_performance.csv')

        ''' record_list  Only for VAE+ properties prediction task'''
        # record_list for recording the loss terms [epoch,vae_loss, reconstruction_loss, prediction_loss,process_type]
        if params['do_prop_pred']:
            pd.DataFrame(self.record_list, columns=['epoch', 'vae_loss', 'reconstruction_loss', 'KL_loss', 'prediction_loss',
                                               'process_type']).to_csv(
                'Loss_record/' + str(dt_string.replace("/", '_')) + '_record_list.csv')




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


'''Read in the training data according to whether the prediction task is performed or not'''
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


def load_data_by_task(params):
    '''
            - Inputs: params (Use hyperparameters.load_params() to load params from expParam.json)
        - Returns:
        A pandas DataFrame include follow items:
            -train_set_final,
            -valid_set_final,
            -test_dataset,
        '''
    if params['do_prop_pred']:
        X_train, X_test, Y_train, Y_test, character_index_lookup_dict = load_dataset(params)
    else:
        X_train, X_test, character_index_lookup_dict = load_dataset(params)

    train_dataset = CustomMoleculeDataset(X_train, Y_train)
    train_set_final, valid_set_final = split_validation_dataset(train_dataset, percentage_train=0.8)
    test_dataset = CustomMoleculeDataset(X_test, Y_test)
    test_set_final, _ = split_validation_dataset(test_dataset, percentage_train=1)# To control the number of percentage
    training_data_dic = pd.DataFrame(list(zip( train_set_final, valid_set_final, test_dataset )),
                      columns=['train_set_final', 'valid_set_final', 'test_dataset'])
    return training_data_dic



'''Loss function '''

def vae_loss(x_decoded_mean, x, z_mean, z_logvar):
    xent_loss = nn.MSELoss()
    vecloss=xent_loss (x_decoded_mean.to(dtype=torch.float32, device=device),x.to(dtype=torch.float32, device=device))
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

    return vecloss,kl_loss

def vae_loss_with_annealing(x_decoded_mean, x, z_mean, z_logvar):
    '''
    This is for calculating the loss value with added annealer
        - Inputs: params (Use hyperparameters.load_params() to load params from expParam.json)
        - Default :cycle sigmoid schedule

    '''
    xent_loss = nn.MSELoss()
    vecloss=xent_loss (x_decoded_mean.to(dtype=torch.float32, device=device),x.to(dtype=torch.float32, device=device))
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    if epoch>=params['vae_annealer_start']:
        return vecloss,frange_cycle_cosine(0.0, 1.0, epoch, 4, 0.5)[0],kl_loss
    else:
        return vecloss,1,kl_loss


def calculate_loss_by_type(decoder_output_data, input_string_data, z_mean, z_logvar, property_prediction_loss, epochs,process_type,logger):
    '''To calculate the loss value '''
    pre_loss = property_prediction_loss
    loss_type = params['loss_type']
    if loss_type == "Variance_only":
        loss = vae_loss(decoder_output_data, input_string_data, z_mean, z_logvar)
    elif loss_type == "pro_prediction_only":
        loss = pre_loss

    elif loss_type == "vae_pre_no_annealing":
        vecloss,kl_loss = vae_loss(decoder_output_data, input_string_data, z_mean, z_logvar)
        #TODO:scaling factor
        vae_loss=vecloss+kl_loss
        loss=vae_loss + pre_loss
        #record_list.append([epochs, vae_loss, reconstruction_loss, prediction_loss, process_type])
        logger.add_record_list([epochs, vae_loss.cpu().detach().numpy(),vecloss.cpu().detach().numpy(),kl_loss.cpu().detach().numpy(), pre_loss.cpu().detach().numpy(), process_type])

    elif loss_type == "vae_pre_with_annealing":
        vecloss,anealing_weight,kl_loss_original = vae_loss_with_annealing(decoder_output_data, input_string_data, z_mean, z_logvar)
        kl_loss=anealing_weight*kl_loss_original
        # TODO:scaling factor
        vae_loss = vecloss + kl_loss
        loss = vae_loss + pre_loss

        # list for record annealing performance
        logger.add_list_for_record_annealing_performance([epochs,kl_loss.cpu().detach().numpy(),kl_loss_original.cpu().detach().numpy(),anealing_weight,process_type])
        logger.add_record_list([epochs, vae_loss.cpu().detach().numpy(), vecloss.cpu().detach().numpy(), kl_loss.cpu().detach().numpy(),
             pre_loss.cpu().detach().numpy(), process_type])

    else:
        loss = None
    return loss


'''========================End of loss module=========================='''


'''Module for visualization'''
@gif.frame
def plot3D(xi, yi, zi, c, label, title, n_min, n_max):
    '''
    :param xi,yi,zi: input data
    :param c: property value
    :param label: property name
    :param title: id of epochs
    :param n_min: min value of input property
    :param n_max: max value of input property

    To save animation to Gif:(ex.)
    logp_frames=[]
        #Training process start
     logp_frames.append(
            plot3D(plot_vector[:, 0], plot_vector[:, 1], plot_vector[:, 2], pre_plot[:, 0],
                   "logP", str(epochs), -7, 9))
        # End of the training process
     gif.save(logp_frames, 'logp.gif', duration=150)

    '''
    data = pd.DataFrame(
        {'x': xi, 'y': yi, 'z': zi, 'c': c})
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    p = ax.scatter(data.x, data.y, data.z, c=data.c, alpha=0.6, s=10, cmap='YlGnBu', vmin=n_min, vmax=n_max)
    ax.view_init(0, 45)
    plt.colorbar(p, label=label)
    # plt.axis('off')
    plt.title('Epoch=' + title)

'''========================End of visualization module=========================='''


'''Train & Test func.'''

def train(epochs,train_set_final, valid_set_final,logger):
    model.train()
    train_loss = 0
    valid_loss = 0
    if params['do_prop_pred']:
        for batch_idx, (data,label) in enumerate(train_set_final):
            data = data.to(dtype=torch.float32, device=device)
            label=label.to(dtype=torch.float32, device=device)
            optimizer.zero_grad()
            output, mean, logvar,pre,z = model(data)
            pre_loss_ca=nn.MSELoss()
            pre_loss=pre_loss_ca(pre,label)
            loss=calculate_loss_by_type(output,data, mean, logvar,pre_loss,epochs,'train',logger)
            loss.backward()
            train_loss += loss
            optimizer.step()
            if batch_idx % len(train_set_final) == 0:
                with torch.no_grad():
                    for data in valid_set_final:
                        valid_smidata, valid_labels = data
                        valid_output, valid_mean, valid_logvar, valid_pre, valid_z = model(valid_smidata.to(dtype=torch.float32, device=device))
                        valid_pre_loss_ca = nn.MSELoss()
                        valid_pre_loss = valid_pre_loss_ca(valid_pre, valid_labels.to(dtype=torch.float32, device=device))
                        valid_loss+= calculate_loss_by_type(valid_output, valid_smidata, valid_mean, valid_logvar,valid_pre_loss, epochs,'valid',logger)
                    if params["SAVE_WEIGHT"]:
                        torch.save(model.state_dict(), 'Weights/' + str(dt_string.replace("/", '_')) + str(epochs) + '_param.pth')

                    print(f'{epochs} / {batch_idx}\t{loss:.4f}')
        return (train_loss.cpu().detach().numpy() / len(train_set_final)), (valid_loss.cpu().detach().numpy() / len(valid_set_final))

def test(test_set,epochs,logger):
    dataset=torch.utils.data.DataLoader(test_set, batch_size=params['batch_size'])
    test_loss=0
    with torch.no_grad():
        for data in dataset:
            test_smidata, test_labels = data
            test_output, test_mean, test_logvar, test_pre, test_z = model(test_smidata.to(dtype=torch.float32, device=device))
            test_pre_loss_ca = nn.MSELoss()
            test_pre_loss = test_pre_loss_ca(test_pre, test_labels.to(dtype=torch.float32, device=device))
            test_loss+=calculate_loss_by_type(test_output,test_smidata,test_mean,test_logvar,test_pre_loss,epochs,'test',logger)
            mae = nn.L1Loss()
            mae_logP = mae(test_labels [:, 0].to(device), test_pre[:, 0].to(device)).cpu().detach().numpy()
            mae_qed = mae(test_labels [:, 1].to(device), test_pre[:, 1].to(device)).cpu().detach().numpy()
            mae_SAS = mae(test_labels [:, 2].to(device), test_pre[:, 2].to(device)).cpu().detach().numpy()

            # performance on separate prediction tasks [epoch,'logP_pre', 'qed_pre', 'SAS_pre', 'mae_logP', 'mae_qed', 'mae_SAS']
            logger.add_separate_prediction([epochs,torch.mean((test_pre[:, 0])).cpu().detach().numpy(), torch.mean((test_pre[:, 1])).cpu().detach().numpy(),
                 torch.mean((test_pre[:, 2])).cpu().detach().numpy(), mae_logP, mae_qed,
                 mae_SAS])

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
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    logger=LogFile()

    for epoch in range(1, epochs + 1):
        training_data_dic=load_data_by_task(params)
        train_set_final=training_data_dic['train_set_final'].to_numpy()
        train_loss, valid_loss = train(epoch, training_data_dic['train_set_final'].tolist(), training_data_dic['valid_set_final'].tolist(),logger)
        test_loss = test(training_data_dic['test_dataset'].tolist(), epoch,logger)
        #print([epoch, train_loss, valid_loss, test_loss])

    logger.save_log(params)

