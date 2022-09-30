import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import argparse
import os

import torch.optim as optim


import torch

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import hyperparameters

from dataloader import load_dataset
import torch.utils.data as data
from pytorch_lightning import seed_everything

from pytorch_lightning.loggers import CSVLogger
import gif
from datetime import datetime
import matplotlib.pyplot as plt
# dd/mm/YY H:M:S
dt_string = datetime.utcnow().strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)
class MolecularVAE(pl.LightningModule):
    def __init__(self):
        super(MolecularVAE, self).__init__()
        self.conv_1 = nn.Conv1d(120, 9, kernel_size=9)
        self.conv_2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv_3 = nn.Conv1d(9, 70, kernel_size=8)
        self.linear_0 = nn.Linear(910, 435)
        self.linear_1 = nn.Linear(435, 292)
        self.linear_2 = nn.Linear(435, 292)
        self.linear_3 = nn.Linear(292, 292)
        self.linear_p1 = nn.Linear(292, 100)
        self.linear_p2 = nn.Linear(100, 3)
        self.visualize_z=nn.Linear(3,2)
        self.gru = nn.GRU(292, 501, 3, batch_first=True)
        self.linear_4 = nn.Linear(501, 36)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.logp_frames = []
        self.qed_frames = []
        self.sas_frames = []
        self.plot_vector = []
        self.pre_plot = []

    def encode(self, x):
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_3(x))
        x = x.view(x.size(0), -1)
        x = F.selu(self.linear_0(x))
        return self.linear_1(x), self.linear_2(x)

    def sampling(self, z_mean, z_logvar):
        epsilon = 1e-2 * torch.randn_like(z_logvar)
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean

    def decode(self, z):
        z = F.selu(self.linear_3(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, 120, 1)
        output, hn = self.gru(z)
        out_reshape = output.contiguous().view(-1, output.size(-1))
        y0 = F.softmax(self.linear_4(out_reshape), dim=1)
        y = y0.contiguous().view(output.size(0), -1, y0.size(-1))

        return y
    def predict_properties(self,z):
        z = F.selu(self.linear_3(z))
        z=F.selu(self.linear_p1(z))
        z=self.linear_p2(z)
        return z

    def vae_loss(self,x_decoded_mean, x, z_mean, z_logvar):
        xent_loss = nn.MSELoss()
        vecloss = xent_loss(x_decoded_mean.to(dtype=torch.float32),
                            x.to(dtype=torch.float32))
        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        return vecloss + kl_loss

    def vae_loss_with_annealing(self,x_decoded_mean, x, z_mean, z_logvar, epoch):
        xent_loss = nn.MSELoss()
        vecloss = xent_loss(x_decoded_mean.to(dtype=torch.float32),
                            x.to(dtype=torch.float32))
        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

        if epoch >= params['vae_annealer_start']:
            return vecloss + self.frange_cycle_sigmoid(0.0, 1.0, epoch, 4, 0.5)[0] * kl_loss
        else:
            return vecloss + kl_loss

    '''============cyclical_annealing for KL vanishing problem==================
    from paper: Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing (NAACL 2019)
    paper link:https://arxiv.org/abs/1903.10145
    Github:https://github.com/haofuml/cyclical_annealing

    KL vanishing:https://alibabatech.medium.com/next-gen-text-generation-alibaba-makes-progress-on-the-kl-vanishing-problem-77ec35f2afa2
    '''

    def frange_cycle_linear(self,start, stop, n_epoch, n_cycle=4, ratio=0.5):
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

    def frange_cycle_sigmoid(self,start, stop, n_epoch, n_cycle=4, ratio=0.5):
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

    def frange_cycle_cosine(self,start, stop, n_epoch, n_cycle=4, ratio=0.5):
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

    def calculate_loss_by_type(self,decoder_output_data, input_string_data, z_mean, z_logvar, property_prediction_loss,
                               epochs):
        '''To calculate the loss value '''
        pre_loss = property_prediction_loss
        loss_type = params['loss_type']
        if loss_type == "Variance_only":
            loss = self.vae_loss(decoder_output_data, input_string_data, z_mean, z_logvar)
        elif loss_type == "pro_prediction_only":
            loss = -pre_loss
        elif loss_type == "vae_pre_no_annealing":
            loss = self.vae_loss(decoder_output_data, input_string_data, z_mean, z_logvar) + pre_loss
        elif loss_type == "vae_pre_with_annealing":
            loss = self.vae_loss_with_annealing(decoder_output_data, input_string_data, z_mean, z_logvar, epochs) + pre_loss
        else:
            loss = None
        return loss

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.sampling(z_mean, z_logvar)
        return self.decode(z), z_mean, z_logvar,self.predict_properties(z),z
    def training_step(self,batch,batch_idx):
        x,y=batch

        output, mean, logvar, pre, z = self(x.to(dtype=torch.float32))
        pre_loss_ca = nn.MSELoss()
        pre_loss = pre_loss_ca(pre, y.to(dtype=torch.float32))
        loss = self.calculate_loss_by_type(output, x, mean, logvar, pre_loss, self.current_epoch)

        self.log("train_loss", loss)
        return {"loss":loss}
    def validation_step(self,batch,batch_idx):
        x,y=batch

        output, mean, logvar, pre, z = self(x.to(dtype=torch.float32))
        pre_loss_ca = nn.MSELoss()
        pre_loss = pre_loss_ca(pre, y.to(dtype=torch.float32))
        val_loss = self.calculate_loss_by_type(output, x, mean, logvar, pre_loss, self.current_epoch)
        self.plot_vector.extend(z.cpu().detach().numpy())
        self.pre_plot.extend(pre.cpu().detach().numpy())
        '''===For plot visualization==='''
        if batch_idx==0:
            plot_vector = np.array(self.plot_vector)
            pre_plot = np.array(self.pre_plot)
            self.logp_frames.append(
                self.plot3D(plot_vector[:, 0], plot_vector[:, 1], plot_vector[:, 2],pre_plot[:, 0],
                       "logP", str(self.current_epoch), -7, 9))
            self.qed_frames.append(
                self.plot3D(plot_vector[:, 0],plot_vector[:, 1], plot_vector[:, 2], pre_plot[:, 1],
                       "QED", str(self.current_epoch), 0, 1))
            self.sas_frames.append(
                self.plot3D(plot_vector[:, 0],plot_vector[:, 1],plot_vector[:, 2], pre_plot[:, 2],
                       "SAS", str(self.current_epoch), 1, 8))
            self.plot_vector=[]
            self.pre_plot=[]

        self.log("val_loss", val_loss)
        return {"val_loss":val_loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        print(np.shape(x))
        output, mean, logvar, pre, z = self(x.to(dtype=torch.float32))
        pre_loss_ca = nn.MSELoss()
        pre_loss = pre_loss_ca(pre, y.to(dtype=torch.float32))
        test_loss = self.calculate_loss_by_type(output, x, mean, logvar, pre_loss, self.current_epoch)
        self.log("test_loss", test_loss)
        return {"test_loss": test_loss}

    @gif.frame
    def plot3D(self,xi, yi, z, c, label, title, n_min, n_max):
        data = pd.DataFrame(
            {'x': xi, 'y': yi, 'z': z, 'c': c})
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        p = ax.scatter(data.x, data.y, data.z, c=data.c, alpha=0.6, s=10, cmap='YlGnBu', vmin=n_min, vmax=n_max)
        ax.view_init(0, 45)
        plt.colorbar(p, label=label)
        # plt.axis('off')
        plt.title('Epoch=' + title)
    def save_gif(self):
        gif.save(self.logp_frames, 'logp.gif', duration=150)
        gif.save(self.qed_frames, 'qed.gif', duration=150)
        gif.save(self.sas_frames, 'sas.gif', duration=150)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer





'''This is for setting a custom dataset for pytorch Dataloder
    ref.:https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
'''
class CustomMoleculeDataset(torch.utils.data.Dataset):
    def __init__(self,smiles_string,properties_label):
        self.smiles_string= smiles_string
        self.properties_label = properties_label
    def __len__(self):
        return len(self.smiles_string)
    def __getitem__(self, idx):
        return self.smiles_string[idx], self.properties_label[idx]


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
    train_dataset = CustomMoleculeDataset(X_train, Y_train)
    train_set_final, valid_set_final = split_validation_dataset(train_dataset, percentage_train=0.9)
    test_dataset = CustomMoleculeDataset(X_test, Y_test)
    test_set_final, _ = split_validation_dataset(test_dataset, percentage_train=1)
    return train_set_final, valid_set_final, test_set_final


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
    return train_set_final, valid_set_final, test_dataset


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
    vae = MolecularVAE()
    logger = CSVLogger("logs", name=dt_string.replace('/','_'))
    seed_everything(params['RAND_SEED'], workers=True)
    train_set_final, valid_set_final, test_dataset = main_property_run(params)
    #logger = TensorBoardLogger("tb_logs", name="my_model")
    #trainer=pl.Trainer(fast_dev_run=True)
    trainer = pl.Trainer(logger=logger,max_epochs=3,devices=1,accelerator="auto",weights_save_path="Weights")

    trainer.fit(vae,train_dataloaders=train_set_final,val_dataloaders=valid_set_final)
    vae.save_gif()