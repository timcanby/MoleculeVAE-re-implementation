
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch



class MolecularVAE(nn.Module):
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

    def forward(self, x):
        '''

        :param x:
        :return: output of decoder,mean vector,variance vector,property predict result ,sampled vector from latent space
        '''
        z_mean, z_logvar = self.encode(x)
        z = self.sampling(z_mean, z_logvar)
        return self.decode(z), z_mean, z_logvar,self.predict_properties(z),z



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

