
import numpy as np
from rdkit import rdBase, Chem
import pandas as pd
import os
import hyperparameters
import torch.utils.data
import argparse
import torch
from sklearn.decomposition import IncrementalPCA
from model import MolecularVAE
from collections import OrderedDict
import matplotlib.pyplot as plt
from dataloader import one_hot_decoder, load_character_index_lookup_dict
from dataloader import one_hot_encoder, caculate_target_values

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MolecularVAE().to(device)
model_path = 'Weights1/23_10_2022 07:27:49150_param.pth'
state_dict = torch.load(model_path)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]  # remove `module.`
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)


def plot(xi, yi, c, label):
    data = pd.DataFrame(
        {'x': xi, 'y': yi, 'c': c})
    cm = plt.cm.get_cmap('RdYlBu')
    p = plt.scatter(data.x, data.y, c=data.c, alpha=0.5, s=3)
    plt.colorbar(p, label=label)
    plt.axis('off')
    plt.savefig(label + '.png')
    plt.clf()


def plot_latent_space(data, n_sample):
    '''

    :param data: samples from latent space size=(n_sample,292)
    :param n_sample: number of samples
    :return: none ( generated SAS.png qed.png logP.png)
    '''

    visual_data = model.predict_properties(data.to(device))
    pca = IncrementalPCA(n_components=2)
    a = torch.nan_to_num(visual_data).cpu().detach().numpy()
    visualize_z = pca.fit_transform(visual_data[~np.isnan(a)].cpu().detach().numpy().reshape(n_sample, -1))
    normalized_z = visualize_z / np.linalg.norm(visualize_z)
    plot(normalized_z[:, 0], normalized_z[:, 1], visual_data[:, 0].cpu().detach().numpy(), 'logP')
    plot(normalized_z[:, 0], normalized_z[:, 1], visual_data[:, 1].cpu().detach().numpy(), 'qed')
    plot(normalized_z[:, 0], normalized_z[:, 1], visual_data[:, 2].cpu().detach().numpy(), 'SAS')


def search_smile(string):
    '''
    :param string: input generated string, because
    :return: list of possible molecular
    '''
    stringlist = []
    for id in range(0, len(string)):
        m = Chem.MolFromSmiles(string[:id], sanitize=True)
        if m is None:
            print('invalid SMILES')
        else:
            try:
                Chem.SanitizeMol(m)
                stringlist.append(string[:id])
            except:
                print('invalid chemistry')
    return stringlist[1:]


def generate_sample(data, n_search, params):
    '''

    :param data: samples from latent space
    :param n_search: search from n generated string
    :param params: params['normalizeSize']
    :return: _Ganerated_mol.csv generated
    '''
    dic = load_character_index_lookup_dict()
    random = data[torch.randint(len(data), (n_search,))]
    test_list = one_hot_decoder(model.decode(random.to(device)).cpu().detach().numpy(), dic)
    mol = []
    for each in test_list:
        mol.extend(search_smile(each))

    generated_mol = list(set(mol))
    one_hot_feature, character_index_lookup_dict = one_hot_encoder(generated_mol, params['normalizeSize'])
    data = torch.tensor(np.array([i.cpu().detach().numpy() for i in one_hot_feature]))
    output, mean, logvar, pre, z = model(data.to(dtype=torch.float32, device=device))

    G = []
    for each in list(set(mol)):
        G.append(caculate_target_values(each))
    data_tuples = list(zip(generated_mol, pre.cpu().detach().numpy(), G))
    print(data_tuples)
    pd.DataFrame(data_tuples, columns=['SMILE_mol', 'properties_prediction_logP_qed_SAS', 'ground_truth']).to_csv(
        '' + str(model_path.replace("/", '_')) + '_Ganerated_mol.csv')


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
    n_sample = 10000
    data = 1e-2 * torch.randn((n_sample, 292))
    n_generate = 1000
    plot_latent_space(data, n_sample)
    generate_sample(data, n_generate, params)