from rdkit import rdBase, Chem

print(rdBase.rdkitVersion)
import pandas as pd
import os
import sys
import hyperparameters

sys.path.append(os.path.join(Chem.RDConfig.RDContribDir, "SA_Score"))
import torch
import torch.nn.functional as F
import torch.utils.data
import sascorer
import argparse
import json
import collections
import numpy as np
from sklearn.model_selection import train_test_split
from rdkit.Chem import QED
import deepchem as dc
import ast

'''
**Constant:ALL_SMILES_CHARACTERS
Unlike dictionary used by https://github.com/aksub99/molecular-vae 

Authors mentioned their SMILES-based text encoding used a subset of
35 different characters for ZINC and 22 different characters for QM9. 
Actually there are more than 24 characters in QM9 Dataset.

In Update version using 35 characters
'''
ALL_SMILES_CHARACTERS = ["S", "B", "\n", "P", "3", "s",
                        "O", "=", ")", "l", "r", "[",
                        "5", "2", "\\", "F", "]", "1",
                        "N", "8", "-", "c", "4", "6",
                        "#", "n", "/", "+", "o", "I",
                        "C", "H", "@", "(", "7", "."]

'''
Read in data:
This project provides two ways to read files, one is to read from a local file(.csv),
the other is to read from Molecule Net (https://moleculenet.org/)
From local:load_dataset_from_file()
From MoleculeNet:load_dataset_from_deepchem():

Use load_dataset() to load them
'''
"""
Read in data:
This project provides two ways to read files, one is to read from a local file(.csv),
the other is to read from Molecule Net (https://moleculenet.org/)
From local:load_dataset_from_file()
From MoleculeNet:load_dataset_from_deepchem():

Use load_dataset() to load them
"""


def load_dataset_from_file(data_path):
    """Loads dataset from local file.
    - Inputs:
        -path: 'Data/250k_rndm_zinc_drugs_clean_3.csv' or user-defined
    - Outputs:
        - dataset: type=pandas object
    """
    return pd.read_csv(data_path)


def load_dataset_from_deepchem(featurizer):
    """Loads dataset from deepcm Default QM9.
    -
        -featurizer: whether use dataset from moleculeNet type=Boolean value default=True
    - Outputs:
        - dataset: type=list
    """
    _, datasets, _ = dc.molnet.load_qm9(featurizer=featurizer)
    return datasets


def load_dataset(params):
    """Loads dataset, either from deepchem or from local file system.
    - Inputs:
        -dataset:path to local dataset
        -from_deepchem: whether use dataset from moleculeNet type=Boolean value default=True
        -featurizer: default='ECFP'
    - Outputs:
        - dataset: type=list
    """
    """This is to load dataset for training"""
    if params["FromDeepchem"]:
        datasets = load_dataset_from_deepchem(params["featurizer"])
        train_dataset, valid_dataset, test_dataset = datasets
        """train_dataset.ids=smile string of molecule (ref:https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html)"""
        smi_onehot_train, character_index_lookup_dict = one_hot_encoder(
            train_dataset.ids, params["normalizeSize"]
        )
        smi_onehot_test, character_index_lookup_dict = one_hot_encoder(
            test_dataset.ids, params["normalizeSize"]
        )
        if params["do_prop_pred"]:
            X_train, Y_train = extract_properties_value(
                smi_onehot_train, train_dataset.ids
            )
            X_test, Y_test = extract_properties_value(smi_onehot_test, test_dataset.ids)
            return X_train, X_test, Y_train, Y_test, character_index_lookup_dict
        else:
            return smi_onehot_train, smi_onehot_test, character_index_lookup_dict
    else:
        datasets = load_dataset_from_file(data_path=params["Data_file"])
        train_docs = datasets["smiles"].tolist()
        logP = datasets["logP"].tolist()
        QED = datasets["qed"].tolist()
        SAS = datasets["SAS"].tolist()
        train_docsy = np.stack((logP, QED, SAS), axis=-1)
        X_train_s, X_test_s, Y_train, Y_test = train_test_split(train_docs, train_docsy)
        X_train, character_index_lookup_dict = one_hot_encoder(
            X_train_s, params["normalizeSize"]
        )
        X_test, character_index_lookup_dict = one_hot_encoder(
            X_test_s, params["normalizeSize"]
        )
        if params["do_prop_pred"]:

            return X_train, X_test, Y_train, Y_test, character_index_lookup_dict
        else:
            return X_train, X_test, character_index_lookup_dict



def one_hot_encoder(smile_data, normalize_size):
    """This is one hot embedding function
    - Inputs:
            -smile_data:list of smile string
            -normalize_size: the maximum number of characters in a single smile string
        - Outputs:
            - one_hot_feature: one hot vector
            - character_index_lookup_dict:"""
    character_index_lookup_dict = load_character_index_lookup_dict()
    one_hot_feature = [
        make_encoding_for_smiles(vec, character_index_lookup_dict, normalize_size)
        for vec in smile_data
    ]
    return one_hot_feature, character_index_lookup_dict


def load_character_index_lookup_dict():
    """This is a dictionary that maps character to a unique index"""
    character_index_lookup_dict = collections.OrderedDict(
        [(a, i) for i, a in enumerate(ALL_SMILES_CHARACTERS)])
    return character_index_lookup_dict


def make_encoding_for_smiles(string_list, character_index_lookup_dict, normalize_size):
    """This is to convert string to one hot vector
    - Inputs:
        -string_list:list of smile string
        -character_index_lookup_dict: a dictionary that maps character to a unique index
        -normalize_size: the maximum number of characters in a single smile string
    - Outputs:
        - onehot_out: one hot vector
    """

    index_list = list(map(lambda a: character_index_lookup_dict[a], [*string_list]))
    num_classes = len(character_index_lookup_dict)
    one_hot_vector = torch.nn.functional.one_hot(
        torch.tensor(index_list), num_classes=num_classes
    )
    padding_size = (
        0,
        0,
        0,
        normalize_size - len(string_list),
    )  # pad last dim by 1 on each side
    onehot_out = F.pad(one_hot_vector, padding_size, "constant", 0)
    """output shape=[normalize_size,len(character_index_lookup_dict)]"""
    return onehot_out



def caculate_target_values(smiles):
    """Calculates logP, QED and SAS for a smiles input;
    - Inputs:
        - smiles:
          str. The smiles of the molecule for which we want to caluclate the values.
    - Returns:
        - logP:
          water−octanol partition coefficient (logP) ref.
        - QED
          synthetic accessibility score (SAS)
        - SAS
          Estimation of Drug-likeness (QED)
    If the molecule cannot be calculated by RDkit, returns error
    These values calculated here sometimes disagree with the values given in the dataset from the original
    dataset.
    """
    m = Chem.MolFromSmiles(smiles)
    logp = Chem.Descriptors.MolLogP(m)
    qed = QED.default(m)
    sas = sascorer.calculateScore(m)

    return logp, qed, sas


def extract_properties_value(smile_onehot, smile_string):
    """Extract properties value for latent space prediction
    - Inputs:
        - smile_onehot: one hot vector for training.
        - smile_string: corresponding smile string.
    - Returns:
        - X_final: one hot vector which has the corresponding properties value
        - Y_values:properties value
    """
    X_final = []
    Y_values = []
    label = []
    if os.path.exists("properties_value.json"):
        with open("properties_value.json", "r", encoding="utf-8") as json_file:
            properties_dic = ast.literal_eval(json.load(json_file))

        for x, s_string in zip(smile_onehot, smile_string):
            if str(s_string) in properties_dic:
                y = properties_dic[str(s_string)]
            else:
                try:
                    y = caculate_target_values(s_string)
                    temp_result = {str(s_string): y}
                    properties_dic.update(temp_result)
                except:
                    continue

            X_final.append(x)
            Y_values.append(y)
            label.append(s_string)
    else:
        for x, s_string in zip(smile_onehot, smile_string):
            try:
                y = caculate_target_values(s_string)
            except:
                continue
            X_final.append(x)
            Y_values.append(y)
            label.append(s_string)
            properties_dic = dict(zip(label, Y_values))
    """creat & update properties_value.json"""
    with open("properties_value.json", "w") as outfile:
        json.dump(str(properties_dic), outfile)
    return X_final, Y_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp_file", help="experiment file", default="expParam.json"
    )
    parser.add_argument("-d", "--directory", help="exp directory", default=None)
    args = vars(parser.parse_args())
    if args["directory"] is not None:
        args["exp_file"] = os.path.join(args["directory"], args["exp_file"])

    params = hyperparameters.load_params(args["exp_file"])
    '''For checking and update the parameters'''